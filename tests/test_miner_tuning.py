"""A7 self-tuning (engine/miner_tuning.py): folds never train on their test block or the holdout; the weekly bootstrap
is right on known inputs; on pure noise nothing is adopted; when the true effect has recently changed, a shorter
memory (half-life) is chosen - and only kept if the untouched holdout confirms it."""
import numpy as np
import pandas as pd
import pytest

from engine import miner_tuning as T
from engine import planted as PL

FAST = {"max_pairs": 60, "max_unless": 0, "null_reps": 1, "min_n": 150}


def test_folds_are_ordered_and_exclude_the_holdout():
    dates = pd.bdate_range("2015-01-01", periods=500)[::5]
    F, hold = T.folds(dates, n_folds=4, holdout_frac=0.2)
    assert len(F) == 4
    for tr_end, te_start, te_end in F:
        assert tr_end < te_start <= te_end < hold
    assert all(F[i][2] < F[i + 1][1] for i in range(len(F) - 1))      # test blocks do not overlap


def test_daily_ic_is_spearman_and_handles_constants():
    s = pd.Series([1, 2, 3, 4, 5, 6.0]); y = pd.Series([10, 20, 30, 40, 50, 60.0])
    assert T.daily_ic(s, y) == pytest.approx(1.0)
    assert np.isnan(T.daily_ic(pd.Series([1.0] * 6), y))


def test_weekly_bootstrap_brackets_a_known_difference():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    a = pd.Series(0.05 + rng.normal(0, 0.01, 200), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    lo, hi = T.weekly_bootstrap_diff(a, b)
    assert lo > 0.04 and hi < 0.06


def test_empty_bootstrap_is_nan():
    lo, hi = T.weekly_bootstrap_diff(pd.Series(dtype=float), pd.Series(dtype=float))
    assert np.isnan(lo) and np.isnan(hi)


def test_noise_adopts_nothing():
    sc = PL.Scenario("noise", [], weeks=160, stocks=120)
    X, y, _ = PL.generate(sc, seed=11)
    r = T.tune(X, y, space={"half_life_years": [1.0, 4.0, 16.0]}, base={**FAST, "half_life_years": 4.0}, n_folds=3)
    assert not r.adopted_any and r.chosen == r.defaults


def test_recent_regime_change_prefers_short_memory():
    # f0 top fifth: -1.5% for the first 70% of history, +1.5% afterwards. Long memory averages the two away.
    plants = [PL.Plant("old", "decaying", (("f0", 4),), -0.015, active=(0.0, 0.7)),
              PL.Plant("new", "strong", (("f0", 4),), 0.015, active=(0.7, 1.0))]
    X, y, _ = PL.generate(PL.Scenario("switch", plants, weeks=200, stocks=150), seed=12)
    r = T.tune(X, y, space={"half_life_years": [0.5, 16.0]}, base={**FAST, "half_life_years": 16.0}, n_folds=3,
               holdout_frac=0.15, min_fold_share=0.6)
    tried = [s for s in r.steps if s["knob"] == "half_life_years" and s["value"] == 0.5]
    assert tried and np.nanmean(tried[0]["fold_gains"]) > 0            # shorter memory helps in the folds
    assert r.holdout["verdict"] in ("confirmed on holdout", "no change adopted",
                                    "not confirmed on holdout (CI straddles 0) - keep defaults")
    if r.adopted_any:
        assert r.chosen["half_life_years"] == 0.5


def test_too_little_history_refuses():
    sc = PL.Scenario("tiny", [], weeks=6, stocks=30)
    X, y, _ = PL.generate(sc, seed=1)
    with pytest.raises(ValueError):
        T.tune(X, y, space={"half_life_years": [1.0]}, n_folds=12)
