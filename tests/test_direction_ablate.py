"""Ablation: a planted informative family must be found, a pure-noise family must not earn a 'helps'."""
import numpy as np
import pandas as pd
import pytest

from engine import direction as D, direction_ablate as A

NOW = pd.Timestamp("2024-06-28")


def data(seed=0, n_dates=700, per=10):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=NOW - pd.Timedelta(days=30), periods=n_dates)
    ix = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(per)]], names=["date", "ticker"])
    n = len(ix)
    a, b = rng.normal(size=n), rng.normal(size=n)                    # two independent real drivers
    up = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(1.2 * a + 1.2 * b)))).astype(float), index=ix)
    F = D.build_inputs(pattern=pd.Series(a + rng.normal(0, .3, n), index=ix), trust=pd.Series(1.0, index=ix),
                       model=pd.Series(b + rng.normal(0, .3, n), index=ix),
                       evidence=pd.Series(rng.normal(size=n), index=ix), index=ix)      # evidence is pure noise
    return F, up


@pytest.fixture(scope="module")
def result():
    F, up = data()
    return A.ablate(F, up, NOW, families=["pattern", "model", "evidence", "analog"], n_boot=800, seed=1)


def test_real_families_help_noise_family_does_not(result):
    r = result.set_index("family")
    assert r.loc["pattern", "verdict"] == "helps" and r.loc["pattern", "lo"] > 0
    assert r.loc["model", "verdict"] == "helps"
    assert r.loc["evidence", "verdict"] != "helps"
    assert abs(r.loc["evidence", "delta"]) < 0.25 * r.loc["pattern", "delta"]


def test_absent_family_is_no_effect(result):
    r = result.set_index("family").loc["analog"]
    assert r["delta"] == pytest.approx(0.0, abs=1e-12) and r["verdict"] == "no_detectable_effect"


def test_paired_rows_and_full_brier_consistent(result):
    assert result["brier_full"].nunique() == 1 and result["n"].nunique() == 1
    assert (result["n_weeks"] > 10).all()


def test_bootstrap_ci_covers_and_is_seeded():
    rng = np.random.default_rng(0)
    ix = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=400), ["a", "b"]])
    d = pd.Series(rng.normal(0.02, 0.1, len(ix)), index=ix)
    b1, b2 = A.week_bootstrap(d, 500, 3), A.week_bootstrap(d, 500, 3)
    assert b1 == b2 and b1["lo"] < 0.02 < b1["hi"]
    zero = A.week_bootstrap(d - d.mean(), 500, 3)
    assert zero["lo"] < 0 < zero["hi"]
    assert np.isnan(A.week_bootstrap(d.iloc[:2], 100, 0)["lo"])


def test_drop_family_semantics():
    F, _ = data(n_dates=20)
    assert A.drop_family(F, "model")["model"].isna().all()
    p = A.drop_family(F, "pattern")
    assert p["pattern"].isna().all() and (p["trust"] == 0).all()
    assert (A.drop_family(F, "trust")["trust"] == 1.0).all() and F["model"].notna().all()


def test_unfittable_engine_returns_empty():
    F, up = data(n_dates=5)
    assert A.ablate(F, up, NOW).empty
