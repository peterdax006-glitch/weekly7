"""Bible Phase 25 (A13/T17): planted-pattern calibration. Synthetic weeks x stocks with KNOWN effects; the miner must
find the real ones, reject the zero ones, keep the sign of the negative one, scope the regime one, and invent almost
nothing when outcomes are pure noise."""
import numpy as np
import pandas as pd
import pytest

from engine.patterns import PatternMiner

WEEKS, STOCKS, NF = 260, 220, 12


def make(seed=0, strong=0.012, weak=0.004, negative=-0.008, regime=0.010, noise_only=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=WEEKS * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(STOCKS)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), NF)), index=idx, columns=[f"f{i}" for i in range(NF)])
    # market context: a per-week regime value (like fear)
    reg = pd.Series(rng.standard_normal(WEEKS), index=dates)
    X["m_vix"] = reg.reindex(X.index.get_level_values(0)).values
    week_shock = rng.normal(0, 0.02, WEEKS)                        # stocks in a week move together
    y = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx) + np.repeat(week_shock, STOCKS)
    if not noise_only:
        q = X[[f"f{i}" for i in range(NF)]].groupby(level=0).rank(pct=True)
        y += strong * (q["f0"] >= 0.8)                              # strong: f0 top fifth
        y += weak * (q["f1"] >= 0.8)                                # weak: f1 top fifth
        y += negative * (q["f2"] >= 0.8)                            # negative: f2 top fifth
        y += regime * ((q["f3"] >= 0.8) & (X["m_vix"] > 0.43))     # only in the top-third regime
    y = y - y.groupby(level=0).transform("mean")
    return X, y


def active(M):
    P = M.patterns
    return P[P["status"].isin(["active", "rescoped"])]


def names(P):
    return set(P["key_named"])


@pytest.fixture(scope="module")
def mined():
    X, y = make()
    M = PatternMiner({"max_pairs": 300, "max_unless": 50, "null_reps": 2, "min_n": 200, "half_life_years": 50}).fit(
        X, y, now=X.index.get_level_values(0).max())
    return M


def test_strong_pattern_found(mined):
    A = active(mined)
    assert any(n.startswith("f0 q4") for n in names(A)), mined.report


def test_negative_pattern_keeps_its_sign(mined):
    A = active(mined)
    neg = A[A["key_named"].str.startswith("f2 q4")]
    assert len(neg) and (neg["effect"] < 0).all(), mined.report


def test_zero_features_not_active_alone(mined):
    A = active(mined)
    singles = [n for n in names(A) if " & " not in n]
    fake = [n for n in singles if n.split()[0] in {f"f{i}" for i in range(5, 12)}]
    assert len(fake) <= 1, fake                                     # at most one false single out of 35


def test_p_real_higher_for_real_than_fake(mined):
    P = mined.patterns
    real = P[P["key_named"] == "f0 q4"]["p_real"]
    fake = P[P["key_named"].str.match(r"^f(5|6|7|8|9|10|11) q\d$")]["p_real"]
    assert len(real) and real.iloc[0] > fake.median()


def test_noise_only_invents_almost_nothing():
    X, y = make(seed=3, noise_only=True)
    M = PatternMiner({"max_pairs": 300, "max_unless": 50, "null_reps": 2, "min_n": 200, "half_life_years": 50}).fit(
        X, y, now=X.index.get_level_values(0).max())
    assert len(active(M)) <= 2, M.report                            # hallucination control


def test_duplicate_feature_does_not_double_count():
    X, y = make(seed=5)
    X["f0_copy"] = X["f0"]
    M = PatternMiner({"max_pairs": 300, "max_unless": 50, "null_reps": 1, "min_n": 200, "half_life_years": 50}).fit(
        X, y, now=X.index.get_level_values(0).max())
    A = active(M)
    both = [n for n in names(A) if n in ("f0 q4", "f0_copy q4")]
    assert len(both) <= 1, both                                     # the copy is a duplicate, not new evidence
