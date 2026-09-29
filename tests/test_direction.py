"""PHASE 13 direction engine: metrics, calibration, and the abstain gate (planted signal vs pure noise)."""
import numpy as np
import pandas as pd
import pytest

from engine import direction as D

NOW = pd.Timestamp("2024-06-28")


def data(seed=0, n_dates=200, per=8, strength=3.0, informative=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=NOW - pd.Timedelta(days=30), periods=n_dates)
    ix = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(per)]], names=["date", "ticker"])
    n = len(ix)
    latent = rng.normal(size=n)
    up = (rng.random(n) < 1 / (1 + np.exp(-strength * latent))).astype(float)
    if not informative:
        latent = rng.normal(size=n)
    F = D.build_inputs(pattern=pd.Series(latent + rng.normal(0, .3, n), index=ix),
                       analog=pd.Series(np.clip(0.5 + 0.2 * latent + rng.normal(0, .1, n), 0, 1), index=ix),
                       trust=pd.Series(1.0, index=ix),
                       model=pd.Series(latent * 0.5 + rng.normal(0, .3, n), index=ix), index=ix)
    return F, pd.Series(up, index=ix)


def test_metrics_known_values():
    p, y = np.array([0.9, 0.1, 0.5]), np.array([1, 0, 1])
    assert D.brier(p, y) == pytest.approx((0.01 + 0.01 + 0.25) / 3)
    assert D.log_loss(p, y) == pytest.approx(-(np.log(.9) + np.log(.9) + np.log(.5)) / 3)
    assert D.log_loss([0.0], [1]) < 20 and np.isfinite(D.log_loss([0.0], [1]))     # clipped, never inf
    lo, hi = D.wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1 and D.wilson(0, 0) == (0.0, 1.0)
    rb = D.reliability_bins(np.array([.05, .06, .95, .96]), np.array([0, 0, 1, 1]), 10)
    assert list(rb["n"]) == [2, 2] and rb["obs_freq"].tolist() == [0.0, 1.0]
    assert D.ece(np.array([.5] * 100), np.array([1, 0] * 50)) == pytest.approx(0.0)
    assert D.ece(np.array([.9] * 100), np.array([1, 0] * 50)) == pytest.approx(0.4)
    assert "<svg" in D.reliability_svg(rb) and D.reliability_bins([], []).empty


def test_planted_signal_opens_gate_and_bets_are_right():
    F, up = data(strength=4.0)
    eng = D.DirectionEngine().fit(F, up, NOW)
    assert eng.open, eng.report()
    t = eng.diag["test"]
    assert t["brier"] < t["brier_base"] and t["acc_gated"] >= 0.8
    out = eng.decide(F)
    assert out["bet"].any() and (out.loc[out["bet"], "conf"] >= 0.8).all()
    assert (out.loc[~out["bet"], "side"] == 0).all()
    late = out.index.get_level_values(0) >= pd.Timestamp(eng.diag["test_start"])
    bets = out[late & out["bet"].values]
    hit = ((bets["side"] > 0) == (up[bets.index] > 0.5)).mean()
    assert hit >= 0.75


def test_noise_abstains_everywhere():
    F, up = data(informative=False, strength=3.0)
    eng = D.DirectionEngine().fit(F, up, NOW)
    assert not eng.open
    out = eng.decide(F)
    assert not out["bet"].any() and out["reason"].str.startswith("engine_closed").all()


def test_weak_signal_never_reaches_gate():
    F, up = data(strength=0.6)               # real but weak: can't honestly reach 80%
    eng = D.DirectionEngine().fit(F, up, NOW)
    assert not eng.decide(F)["bet"].any()
    assert "NOT" in eng.report() or not eng.open


def test_overconfident_input_is_calibrated():
    F, up = data(seed=3, strength=1.5)
    F["analog"] = np.where(F["pattern"] > 0, 0.99, 0.01)      # wildly overconfident direction call
    eng = D.DirectionEngine().fit(F, up, NOW)
    p = eng.predict(F)
    assert p.max() <= 0.98 and p.min() >= 0.02
    assert eng.diag["test"]["ece"] < 0.5 * D.ece(F["analog"].values, up.values)   # far better than the raw claim


def test_zero_trust_neutralizes_pattern():
    F, up = data(strength=4.0)
    F = F.assign(analog=np.nan, model=np.nan)
    eng = D.DirectionEngine().fit(F.assign(trust=0.0), up, NOW)
    assert not eng.open
    assert not eng.decide(F.assign(trust=0.0))["bet"].any()


def test_non_movers_and_unclosed_windows_ignored():
    F, up = data(strength=4.0)
    movers = pd.Series(True, index=F.index)
    movers.iloc[::2] = False
    poisoned = up.copy()
    poisoned.iloc[::2] = 1 - poisoned.iloc[::2]            # nonsense labels on non-movers must not matter
    a = D.DirectionEngine().fit(F, up, NOW, movers)
    b = D.DirectionEngine().fit(F, poisoned, NOW, movers)
    assert a.diag["test"]["brier"] == pytest.approx(b.diag["test"]["brier"])
    early = D.DirectionEngine().fit(F, up, F.index.get_level_values(0)[0] + pd.Timedelta(days=3))
    assert early.diag["n_rows"] < 300 and not early.open


def test_degenerate_inputs():
    F, up = data(n_dates=5)
    eng = D.DirectionEngine().fit(F, up, NOW)
    assert not eng.open and "insufficient" in eng.reason
    out = eng.decide(F)
    assert len(out) == len(F) and not out["bet"].any()
    assert "NOT FITTED" in eng.report()
    e2 = D.DirectionEngine().fit(F.iloc[:0], up.iloc[:0], NOW)
    assert not e2.open
    with pytest.raises(RuntimeError):
        e2.predict(F)
    with pytest.raises(ValueError):
        D.build_inputs()


def test_too_few_inputs_abstains_even_when_open():
    F, up = data(strength=4.0)
    eng = D.DirectionEngine().fit(F, up, NOW)
    assert eng.open
    sparse = F.assign(analog=np.nan, model=np.nan, mw=np.nan, evidence=np.nan)
    out = eng.decide(sparse)
    assert not out["bet"].any() and (out["reason"] == "too_few_inputs").all()
