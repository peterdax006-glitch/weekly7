"""Timeline dial (Phase 17): bounds, no calendar, no look-ahead, and a planted volatility-clustering effect it must exploit."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from engine import timeline as T, objective as O


def make_windows(n, seed, sig=(0.04, 0.06, 0.16), block=13, weeks=52, mu=0.01):
    """Weekly series whose volatility switches by block (persistent regimes) - the effect a pacing dial can use."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = np.repeat(rng.choice(sig, size=weeks // block), block)
        out.append({"id": f"w{i}", "end": i, "weeks": mu + s * rng.standard_normal(len(s))})
    return out


def test_outputs_inside_bounds_for_extreme_inputs():
    p = T.DialParams()
    st = T.DialState()
    rng = np.random.default_rng(0)
    hist = []
    for i in range(120):
        hist.append(float(rng.choice([-0.6, -0.2, 0.0, 0.3, 1.5])))
        out, st = T.step(hist, {"stress": float(rng.uniform(0, 4))}, {"mean": 0.3, "sd": 0.5}, st, p)
        assert p.k_bounds[0] <= out.k <= p.k_bounds[1]
        assert p.exposure_bounds[0] * p.brake_exposure - 1e-9 <= out.exposure <= p.exposure_bounds[1]
        assert p.pool_bounds[0] - 1e-9 <= out.pool_q <= p.pool_bounds[1] + 1e-9
        assert -1 <= out.aggr <= 1 and out.brake in (None, p.brake_level)


def test_empty_history_is_neutral_and_valid():
    out, st = T.step([], None, None)
    assert out.aggr == 0.0 and not out.brake_on and 1 <= out.k <= 6
    out2, _ = T.step(np.array([]), {"stress": 1.0}, {})
    assert out2.exposure == out.exposure


def test_calm_raises_and_overshoot_lowers_aggressiveness():
    st = T.DialState()
    for _ in range(6):
        calm, st = T.step([0.01] * 8, None, None, st)
    st = T.DialState()
    for _ in range(6):
        wild, st = T.step([0.15, -0.14, 0.18, -0.2, 0.16, -0.15, 0.17, -0.12], None, None, st)
    assert calm.aggr > 0.3 and calm.k < wild.k and wild.aggr < 0 and wild.brake is not None


def test_rate_limit_and_deadband():
    p = T.DialParams(max_step=0.2, deadband=0.05)
    out, st = T.step([0.0] * 8, None, None, T.DialState(), p)
    assert abs(out.aggr) <= 0.2 + 1e-12
    quiet = T.DialParams(deadband=0.9)
    o2, _ = T.step([0.0] * 8, None, None, T.DialState(0.1), quiet)
    assert o2.aggr == 0.1


def test_brake_trips_caps_and_releases():
    p = T.DialParams(brake_release=2)
    out, st = T.step([0.05, -0.2, -0.2], None, None, T.DialState(0.9), p)     # 36% drawdown
    assert out.brake_on and out.aggr <= -0.5 and out.exposure < p.exposure_bounds[0]
    hist = [0.05, -0.2, -0.2] + [0.0] * 12                                     # drawdown leaves the trailing window
    for i in range(3, len(hist) + 1):
        out, st = T.step(hist[:i], None, None, st, p)
    assert not out.brake_on


def test_progress_losing_does_not_gamble():
    losing, _ = T.step([-0.02] * 8, None, None, T.DialState())
    on_path, _ = T.step([0.07] * 8, None, None, T.DialState())
    assert losing.terms["prog"] < 0 <= on_path.terms["prog"] + 1e-9


def test_forecast_and_regime_move_the_dial():
    base, _ = T.step([0.03] * 6, None, None)
    hot, _ = T.step([0.03] * 6, {"stress": 2.0}, {"mean": 0.0, "sd": 0.2})
    cold, _ = T.step([0.03] * 6, {"stress": 0.6}, {"mean": 0.0, "sd": 0.01})
    assert cold.terms["goal"] > base.terms["goal"] > hot.terms["goal"]
    assert T.expected_abs_move(0, 0.1) == pytest.approx(0.1 * np.sqrt(2 / np.pi))
    rng = np.random.default_rng(1)
    assert T.expected_abs_move(0.05, 0.1) == pytest.approx(np.abs(rng.normal(0.05, 0.1, 400000)).mean(), rel=0.01)


def test_calendar_information_is_rejected():
    for bad in (dict(weeks=pd.Series([0.1], index=pd.date_range("2020-01-06", periods=1))),
                dict(weeks=[0.1], regime={"stress": 1.0, "as_of": dt.date(2020, 1, 1)}),
                dict(weeks=[0.1], forecast={"mean": pd.Timestamp("2020-01-01")}),
                dict(weeks=pd.DataFrame({"a": [0.1]}))):
        with pytest.raises(TypeError):
            T.step(**bad)


def test_decisions_do_not_depend_on_absolute_time_or_future():
    w = make_windows(1, 3)[0]["weeks"]
    a, _ = T.simulate(w)
    w2 = w.copy()
    w2[30:] = 5.0                                       # rewrite the future
    b, _ = T.simulate(w2)
    assert np.allclose(a[:30], b[:30])                  # nothing before week 30 changed: no look-ahead
    shifted = np.r_[w[:0], w]                           # same series again: identical (no hidden clock)
    assert np.array_equal(T.simulate(shifted)[0], a)


def test_dial_bounds_are_validated():
    for bad in (T.DialParams(k_bounds=(0, 3)), T.DialParams(exposure_bounds=(1.0, 0.5)), T.DialParams(target=0.9),
                T.DialParams(max_step=0)):
        with pytest.raises(ValueError):
            T.step([0.0], None, None, None, bad)


def test_planted_effect_dial_passes_gate_on_unseen_windows():
    wins = make_windows(24, 11, sig=(0.05, 0.2))          # regimes far enough apart that pacing has something to fix
    train, test = wins[:10], wins[10:]
    params, _ = T.train_dial(train, n_draws=25, seed=0)
    rep = T.promotion_gate(train, test, params)
    assert rep.passed, rep.summary()
    assert rep.numbers["gain_test"] > 0 and rep.numbers["boot_lo"] > 0


def test_gate_rejects_dial_when_no_effect_to_exploit():
    """Returns already in band every week (nothing to pace): the dial can only add noise, so the gate must fail."""
    rng = np.random.default_rng(5)
    wins = [{"id": i, "end": i, "weeks": rng.choice([-1, 1], 52) * rng.uniform(0.055, 0.095, 52)} for i in range(16)]
    # calibrate the proxy so the neutral dial is exactly neutral: leverage 1 at aggr 0
    kw = dict(k_ref=4)
    p = T.DialParams(exposure_bounds=(1.0, 1.0), k_bounds=(4, 4))
    assert T.promotion_gate(wins[:6], wins[6:], p, **kw).passed is False


def test_gate_fails_closed_and_checks_order():
    wins = make_windows(6, 2)
    assert not T.promotion_gate([], wins, T.DialParams()).passed
    assert not T.promotion_gate(wins[:5], wins[5:], T.DialParams()).passed                  # only 1 test window
    with pytest.raises(ValueError):
        T.promotion_gate(wins[3:], wins[:3], T.DialParams())                                 # train after test


def test_gate_catches_a_risk_deteriorating_dial():
    """Planted: a dial pinned at max leverage improves nothing safely - on wild windows it must trip the risk check."""
    wins = make_windows(18, 4, sig=(0.16,))
    p = T.DialParams(k_bounds=(1, 1), exposure_bounds=(1.0, 1.0))                            # always k=1, full exposure
    rep = T.promotion_gate(wins[:6], wins[6:], p)
    assert not rep.passed and (not rep.checks["risk"] or not rep.checks["improvement"])


def test_train_dial_is_deterministic_and_requires_data():
    wins = make_windows(6, 8)
    assert T.train_dial(wins, 8, seed=1)[0] == T.train_dial(wins, 8, seed=1)[0]
    with pytest.raises(ValueError):
        T.train_dial([])


def test_cfg_overrides_keys():
    out, _ = T.step([0.0] * 5)
    assert set(T.cfg_overrides(out)) == {"k", "pool_q", "brake"}


def test_gate_flags_a_dial_that_only_works_in_one_era():
    """Planted: dial-friendly regimes only in the first third of test windows; the rest are neutral noise. The average
    gain can look fine, stability across eras must not."""
    good = make_windows(6, 21, sig=(0.05, 0.2))
    rng = np.random.default_rng(22)
    flat = [{"id": f"f{i}", "end": 100 + i, "weeks": rng.choice([-1, 1], 52) * rng.uniform(0.055, 0.095, 52)} for i in range(12)]
    train = make_windows(8, 23, sig=(0.05, 0.2))
    for i, w in enumerate(train):
        w["end"] = -50 + i
    rep = T.promotion_gate(train, good + flat, T.DialParams())
    assert not rep.checks["stable_eras"] and not rep.passed


def test_step_is_pure():
    """Same inputs, same state, same output; the state object is not mutated."""
    st = T.DialState(0.3, False, 1)
    a = T.step([0.05, -0.04, 0.1], {"stress": 1.4}, {"mean": 0.01, "sd": 0.09}, st)
    b = T.step([0.05, -0.04, 0.1], {"stress": 1.4}, {"mean": 0.01, "sd": 0.09}, st)
    assert a == b and st == T.DialState(0.3, False, 1)


def test_over_limit_forces_derisking_even_with_bullish_forecast():
    hist = [0.2, -0.18, 0.25, -0.2, 0.22, -0.19, 0.21, -0.17]
    out, _ = T.step(hist, {"stress": 0.5}, {"mean": 0.0, "sd": 0.001}, T.DialState(0.9))
    assert out.terms["goal"] <= -0.25 and out.aggr < 0.9


def test_brake_exposure_never_exceeds_normal_exposure():
    p = T.DialParams()
    for a0 in (-1.0, 0.0, 1.0):
        on, _ = T.step([0.1, -0.25, -0.1], None, None, T.DialState(a0), p)
        off, _ = T.step([0.0] * 3, None, None, T.DialState(a0), p)
        assert on.brake_on and on.exposure <= off.exposure + 1e-12


def test_simulate_proxy_leverage_scales_with_k():
    base = np.full(10, 0.05)
    lo, _ = T.simulate(base, T.DialParams(k_bounds=(1, 1), exposure_bounds=(1.0, 1.0)), k_ref=3)
    hi, _ = T.simulate(base, T.DialParams(k_bounds=(6, 6), exposure_bounds=(1.0, 1.0)), k_ref=3)
    assert (lo > hi).all() and lo[0] == pytest.approx(0.05 * np.sqrt(3))


def test_gate_rows_matches_promotion_gate_and_rejects_unpaired():
    wins = make_windows(24, 11, sig=(0.05, 0.2))
    train, test = wins[:10], wins[10:]
    p = T.DialParams()
    a = T.promotion_gate(train, test, p)
    b = T.gate_rows([O.week_row(w["weeks"]) for w in train], T._rows(train, p), [O.week_row(w["weeks"]) for w in test],
                    T._rows(test, p))
    assert (a.passed, a.checks) == (b.passed, b.checks) and a.numbers["gain_test"] == b.numbers["gain_test"]
    with pytest.raises(ValueError):
        T.gate_rows([O.week_row(w["weeks"]) for w in train], T._rows(train, p), [O.week_row(w["weeks"]) for w in test], [])
    assert not T.gate_rows([], [], [], []).passed
