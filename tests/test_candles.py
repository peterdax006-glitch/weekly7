"""A1 (canon C35, Bible Phase 32): candle and micro-signal features. Each signal is checked on a hand-built bar whose
answer is known, and the whole block is checked point-in-time: features up to day t must not change when later days
are appended (a feature that peeks ahead fails the truncation test)."""
import numpy as np
import pandas as pd
import pytest

from engine import candles


def bars(ohlc):
    """ohlc: list of (open, high, low, close) for one ticker 'A'; a second ticker 'B' is a flat control."""
    idx = pd.bdate_range("2021-01-04", periods=len(ohlc))
    a = np.array(ohlc, dtype=float)
    out = {}
    for j, k in enumerate(("Open", "High", "Low", "Close")):
        out[k] = pd.DataFrame({"A": a[:, j], "B": 10.0}, index=idx)
    return out


def rand_bars(n=160, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
    o = c * np.exp(rng.normal(0, 0.01, (n, k)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, (n, k))))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, (n, k))))
    cols = [f"T{i}" for i in range(k)]
    return {nm: pd.DataFrame(v, index=idx, columns=cols) for nm, v in (("Open", o), ("High", h), ("Low", l), ("Close", c))}


def test_full_green_body_is_plus_one():
    F = candles.build(bars([(10, 11, 9.5, 10.5), (10, 11, 10, 11)]))
    assert F["cd_body"]["A"].iloc[-1] == pytest.approx(1.0)          # opened at the low, closed at the high
    assert F["cd_upwick"]["A"].iloc[-1] == pytest.approx(0.0)
    assert F["cd_pos"]["A"].iloc[-1] == pytest.approx(1.0)


def test_full_red_body_is_minus_one():
    F = candles.build(bars([(10, 11, 9.5, 10.5), (11, 11, 10, 10)]))
    assert F["cd_body"]["A"].iloc[-1] == pytest.approx(-1.0)
    assert F["cd_pos"]["A"].iloc[-1] == pytest.approx(0.0)


def test_wicks_and_body_sum_to_one():
    B = rand_bars()
    F = candles.build(B)
    for tag in ("cd", "cw", "cm"):
        tot = (F[f"{tag}_body"].abs() + F[f"{tag}_upwick"] + F[f"{tag}_lowwick"]).stack().dropna()
        assert np.allclose(tot.values, 1.0, atol=1e-4), tag


def test_flat_bar_gives_nan_not_inf():
    F = candles.build(bars([(10, 10, 10, 10), (10, 10, 10, 10)]))
    for k in ("cd_body", "cd_pos", "cd_upwick"):
        v = F[k]["B"].iloc[-1]
        assert np.isnan(v), k                                         # zero range: undefined, never infinite


def test_streak_counts_consecutive_days():
    closes = [10, 11, 12, 13, 12, 11]
    F = candles.build(bars([(c, c + 0.5, c - 0.5, c) for c in closes]))
    s = F["streak"]["A"].tolist()
    assert s[1:4] == [1, 2, 3]
    assert s[4:6] == [-1, -2]


def test_inside_and_outside_day():
    F = candles.build(bars([(10, 12, 8, 11), (10.5, 11, 9, 10), (10, 13, 7, 12)]))
    assert F["inside_day"]["A"].tolist()[1:] == [1.0, 0.0]
    assert F["outside_day"]["A"].tolist()[1:] == [0.0, 1.0]


def test_bullish_and_bearish_engulfing():
    bull = candles.build(bars([(11, 11.2, 9.8, 10), (9.9, 11.5, 9.8, 11.3)]))
    bear = candles.build(bars([(10, 11.2, 9.8, 11), (11.1, 11.3, 9.7, 9.8)]))
    assert bull["engulf"]["A"].iloc[-1] == 1.0
    assert bear["engulf"]["A"].iloc[-1] == -1.0


def test_gap_and_gap_fill():
    F = candles.build(bars([(10, 10.2, 9.8, 10), (11, 11.5, 9.9, 11.2), (12, 12.5, 11.8, 12.2)]))
    assert F["gap"]["A"].iloc[1] == pytest.approx(0.1)
    assert F["gap_filled"]["A"].iloc[1] == 1.0                         # traded back down to yesterday's close
    assert F["gap_filled"]["A"].iloc[2] == 0.0                         # gap up that never came back


def test_weekly_candle_spans_five_sessions():
    closes = [10, 11, 12, 13, 14, 15]
    F = candles.build(bars([(c - 0.5, c + 0.5, c - 1, c) for c in closes]))
    # weekly candle at day 5: open of day 1 (10.5), high max(11.5..15.5), low min(10..14), close 15
    o, h, l, c = 10.5, 15.5, 10.0, 15.0
    assert F["cw_body"]["A"].iloc[-1] == pytest.approx((c - o) / (h - l), rel=1e-5)


def changed(part, full):
    """Cells that differ between a truncated and a full computation. NaN-vs-value counts as a change: a feature that
    peeks ahead shows up as NaN at the truncation edge, which a plain `abs diff > eps` silently skips."""
    a, b = part.to_numpy(float), full.iloc[:len(part)].to_numpy(float)
    return int(((np.isnan(a) != np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 1e-6)).sum())


def test_point_in_time_truncation():
    """Appending future days must never change any feature value on an earlier day."""
    B = rand_bars(n=200)
    full = candles.build(B)
    cut = 120
    part = candles.build({k: v.iloc[:cut] for k, v in B.items()})
    for k in part:
        n = changed(part[k], full[k])
        assert n == 0, f"{k}: {n} cells changed when future rows were appended"


def test_truncation_check_catches_a_peeking_feature():
    """Control: the truncation check above must be able to fail. A centred (future-peeking) average fails it."""
    B = rand_bars(n=200)
    peek = lambda C: C.rolling(5, center=True).mean()
    full, part = peek(B["Close"]), peek(B["Close"].iloc[:120])
    assert changed(part, full) > 0
    shift = lambda C: C.shift(-1)                                       # tomorrow's close: the classic leak
    assert changed(shift(B["Close"].iloc[:120]), shift(B["Close"])) > 0


def test_empty_frame_does_not_crash():
    B = {k: v.iloc[:0] for k, v in rand_bars().items()}
    F = candles.build(B)
    assert all(len(v) == 0 for v in F.values())
