"""Canon C33 for training labels: engine.features.labels. The default (close entry) is unchanged so the Live champion is
unaffected; entry="open" measures from the next session's open, so an overnight gap the system cannot capture is not
rewarded. Barrier logic (target first / stop first / neither) is checked on hand-built paths."""
import numpy as np
import pandas as pd
import pytest

from engine import features


def frames(closes, opens=None, highs=None, lows=None):
    idx = pd.bdate_range("2022-01-03", periods=len(closes))
    C = pd.DataFrame({"A": closes}, index=idx, dtype=float)
    O = pd.DataFrame({"A": opens if opens is not None else closes}, index=idx, dtype=float)
    H = pd.DataFrame({"A": highs if highs is not None else np.maximum(C["A"], O["A"])}, index=idx, dtype=float)
    L = pd.DataFrame({"A": lows if lows is not None else np.minimum(C["A"], O["A"])}, index=idx, dtype=float)
    atr = pd.DataFrame({"A": 1.0}, index=idx)
    return {"Close": C, "Open": O, "High": H, "Low": L}, atr


def test_default_is_close_entry_and_unchanged():
    st, atr = frames([100, 101, 102, 103, 104, 105, 106])
    y, fwd = features.labels(st, atr, horizon=5)
    assert fwd["A"].iloc[0] == pytest.approx(np.log(105 / 100))
    y2, fwd2 = features.labels(st, atr, horizon=5, entry="close")
    assert fwd.equals(fwd2) and y.equals(y2)


def test_open_entry_excludes_the_overnight_gap():
    # closes flat at 100, but day 1 gaps up to open 110 and stays: close entry "earns" 10%, open entry earns 0
    closes = [100, 110, 110, 110, 110, 110, 110]
    opens = [100, 110, 110, 110, 110, 110, 110]
    st, atr = frames(closes, opens)
    _, fc = features.labels(st, atr, horizon=5, entry="close")
    _, fo = features.labels(st, atr, horizon=5, entry="open")
    assert fc["A"].iloc[0] == pytest.approx(np.log(1.10))
    assert fo["A"].iloc[0] == pytest.approx(0.0)


def test_open_entry_target_barrier_from_the_open():
    # entry at open 100 on day 1; day 3 high 108 touches +7% from the open (but only +5.9% from close 102)
    closes = [102, 100, 101, 104, 103, 103, 103]
    opens = [102, 100, 100, 101, 104, 103, 103]
    highs = [102, 101, 102, 108, 104, 103, 103]
    lows = [102, 99.5, 99.5, 101, 102, 103, 103]
    st, atr = frames(closes, opens, highs, lows)
    yo, _ = features.labels(st, atr, horizon=5, entry="open", stop_atr=5.0)
    yc, _ = features.labels(st, atr, horizon=5, entry="close", stop_atr=5.0)
    assert yo["A"].iloc[0] == 1.0
    assert yc["A"].iloc[0] == 0.0                                    # 108/102 = +5.9%: short of the target


def test_stop_barrier_first_gives_minus_one():
    closes = [100, 99, 96, 97, 110, 110, 110]
    lows = [100, 98.5, 95.5, 96, 97, 110, 110]
    highs = [100, 99.5, 99, 98, 111, 110, 110]
    st, atr = frames(closes, closes, highs, lows)
    y, _ = features.labels(st, atr, horizon=5, entry="close", stop_atr=3.0)   # stop 97; low 95.5 on day 2 first
    assert y["A"].iloc[0] == -1.0


def test_labels_are_nan_where_the_horizon_runs_out():
    st, atr = frames([100, 101, 102, 103, 104, 105, 106])
    y, fwd = features.labels(st, atr, horizon=5, entry="open")
    assert fwd["A"].iloc[-5:].isna().all() and y["A"].iloc[-5:].isna().all()


def test_unknown_entry_is_refused():
    st, atr = frames([100, 101, 102, 103, 104, 105, 106])
    with pytest.raises(ValueError):
        features.labels(st, atr, entry="vwap")
