"""creator/phonetap.py with FakeDevice: homing, relative moves, taps, swipes, calibration, frame mapping. No board, no phone."""
from __future__ import annotations

import pytest

from creator import phonetap as T


def test_home_then_tap_lands_on_target():
    dev = T.FakeDevice(start=(333, 777))
    pt = T.PhoneTap(dev)
    pt.tap(100, 200)
    (x, y), = dev.taps
    assert abs(x - 100) <= 1 and abs(y - 200) <= 1
    assert all(abs(int(v)) <= 127 for ln in dev.lines if ln.startswith("M ") for v in ln.split()[1:])


def test_many_taps_rehome_and_stay_accurate():
    dev = T.FakeDevice()
    pt = T.PhoneTap(dev, rehome_every=3)
    targets = [(20, 30), (370, 820), (195, 422), (10, 800), (380, 10), (200, 600), (50, 50)]
    for x, y in targets:
        pt.tap(x, y)
    for (x, y), (tx, ty) in zip(dev.taps, targets):
        assert abs(x - tx) <= 1 and abs(y - ty) <= 1


def test_calibration_fixes_unknown_gain_and_acceleration():
    dev = T.FakeDevice(gain=1.7, accel=0.03)                                               # the real phone's numbers are unknown
    pt = T.PhoneTap(dev)
    g = pt.calibrate(lambda: (dev.x, dev.y))
    assert 1.9 < g < 2.1                                                                    # 1.7 * (1 + 0.03*6)
    pt.tap(300, 700)
    x, y = dev.taps[-1]
    assert abs(x - 300) <= 3 and abs(y - 700) <= 3


def test_uncalibrated_gain_misses_which_is_why_calibration_exists():
    dev = T.FakeDevice(gain=1.7)
    T.PhoneTap(dev).tap(100, 100)
    assert abs(dev.taps[0][0] - 100) > 30


def test_swipe_presses_moves_releases():
    dev = T.FakeDevice()
    pt = T.PhoneTap(dev)
    pt.swipe(195, 700, 195, 150)
    assert not dev.down and len(dev.drags) == 1
    path = dev.drags[0]
    assert abs(path[0][1] - 700) <= 1 and abs(path[-1][1] - 150) <= 1 and len(path) > 10


def test_frame_mapping_and_bounds():
    pt = T.PhoneTap(T.FakeDevice())
    assert pt.frame_to_points(250, 540, 500, 1080) == (195.0, 422.0)
    with pytest.raises(T.TapError):
        pt.tap(500, 10)


def test_unpaired_device_raises_clearly():
    dev = T.FakeDevice(paired=False)
    pt = T.PhoneTap(dev)
    assert not pt.paired()
    with pytest.raises(T.TapError, match="not paired"):
        pt.tap(10, 10)


def test_bad_calibration_is_rejected():
    dev = T.FakeDevice()
    with pytest.raises(T.TapError):
        T.PhoneTap(dev).calibrate(lambda: (0.0, 0.0))
