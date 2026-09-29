"""Per-type calibration, Venn-Abers and conformal gates - and a miscalibrated model the gates must refuse."""
import numpy as np
import pandas as pd
import pytest

from engine import direction as D, direction_calib as C

NOW = pd.Timestamp("2024-06-28")


def scores(n, seed, strength=2.5, flip=False, overconf=1.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-strength * z * (-1 if flip else 1)))).astype(float)
    raw = 1 / (1 + np.exp(-overconf * strength * z))
    return raw, y


def test_perfect_model_all_routes_calibrated_and_bets_are_right():
    rc, yc = scores(1500, 1)
    rt, yt = scores(3000, 2)
    tab = C.compare_gates(rc, yc, rt, yt).set_index("method")
    assert set(tab.index) >= {"none", "platt", "isotonic", "venn_abers_gate", "conformal_gate"}
    for m in ("platt", "isotonic"):
        assert tab.loc[m, "ece"] < 0.05
    for m in ("platt", "venn_abers_gate", "conformal_gate"):
        assert tab.loc[m, "n_bet"] > 50 and tab.loc[m, "acc_bet"] >= 0.78, tab


def test_overconfident_raw_is_worse_than_calibrated():
    rc, yc = scores(1500, 3, strength=1.0, overconf=4.0)
    rt, yt = scores(3000, 4, strength=1.0, overconf=4.0)
    tab = C.compare_gates(rc, yc, rt, yt).set_index("method")
    assert tab.loc["platt", "brier"] < tab.loc["none", "brier"]
    assert tab.loc["none", "ece"] > 3 * tab.loc["platt", "ece"]


def test_miscalibrated_model_is_refused_by_conservative_gates_under_flip():
    """Calibration block: signal works. Test block: relationship inverts. Raw and Platt routes bet confidently and are
    wrong; the DirectionEngine's out-of-sample gate must be closed, so it refuses."""
    rng = np.random.default_rng(9)
    dates = pd.bdate_range(end=NOW - pd.Timedelta(days=30), periods=240)
    ix = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(8)]], names=["date", "ticker"])
    z = rng.normal(size=len(ix))
    flip = np.asarray(ix.get_level_values(0) >= dates[int(240 * 0.8)])       # last 20% (the engine's test block) inverts
    p = 1 / (1 + np.exp(-4 * z * np.where(flip, -1, 1)))
    up = pd.Series((rng.random(len(ix)) < p).astype(float), index=ix)
    F = D.build_inputs(pattern=pd.Series(z, index=ix), trust=pd.Series(1.0, index=ix), index=ix)
    eng = D.DirectionEngine().fit(F, up, NOW)
    assert not eng.open
    assert not eng.decide(F)["bet"].any()
    assert "gated" in eng.reason or "miscalibrated" in eng.reason or "skill" in eng.reason


def test_per_type_calibration_fixes_type_specific_bias():
    rng = np.random.default_rng(5)
    n = 4000
    types = np.where(rng.random(n) < 0.5, "A", "B")
    z = rng.normal(size=n)
    raw = 1 / (1 + np.exp(-2 * z))
    # type B's truth is much weaker than the score says; A is exactly as claimed
    y = (rng.random(n) < np.where(types == "A", raw, 0.5 + 0.25 * (raw - 0.5))).astype(float)
    cut = n // 2
    glob = D._Cal("platt").fit(raw[:cut], y[:cut])
    pt = C.PerTypeCalibrator().fit(raw[:cut], y[:cut], types[:cut])
    mB = types[cut:] == "B"
    bg = D.brier(np.clip(glob(raw[cut:]), .02, .98)[mB], y[cut:][mB])
    bp = D.brier(pt(raw[cut:], types[cut:])[mB], y[cut:][mB])
    assert bp < bg
    assert set(pt.corrections()["type"]) == {"A", "B"}


def test_per_type_thin_type_gets_global_map():
    rng = np.random.default_rng(6)
    raw, y = scores(500, 7)
    types = np.array(["big"] * 495 + ["tiny"] * 5)
    pt = C.PerTypeCalibrator().fit(raw, y, types)
    assert "tiny" not in pt.ab
    same = pt(raw[-5:], types[-5:])
    assert np.allclose(same, np.clip(pt.glob(raw[-5:]), .02, .98))


def test_venn_abers_interval_brackets_and_widens_with_little_data():
    rc, yc = scores(2000, 11)
    p0, p1, pm = C.venn_abers(rc, yc, np.linspace(0.02, 0.98, 25))
    assert (p0 <= p1 + 1e-12).all() and (p0 <= pm + 1e-12).all() and (pm <= p1 + 1e-12).all()
    w_big = (p1 - p0).mean()
    q0, q1, _ = C.venn_abers(rc[:40], yc[:40], np.linspace(0.02, 0.98, 25))
    assert (q1 - q0).mean() > w_big
    with pytest.raises(ValueError):
        C.venn_abers([], [], [0.5])


def test_conformal_abstains_on_noise_and_bets_on_signal():
    rng = np.random.default_rng(12)
    pc, yc = rng.random(1000), (rng.random(1000) < 0.5).astype(float)      # scores carry no information
    pn = rng.random(1000)
    plat = D._Cal("platt").fit(pc, yc)
    assert not C.ConformalGate().fit(plat(pc), yc).decide(plat(pn))["bet"].any()
    # coverage-only conformal is fooled by spread-out but uninformative scores; that is why require_conf exists
    naive = C.ConformalGate(require_conf=False).fit(pc, yc).decide(pn)
    assert naive["bet"].any()
    rc, yc = scores(2000, 13, strength=4.0)
    rt, yt = scores(2000, 14, strength=4.0)
    d = C.ConformalGate().fit(rc, yc).decide(rt)
    assert d["bet"].mean() > 0.2
    hit = ((d.loc[d["bet"], "side"] > 0) == (yt[d["bet"].values] > 0.5)).mean()
    assert hit >= 0.8


def test_venn_abers_gate_abstains_on_noise_and_small_calibration():
    rng = np.random.default_rng(15)
    pc, yc = rng.random(800), (rng.random(800) < 0.5).astype(float)
    assert not C.VennAbersGate().fit(pc, yc).decide(rng.random(300))["bet"].any()
    rc, yc = scores(30, 16, strength=4.0)
    rt, _ = scores(300, 17, strength=4.0)
    small = C.VennAbersGate().fit(rc, yc).decide(rt)["bet"].mean()
    big = C.VennAbersGate().fit(*scores(3000, 18, strength=4.0)).decide(rt)["bet"].mean()
    assert small < big


def test_compare_degenerate_single_class():
    rt, yt = scores(50, 1)
    tab = C.compare_gates(np.array([0.6, 0.7, 0.4]), np.array([1.0, 1.0, 1.0]), rt, yt)
    assert list(tab["method"]) == ["none"]
