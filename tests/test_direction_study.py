"""Study plumbing on synthetic bars: labels match scripts/movers.py, features are point-in-time, analog history is
leak-free, and a planted direction effect is picked up by the fold code."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import direction_study as DS  # noqa: E402


def bars(n_days=900, n_t=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    tick = [f"T{i:02d}" for i in range(n_t)]
    r = rng.normal(0, 0.03, (n_days, n_t))
    C = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), index=idx, columns=tick)
    O = C.shift(1).fillna(C.iloc[0]) * (1 + rng.normal(0, .005, C.shape))
    Hh, L = np.maximum(C, O) * 1.01, np.minimum(C, O) * 0.99
    V = pd.DataFrame(rng.uniform(1e5, 2e5, C.shape), index=idx, columns=tick)
    return dict(Open=O, High=Hh, Low=L, Close=C, Volume=V)


def sic_for(b):
    return pd.DataFrame({"ticker": b["Close"].columns, "sic": np.where(np.arange(b["Close"].shape[1]) % 2, 6021, 3674)})


def test_labels_identical_to_movers_script():
    spec = importlib.util.spec_from_file_location("movers_script", ROOT / "scripts" / "movers.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    b = bars(120, 6)
    for a, c in zip(DS.labels(b), m.labels(b)):
        pd.testing.assert_frame_equal(a, c)
    assert (DS.H, DS.MOVE) == (m.H, m.MOVE)


def test_features_are_point_in_time():
    b = bars()
    P1 = DS.build_panel(b, sic_for(b), min_dv=0)
    cut = P1.index.get_level_values(0)[len(P1) // 2]
    b2 = {k: v.copy() for k, v in b.items()}
    for k in b2:
        b2[k].loc[b2[k].index > cut] *= 1.7          # wreck everything after the cut
    P2 = DS.build_panel(b2, sic_for(b), min_dv=0)
    cols = ["pattern", "model", "evidence", "analog", "size", "vol", "trend", "attention"]
    early = P1.index.get_level_values(0) <= cut
    e2 = P2.index.get_level_values(0) <= cut
    a, c = P1[early][cols], P2[e2][cols]
    common = a.index.intersection(c.index)
    assert len(common) > 100
    # rows near the cut differ only through labels (excluded); features must match exactly
    pd.testing.assert_frame_equal(a.loc[common], c.loc[common])


def test_analog_only_uses_resolved_events():
    b = bars(400, 10)
    P = DS.build_panel(b, sic_for(b), min_dv=0)
    up, dn, ret = DS.labels(b)
    touch = ((up >= DS.MOVE) | (dn <= -DS.MOVE)) & ret.notna()
    tk, d = P.index[len(P) // 2][1], P.index[len(P) // 2][0]
    pos = b["Close"].index.get_loc(d)
    known = touch[tk].iloc[: pos - DS.H]            # events whose window closed by d (event date <= d-6)
    ups = (known & (ret[tk].iloc[: pos - DS.H] > 0)).sum()
    n = known.sum()
    if n >= 3:
        assert abs(P.loc[(d, tk), "analog"] - (ups + 1) / (n + 2)) < 1e-12
    else:
        assert np.isnan(P.loc[(d, tk), "analog"])


def test_fold_finds_planted_direction_effect_and_opens_gate():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2005-01-07", "2016-12-30", freq="W-FRI")
    ix = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(30)]], names=["date", "ticker"])
    n = len(ix)
    z = rng.normal(size=n)
    up = (rng.random(n) < 1 / (1 + np.exp(-4 * z))).astype(float)
    panel = pd.DataFrame({"pattern": z + rng.normal(0, .3, n), "analog": np.nan, "trust": 1.0, "model": z + rng.normal(0, .3, n),
                          "evidence": np.nan, "up": up, "mover": True, "sic": "D"}, index=ix)
    rec, alt = DS.fold(panel, DS.inputs(panel), 2016)
    assert rec["engine_open"] and rec["brier"] < rec["brier_base"] and rec["acc_bet"] > 0.8
    assert {"platt", "venn_abers_gate", "conformal_gate"} <= set(alt["method"])


def test_fold_refuses_on_noise_and_on_too_little_history():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2005-01-07", "2016-12-30", freq="W-FRI")
    ix = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(30)]], names=["date", "ticker"])
    n = len(ix)
    panel = pd.DataFrame({"pattern": rng.normal(size=n), "analog": np.nan, "trust": 1.0, "model": np.nan, "evidence": np.nan,
                          "up": (rng.random(n) < 0.5).astype(float), "mover": True, "sic": "D"}, index=ix)
    rec, _ = DS.fold(panel, DS.inputs(panel), 2016)
    assert not rec["engine_open"] and rec["n_bet"] == 0
    rec, alt = DS.fold(panel, DS.inputs(panel), 2005)          # no earlier history at all
    assert not rec["engine_open"] and alt is None


def test_era_summary_pools_by_rows():
    f = pd.DataFrame({"year": [2005, 2006], "n_test": [100, 300], "brier": [0.2, 0.3], "brier_base": [.25, .25],
                      "ece": [.1, .0], "acc": [.6, .5], "engine_open": [True, False], "n_bet": [10, 0], "acc_bet": [0.9, np.nan]})
    e = DS.era_summary(f, {"x": (2005, 2006)}).iloc[0]
    assert e["n"] == 400 and abs(e["brier"] - 0.275) < 1e-12 and e["n_bet"] == 10 and abs(e["acc_bet"] - 0.9) < 1e-12
