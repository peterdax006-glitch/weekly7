"""F23 part 1 (C75 Phase 1E, C68): the planted world's band-eligible mechanism. Directional runs start only in a HOT volatility
state, move ~2%/session for a geometric number of sessions, and show in r5 / r20 only after they start - so a 5-10% week is knowable
in advance from past prices for a minority of names, and not at all in the null world. Fast checks on small planted worlds; the
real-loop proof (positions, the C68 chain) is scripts/f23_planted_trades.py (results in state/research/f23_planted_trades/)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.research import feeds as FD

SMALL = dict(n_names=40, n_days=700)


def fwd_table(w):
    """Past-only signals at each close and the next week's return from the next open (the decision convention)."""
    C, O = w.bars["Close"], w.bars["Open"]
    r20 = C / C.shift(20) - 1
    vol20 = C.pct_change(fill_method=None).rolling(20).std()
    fwd = C.shift(-5) / O.shift(-1) - 1
    return pd.DataFrame({"r20": r20.stack(), "vol20": vol20.stack(), "fwd": fwd.stack()}).dropna()


def run_mask(w):
    """(session x name) True while a planted run is active, from the exact truth."""
    C = w.bars["Close"]
    m = np.zeros(C.shape, bool)
    pos = {d: i for i, d in enumerate(C.index.strftime("%Y-%m-%d"))}
    col = {t: j for j, t in enumerate(C.columns)}
    for d, t, s, k in w.truth["trend_runs"]:
        m[pos[d]:pos[d] + k, col[t]] = True
    return pd.DataFrame(m, index=C.index, columns=C.columns)


def test_default_world_has_runs_for_a_minority_and_names_them():
    w = FD.planted_world(FD.PlantConfig(seed=0, **SMALL))
    share = w.truth["trend_share"]
    assert 0.03 < share < 0.35, share                                   # a minority of name-days
    assert w.truth["genuine_direction"] == ("rel_r20", "rel_r5") and "abs_rel_r20" in w.truth["genuine"]
    signs = [s for _, _, s, _ in w.truth["trend_runs"]]
    assert {-1, 1} <= set(signs)                                          # both directions: the sign is not a free base rate


def test_null_world_has_no_run_and_no_direction_truth():
    w = FD.planted_world(FD.PlantConfig(seed=0, **SMALL, **FD.NULL_PLANT))
    assert w.truth["trend_runs"] == [] and w.truth["trend_share"] == 0.0 and "genuine_direction" not in w.truth


def test_mechanism_off_leaves_every_other_mechanism_bit_identical():
    on = FD.planted_world(FD.PlantConfig(seed=2, **SMALL))
    off = FD.planted_world(FD.PlantConfig(seed=2, trend_rate=0.0, **SMALL))
    ran = run_mask(on).any(axis=0)
    never = ran.index[~ran.to_numpy()]
    assert len(never) >= 5
    for f in FD.BAR_FIELDS:                                               # names that never ran: exactly the pre-F23 world
        pd.testing.assert_frame_equal(on.bars[f][never], off.bars[f][never])
    assert on.truth["earnings"] == off.truth["earnings"] and on.truth["cheap"] == off.truth["cheap"]
    assert not on.bars["Close"][ran.index[ran.to_numpy()]].equals(off.bars["Close"][ran.index[ran.to_numpy()]])


def test_runs_start_only_in_a_hot_volatility_state():
    pc = FD.PlantConfig(seed=1, **SMALL)
    rng = np.random.default_rng(5)
    h = np.zeros((300, 10))
    h[:, :3] = 2.0                                                        # three names always hot, seven never
    drift, runs = FD.trend_runs(h, pc)
    assert runs and {j for _, j, _, _ in runs} <= {0, 1, 2}
    assert not drift[:, 3:].any()
    assert FD.trend_runs(rng.normal(0, 1, (300, 10)), dataclasses.replace(pc, vol_state_sd=0.0))[1] == []


def test_run_schedule_is_forward_only():
    """Drift up to session t depends only on the state up to t: scrambling everything after t changes nothing before it."""
    pc = FD.PlantConfig(seed=4, **SMALL)
    rng = np.random.default_rng(0)
    h = rng.normal(0.3, 0.8, (400, 12))
    a, _ = FD.trend_runs(h, pc)
    h2 = h.copy()
    h2[250:] = rng.normal(-3, 1, h2[250:].shape)                          # the future turns cold everywhere
    b, _ = FD.trend_runs(h2, pc)
    assert np.array_equal(a[:250], b[:250])


def test_a_five_to_ten_percent_week_is_knowable_from_past_prices_only():
    """Planted effect: among volatile names, a strong past r20 predicts a 5-10% next week (from the next open). The same past-only
    screen on the null world predicts nothing."""
    out = {}
    for name, extra in (("default", {}), ("null", FD.NULL_PLANT)):
        rows = []
        for seed in (0, 1):
            T = fwd_table(FD.planted_world(FD.PlantConfig(seed=seed, **SMALL, **extra)))
            g = T.groupby(level=0)
            hot = T["vol20"] >= g["vol20"].transform(lambda v: v.quantile(0.75))
            strong = T["r20"] >= 0.25
            rows.append(T[hot & strong]["fwd"])
        out[name] = pd.concat(rows)
    d = out["default"]
    assert len(d) > 200 and 0.04 < d.mean() < 0.10 and (d > 0).mean() > 0.6, (len(d), d.mean())
    n = out["null"]
    assert len(n) < 50 or abs(n.mean()) < 0.01                            # the null world never has r20 >= 25% often, nor an edge


def test_in_run_weeks_land_in_the_band_on_average():
    w = FD.planted_world(FD.PlantConfig(seed=3, **SMALL))
    T = fwd_table(w)
    m = run_mask(w).stack().reindex(T.index).fillna(False).to_numpy(bool)
    up = {(pd.Timestamp(d), t) for d, t, s, _ in w.truth["trend_runs"] if s > 0}
    assert m.sum() > 100 and up
    sgn = np.sign(T["r20"].to_numpy())
    signed = T["fwd"].to_numpy()[m] * sgn[m]
    assert 0.04 < float(np.mean(signed)) < 0.10                           # continuation, in the band, on average


def test_config_validation_and_degenerate_worlds():
    assert FD.PlantConfig().validate() == []
    assert FD.PlantConfig(trend_rate=1.5).validate() and FD.PlantConfig(trend_drift=0.2).validate()
    with pytest.raises(ValueError):
        FD.planted_world(FD.PlantConfig(trend_len=0.5))
    with pytest.raises(ValueError):
        FD.planted_world(FD.PlantConfig(n_names=4))
    drift, runs = FD.trend_runs(np.zeros((0, 5)), FD.PlantConfig())
    assert drift.shape == (0, 5) and runs == []
