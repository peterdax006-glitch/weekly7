"""PHASE 12 trust tables: planted effect in one type is found and trusted, absent elsewhere is neutralized."""
import numpy as np
import pandas as pd
import pytest

from engine import trust as T

NOW = pd.Timestamp("2024-06-28")


def panel(seed=0, weeks=120, n=60, planted=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=NOW - pd.Timedelta(days=30), periods=weeks * 5)
    dates = dates[dates.dayofweek == 4]                     # one weekly decision date
    tick = [f"T{i:02d}" for i in range(n)]
    ix = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    sic = np.where(np.arange(n) < n // 2, 2834, 3674)       # divisions D both -> use 6000 (H) for half
    sic = np.where(np.arange(n) < n // 2, 6021, 3674)
    info = pd.DataFrame({"sic": np.tile(sic, len(dates)), "size": rng.normal(size=len(ix)),
                         "vol": rng.normal(size=len(ix))}, index=ix)
    X = pd.DataFrame({"good": rng.normal(size=len(ix)), "noise": rng.normal(size=len(ix))}, index=ix)
    y = rng.normal(size=len(ix)) * 0.05
    if planted:
        is_h = (info["sic"] == 6021).values
        y = y + 0.05 * 0.5 * X["good"].values * is_h        # 'good' works only for division H
    return X, pd.Series(y, index=ix), info


def test_sic_division_and_bad_inputs():
    assert T.sic_division(2834) == "D" and T.sic_division("6021") == "H" and T.sic_division(73) == "I"
    assert T.sic_division(None) == "NA" and T.sic_division(float("nan")) == "NA" and T.sic_division(-5) == "NA"


def test_terciles_are_per_date_and_na_safe():
    X, y, info = panel(weeks=6)
    info.iloc[:5, info.columns.get_loc("size")] = np.nan
    L = T.assign_types(info)
    assert (L["size"].iloc[:5] == "NA").all()
    d0 = L.loc[L.index.get_level_values(0)[-1], "size"].value_counts()
    assert d0.max() - d0.min() <= 1
    assert T.parent_key("D|S|lowvol|up|hot|hyped") == "D|S|lowvol|up" and T.parent_key("D") == "ALL"


def test_planted_effect_found_and_noise_neutralized():
    X, y, info = panel()
    tt = T.TrustTable(min_weeks=26, min_obs=300).fit(X, y, info, NOW)
    t = tt.table
    good = t[(t.indicator == "good") & (t.type == "H")].iloc[0]
    other = t[(t.indicator == "good") & (t.type == "D")].iloc[0]
    assert good.reliable and good.post > 0.05
    assert not other.reliable                                # same indicator, type where it does not work
    assert not t[(t.indicator == "noise") & (t.level == "sic")].reliable.any()
    # neutralization: noise scores multiplied by ~0 in every type
    day = info.xs(info.index.get_level_values(0)[-1])
    sc = pd.DataFrame({"good": 1.0, "noise": 1.0}, index=day.index)
    out = tt.neutralize(sc, day)
    assert (out["noise"] == 0).all()
    assert out.loc[day["sic"] == 6021, "good"].gt(0).all()
    assert (out.loc[day["sic"] == 3674, "good"] == 0).all()


def test_no_lookahead_recent_rows_excluded():
    X, y, info = panel()
    tt_a = T.TrustTable().fit(X, y, info, NOW)
    # poison the last 8 weeks' returns: rows whose window had not closed at `now`-10d must not matter
    cut = NOW - pd.Timedelta(days=60)
    y2 = y.copy()
    late = y2.index.get_level_values(0) > cut
    y2[late] = 99.0
    a = T.TrustTable().fit(X, y, info, cut).table
    b = T.TrustTable().fit(X, y2, info, cut).table
    pd.testing.assert_frame_equal(a, b)
    assert len(tt_a.table) > 0


def test_shrinkage_pulls_thin_cell_toward_parent():
    X, y, info = panel()
    tt = T.TrustTable(min_weeks=5, min_obs=10, min_per_week=3).fit(X, y, info, NOW)
    t = tt.table.set_index(["type", "indicator"])
    leaf = t[(t.level == "sic|size|vol|trend|theme|attn") & (t.index.get_level_values(1) == "good")]
    assert len(leaf) > 0
    # posterior sd never exceeds the tau-scale prior and shrinks toward the (positive) parent for H leaves
    assert (leaf["post_sd"] <= np.sqrt(0.03 ** 2 + 0.03 ** 2) + 1e-9).all()
    hleaf = leaf[[k[0].startswith("H") for k in leaf.index]]
    assert hleaf["post"].abs().max() < hleaf["raw"].abs().max()


def test_empty_and_degenerate():
    X, y, info = panel(weeks=3)
    tt = T.TrustTable().fit(X.iloc[:0], y.iloc[:0], info.iloc[:0], NOW)
    assert tt.table.empty and "empty" in tt.report()
    tt = T.TrustTable().fit(X, y, info, NOW)               # 3 weeks << min_weeks: nothing may be reliable
    assert not tt.table["reliable"].any()
    assert (tt.table["reason"].str.startswith("support")).all() or tt.table.empty
    assert tt.lookup(["nope"], "good")["reason"] == "type never observed"
    W = tt.weights(info.xs(info.index.get_level_values(0)[-1]))
    assert (W.values == 0).all()


def test_json_roundtrip(tmp_path):
    X, y, info = panel(weeks=60)
    tt = T.TrustTable(min_weeks=10, min_obs=50).fit(X, y, info, NOW)
    p = tmp_path / "t.json"
    tt.to_json(p)
    t2 = T.TrustTable.from_json(p)
    assert len(t2.table) == len(tt.table) and t2.lookup(["H"], "good")["weight"] == pytest.approx(tt.lookup(["H"], "good")["weight"])
