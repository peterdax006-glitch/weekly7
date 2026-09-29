"""Trust store: versioning, tamper detection, diff, planted drift, report."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import trust as T, trust_store as S

NOW1 = pd.Timestamp("2022-06-24")
NOW2 = pd.Timestamp("2025-06-27")


def panel(weeks=260, n=40, seed=0, flip_after=None):
    """Indicator 'good' works in division H (6021). If flip_after is set, its sign inverts after that date (drift)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-03", periods=weeks, freq="W-FRI")
    ix = pd.MultiIndex.from_product([dates, [f"T{i:02d}" for i in range(n)]], names=["date", "ticker"])
    sic = np.tile(np.where(np.arange(n) < n // 2, 6021, 3674), len(dates))
    info = pd.DataFrame({"sic": sic, "size": rng.normal(size=len(ix)), "vol": rng.normal(size=len(ix))}, index=ix)
    X = pd.DataFrame({"good": rng.normal(size=len(ix)), "noise": rng.normal(size=len(ix))}, index=ix)
    sign = np.ones(len(ix))
    if flip_after is not None:
        sign = np.where(ix.get_level_values(0) > flip_after, -1.0, 1.0)
    y = rng.normal(size=len(ix)) * 0.05 + 0.025 * X["good"].values * (sic == 6021) * sign
    return X, pd.Series(y, index=ix), info


def fit(X, y, info, now):
    return T.TrustTable(min_weeks=20, min_obs=200, half_life_weeks=40).fit(X, y, info, now)


@pytest.fixture(scope="module")
def two():
    X, y, info = panel(flip_after=pd.Timestamp("2022-12-30"))
    return fit(X, y, info, NOW1), fit(X, y, info, NOW2)


def test_save_load_roundtrip_and_chain(tmp_path, two):
    a, b = two
    st = S.TrustStore(tmp_path)
    assert st.save(a, "first") == 1 and st.save(b, "second") == 2
    assert [v[0] for v in st.versions()] == [1, 2]
    got = st.load(1)
    assert len(got.table) == len(a.table) and got.now == a.now
    assert got.lookup(["H"], "good")["weight"] == pytest.approx(a.lookup(["H"], "good")["weight"])
    assert st.load().now == b.now and st.verify()


def test_refuses_empty_and_duplicate(tmp_path, two):
    st = S.TrustStore(tmp_path)
    with pytest.raises(S.StoreError):
        st.save(T.TrustTable())
    st.save(two[0])
    with pytest.raises(S.StoreError, match="identical"):
        st.save(two[0])
    with pytest.raises(S.StoreError):
        S.TrustStore(tmp_path / "empty").load()


def test_tampering_is_detected(tmp_path, two):
    st = S.TrustStore(tmp_path)
    st.save(two[0]); st.save(two[1])
    p = st.versions()[0][2]
    rec = json.loads(p.read_text())
    rec["table"][0]["post"] += 0.5                       # quietly inflate a reliability
    p.write_text(json.dumps(rec))
    with pytest.raises(S.StoreError, match="edited"):
        st.load()
    # deleting a middle version breaks the chain
    st2 = S.TrustStore(tmp_path / "b")
    st2.save(two[0]); st2.save(two[1])
    st2.versions()[0][2].unlink()
    with pytest.raises(S.StoreError, match="gap"):
        st2.verify()


def test_planted_drift_is_flagged_and_stable_cells_are_not(two):
    a, b = two
    d = S.drift(a, b)
    hit = d[(d["indicator"] == "good") & d["type"].str.startswith("H")]
    assert len(hit) > 0 and hit["flag"].isin(["sign_flip", "moved", "lost_trust"]).all()
    assert (d[d["indicator"] == "noise"]["flag"] != "sign_flip").all()
    top = d.iloc[0]
    assert top["indicator"] == "good"


def test_no_drift_between_identical_tables(two):
    a, _ = two
    assert S.drift(a, a).empty
    df = S.diff(a, a)
    assert (df["change"] == "unchanged").all() and (df["delta"] == 0).all()


def test_diff_added_removed(two):
    a, b = two
    small = T.TrustTable(min_weeks=20, min_obs=200)
    small.table = a.table[a.table["level"] == "ALL"].reset_index(drop=True)
    small.now = a.now
    small._reindex()
    d = S.diff(small, a)
    assert (d["change"] == "added").sum() == len(a.table) - len(small.table)
    assert (S.diff(a, small)["change"] == "removed").sum() == len(a.table) - len(small.table)
    assert S.drift(T.TrustTable(), T.TrustTable()).empty


def test_report_renders_and_escapes(tmp_path, two):
    a, b = two
    d = S.drift(a, b)
    a.table.loc[0, "type"] = "<script>x</script>"
    a._reindex()
    out = S.render_html(a, tmp_path / "docs" / "r.html", d)
    txt = open(out, encoding="utf-8").read()
    assert "<svg" in txt and "Drift since previous version" in txt and "<script>" not in txt
    empty = S.render_html(T.TrustTable(), tmp_path / "e.html")
    assert "no cells" in open(empty, encoding="utf-8").read()
