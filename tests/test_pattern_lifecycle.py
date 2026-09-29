"""Bible Phase 4: lifecycle state machine. Synthetic panels with KNOWN histories: a stable pattern must stay active, a
pattern that only survives in one market regime must be rescoped to that regime, a pattern that reverses everywhere must
be discarded with a reason, and nothing may be left in a transient state or vanish from the log."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.pattern_lifecycle import (ALLOWED, STATES, Lifecycle, LifecycleError, Panel, detect_failure, parse_key_named,
                                      pattern_id, run_tests)
from engine.patterns import PatternMiner

WEEKS, STOCKS = 300, 120


def make_panel_data(seed=0, weeks=WEEKS, stocks=STOCKS):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=weeks * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(stocks)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), 4)), index=idx, columns=["f0", "f1", "f2", "f3"])
    reg = pd.Series(rng.standard_normal(weeks), index=dates)
    X["m_vix"] = reg.reindex(idx.get_level_values(0)).values
    y = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx) + np.repeat(rng.normal(0, 0.02, weeks), stocks)
    q = X[["f0", "f1", "f2"]].groupby(level=0).rank(pct=True) >= 0.8
    wk = np.repeat(np.arange(weeks), stocks)
    high = (X["m_vix"] > 0.43).values
    late = wk >= int(weeks * 0.75)
    y += 0.012 * q["f0"]                                                    # stable
    y += q["f1"] * np.where(high, 0.02, np.where(late, -0.03, 0.0))         # holds only in high-vix weeks
    y += q["f2"] * np.where(wk < int(weeks * 0.6), 0.012, -0.015)           # reverses everywhere
    return X, y, dates


@pytest.fixture(scope="module")
def data():
    return make_panel_data()


def frame(rows):
    cols = ["key_named", "effect", "t_disc", "t_conf", "p_real", "p_hallucinated", "p_coincidence", "status", "scope"]
    out = []
    for name, effect, status, *sc in rows:
        out.append([name, effect, 4.0, 3.0, 0.95, 0.01, 0.001, status, sc[0] if sc else None])
    return pd.DataFrame(out, columns=cols)


def seeded(as_of, params=None):
    life = Lifecycle(params=params)
    life.ingest(frame([("f0 q4", 0.006, "active"), ("f1 q4", 0.004, "active"), ("f2 q4", 0.005, "active")]), as_of)
    return life


def states(life):
    return {r["name"]: r["state"] for r in life.records.values()}


# ---------------------------------------------------------------- state machine
def test_transition_table_is_closed_and_reachable():
    for frm, tos in ALLOWED.items():
        assert frm is None or frm in STATES
        assert tos <= set(STATES)
    reach, todo = set(), [None]
    while todo:
        s = todo.pop()
        for t in ALLOWED[s]:
            if t not in reach:
                reach.add(t)
                todo.append(t)
    assert reach == set(STATES)


def test_illegal_transition_is_refused():
    life = seeded("2020-01-01")
    rec = life.records[pattern_id(("s", "f0", 4))]
    with pytest.raises(LifecycleError):
        life._move(rec, "rescoped", "2020-01-02", "skip the cause search")     # active -> rescoped is not a road
    with pytest.raises(LifecycleError):
        life._move(rec, "discarded", "2020-01-02", "straight to discarded")
    assert rec["state"] == "active" and life.invariants() == []


def test_parse_key_named_roundtrips_the_miner_naming():
    m = PatternMiner()
    m.feats = ["f0", "ret_5d", "vol"]
    for key in [("s", 1, 3), ("p", 0, 4, 2, 1), ("u", 0, 4, 1, 2, 2, 0)]:
        assert parse_key_named(m._name(key)) == m.key_names(key)
    for bad in ["", "f0", "f0 q9", "a q1 & b q2 & c q3", "a q1 unless b q2"]:
        with pytest.raises(ValueError):
            parse_key_named(bad)


# ---------------------------------------------------------------- ingesting miner frames
def test_ingest_maps_every_miner_status_with_a_legal_logged_path():
    rows = [("f0 q4", 0.01, "active"), ("f1 q1", 0.01, "rejected"), ("f2 q1", 0.01, "duplicate"),
            ("f3 q1", 0.01, "no_gain"), ("f0 q1", 0.01, "discarded"), ("f1 q2", 0.01, "rescoped", (0, "high", -0.3, 0.4))]
    life = Lifecycle()
    life.ingest(frame(rows), "2020-06-01")
    assert states(life) == {"f0 q4": "active", "f1 q1": "rejected", "f2 q1": "duplicate", "f3 q1": "no_gain",
                            "f0 q1": "discarded", "f1 q2": "rescoped"}
    r = life.records[pattern_id(("s", "f1", 2))]
    assert r["scope"] == {"col": "m_vix", "label": "high", "lo": -0.3, "hi": 0.4}
    assert life.records[pattern_id(("s", "f0", 1))]["discard_reason"]
    assert life.invariants() == []
    paths = life.log_frame().query("name == 'f1 q2'")["to"].tolist()
    assert paths == ["candidate", "active", "failed", "cause_search", "rescoped"]


def test_pattern_the_miner_stops_finding_goes_to_watch_not_missing():
    life = seeded("2020-01-01")
    life.ingest(frame([("f0 q4", 0.006, "active")]), "2020-02-01")
    assert states(life)["f1 q4"] == "watch" and states(life)["f2 q4"] == "watch" and len(life.records) == 3
    life.ingest(frame([("f0 q4", 0.006, "active"), ("f1 q4", 0.004, "rejected")]), "2020-03-01")
    assert states(life)["f1 q4"] == "watch"                                     # rejected by miner: still flagged, kept
    assert life.invariants() == []


def test_ingest_empty_none_and_garbage_rows():
    life = Lifecycle()
    assert life.ingest(pd.DataFrame(), "2020-01-01")["seen"] == 0
    assert life.ingest(None, "2020-01-01")["seen"] == 0
    out = life.ingest(frame([("not a pattern", 0.1, "active"), ("f0 q4", 0.1, "active")]), "2020-01-02")
    assert out["skipped"] == 1 and len(life.records) == 1
    assert any("skipped row" in e["reason"] for e in life.log)                  # the bad row is logged, not swallowed


def test_real_miner_output_ingests_cleanly():
    X, y, _ = make_panel_data(seed=3, weeks=150, stocks=60)
    last = X.index.get_level_values(0).max()
    M = PatternMiner({"min_n": 100, "max_pairs": 150, "max_unless": 20, "null_reps": 1}).fit(X, y, last)
    life = Lifecycle()
    life.ingest(M.patterns, last)
    assert len(life.records) == len(M.patterns)
    assert life.invariants() == []
    assert int((M.patterns["status"] == "active").sum()) == life.counts()["active"]


# ---------------------------------------------------------------- review on planted histories
def test_planted_review_keeps_stable_rescopes_regime_and_discards_reversed(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = seeded(as_of)
    summ = life.review(Panel.build(X, y, as_of), as_of)
    s = states(life)
    f1 = life.records[pattern_id(("s", "f1", 4))]
    assert s["f0 q4"] == "active"
    assert s["f1 q4"] == "rescoped"
    assert f1["scope"]["col"] == "m_vix" and f1["scope"]["label"] == "high"
    assert s["f2 q4"] == "discarded"
    assert "no context split held" in life.records[pattern_id(("s", "f2", 4))]["discard_reason"]
    assert summ["failed"] == 2 and summ["rescoped"] == 1 and summ["discarded"] == 1
    assert life.invariants() == []
    assert life.log_frame().query("name == 'f1 q4'")["to"].tolist() == ["candidate", "active", "failed", "cause_search", "rescoped"]
    assert f1["failures"]                                                       # failure evidence is kept


def test_rescoped_form_is_promoted_then_fails_when_scope_dies(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = seeded(as_of)
    life.review(Panel.build(X, y, as_of), as_of)
    life.review(Panel.build(X, y, as_of), as_of + pd.Timedelta(days=7))
    rec = life.records[pattern_id(("s", "f1", 4))]
    assert rec["state"] == "active" and rec["scope"] is not None               # rescoped -> active, scope kept
    # planted defect: the effect disappears (goes negative) for the last stretch, inside the scope too
    y2 = y.copy()
    late = X.index.get_level_values(0) >= dates[int(len(dates) * 0.8)]
    hit = (X.groupby(level=0)["f1"].rank(pct=True) >= 0.8).values & late
    y2[hit] = y2[hit] - 0.04
    later = as_of + pd.Timedelta(days=120)
    life.review(Panel.build(X, y2, later), later)
    assert rec["state"] == "discarded" and rec["discard_reason"]
    assert rec["failures"][-1]["as_of"].startswith(str(later.date()))
    assert life.invariants() == []


def test_no_lookahead_rows_after_as_of_or_with_open_horizon_are_invisible(data):
    X, y, dates = data
    cut = dates[200]
    y_poison = y.copy()
    y_poison[X.index.get_level_values(0) > cut - pd.Timedelta(days=7)] = 5.0    # everything not yet resolved at `cut`
    a = Panel.build(X, y, cut)
    b = Panel.build(X, y_poison, cut)
    assert a.dates.max() <= cut - pd.Timedelta(days=7)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.Q, b.Q)
    l1, l2 = seeded(cut), seeded(cut)
    l1.review(a, cut)
    l2.review(b, cut)
    assert l1.log == l2.log


def test_degenerate_panels_are_noted_never_silently_passed():
    life = seeded("2020-01-01")
    ix = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
    empty = Panel.build(pd.DataFrame({"f0": [], "m_vix": []}, index=ix), pd.Series([], dtype=float, index=ix), "2020-01-01")
    out = life.review(empty, "2020-01-02")
    assert out["insufficient"] == 3 and states(life) == {"f0 q4": "active", "f1 q4": "active", "f2 q4": "active"}
    assert life.log[-1]["kind"] == "note" and "too short" in life.log[-1]["reason"]
    assert Lifecycle().review(empty, "2020-01-02")["reviewed"] == 0


def test_mismatched_index_is_refused():
    X, y, _ = make_panel_data(weeks=45, stocks=5)
    with pytest.raises(LifecycleError):
        Panel.build(X, y.iloc[::-1], "2030-01-01")


def test_missing_feature_is_reported_unavailable_not_failed(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = Lifecycle()
    life.ingest(frame([("ghost q4", 0.01, "active")]), as_of)
    out = life.review(Panel.build(X, y, as_of), as_of)
    assert out["unavailable"] == 1 and states(life) == {"ghost q4": "active"}


def test_watch_recovers_or_expires(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    p = Panel.build(X, y, as_of)
    ok = Lifecycle(params={"max_watch": 1})
    ok.ingest(frame([("f0 q4", 0.006, "active")]), as_of)
    rec = ok.records[pattern_id(("s", "f0", 4))]
    ok._move(rec, "watch", as_of, "planted")
    ok.review(p, as_of)
    assert rec["state"] == "active"                                             # evidence recovered
    # a watch that never recovers expires into the failure path after max_watch reviews
    recent = X.index.get_level_values(0) >= dates[int(len(dates) * 0.8)]
    y_flat = y - 0.012 * ((X.groupby(level=0)["f0"].rank(pct=True) >= 0.8).values & recent)      # fades, does not reverse
    w = Lifecycle(params={"max_watch": 1})
    w.ingest(frame([("f0 q4", 0.006, "active")]), as_of)
    r = w.records[pattern_id(("s", "f0", 4))]
    w._move(r, "watch", as_of, "planted")
    pf = Panel.build(X, y_flat, as_of)
    w.review(pf, as_of)
    assert r["state"] == "watch"                                                # first bad review: still only watched
    w.review(pf, as_of)
    assert r["state"] == "discarded" and w.invariants() == []


def test_discarded_pattern_is_revived_when_it_passes_again(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = Lifecycle()
    life.ingest(frame([("f0 q4", 0.006, "discarded")]), as_of)
    rec = life.records[pattern_id(("s", "f0", 4))]
    assert rec["state"] == "discarded"
    out = life.review(Panel.build(X, y, as_of), as_of)
    assert out["revived"] == 1 and rec["state"] == "active" and rec["discard_reason"] is None
    assert life.log_frame()["to"].tolist()[-2:] == ["candidate", "active"]
    off = Lifecycle(params={"revive": False})
    off.ingest(frame([("f0 q4", 0.006, "discarded")]), as_of)
    off.review(Panel.build(X, y, as_of), as_of)
    assert states(off) == {"f0 q4": "discarded"}


# ---------------------------------------------------------------- the individual tests and detector
def fake_panel(n=100):
    idx = np.arange(n)
    return SimpleNamespace(disc=idx < 70, conf=idx >= 70, recent=idx >= 80, years=2015 + idx // 10)


def test_sign_consistency_catches_an_effect_that_lives_in_three_years():
    rng = np.random.default_rng(1)
    p = fake_panel()
    steady = 0.01 + rng.normal(0, 0.003, 100)
    burst = rng.normal(0, 0.003, 100)
    burst[:30] += 0.03                                                          # three strong years
    burst[30:80] += -0.002                                                      # five slightly wrong years
    burst[80:] += 0.01
    assert run_tests(steady, p, +1)["pass"]
    r = run_tests(burst, p, +1)
    assert r["failed"] == ["sign_consistency"]                                  # every other test is fooled by the burst
    assert r["stats"]["years"] == {"blocks": 10, "hits": 5}


def test_tests_are_sign_aware_and_report_each_failure():
    rng = np.random.default_rng(2)
    p = fake_panel()
    neg = -0.01 + rng.normal(0, 0.01, 100)
    assert run_tests(neg, p, -1)["pass"] and not run_tests(neg, p, +1)["pass"]
    assert set(run_tests(neg, p, +1)["failed"]) == {"long_run", "discovery", "confirmation", "recent", "sign_consistency"}


def test_detector_verdicts_ok_watch_failed_insufficient():
    rng = np.random.default_rng(3)
    p = fake_panel()
    good = 0.01 + rng.normal(0, 0.005, 100)
    assert detect_failure(good, p, +1)["verdict"] == "ok"
    faded = good.copy()
    faded[80:] = rng.normal(0.0005, 0.005, 20)
    assert detect_failure(faded, p, +1)["verdict"] == "watch"
    rev = good.copy()
    rev[80:] = -0.01 + rng.normal(0, 0.005, 20)
    d = detect_failure(rev, p, +1)
    assert d["verdict"] == "failed" and "contradicts" in d["reason"]
    thin = good.copy()
    thin[80:] = np.nan
    thin[80:83] = -0.05
    assert detect_failure(thin, p, +1)["verdict"] == "insufficient"


def test_cause_search_skips_a_scope_that_already_failed(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = seeded(as_of)
    rec = life.records[pattern_id(("s", "f1", 4))]
    panel = Panel.build(X, y, as_of)
    first = life.cause_search(panel, rec, as_of)
    assert first["scope"]["label"] == "high" and first["evidence"]["n_tried"] == len(first["tried"]) == 3      # one context column x three terciles
    rec["scope"] = first["scope"]
    second = life.cause_search(panel, rec, as_of)
    assert second["scope"] is None or (second["scope"]["col"], second["scope"]["label"]) != ("m_vix", "high")


# ---------------------------------------------------------------- scope permutation null
def noise_context_panel(seed, weeks=300, stocks=100, nf=30, nctx=7):
    """Thirty patterns that all reverse in a random ~half of late weeks, with seven context columns that are pure noise:
    any 'cause' the search finds is a false one."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=weeks * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(stocks)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), nf)), index=idx, columns=[f"f{i}" for i in range(nf)])
    for c in range(nctx):
        X[f"m_c{c}"] = pd.Series(rng.standard_normal(weeks), index=dates).reindex(idx.get_level_values(0)).values
    y = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx) + np.repeat(rng.normal(0, 0.02, weeks), stocks)
    q = X[[f"f{i}" for i in range(nf)]].groupby(level=0).rank(pct=True) >= 0.8
    late = np.repeat(np.arange(weeks), stocks) >= int(weeks * 0.75)
    for i in range(nf):
        y += q[f"f{i}"] * np.where(late & np.repeat(rng.random(weeks) < 0.55, stocks), -0.03, 0.012)
    return X, y, dates


def test_false_rescope_rate_on_meaningless_context_is_low():
    X, y, dates = noise_context_panel(1)
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = Lifecycle()
    life.ingest(frame([(f"f{i} q4", 0.005, "active") for i in range(30)]), as_of)
    out = life.review(Panel.build(X, y, as_of), as_of)
    assert out["failed"] >= 28                                                  # the planted reversal is detected
    assert life.counts()["rescoped"] <= 3                                       # measured 1 of 30; a cause is not conjured
    assert life.counts()["discarded"] >= 26 and life.invariants() == []
    rep = life.report()
    assert rep["failure_causes"] and sum(rep["discard_reasons"].values()) == life.counts()["discarded"]


def test_null_gate_rejects_a_scope_when_shuffled_context_finds_one_as_often(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    panel = Panel.build(X, y, as_of)
    real = seeded(as_of)
    rec = real.records[pattern_id(("s", "f1", 4))]
    found = real.cause_search(panel, rec, as_of)
    assert found["scope"] is not None and found["evidence"]["p_null"] <= 0.15 and found["evidence"]["null_reps"] == 20
    strict = seeded(as_of, {"scope_null_max": 0.0})                             # nothing can beat p_null = 1/21
    rec2 = strict.records[pattern_id(("s", "f1", 4))]
    refused = strict.cause_search(panel, rec2, as_of)
    assert refused["scope"] is None and "shuffled context" in refused["reason"]
    strict.review(panel, as_of)
    assert "not accepted as a cause" in strict.records[pattern_id(("s", "f1", 4))]["discard_reason"]
    off = seeded(as_of, {"scope_null_reps": 0})
    ev = off.cause_search(panel, off.records[pattern_id(("s", "f1", 4))], as_of)["evidence"]
    assert ev["p_null"] is None and ev["null_reps"] == 0                        # gate can be switched off, and says so


def test_null_gate_is_deterministic(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    panel = Panel.build(X, y, as_of)
    a, b = seeded(as_of), seeded(as_of)
    ra = a.cause_search(panel, a.records[pattern_id(("s", "f1", 4))], as_of)["evidence"]
    rb = b.cause_search(panel, b.records[pattern_id(("s", "f1", 4))], as_of)["evidence"]
    assert ra == rb


def test_transition_matrix_and_report_agree_with_the_log(data):
    X, y, dates = data
    as_of = dates[-1] + pd.Timedelta(days=30)
    life = seeded(as_of)
    life.review(Panel.build(X, y, as_of), as_of)
    M = life.transition_matrix()
    assert M.loc["start", "candidate"] == 3 and M.loc["candidate", "active"] == 3
    assert M.loc["active", "failed"] == 2 and M.loc["failed", "cause_search"] == 2
    assert M.loc["cause_search", "rescoped"] == 1 and M.loc["cause_search", "discarded"] == 1
    assert int(M.values.sum()) == life.report()["transitions"]
    assert Lifecycle().transition_matrix().values.sum() == 0 and Lifecycle().report()["transitions"] == 0
