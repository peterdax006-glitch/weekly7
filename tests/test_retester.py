"""Phase 23: the re-tester must accept a faithful replay, and catch every planted divergence."""
import copy

import numpy as np
import pandas as pd
import pytest

from engine import retester as R

PROV = {"code_hash": "abc123", "seed": 7}


def make_run(seed=0):
    rng = np.random.default_rng(seed)
    dates = [str(d.date()) for d in pd.bdate_range("2101-01-04", periods=3)]
    tick = ["S0001", "S0002", "S0003"]
    hold = pd.DataFrame([{"date": d, "ticker": t, "weight": 1 / 3 + 0.001 * i}
                         for i, d in enumerate(dates) for t in tick])
    trades = pd.DataFrame([{"date": d, "ticker": t, "side": "buy", "qty": 10.0 + i, "price": 50.0 + rng.random()}
                           for i, d in enumerate(dates) for t in tick[:2]])
    scores = pd.DataFrame([{"date": d, "ticker": t, "score": float(s)} for d in dates
                           for t, s in zip(tick, rng.permutation(3) + 1.0)])
    return {"provenance": dict(PROV), "holdings": hold, "trades": trades, "scores": scores,
            "weekly_returns": {"2101-W1": 0.071, "2101-W2": -0.02, "2101-W3": 0.0001},
            "adaptation_events": [{"week": 1, "kind": "tighten_stop", "value": 0.05},
                                  {"week": 3, "kind": "pace_up", "value": 0.2}],
            "pattern_activation": {"p1": {"status": "active", "effect": 0.012},
                                   "p2": {"status": "rescoped", "effect": -0.004}},
            "memory_state": {"arms": {"a": {"n": 12, "mean": 0.031}, "b": {"n": 3, "mean": -0.01}}, "ctx": ["x", "y"]}}


def test_identical_replay_passes():
    a = make_run()
    v = R.compare_runs(a, copy.deepcopy(a))
    assert v.ok and v.status == R.PASS and not v.divergences
    assert set(v.compared) == set(R.COMPONENTS) and all(n > 0 for n in v.compared.values())


def test_within_tolerance_passes_beyond_fails():
    a, b = make_run(), make_run()
    b["weekly_returns"]["2101-W1"] = 0.071 * 1.004
    b["memory_state"]["arms"]["a"]["mean"] = 0.031 * 1.004
    assert R.compare_runs(a, b).ok
    b["weekly_returns"]["2101-W1"] = 0.071 * 1.02
    v = R.compare_runs(a, b)
    assert v.status == R.FAIL and v.unexplained()[0].component == "weekly_returns"


def test_stale_code_is_not_a_parity_failure():
    a, b = make_run(), make_run()
    b["provenance"]["code_hash"] = "zzz999"
    b["weekly_returns"]["2101-W1"] = 0.5                          # enormous divergence, but the code differs
    v = R.compare_runs(a, b)
    assert v.status == R.STALE and "stale code" in v.note and not v.divergences and v.compared == {}


def test_missing_hash_never_matches_and_disk_hash_checked():
    a, b = make_run(), make_run()
    del a["provenance"]["code_hash"]
    assert R.compare_runs(a, b).status == R.STALE
    a, b = make_run(), make_run()
    assert R.compare_runs(a, b, current_hash="abc123").ok
    assert R.compare_runs(a, b, current_hash="newer").status == R.STALE


@pytest.mark.parametrize("mutate,component", [
    (lambda r: r["holdings"].__setitem__("weight", r["holdings"]["weight"] * 1.02), "holdings"),
    (lambda r: r["holdings"].__setitem__("ticker", r["holdings"]["ticker"].replace("S0003", "S0009")), "holdings"),
    (lambda r: r.__setitem__("trades", r["trades"].iloc[:-1].copy()), "trades"),
    (lambda r: r["trades"].__setitem__("price", r["trades"]["price"] * 1.01), "trades"),
    (lambda r: r["scores"].__setitem__("score", r["scores"]["score"].iloc[::-1].values), "scores"),
    (lambda r: r["adaptation_events"][0].__setitem__("kind", "loosen_stop"), "adaptation_events"),
    (lambda r: r["adaptation_events"].pop(), "adaptation_events"),
    (lambda r: r["pattern_activation"]["p1"].__setitem__("status", "discarded"), "pattern_activation"),
    (lambda r: r["pattern_activation"].pop("p2"), "pattern_activation"),
    (lambda r: r["memory_state"]["arms"]["b"].__setitem__("n", 4), "memory_state"),
    (lambda r: r["memory_state"]["ctx"].append("z"), "memory_state"),
])
def test_every_component_catches_a_planted_defect(mutate, component):
    a, b = make_run(), make_run()
    mutate(b)
    v = R.compare_runs(a, b)
    assert v.status == R.FAIL
    assert component in {d.component for d in v.unexplained()}


def test_explanation_waives_only_the_named_divergence():
    a, b = make_run(), make_run()
    b["weekly_returns"]["2101-W1"] = 0.09
    b["pattern_activation"]["p1"]["status"] = "discarded"
    ex = [{"component": "weekly_returns", "key": "2101-W1", "reason": "vendor restated a split"}]
    v = R.compare_runs(a, b, explanations=ex)
    assert v.status == R.FAIL                                                     # p1 still unexplained
    assert [d.component for d in v.unexplained()] == ["pattern_activation"]
    ex.append({"component": "pattern_activation", "key": "p1", "reason": "known reseed"})
    assert R.compare_runs(a, b, explanations=ex).ok
    blank = [{"component": "weekly_returns", "key": "2101-W1", "reason": ""}]     # no reason = no waiver
    assert R.compare_runs(a, b, explanations=blank).status == R.FAIL


def test_empty_archives_are_not_a_pass():
    empty = {"provenance": dict(PROV), **{c: ([] if c == "adaptation_events" else {}) for c in R.COMPONENTS}}
    empty["holdings"] = pd.DataFrame(); empty["trades"] = pd.DataFrame(); empty["scores"] = pd.DataFrame()
    v = R.compare_runs(empty, copy.deepcopy(empty))
    assert v.status == R.FAIL and "nothing was compared" in v.note


def test_missing_component_is_a_failure():
    a, b = make_run(), make_run()
    del b["memory_state"]
    v = R.compare_runs(a, b)
    assert v.status == R.FAIL and v.unexplained()[0].detail.startswith("component missing")


def test_rel_diff_edges():
    assert R.rel_diff(0, 1e-12) == 0 and R.rel_diff(float("nan"), float("nan")) == 0
    assert R.rel_diff(float("nan"), 1) == float("inf")
    assert R.rel_diff(100, 100.5) == pytest.approx(0.005, abs=1e-4)
    assert R.rel_diff(-1, 1) == 2.0


def test_summary_text():
    a, b = make_run(), make_run()
    assert R.compare_runs(a, b).summary().startswith("PASS")
    b["scores"]["score"] = b["scores"]["score"] + 5
    assert "unexplained" in R.compare_runs(a, b).summary()


def test_archive_roundtrip_and_retest(tmp_path):
    a = make_run()
    a.update({"window": "W1", "seed": 7, "config": {"k": 1}})
    p = tmp_path / "arch.json"
    R.save_archive(p, a)
    seen = {}

    def faithful(blind):
        seen.update(blind)
        return make_run()

    v = R.retest(p, faithful)
    assert v.ok
    assert set(seen) == {"window", "seed", "config"}                              # results were hidden from the runner

    def drifting(blind):
        r = make_run(); r["trades"].loc[0, "qty"] *= 1.05
        return r

    assert R.retest(p, drifting).status == R.FAIL


def test_retest_many_records_runner_crash_as_fail(tmp_path):
    a = make_run()
    R.save_archive(tmp_path / "a.json", a)
    R.save_archive(tmp_path / "b.json", a)
    calls = []

    def runner(blind):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("worker died")
        return make_run()

    res, tally = R.retest_many({"a": tmp_path / "a.json", "b": tmp_path / "b.json"}, runner)
    assert res["a"].ok and res["b"].status == R.FAIL and "worker died" in res["b"].note
    assert tally == {R.PASS: 1, R.FAIL: 1, R.STALE: 0}


# ---------------- diagnostics
def test_divergence_table_orders_worst_first():
    a, b = make_run(), make_run()
    b["weekly_returns"]["2101-W1"] = 0.09
    b["memory_state"]["arms"]["b"]["n"] = 4
    b["pattern_activation"].pop("p2")
    v = R.compare_runs(a, b)
    t = R.divergence_table(v)
    assert t.iloc[0]["magnitude"] != t.iloc[0]["magnitude"] or t.iloc[0]["magnitude"] >= t["magnitude"].max()
    assert set(t["component"]) == {"weekly_returns", "memory_state", "pattern_activation"}
    w = R.worst_by_component(v)
    assert w["holdings"] == (0, 0.0) and w["weekly_returns"][0] == 1
    assert len(R.divergence_table(R.compare_runs(a, make_run()))) == 0


def test_tolerance_sweep_and_noise_floor():
    a, b = make_run(), make_run()
    assert R.noise_floor(a, b) == 0.0                                            # bit-exact
    b["memory_state"]["arms"]["a"]["mean"] = 0.031 * (1 + 3e-4)
    assert R.noise_floor(a, b) == 1e-3
    sweep = dict((t, s) for t, s, _ in R.tolerance_sweep(a, b))
    assert sweep[0.0] == R.FAIL and sweep[0.005] == R.PASS
    b["memory_state"]["arms"]["a"]["mean"] = 0.031 * 3
    assert R.noise_floor(a, b) is None
    stale = make_run(); stale["provenance"]["code_hash"] = "other"
    assert {s for _, s, _ in R.tolerance_sweep(a, stale)} == {R.STALE}


def test_first_divergence_week_and_gap():
    a = pd.Series([0.01, 0.02, 0.03, 0.04])
    b = a.copy(); b.iloc[2] = 0.031; b.iloc[3] = 0.05
    assert R.first_divergence_week(a, b) == 2
    assert R.first_divergence_week(a, a.copy()) is None
    assert R.first_divergence_week(a, a.iloc[:3]) == 3                          # replay stopped early
    g = R.cumulative_gap(a, b)
    assert g["gap"] > 0 and g["archive"] == pytest.approx(1.01 * 1.02 * 1.03 * 1.04 - 1)
    assert R.per_week_divergence(a, b).abs().max() == pytest.approx(0.01)


def test_turnover_distinguishes_same_names_different_trading():
    a = make_run()["holdings"]
    same = a.copy()
    churn = a.copy()
    churn.loc[churn["date"] == churn["date"].unique()[1], "weight"] *= 0.0
    assert R.turnover(a) == R.turnover(same)
    assert R.turnover(churn) > R.turnover(a)
    assert R.turnover(pd.DataFrame()) == 0.0


def test_type_and_era_breakdown():
    a, b = make_run(), make_run()
    b["scores"]["score"] = b["scores"]["score"] + 50
    tb = R.type_breakdown(a, b).set_index("component")
    assert tb.loc["scores", "status"] == "FAIL" and tb.loc["holdings", "status"] == "PASS"
    stale = make_run(); stale["provenance"]["code_hash"] = "x"
    assert set(R.type_breakdown(a, stale)["status"]) == {"n/a"}
    v = {"w1": R.compare_runs(a, make_run()), "w2": R.compare_runs(a, b), "w3": R.compare_runs(a, stale)}
    starts = {"w1": "1980-01-01", "w2": "1999-01-01", "w3": "2010-01-01"}
    e = R.era_breakdown(v, starts)
    assert e["pre1997"][R.PASS] == 1 and e["1997-2000"][R.FAIL] == 1 and e["2001+"][R.STALE] == 1


def test_reports_written(tmp_path):
    a, b = make_run(), make_run()
    b["weekly_returns"]["2101-W2"] = 0.05
    v = R.compare_runs(a, b)
    md = R.markdown_report("w1", v, a, b)
    assert "FAIL" in md and "2101-W2" in md and "First divergent week" in md
    stale = make_run(); stale["provenance"]["code_hash"] = "x"
    assert "different code" in R.markdown_report("w2", R.compare_runs(a, stale))
    summ = R.write_reports({"w1": v, "w2": R.compare_runs(a, make_run())}, tmp_path / "out", stamp={"code_hash": "abc"})
    assert summ["tally"] == {R.PASS: 1, R.FAIL: 1, R.STALE: 0}
    assert (tmp_path / "out" / "w1.md").exists() and (tmp_path / "out" / "summary.json").exists()


# ---------------- provenance details, component subsets, archived live runs
from engine import provenance


def _res(prov=None, weekly=True):
    d = {"diagnosis": {"year_return": 0.31, "mean_week": 0.0052, "weeks_ge_7": 4, "max_dd": -0.12}, "config": {}}
    if prov is not None:
        d["provenance"] = prov
    if weekly:
        d["weekly_returns"] = {"w000": 0.02, "w001": -0.01, "w002": 0.05}
    return d


def _replay(**over):
    r = {"year_return": 0.31, "mean_week": 0.0052, "weeks_ge_7": 4, "max_dd": -0.12,
         "weekly_returns": {"w000": 0.02, "w001": -0.01, "w002": 0.05}}
    r.update(over)
    return r


def _stamp(files=("engine/provenance.py",)):
    return {"code_hash": provenance.code_hash(list(files)), "code_files": list(files), "code_mixed": []}


def test_code_mixed_makes_a_run_stale():
    a, b = make_run(), make_run()
    a["provenance"] = {"code_hash": "h", "code_mixed": ["engine/memory.py"]}
    b["provenance"] = {"code_hash": "h", "code_mixed": []}
    v = R.compare_runs(a, b)
    assert v.status == R.STALE and "engine/memory.py" in v.note
    assert R.compare_runs(b, a).status == R.STALE                          # either side


def test_disk_check_uses_the_archived_file_list():
    a, b = make_run(), make_run()
    a["provenance"] = b["provenance"] = _stamp()
    assert R.compare_runs(a, b, disk=True).ok
    a["provenance"] = b["provenance"] = dict(_stamp(), code_hash="0" * 16)     # code changed since the run
    assert R.compare_runs(a, b, disk=True).status == R.STALE
    assert R.compare_runs(a, b, disk=False).ok                                 # the two stamps agree with each other


def test_component_subset_and_summary():
    a, b = make_run(), make_run()
    a["summary"], b["summary"] = {"year_return": 0.3, "n": 4}, {"year_return": 0.3, "n": 4}
    v = R.compare_runs(a, b, components=("summary",))
    assert v.ok and set(v.compared) == {"summary"}
    b["summary"]["year_return"] = 0.31
    assert R.compare_runs(a, b, components=("summary",)).status == R.FAIL
    del b["summary"]
    assert R.compare_runs(a, b, components=("summary",)).status == R.FAIL       # a missing requested component fails
    b["weekly_returns"]["2101-W1"] = 9.0                                         # not requested: not looked at
    assert R.compare_runs(a, make_run(), components=("holdings",)).ok


def test_judge_archived_pass_stale_unstamped():
    stamped, v = R.judge_archived(_res(_stamp()), _replay())
    assert stamped and v.ok and set(v.compared) == {"summary", "weekly_returns"}
    stamped, v = R.judge_archived(_res(dict(_stamp(), code_hash="f" * 16)), _replay(year_return=0.9))
    assert stamped and v.status == R.STALE and v.compared == {}                # huge gap, but the code changed
    stamped, v = R.judge_archived(_res(dict(_stamp(), code_mixed=["engine/x.py"])), _replay())
    assert v.status == R.STALE
    stamped, v = R.judge_archived(_res(None), _replay())
    assert not stamped and v.ok                                                  # compared, but not 'verified'
    stamped, v = R.judge_archived(_res(None), _replay(year_return=0.5))
    assert not stamped and v.status == R.FAIL


def test_judge_archived_relative_half_percent():
    ok = R.judge_archived(_res(_stamp()), _replay(year_return=0.31 * 1.004))[1]
    assert ok.ok
    bad = R.judge_archived(_res(_stamp()), _replay(year_return=0.31 * 1.01))[1]
    assert bad.status == R.FAIL and bad.unexplained()[0].key == "year_return"
    # the OLD absolute rule (|diff| < 0.005) would have passed this: 0.31 vs 0.3131 is 1% relative
    assert abs(0.31 - 0.31 * 1.01) < 0.005
    w = R.judge_archived(_res(_stamp()), _replay(weekly_returns={"w000": 0.02, "w001": -0.01, "w002": 0.056}))[1]
    assert w.status == R.FAIL and w.unexplained()[0].component == "weekly_returns"
    short = R.judge_archived(_res(_stamp()), _replay(weekly_returns={"w000": 0.02}))[1]
    assert short.status == R.FAIL                                                # a replay that lost weeks fails
    nowk = R.judge_archived(_res(_stamp(), weekly=False), _replay())[1]
    assert nowk.ok and set(nowk.compared) == {"summary"}                         # older archives: headline numbers only


def test_summary_divergences_are_labelled_summary():
    v = R.judge_archived(_res(_stamp()), _replay(mean_week=0.009))[1]
    assert v.status == R.FAIL and {d.component for d in v.unexplained()} == {"summary"}
    waived = R.compare_runs({"provenance": _stamp(), "summary": {"a": 1.0}}, {"provenance": _stamp(), "summary": {"a": 2.0}},
                            components=("summary",), explanations=[{"component": "summary", "key": "a", "reason": "known"}])
    assert waived.ok
