import json
import numpy as np
import pandas as pd
import pytest
from engine import run_report as R

WIN = {"train": {"start": "2010-01-01", "end": "2015-12-31"}, "validation": {"start": "2016-01-01", "end": "2016-12-31"},
       "test": {"start": "2017-01-01", "end": "2019-12-31"}}
GATES = {g: True for g in R.GATES}


def weekly(n=150, mu=0.07, sd=0.03, seed=0):
    return np.random.default_rng(seed).normal(mu, sd, n)


def patterns():
    return pd.DataFrame({"status": ["active", "active", "rescoped", "discarded", "rejected", "no_gain"],
                         "p_real": [.9, .8, .5, .1, .05, .2], "p_hallucinated": [.05, .1, .3, .6, .7, .5],
                         "p_coincidence": [.05, .1, .2, .3, .25, .3]})


def run(**kw):
    r = dict(run_id="R1", git_commit="abc", canon_hash="c", blueprint_version="4.0", config_hash="h", data_snapshot="d",
             seed=7, windows=WIN, window_ids=["w1"], weekly=weekly(), turnover=0.5, costs=0.001, gates=dict(GATES),
             pred_dir=[1, -1, 1, 1], actual_ret=[.02, -.01, -.03, .04], prob=[.9, .2, .6, .7], outcome=[1, 0, 1, 1],
             patterns=patterns(), analog_distances=[.1, .2, .3, .4], analog_forecast=[.01, .02, -.01, .03],
             analog_realised=[.02, .01, -.02, .02], memory=dict(lesson_count=3, accepted_lessons=2, rejected_lessons=1,
                                                                 shock_events=0, memory_switches=1),
             adaptation=dict(parameter_changes=2, reverts=0, evidence="e", confidence=0.7))
    r.update(kw)
    return r


def test_tier_maths_on_known_series():
    w = [0.06, 0.08, 0.12, 0.02, -0.10]
    t1, t2 = R.tier1(w), R.tier2(w)
    assert t1["share_in_band"] == pytest.approx(.4) and t1["share_gt_10"] == pytest.approx(.2) and t1["share_lt_5"] == pytest.approx(.4)
    assert t1["share_in_band"] + t1["share_gt_10"] + t1["share_lt_5"] == pytest.approx(1)
    assert t2["worst_week"] == -0.10 and t2["catastrophic_losses"] == 0
    assert R.max_drawdown([0.5, -0.5]) == pytest.approx(-0.5) and R.max_drawdown([0.1, 0.1]) == 0.0


def test_eight_of_ten_and_calibration():
    assert R.eight_of_ten([1] * 10 + [-1] * 10) == pytest.approx(3 / 11)      # windows 0..2 have >=8 positives
    assert R.eight_of_ten([1] * 5) is None
    perfect = R.calibration([0, 1, 0, 1], [0, 1, 0, 1])
    assert perfect["brier"] == 0 and perfect["ece"] == 0
    assert R.calibration([.9, .9, .9, .9], [0, 0, 0, 0])["ece"] == pytest.approx(.9)
    with pytest.raises(ValueError):
        R.calibration([1.5], [1])


def test_full_report_adopts_and_renders_every_section():
    rep = R.build_report(run(), "2026-01-01")
    assert rep["decision"] == "ADOPT" and rep["missing"] == []
    txt = R.render_text(rep)
    for h in ("RUN ID", "GIT COMMIT", "CANON HASH", "BLUEPRINT VERSION", "CONFIG HASH", "DATA SNAPSHOT", "RANDOM SEED", "WINDOWS",
              "TRAIN PERIOD", "VALIDATION PERIOD", "TEST PERIOD", "TIER 1", "TIER 2", "TIER 3", "ALGORITHM", "ANALOGS", "MEMORY",
              "ADAPTATION", "GATES", "DECISION", "REASON", "share 5-10%", "8/10 rate", "P(real) distribution", "future scramble"):
        assert h in txt, h
    assert rep["algorithm"]["active"] == 2 and rep["algorithm"]["failed"] == 2
    assert rep["algorithm"]["false_discovery"]["expected_false_active"] == pytest.approx(0.3)
    json.dumps(rep, allow_nan=False)


def test_planted_leak_and_failed_gate_reject():
    leak = {**WIN, "test": {"start": "2015-06-01", "end": "2019-12-31"}}
    assert R.build_report(run(windows=leak), "t")["decision"] == "REJECT"
    g = {**GATES, "future_scramble": False}
    rep = R.build_report(run(gates=g), "t")
    assert rep["decision"] == "REJECT" and "future_scramble" in rep["reason"]


def test_missing_gate_or_short_evidence_continues_testing_not_adopts():
    g = dict(GATES)
    del g["retester"]
    rep = R.build_report(run(gates=g), "t")
    assert rep["decision"] == "CONTINUE TESTING" and "gates.retester" in rep["missing"]
    assert R.build_report(run(weekly=weekly(20)), "t")["decision"] == "CONTINUE TESTING"


def test_requested_adopt_is_overruled_and_bad_results_reject():
    rep = R.build_report(run(weekly=weekly(mu=0.01), requested_decision="ADOPT"), "t")
    assert rep["decision"] == "REJECT" and "overruled" in rep["reason"]
    crash = weekly()
    crash[3] = -0.4
    assert R.build_report(run(weekly=crash), "t")["decision"] == "REJECT"


def test_empty_report_is_all_missing_never_zero():
    rep = R.build_report({"run_id": "E"}, "t")
    assert rep["decision"] == "CONTINUE TESTING" and rep["tier1"]["mean_week"] is None and rep["tier1"]["n_weeks"] == 0
    assert "tier1.mean_week" in rep["missing"] and "n/a" in R.render_text(rep)
    json.dumps(rep, allow_nan=False)


def test_nan_weeks_are_dropped_and_direction_accuracy():
    w = np.append(weekly(120), [np.nan, np.nan])
    assert R.tier1(w)["n_weeks"] == 120
    assert R.tier3(w, [1, -1, 1], [.1, -.2, -.3])["direction_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        R.tier3(w, [1], [.1, .2])


def test_write_report_never_overwrites(tmp_path):
    rep = R.build_report(run(), "t")
    j, t = R.write_report(rep, tmp_path)
    assert json.loads(j.read_text())["run_id"] == "R1" and "DECISION" in t.read_text()
    with pytest.raises(FileExistsError):
        R.write_report(rep, tmp_path)
    with pytest.raises(ValueError):
        R.write_report({**rep, "run_id": "../x"}, tmp_path)


# ---- regimes, eras, baseline, unproven, html ----
from engine import baseline as B
from engine.registry import Registry


def test_breakdown_flags_thin_and_dominance():
    w = np.concatenate([np.full(100, 0.07), np.full(10, 0.01)])
    lab = ["bull"] * 100 + ["crash"] * 10
    b = R.breakdown(w, lab)
    assert b["groups"]["crash"]["thin"] and not b["groups"]["bull"]["thin"] and b["n_thin"] == 1
    assert b["dominance"] == pytest.approx(7 / 7.1)
    with pytest.raises(ValueError):
        R.breakdown(w, lab[:5])


def test_breakdown_dominance_undefined_when_total_not_positive():
    assert R.breakdown([-0.1] * 30, ["a"] * 30)["dominance"] is None


def test_era_labels():
    assert R.era_labels(["1995-05-01", "2005-01-01", "2015-06-01", "2024-01-01"]) == ["<2000", "2000-2009", "2010-2019", "2020+"]


def test_regime_and_era_in_report_and_unproven():
    n = 150
    dates = pd.date_range("2018-01-05", periods=n, freq="7D")
    lab = ["bull"] * 140 + ["bear"] * 10
    rep = R.build_report(run(weekly=weekly(n), week_dates=list(dates), regimes=lab, unproven=["live fills untested"]), "t")
    assert rep["regimes"]["groups"]["bear"]["thin"] and rep["eras"]["groups"]
    u = " | ".join(rep["unproven"])
    assert "regimes with too few weeks" in u and "live fills untested" in u and "no comparison with the frozen baseline" in u
    txt = R.render_text(rep)
    assert "REGIMES" in txt and "(THIN)" in txt and "UNPROVEN (" in txt
    no = R.build_report(run(), "t")
    assert "no regimes breakdown supplied" in " ".join(no["unproven"])


def _baseline(tmp_path, mean=0.002):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"event": "historical_test", "mean_week": mean, "median_week": mean, "worst_week": -0.2,
                             "max_dd": -0.6, "pct_ge_7": 0.02, "pct_le_m7": 0.03, "win_weeks": 0.5}) + "\n")
    d, _ = B.freeze({"cfg": 1}, "2026-01-01", root=tmp_path / "bl", registry=Registry(p))
    return str(d)


def test_baseline_comparison_and_decision(tmp_path):
    rep = R.build_report(run(baseline_dir=_baseline(tmp_path)), "t")
    bl = rep["baseline"]
    assert bl["rows"]["mean_week"]["verdict"] == "better" and rep["decision"] == "ADOPT"
    assert "VS BASELINE" in R.render_text(rep)
    assert any("baseline never measured: mover_accuracy" in u for u in rep["unproven"])
    (tmp_path / "w").mkdir()
    worse = R.build_report(run(baseline_dir=_baseline(tmp_path / "w", mean=0.09)), "t")
    assert worse["decision"] == "REJECT" and "worse than the frozen baseline" in worse["reason"]


def test_corrupt_baseline_blocks_adoption(tmp_path):
    rep = R.build_report(run(baseline_dir=str(tmp_path / "missing")), "t")
    assert rep["decision"] == "CONTINUE TESTING" and "error" in rep["baseline"]
    assert any("baseline comparison unavailable" in u for u in rep["unproven"])


def test_html_is_escaped_and_carries_decision_and_unproven(tmp_path):
    rep = R.build_report(run(run_id="R<b>1", unproven=["<script>alert(1)</script>"]), "t")
    h = R.render_html(rep)
    assert "<script>" not in h and "&lt;script&gt;" in h and "UNPROVEN" in h and rep["decision"] in h
    assert h.count("<h2>") >= 8
    rep2 = R.build_report(run(), "t")
    j, t = R.write_report(rep2, tmp_path / "out", html_dir=tmp_path / "docs" / "reports")
    assert (tmp_path / "docs" / "reports" / "R1.html").read_text().startswith("<!doctype html>")
    with pytest.raises(FileExistsError):
        R.write_report(rep2, tmp_path / "out2", html_dir=tmp_path / "docs" / "reports")
