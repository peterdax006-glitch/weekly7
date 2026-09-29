"""Phase 29/39 dashboard builder (scripts/dashboard_build.py) and the Phase 36 run-report additions (direction accuracy,
analog distance distribution, memory shock events). Synthetic roots only; nothing reads the real caches."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import dashboard_build as D
from scripts import site_safety as S
from engine import resources as R
from engine import run_report as RR

NOW = 1_800_000_000.0
FOUNDATION = """# Foundation status

| unit | phases | range | lines | status | tests |
|---|---|---|---|---|---|
| B01_x | 1 | 1,500-3,000 | 2,376 | IN RANGE | - |
| B02_y | 2 | 1,000-2,000 | 400 | BELOW RANGE | - |
"""
CHECKLIST = """# Checklist
## Phase 0
- [x] P0.1 Canon integrity. Evidence: verified.
  - [!] nested item is not counted
- [~] P0.2 Registry with provenance.
## Test
- [!] T17 Planted calibration NOT VALIDATED.
- [ ] T3 Tiered objective reached.
- [?] T20 Something unproven.
"""


def make_root(tmp_path, checklist=CHECKLIST, foundation=FOUNDATION, full=True):
    r = tmp_path / "repo"
    for d in ("state/build", "state/reports", "state/research", "state/quality", "docs"):
        (r / d).mkdir(parents=True, exist_ok=True)
    (r / "state/build/FOUNDATION.md").write_text(foundation, encoding="utf-8")
    (r / "state/CHECKLIST.md").write_text(checklist, encoding="utf-8")
    (r / "state/heartbeat.json").write_text(json.dumps({"t": "2026-09-28T00:00:00", "jobs": []}), encoding="utf-8")
    if full:
        (r / "state/build/bible_trace.json").write_text(json.dumps({
            "generated": "2026-09-29T05:05Z", "index": {"modules": 3, "test_files": 2, "evidence_files": 4},
            "summary": {"counts": {"[x]": 5, "[~]": 3, "[?]": 1, "[!]": 2, "[ ]": 4},
                        "per_phase": {"1": {"counts": {"[x]": 3, "[!]": 2}}, "2": {"counts": {"[ ]": 4, "[x]": 2}}}},
            "top_missing": [{"id": "2.0.7", "phase": 2, "text": "maximum relative error"}],
            "failing_tests_cache": {"tests/test_a.py": ["test_one"]}, "integration_pending": {"adaptive": ["hook"]}}), encoding="utf-8")
        (r / "state/reports/final_report.json").write_text(json.dumps({
            "leakage": {"verdict": "PASS", "checks": {"feature_parity": "pass"}},
            "reproducibility": {"verdict": "FAIL", "checks": {"fresh_process": "fail"}},
            "validation": {"quality_gate": {"exit_code": 0, "failing": 0}, "planted_calibration": {"validated": False, "failed": ["fdr"]}, "parity": {"passed": True, "failed": []}},
            "checklist": {"counts": {"validated": 3}},
            "performance": {"weekly": {"mean": 0.0076, "median": 0.0027, "share_in_band_5_10": 0.23, "share_positive": 0.52, "worst": -0.51, "weeks": 2049}}}), encoding="utf-8")
        (r / "state/research/backtest_summary.json").write_text(json.dumps({"weekly7": {"mean_week": 0.0059, "pct_ge_7": 0.045, "pct_le_m7": 0.02, "win_weeks": 0.57, "worst_week": -0.21, "max_dd": -0.46}}), encoding="utf-8")
        rows = [{"t": "2026-09-01T10:00:00", "event": "frontier", "outcome": "reject", "reason": "drawdown too deep", "experiment_id": "E1",
                 "metrics": {"mean_week": 0.002}, "train_range": "2008-01-01..2015-12-31", "test_range": "2016-01-01..2019-12-31"},
                {"t": "2026-09-02T10:00:00", "event": "livesim_cycle", "outcome": None, "reason": "SEALEDROW_MARKER", "experiment_id": "E2"}]
        (r / "state/experiments.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    return r


def build(r, **kw):
    kw.setdefault("built_at", "2026-09-28T00:00:00Z")
    kw.setdefault("now", NOW)
    kw.setdefault("free_fn", lambda: 12.0)
    return D.build(r, **kw)


# ---------------------------------------------------------------- parsers
def test_parse_foundation_and_checklist():
    rows = D.parse_foundation(FOUNDATION)
    assert [r["unit"] for r in rows] == ["B01_x", "B02_y"] and rows[1]["status"] == "BELOW RANGE"
    ck = D.parse_checklist(CHECKLIST)
    assert ck["counts"] == {"validated": 1, "implemented / testing": 1, "failed": 1, "not started": 1, "unproven": 1}
    assert all("nested" not in i["text"] for i in ck["items"])
    assert D.parse_foundation("")== [] and D.parse_checklist(None)["items"] == []


# ---------------------------------------------------------------- build
def test_build_writes_a_page_with_every_section_and_honest_counts(tmp_path):
    r = make_root(tmp_path)
    res = build(r)
    page = (r / "docs/dashboard.html").read_text(encoding="utf-8")
    for h in ("Foundation status", "Bible trace", "Master checklist", "Final-report audits", "Latest research numbers", "Experiment memory", "Running jobs"):
        assert f"<h2>{h}</h2>" in page
    assert "1 of 2 units are in their Bible line range" in page and "below range: B02_y" in page
    assert "test_one" in page and "maximum relative error" in page
    assert "Planted calibration" in page and "fdr" in page
    assert "0.76%" in page and "drawdown too deep" in page
    assert res["problems"] == [] and len(res["sha256"]) == 64
    assert "\r" not in page                                             # byte-exact LF so hashes are portable


def test_reproducibility_fail_is_shown_as_fail_not_hidden(tmp_path):
    r = make_root(tmp_path)
    build(r)
    page = (r / "docs/dashboard.html").read_text(encoding="utf-8")
    assert 'b-bad">FAIL' in page and "Audits passing</div><div class=\"v\">1/2" in page


def test_livesim_rows_only_counted_never_shown(tmp_path):
    r = make_root(tmp_path)
    build(r)
    page = (r / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "SEALEDROW_MARKER" not in page and "1 blind-window cycle records" in page


def test_missing_sources_are_stated_not_zeroed_and_never_crash(tmp_path):
    r = make_root(tmp_path, full=False)
    res = build(r)
    page = (r / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "Not available" in page and "state/build/bible_trace.json" in page
    assert len(res["problems"]) >= 3 and "Sources that were not available" in page
    empty = tmp_path / "empty"
    (empty / "docs").mkdir(parents=True)
    build(empty)                                                        # nothing exists at all
    assert "Status board" in (empty / "docs/dashboard.html").read_text(encoding="utf-8")


def test_build_is_deterministic_and_dry_run_writes_nothing(tmp_path):
    r = make_root(tmp_path)
    a = build(r, dry_run=True)
    assert not (r / "docs/dashboard.html").exists()
    b = build(r)
    c = build(r)
    assert a["sha256"] == b["sha256"] == c["sha256"]
    assert build(r, built_at="2026-10-01T00:00:00Z")["sha256"] != b["sha256"]


def test_html_is_escaped(tmp_path):
    r = make_root(tmp_path, checklist="## S\n- [!] T1 <script>alert(1)</script> broke\n")
    build(r)
    page = (r / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "<script>alert" not in page and "&lt;script&gt;alert(1)" in page


# ---------------------------------------------------------------- safety
@pytest.mark.parametrize("payload,kind", [
    ("- [!] T1 key sk-ABCDEFGHIJKLMNOPQRSTUVWX leaked", "openai"),
    ("- [!] T1 alpaca PKABCDEFGHIJKLMNOPQR", "alpaca"),
    ("- [!] T1 password = hunter2hunter2hunter2", "assignment"),
    ("- [!] T1 window 2154-03-05 revealed", "disguised date"),
    ("- [!] T1 stock S0123 was bought", "disguised code"),
])
def test_a_secret_or_sealed_mark_stops_the_build_and_writes_nothing(tmp_path, payload, kind):
    r = make_root(tmp_path, checklist=f"## S\n{payload}\n")
    with pytest.raises(S.Leak):
        build(r)
    assert not (r / "docs/dashboard.html").exists(), f"{kind}: nothing may be written when the audit fails"
    assert not (r / "state/research/dashboard/manifest.json").exists()


def test_a_failed_rebuild_leaves_the_previous_page_intact(tmp_path):
    r = make_root(tmp_path)
    build(r)
    before = (r / "docs/dashboard.html").read_bytes()
    (r / "state/CHECKLIST.md").write_text("## S\n- [!] T1 leak sk-ABCDEFGHIJKLMNOPQRSTUVWX\n", encoding="utf-8")
    with pytest.raises(S.Leak):
        build(r)
    assert (r / "docs/dashboard.html").read_bytes() == before


def test_sealed_and_cache_trees_are_unreachable_through_the_loader():
    src = D.Sources(ROOT)
    with pytest.raises(S.SealedAccess):
        src.text("state/livesim/anything.json")
    with pytest.raises(S.SealedAccess):
        src.json("data/cache/prices.parquet")
    with pytest.raises(S.SealedAccess):
        src.text("state/.env")


def test_real_repo_page_passes_its_own_audit():
    res = D.build(ROOT, dry_run=True, free_fn=lambda: 10.0, info=R.process_info)
    assert len(res["sha256"]) == 64                                     # build() audits before returning; a leak would raise


# ---------------------------------------------------------------- check
def test_check_detects_stale_edited_and_leaky_pages(tmp_path):
    r = make_root(tmp_path)
    assert D.check(r)["ok"] is False                                    # never built
    build(r)
    assert D.check(r)["ok"] is True
    (r / "state/CHECKLIST.md").write_text(CHECKLIST + "- [x] T99 new item\n", encoding="utf-8")
    c = D.check(r)
    assert not c["ok"] and any("CHECKLIST.md changed" in x for x in c["reasons"])
    build(r)
    page = r / "docs/dashboard.html"
    page.write_text(page.read_text(encoding="utf-8") + "<!-- hand edit -->", encoding="utf-8")
    assert any("differs from the manifest" in x for x in D.check(r)["reasons"])
    build(r)
    page.write_text(page.read_text(encoding="utf-8") + "sk-ABCDEFGHIJKLMNOPQRSTUVWX", encoding="utf-8")
    assert any(x.startswith("leak") for x in D.check(r)["reasons"])
    (r / "state/build/FOUNDATION.md").unlink()
    build(r)
    (r / "state/CHECKLIST.md").unlink()
    assert any("is gone" in x for x in D.check(r)["reasons"])


# ---------------------------------------------------------------- running jobs on the page
def test_running_jobs_show_age_and_stale_findings(tmp_path):
    r = make_root(tmp_path)
    reg = R.ProcRegistry(r / "state/procs.json")
    reg.register("alpha", 4321, "python a.py", r / "out", 2.0, now=NOW - 7300, ppid=1, create_time=100.0)
    reg.heartbeat("alpha", NOW - 4000)
    info = lambda pid: {"alive": True, "create_time": 100.0, "ppid": 1, "rss_gb": 1.5} if pid in (4321,) else {"alive": pid == 1, "create_time": 1.0, "ppid": 0, "rss_gb": None}
    build(r, info=info)
    page = (r / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "alpha" in page and "2h 1m" in page and "1.50 GB" in page       # age from the registry, rss from the process
    assert "silent" in page                                                # 4000 s without a heartbeat (limit 300 s)
    assert "12.0 GB" in page                                               # injected free memory


def test_no_procs_file_is_said_plainly(tmp_path):
    r = make_root(tmp_path)
    build(r)
    assert "state/procs.json does not exist yet" in (r / "docs/dashboard.html").read_text(encoding="utf-8")


def test_age_formatting():
    assert D.age(None) == "n/a" and D.age(59) == "0m 59s" and D.age(3700) == "1h 1m" and D.age(90000) == "1d 1h" and D.age(-5) == "0m 0s"


# ================================================================== Phase 36 additions to engine/run_report.py
def test_direction_accuracy_separates_skill_from_luck_and_from_the_majority_guess():
    rng = np.random.default_rng(0)
    ret = rng.normal(0.002, 0.03, 4000)
    skilled = np.where(rng.random(4000) < 0.7, np.sign(ret), -np.sign(ret))
    r = RR.direction_accuracy_report(skilled, ret)
    assert 0.67 < r["accuracy"] < 0.73 and r["ci_low"] < r["accuracy"] < r["ci_high"] and r["beats_majority"]
    coin = RR.direction_accuracy_report(rng.choice([-1, 1], 4000), ret)
    assert abs(coin["accuracy"] - 0.5) < 0.04 and not coin["beats_majority"]
    always_up = RR.direction_accuracy_report(np.ones(4000), 0.05 + np.abs(ret) * (rng.random(4000) < 0.7) - np.abs(ret) * (rng.random(4000) < 0.0) * 0)
    assert always_up["accuracy"] == pytest.approx(1.0) and always_up["majority_baseline"] == pytest.approx(1.0) and not always_up["beats_majority"]


def test_direction_accuracy_by_confidence_and_the_80_percent_bar():
    rng = np.random.default_rng(1)
    n = 6000
    conf = rng.uniform(0.5, 1.0, n)
    ret = rng.normal(0, 0.02, n)
    hit = rng.random(n) < conf                                          # calibrated: accuracy == confidence
    pred = np.where(hit, np.sign(ret), -np.sign(ret))
    r = RR.direction_accuracy_report(pred, ret, conf)
    assert r["at_or_above_0.8"]["n"] > 800 and 0.86 < r["at_or_above_0.8"]["accuracy"] < 0.94 and r["at_or_above_0.8"]["meets_80"]
    accs = [b["accuracy"] for b in r["by_confidence"]]
    assert accs == sorted(accs) and len(accs) == 5
    over = RR.direction_accuracy_report(pred, ret, np.full(n, 0.95))    # claims 95% but hits ~75%: the bar is not met
    assert not over["at_or_above_0.8"]["meets_80"]


def test_direction_accuracy_degenerate_inputs_are_none_not_zero():
    e = RR.direction_accuracy_report([0, 0], [0.1, 0.2])
    assert e["n"] == 0 and e["accuracy"] is None and e["by_confidence"] == []
    assert RR.direction_accuracy_report([], [])["accuracy"] is None
    assert RR.direction_accuracy_report([1, -1], [0.0, np.nan])["n"] == 0
    with pytest.raises(ValueError):
        RR.direction_accuracy_report([1, 1], [0.1])
    with pytest.raises(ValueError):
        RR.direction_accuracy_report([1, 1], [0.1, 0.1], [0.6])


def test_analog_distance_distribution_flags_far_neighbours():
    rng = np.random.default_rng(2)
    d = np.sort(rng.gamma(2.0, 0.5, (300, 5)), axis=1)                  # ranked neighbours per query
    a = RR.analog_distance_distribution(d)
    q = a["quantiles"]
    assert a["n"] == 1500 and q["p0"] <= q["p50"] <= q["p100"] and sum(a["histogram"]["counts"]) == 1500
    assert a["nearest"]["median"] < q["p50"] and a["kth_over_first"] > 1.0
    assert a["far_share"] >= 0.09 and a["near_share"] >= 0.09
    fixed = RR.analog_distance_distribution(d, near=0.1, far=1.5)
    assert fixed["far_threshold"] == 1.5 and 0 < fixed["far_share"] < 0.5


def test_analog_distance_empty_and_nan_cases():
    assert RR.analog_distance_distribution([])["quantiles"] is None
    assert RR.analog_distance_distribution([np.nan, np.nan])["n"] == 0
    one = RR.analog_distance_distribution([0.3, 0.3, 0.3])                # zero spread must not divide by zero
    assert one["quantiles"]["p50"] == pytest.approx(0.3)


def test_shock_events_finds_planted_spikes_and_uses_no_lookahead():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    x[100] += 12
    x[150] -= 12
    dates = [f"2020-01-{1 + i % 28:02d}" if False else str(d.date()) for i, d in enumerate(__import__("pandas").date_range("2020-01-06", periods=200, freq="W"))]
    r = RR.memory_shock_events(dates, x)
    assert r["count"] == 2 and {e["direction"] for e in r["events"]} == {"up", "down"}
    assert r["events"][0]["date"] == dates[100] and r["events"][0]["z"] > 8
    # no look-ahead: changing the future must not change an earlier event
    y = x.copy()
    y[180:] += 50
    assert RR.memory_shock_events(dates, y)["events"][0] == r["events"][0]
    # control: pure noise yields (almost) nothing at |z| >= 4
    assert RR.memory_shock_events(dates, rng.normal(0, 1, 200), z_thr=4.0)["count"] == 0


def test_shock_cooldown_merges_a_burst_and_flat_series_is_silent():
    x = np.zeros(60)
    x[:40] = np.random.default_rng(4).normal(0, 1, 40)
    x[45], x[46], x[47] = 30, 45, 25
    dates = [str(d.date()) for d in __import__("pandas").date_range("2021-01-04", periods=60, freq="W")]
    r = RR.memory_shock_events(dates, x, cooldown=3)
    assert r["count"] == 1 and r["events"][0]["value"] == 30          # one event; later spikes inflate the past-only baseline so the first has the largest z
    flat = RR.memory_shock_events(dates, np.ones(60))
    assert flat["count"] == 0 and flat["n_scored"] > 0
    assert RR.memory_shock_events([], [])["count"] == 0
    with pytest.raises(ValueError):
        RR.memory_shock_events(["2020-01-01"], [1.0, 2.0])


def test_extended_sections_and_text_rendering():
    rng = np.random.default_rng(5)
    ret = rng.normal(0, 0.02, 500)
    x = rng.normal(0, 1, 100)
    x[60] = 15
    run = {"pred_dir": np.sign(ret), "actual_ret": ret, "confidence": rng.uniform(0.5, 1, 500),
           "analog_distances": np.sort(rng.random((50, 3)), axis=1),
           "memory_dates": [str(d.date()) for d in __import__("pandas").date_range("2020-01-06", periods=100, freq="W")], "memory_signal": x}
    ext = RR.extended_sections(run)
    txt = RR.render_extended_text(ext)
    for needle in ("DIRECTION (Tier 3)", "ANALOG DISTANCE", "MEMORY SHOCK EVENTS (1 in", "at or above 0.8 confidence"):
        assert needle in txt
    none = RR.extended_sections({})
    assert none == {"direction": None, "analog_distance": None, "shocks": None} and RR.render_extended_text(none) == ""
    # the shock count can be fed straight into the existing report's memory section
    rep = RR.build_report({"memory": {"shock_events": ext["shocks"]["count"]}}, "2026-09-28")
    assert rep["memory"]["shock_events"] == 1
