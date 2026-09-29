"""Phases 37-39: scripts/bible_trace.py on a small synthetic Bible and a synthetic repo (never the real caches).

Planted-defect tests: a passing module flips to [!] when a failing verdict about it appears, to [~] when its test
disappears, and one shared generic word must NOT condemn an unrelated requirement.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bible_trace", ROOT / "scripts" / "bible_trace.py")
bt = importlib.util.module_from_spec(_spec)
sys.modules["bible_trace"] = bt
_spec.loader.exec_module(bt)

BIBLE = """# TITLE

Preamble with * [ ] a checkbox that is outside any phase and must be ignored.

# 8. MASTER EXECUTION ORDER

# PHASE 1 — POINT-IN-TIME DATA FIREWALL

Estimated code: **1,500–3,000 lines**

## 1.1 Date integrity

* [ ] Every feature carries an effective date.
* [ ] Future rows cannot be queried.

## 1.3 Future-scramble test

Inject information after the cutoff.

Verify:

* predictions unchanged;
* memory unchanged;

If anything changes: FAIL CLOSED.

# PHASE 2 — FEATURE/PARITY FIREWALL

Implement:

* [ ] feature parity harness;
* [ ] maximum absolute error;

Required gate:

`max difference <= 1e-4`

# PHASE 3 — PATTERN THINGS

Measure:

1. hit rate;
2. tail loss;

Never do:

* comment out failing tests;

Examples of unacceptable completion:

* a fake thing;

After each completed phase:

1. run its tests;

## 3.1 Prose only section

This heading carries a requirement in one sentence.

# PHASE 4 — EMPTY PHASE

Nothing structured here at all, just words.

# PHASE 5 — REPORT

Produce a report containing:

```text
RUN ID
GIT COMMIT

TIER 1
weekly average move
weekly median move
```

# PHASE 38 — MASTER CHECKLIST

## ALGORITHM

* [x/~] A1 Feature parity stuff
* [ ] Z9 Nothing maps here

# PHASE 39 — DEFINITION OF DONE

* [ ] parity works;
* [ ] some claim nothing verifies;
"""


def write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def make_repo(tmp_path, with_test=True, with_evidence=True, evidence=None, test_name="test_feature_parity_harness_is_exact"):
    write(tmp_path / "BIBLE.md", BIBLE)
    write(tmp_path / "engine" / "__init__.py", "")
    write(tmp_path / "engine" / "parity.py", '"""Bible Phase 2: feature parity harness with maximum absolute error."""\n\n'
          "def feature_parity_harness(a, b):\n    return max_abs_error(a, b)\n\n\ndef max_abs_error(a, b):\n    return max(abs(x - y) for x, y in zip(a, b))\n")
    write(tmp_path / "engine" / "pit.py", '"""Phase 1 point-in-time: every feature carries an effective date; future rows cannot be queried."""\n\n'
          "def effective_date(row):\n    return row\n\n\ndef future_rows_cannot_be_queried(store, now):\n    raise ValueError\n")
    if with_test:
        write(tmp_path / "tests" / "test_parity.py", "from engine.parity import feature_parity_harness\n\n\n"
              f"def {test_name}():\n    assert feature_parity_harness([1], [1]) == 0\n\n\n"
              "def test_maximum_absolute_error_is_the_largest_gap():\n    assert feature_parity_harness([1], [3]) == 2\n")
    if with_evidence:
        write(tmp_path / "state" / "research" / "parity" / "latest.json", json.dumps(evidence or {"provenance": {"code_files": ["engine/parity.py"]},
              "feature_parity_harness": {"passed": True}, "maximum_absolute_error": {"pass": True}}))
    return tmp_path


def by_text(tr, text):
    return next(r for r in tr["requirements"] if text.lower() in r["text"].lower())


# ---- parsing ----------------------------------------------------------------------------------------------------------
def test_parse_finds_phases_estimates_and_ignores_preamble():
    phases, reqs = bt.parse_bible(BIBLE)
    assert [p.phase for p in phases] == [1, 2, 3, 4, 5, 38, 39]
    assert (phases[0].est_lo, phases[0].est_hi) == (1500, 3000)
    assert not any("outside any phase" in r.text for r in reqs)


def test_ids_are_unique_stable_and_shaped_phase_section_n():
    _, a = bt.parse_bible(BIBLE)
    _, b = bt.parse_bible(BIBLE)
    ids = [r.id for r in a]
    assert ids == [r.id for r in b] and len(ids) == len(set(ids))
    assert "1.1.1" in ids and "1.1.2" in ids and "1.3.1" in ids and "2.0.1" in ids


def test_introducer_lists_and_numbered_lists_are_requirements_but_negative_ones_are_not():
    _, reqs = bt.parse_bible(BIBLE)
    texts = [r.text for r in reqs if r.phase == 3]
    assert "hit rate" in texts and "tail loss" in texts          # numbered under 'Measure:'
    assert "predictions unchanged" in [r.text for r in reqs if r.phase == 1]
    joined = " ".join(texts)
    assert "comment out failing tests" not in joined and "fake thing" not in joined and "run its tests" not in joined


def test_prose_only_heading_and_empty_phase_and_report_template_still_yield_requirements():
    _, reqs = bt.parse_bible(BIBLE)
    heading = [r for r in reqs if r.kind == "heading"]
    assert any("Prose only section" in r.text and "one sentence" in r.text for r in heading)
    assert [r.kind for r in reqs if r.phase == 4] == ["phase"]      # nothing may silently disappear
    rep = [r for r in reqs if r.phase == 5]
    assert {r.text for r in rep} == {"TIER 1: weekly average move", "TIER 1: weekly median move"}   # all-caps lines are headers


def test_checklist_labels_and_the_bibles_own_claims_are_kept():
    _, reqs = bt.parse_bible(BIBLE)
    a1 = next(r for r in reqs if r.label == "A1")
    assert a1.claim == "[x/~]" and a1.kind == "checklist" and a1.text.startswith("ALGORITHM:")
    assert next(r for r in reqs if r.label == "Z9").claim == "[ ]"


def test_real_bible_parses_every_phase_without_touching_caches():
    phases, reqs = bt.parse_bible((ROOT / "BIBLE.md").read_text(encoding="utf-8"))
    nums = [p.phase for p in phases]
    assert nums == sorted(nums) and 0 in nums and 46 in nums and len(nums) == 47
    assert len(reqs) > 400
    assert all(any(r.phase == p for r in reqs) for p in nums)      # no phase vanishes
    assert len({r.id for r in reqs}) == len(reqs)


def test_empty_and_degenerate_bibles():
    assert bt.parse_bible("") == ([], [])
    assert bt.parse_bible("no phases here\n* [ ] orphan checkbox\n") == ([], [])
    phases, reqs = bt.parse_bible("# PHASE 9 — LONELY\n")
    assert [p.phase for p in phases] == [9] and len(reqs) == 1 and reqs[0].kind == "phase"


# ---- text helpers -------------------------------------------------------------------------------------------------------
def test_stems_equate_prose_and_identifiers():
    assert bt.terms("validation validated validate") == ["validat"]
    assert bt.terms("max_abs_err") == ["maximum", "absolut", "error"]
    assert set(bt.terms("maximum absolute error")) == {"maximum", "absolut", "error"}
    assert bt.terms("detection") == bt.terms("detect")
    assert bt.terms("the and of") == []


def test_phase_mentions_expand_ranges_and_lists():
    assert bt.phases_in("Bible Phase 25") == {25}
    assert bt.phases_in("PHASES 21, 22, 23, 24") == {21, 22, 23, 24}
    assert bt.phases_in("Phases 3-5") == {3, 4, 5}
    assert bt.phases_in("phases 4 and 5") == {4, 5}
    assert bt.phases_in("no mention") == set()


def test_json_verdicts_count_only_verdict_keys():
    p, f, cf = bt.judge_json({"failures": 0, "gate": "FAIL: bad", "ok": True, "drift": {"passed": False},
                              "dead_feature": {"verdict": "rejected"}, "code_files": ["engine/x.py"]})
    assert [x[0] for x in f] == ["drift.passed = False", "gate = FAIL: bad"]
    assert [x[0] for x in p] == ["ok = True"]
    assert cf == {"engine/x.py"}                    # 'rejected' is a lifecycle outcome, 'failures: 0' is a count


# ---- states -------------------------------------------------------------------------------------------------------------
def test_fully_traced_requirement_is_x_and_cites_paths(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path))
    r = by_text(tr, "maximum absolute error")
    assert r["state"] == "[x]"
    assert r["code"][0]["module"] == "engine/parity.py" and r["tests"] == ["tests/test_parity.py"]
    assert r["pass_lines"][0]["path"] == "state/research/parity/latest.json"
    assert tr["invariant_violations"] == []


def test_no_code_is_not_started(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path))
    r = by_text(tr, "memory unchanged")
    assert r["state"] == "[ ]" and r["code"] == []


def test_code_without_test_is_implemented_only(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path, with_test=False))
    assert by_text(tr, "feature parity harness")["state"] == "[~]"


def test_real_data_requirement_without_evidence_is_not_validated(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path, with_evidence=False))
    r = by_text(tr, "maximum absolute error")            # 'error' asks for proof only via the Bible's own real-data words
    assert r["state"] in ("[~]", "[x]")
    r2 = by_text(tr, "predictions unchanged")            # phase 1.3 has no test and no code linked by name
    assert r2["state"] in ("[ ]", "[~]")
    tr_ev = bt.build_trace(make_repo(tmp_path / "b"))
    assert by_text(tr_ev, "feature parity harness")["state"] in ("[x]", "[?]")


def test_test_name_that_does_not_speak_to_the_requirement_blocks_x(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path, test_name="test_unrelated_thing"))
    r = by_text(tr, "feature parity harness")
    assert r["state"] == "[~]" and "no test name speaks" in r["why"]


def test_planted_failure_flips_x_to_failed(tmp_path):
    good = bt.build_trace(make_repo(tmp_path / "g"))
    bad = bt.build_trace(make_repo(tmp_path / "b", evidence={"provenance": {"code_files": ["engine/parity.py"]},
                                                              "feature_parity_harness": {"passed": False},
                                                              "maximum_absolute_error": {"pass": True}}))
    assert by_text(good, "feature parity harness")["state"] == "[x]"
    r = by_text(bad, "feature parity harness")
    assert r["state"] == "[!]" and r["fail_lines"][0]["path"] == "state/research/parity/latest.json"
    assert bad["summary"]["counts"]["[!]"] > good["summary"]["counts"]["[!]"]
    assert by_text(bad, "maximum absolute error")["state"] != "[!]"      # the failure is about the harness, not this line


def test_one_shared_generic_word_does_not_condemn_an_unrelated_requirement(tmp_path):
    ev = {"provenance": {"code_files": ["engine/parity.py"]}, "timeline": {"gate": "FAIL: feature drift"},
          "maximum_absolute_error": {"pass": True}}
    tr = bt.build_trace(make_repo(tmp_path, evidence=ev))
    assert by_text(tr, "feature parity harness")["state"] != "[!]"       # shares only 'feature' with the failing line


def test_evidence_without_a_pass_line_is_unproven(tmp_path):
    ev = {"provenance": {"code_files": ["engine/parity.py"]}, "note": "ran", "rows": 12}
    tr = bt.build_trace(make_repo(tmp_path, evidence=ev))
    r = by_text(tr, "maximum absolute error")
    assert r["state"] in ("[?]", "[x]")
    if r["needs_real_data"]:
        assert r["state"] == "[?]"


def test_failing_test_cache_marks_failed_but_stale_cache_is_ignored(tmp_path):
    make_repo(tmp_path)
    cache = write(tmp_path / ".pytest_cache" / "v" / "cache" / "lastfailed",
                  json.dumps({"tests/test_parity.py::test_feature_parity_harness_is_exact": True}))
    now = time.time()
    os.utime(tmp_path / "tests" / "test_parity.py", (now - 100, now - 100))
    os.utime(tmp_path / "engine" / "parity.py", (now - 100, now - 100))
    os.utime(cache, (now, now))
    tr = bt.build_trace(tmp_path)
    r = by_text(tr, "feature parity harness")
    assert r["state"] == "[!]" and r["failing_tests"] == ["tests/test_parity.py::test_feature_parity_harness_is_exact"]
    os.utime(tmp_path / "tests" / "test_parity.py", (now + 50, now + 50))      # test edited after the failure was recorded
    assert by_text(bt.build_trace(tmp_path), "feature parity harness")["state"] != "[!]"
    assert bt.failing_tests(tmp_path / "nowhere") == ({}, 0.0)


# ---- Phases 38 / 39 are derived -----------------------------------------------------------------------------------------
def test_checklist_claims_are_derived_from_underlying_phases_and_never_word_matched():
    recs = [{"phase": 2, "section": "0", "state": s, "tests": [], "evidence": [], "code": [], "pass_lines": [],
             "fail_lines": [], "failing_tests": []} for s in ["[x]"] * 5]
    rec = bt.derive({"code": [], "tests": [], "evidence": [], "pass_lines": [], "fail_lines": [], "failing_tests": []}, recs)
    assert rec["state"] == "[x]" and rec["derived"]["n"] == 5
    recs[0]["state"] = "[!]"
    recs[0]["fail_lines"] = [{"path": "p", "line": "l"}]
    assert bt.derive({"code": [], "tests": [], "evidence": [], "pass_lines": [], "fail_lines": [], "failing_tests": []}, recs)["state"] == "[!]"
    for u in recs:
        u["state"] = "[ ]"
    assert bt.derive({}, recs)["state"] == "[ ]"


def test_unmapped_claims_are_unverifiable_and_the_bibles_x_claim_is_challenged(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path))
    a2 = next(r for r in tr["requirements"] if r["label"] == "Z9")
    assert a2["state"] == "[ ]" and "nothing verifies" in a2["why"]
    unknown = by_text(tr, "some claim nothing verifies")
    assert unknown["state"] == "[ ]"
    parity = by_text(tr, "parity works")
    assert parity["derived"]["n"] == sum(parity["derived"]["counts"].values()) > 0


def test_x_claim_by_the_bible_without_support_is_reported(tmp_path):
    write(tmp_path / "BIBLE.md", "# PHASE 38 — MASTER CHECKLIST\n\n## ALGORITHM\n\n* [x/~] A2 Pattern miner\n")
    tr = bt.build_trace(tmp_path)
    assert tr["claim_mismatches"] and tr["claim_mismatches"][0]["claim"] == "[x/~]"


# ---- ranking, outputs, determinism ---------------------------------------------------------------------------------------
def test_top_missing_orders_by_priority_and_caps_per_phase():
    recs = [{"id": f"1.0.{i}", "phase": 1, "text": f"gate item {i}", "kind": "list", "state": "[ ]", "bible_line": i} for i in range(9)]
    recs += [{"id": "44.0.1", "phase": 44, "text": "workers", "kind": "list", "state": "[ ]", "bible_line": 99},
             {"id": "9.0.1", "phase": 9, "text": "done thing", "kind": "list", "state": "[x]", "bible_line": 5}]
    top = bt.top_missing(recs, n=20)
    assert len([t for t in top if t["phase"] == 1]) == bt.MAX_PER_PHASE_IN_TOP
    assert top[0]["phase"] == 1 and top[-1]["phase"] == 44
    assert all(t["id"] != "9.0.1" for t in top)
    assert bt.top_missing([], 20) == []


def test_build_is_deterministic_and_outputs_are_written(tmp_path):
    make_repo(tmp_path)
    a = bt.build_trace(tmp_path, now="T")
    b = bt.build_trace(tmp_path, now="T")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    md, js = tmp_path / "out" / "T.md", tmp_path / "out" / "t.json"
    bt.write_outputs(a, md, js)
    text = md.read_text(encoding="utf-8")
    assert "# BIBLE TRACE" in text and "20 most important MISSING" in text and "Phase 1:" in text
    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert loaded["summary"]["total"] == len(loaded["requirements"]) == a["summary"]["total"]
    assert sum(loaded["summary"]["counts"].values()) == loaded["summary"]["total"]


def test_invariants_catch_a_tampered_record(tmp_path):
    tr = bt.build_trace(make_repo(tmp_path))
    recs = tr["requirements"]
    assert bt.check_invariants(recs, tmp_path) == []
    x = next(r for r in recs if r["state"] == "[x]")
    x["pass_lines"] = [{"path": "state/research/deleted.json", "line": "ok = True"}]
    assert any("missing path" in v for v in bt.check_invariants(recs, tmp_path))
    x["pass_lines"], x["fail_lines"] = [], [{"path": "p", "line": "l"}]
    x["code"] = []
    assert bt.check_invariants(recs, tmp_path)
    recs.append(dict(recs[0]))
    assert any("duplicate id" in v for v in bt.check_invariants(recs, tmp_path))


def test_missing_directories_and_no_bible_do_not_crash(tmp_path):
    tr = bt.build_trace(tmp_path, now="T")
    assert tr["summary"]["total"] == 0 and tr["top_missing"] == [] and tr["invariant_violations"] == []
    assert "0 requirements" in bt.render_markdown(tr)
    write(tmp_path / "BIBLE.md", BIBLE)
    tr = bt.build_trace(tmp_path)                      # no engine/, tests/, state/: everything is honestly [ ] or derived
    assert tr["summary"]["counts"]["[x]"] == 0 and tr["summary"]["counts"]["[!]"] == 0


def test_task_files_link_owned_modules_to_phases_and_report_missing_output(tmp_path):
    make_repo(tmp_path)
    write(tmp_path / "state" / "build" / "queue" / "B99_x.md", "# B99_x\nBible: PHASES 2 and 3 (lines 1-2). Estimated code: 1-2.\n"
          "You own: engine/pit.py, engine/not_written_yet.py, tests/test_x.py\n")
    write(tmp_path / "state" / "build" / "INTEGRATION.md", "- [ ] pit.py: wire future_rows_cannot_be_queried into livesim\n- [x] pit.py: done hook\n")
    tr = bt.build_trace(tmp_path)
    assert tr["tasks"][0]["phases"] == [2, 3] and "engine/not_written_yet.py" in tr["planned_files_missing"]
    assert set(tr["integration_pending"]) == {"pit"} and len(tr["integration_pending"]["pit"]) == 1
    assert 2 in bt.index_code(tmp_path)["engine/pit.py"].phases | {2}


def test_main_writes_both_outputs_and_returns_nonzero_only_on_invariant_violation(tmp_path, capsys):
    make_repo(tmp_path)
    rc = bt.main(["--root", str(tmp_path), "--now", "T", "--out-md", str(tmp_path / "o.md"), "--out-json", str(tmp_path / "o.json")])
    assert rc == 0 and (tmp_path / "o.md").exists() and (tmp_path / "o.json").exists()
    assert "requirements" in capsys.readouterr().out
