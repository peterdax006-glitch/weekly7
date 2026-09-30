"""Tests for scripts/survivorship_gate.py (C69 ledger W-03): survivor-only results may not become final evidence unlabelled.
Synthetic trees only (tmp_path); nothing reads the real caches or the real checklists."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("survivorship_gate", Path(__file__).resolve().parents[1] / "scripts" / "survivorship_gate.py")
G = importlib.util.module_from_spec(SPEC)
sys.modules["survivorship_gate"] = G
SPEC.loader.exec_module(G)


def _tree(tmp_path, rows, reports=None, artefacts=()):
    (tmp_path / "state" / "build").mkdir(parents=True)
    (tmp_path / "state" / "build" / "RESEARCH_BRAIN_CHECKLIST.json").write_text(json.dumps({"items": rows}), encoding="utf-8")
    for rel in list(artefacts) + list((reports or {}).keys()):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text((reports or {}).get(rel, "walk-forward table"), encoding="utf-8")
    return tmp_path


def _row(i, status="VALIDATED", evidence=("state/research/fv/run1/report.md",), notes=""):
    return {"id": i, "status": status, "evidence": list(evidence), "notes": notes}


ART = ("state/research/fv/run1/report.md",)


def test_planted_unlabelled_validated_row_fails_the_gate(tmp_path):
    root = _tree(tmp_path, [_row("RS01")], artefacts=ART)
    res = G.run(root)
    assert res["exit_code"] == 1
    assert [v["id"] for v in res["violations"]] == ["RS01"]
    assert "SURVIVOR_ONLY" in res["violations"][0]["why"]


def test_the_same_row_passes_once_labelled(tmp_path):
    root = _tree(tmp_path, [_row("RS01", notes="SURVIVOR_ONLY: no delisted names")], artefacts=ART)
    assert G.run(root)["exit_code"] == 0


def test_survivor_corrected_label_also_passes(tmp_path):
    root = _tree(tmp_path, [_row("RS01", notes="SURVIVOR_CORRECTED: re-measured with 77 delisted names")], artefacts=ART)
    assert G.run(root)["exit_code"] == 0


def test_notes_alone_can_mark_dependence(tmp_path):
    root = _tree(tmp_path, [_row("RS09", evidence=("state/misc/x.md",), notes="Artefact x. walk-forward on the real panel")],
                 artefacts=("state/misc/x.md",))
    res = G.run(root)
    assert res["exit_code"] == 1 and "real panel" in res["violations"][0]["why"]


def test_planted_world_evidence_is_not_flagged(tmp_path):
    ev = "state/research/algorithm/planted/report.md"
    root = _tree(tmp_path, [_row("I06", evidence=(ev,), notes="strong_detection 1.000 on the planted world")], artefacts=(ev,))
    assert G.run(root)["exit_code"] == 0


def test_only_validated_rows_are_gated(tmp_path):
    root = _tree(tmp_path, [_row("RS02", status="IN_PROGRESS"), _row("RS03", status="FAILED")], artefacts=ART)
    assert G.run(root)["exit_code"] == 0


def test_validated_row_without_evidence_on_disk_warns_and_strict_fails(tmp_path):
    root = _tree(tmp_path, [_row("A12", evidence=("pytest run 2026-09-29: 170 passed",))])
    assert G.run(root) ["exit_code"] == 0 and G.run(root)["warnings"]
    assert G.run(root, strict_evidence=True)["exit_code"] == 1


def test_unlabelled_final_report_in_a_real_panel_dir_fails(tmp_path):
    rep = "state/research/direction2/run9/report.md"
    root = _tree(tmp_path, [_row("X1", status="IN_PROGRESS")], {rep: "FINAL: accuracy 0.62, VALIDATED"})
    res = G.run(root)
    assert res["exit_code"] == 1 and res["violations"][0]["where"] == rep


def test_report_that_mentions_survivorship_or_is_not_final_passes(tmp_path):
    reports = {"state/research/direction2/a/report.md": "FINAL result. Survivor-only panel, see RG04.",
               "state/research/direction2/b/report.md": "intermediate table, no verdict yet",
               "state/research/algorithm/planted/report.md": "FINAL planted world"}
    assert G.run(_tree(tmp_path, [_row("X1", status="IN_PROGRESS")], reports))["exit_code"] == 0


def test_a_labelled_row_citing_the_report_covers_it(tmp_path):
    rep = "state/research/memory_adapter/summary.txt"
    row = _row("D13", status="FAILED", evidence=(rep,), notes="SURVIVOR_ONLY: negative result on a biased panel")
    root = _tree(tmp_path, [row], {rep: "VERDICT helped: False"})
    assert G.run(root)["exit_code"] == 0
    row["notes"] = ""
    (tmp_path / "state" / "build" / "RESEARCH_BRAIN_CHECKLIST.json").write_text(json.dumps({"items": [row]}), encoding="utf-8")
    assert G.run(root)["exit_code"] == 1


def test_empty_scan_fails_closed_and_null_tree_is_clean(tmp_path):
    assert G.run(tmp_path)["exit_code"] == 2
    root = _tree(tmp_path / "t", [_row("Q1", status="NOT_STARTED", evidence=())])
    res = G.run(root)
    assert res["exit_code"] == 0 and res["violations"] == [] and res["rows_scanned"] == 1


def test_corrupt_checklist_is_a_violation_not_a_crash(tmp_path):
    (tmp_path / "state" / "build").mkdir(parents=True)
    (tmp_path / "state" / "build" / "RESEARCH_BRAIN_CHECKLIST.json").write_text("{not json", encoding="utf-8")
    res = G.run(tmp_path)
    assert res["exit_code"] == 1 and "unreadable" in res["violations"][0]["why"]


def test_cli_exit_codes_and_json_output(tmp_path, capsys):
    root = _tree(tmp_path, [_row("RS01")], artefacts=ART)
    out = tmp_path / "out" / "gate.json"
    assert G.main(["--root", str(root), "--json", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["violations"][0]["id"] == "RS01"
    assert "VIOLATION" in capsys.readouterr().out
    assert G.main(["--root", str(tmp_path / "nothing")]) == 2


def test_no_reports_flag_skips_report_scan(tmp_path):
    rep = "state/research/direction2/run9/report.md"
    root = _tree(tmp_path, [_row("X1", status="IN_PROGRESS")], {rep: "FINAL"})
    assert G.main(["--root", str(root), "--no-reports"]) == 0
    assert G.main(["--root", str(root)]) == 1


@pytest.mark.parametrize("path,expected", [("state/research/fv/run1/r.md", True), ("state/research/delisted/report.md", False),
                                           ("state/research/acceptance_mini/s.json", False), ("engine/x.py", False),
                                           ("state\\research\\exits_stops\\report.txt", True)])
def test_real_panel_path_classifier(path, expected):
    assert G._real_panel_evidence(path) is expected


# --- the real checklists (small JSON files; no caches) --------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]


def _real(name):
    return {it["id"]: it for it in json.loads((REPO / "state" / "build" / name).read_text(encoding="utf-8"))["items"]}


def test_real_checklists_pass_the_row_gate():
    res = G.run(REPO, reports=False)
    assert res["exit_code"] == 0, res["violations"]


def test_survivor_only_validated_rows_carry_the_label():
    for i in ("RS01", "RS05", "RS06", "RS07", "RS08", "RS09", "RS10", "RS11", "RS12", "RS14"):
        assert "SURVIVOR_ONLY" in (_real("RESEARCH_BRAIN_CHECKLIST.json")[i]["notes"]), i


AT_LEAST_IMPLEMENTED = {"IMPLEMENTED", "TESTING", "VALIDATED"}      # a row may move past IMPLEMENTED (C75 Phase 0: TESTING), never back


def test_stale_c66_rows_no_longer_claim_not_started():
    rows = _real("RESEARCH_BRAIN_CHECKLIST.json")
    for i in ("RF03", "RF08", "RF19", "RF21", "RF23", "RF32", "RT19", "RA11"):
        assert rows[i]["status"] in AT_LEAST_IMPLEMENTED, (i, rows[i]["status"])
    for i in ("RF03", "RF08", "RF21", "RF23", "RF32"):
        assert rows[i]["code_paths"] and rows[i]["tests"]


def test_c68_pz_rows_are_implemented_with_existing_code_and_tests():
    rows = _real("PREDICTION_ERROR_CHECKLIST.json")
    for n in range(1, 20):
        r = rows[f"PZ{n:02d}"]
        assert r["status"] in AT_LEAST_IMPLEMENTED, (r["id"], r["status"])
        assert r["code_paths"] and r["tests"], r["id"]
        for p in r["code_paths"] + r["tests"]:
            assert (REPO / p).exists(), (r["id"], p)


def test_no_row_is_validated_without_an_artefact_on_disk():
    assert G.run(REPO, reports=False, strict_evidence=True)["exit_code"] == 0
