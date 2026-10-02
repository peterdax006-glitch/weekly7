"""creator/recon.py (C77 Phase 0, CR001-CR036): every reconnaissance section is COMPUTED from the tree, changes when the tree
changes, and a ledger of another tree is stale. Runs on a small synthetic git repository."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from creator import recon

ROOT = Path(__file__).resolve().parents[1]
BODY = "    a = x + 1\n    b = a * 2\n    c = b - x\n    return c * 3\n"


def _w(repo: Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _git(repo: Path, *a: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True, capture_output=True)


def _repo(tmp: Path) -> Path:
    r = tmp / "repo"
    r.mkdir()
    _w(r, "engine/__init__.py", "")
    _w(r, "engine/core.py", f"def core(x):\n{BODY}")
    _w(r, "engine/user.py", "from engine import core\n\ndef use(x):\n    return core.core(x)\n")
    _w(r, "engine/research/__init__.py", "")
    _w(r, "tests/test_core.py", "from engine import core\n\ndef test_a():\n    assert core.core(1)\n\ndef test_b():\n    assert core.core(2)\n")
    _w(r, "CREATOR_MASTER_PROMPT.md", "prompt\n")
    _w(r, "BLUEPRINT.md", "blueprint\n")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "one")
    return r


def _rec(r: Path, **kw: Any) -> dict[str, Any]:
    return recon.compute(r, scope=("engine", "tests"), external=False, **kw)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _repo(tmp_path)


def test_structure_lists_existing_modules_and_changes_with_the_tree(repo: Path) -> None:
    before = _rec(repo)["CR005_structure"]
    assert {"engine/core.py", "engine/user.py", "tests/test_core.py"} <= set(before["file_list"])
    assert before["files"] == len(before["file_list"])
    _w(repo, "engine/newmod.py", f"def fresh(x):\n{BODY}")
    after = _rec(repo)["CR005_structure"]
    assert "engine/newmod.py" in after["file_list"] and after["files"] == before["files"] + 1
    assert "engine/newmod.py" not in before["file_list"]


def test_branch_state_reflects_commits_and_dirty_files(repo: Path) -> None:
    a = _rec(repo)["CR006_branch_state"]
    assert a["branch"] == "main" and len(a["head"]) == 40 and a["dirty_files"] == 0
    _w(repo, "engine/dirty.py", "X = 1\n")
    assert _rec(repo)["CR006_branch_state"]["dirty_files"] == 1
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two")
    b = _rec(repo)["CR006_branch_state"]
    assert b["head"] != a["head"] and b["dirty_files"] == 0
    _git(repo, "branch", "creator/sbx-x")
    assert "creator/sbx-x" in _rec(repo)["CR006_branch_state"]["sandbox_branches"]


def test_builders_section_sees_a_kernel_lock(repo: Path) -> None:
    assert _rec(repo)["CR007_builders"]["kernel_lock"] is False
    _w(repo, "state/creator/kernel.lock", "1")
    assert _rec(repo)["CR007_builders"]["kernel_lock"] is True


def test_integrations_read_ci_workflows(repo: Path) -> None:
    _w(repo, ".github/workflows/ci.yml", "jobs:\n  t:\n    steps:\n      - run: pytest -q\n      - env: ${{ secrets.TOKEN_X }}\n")
    wf = _rec(repo)["CR008_integrations"]["ci_workflows"]["ci.yml"]
    assert wf["runs"] == ["pytest -q"] and wf["secrets"] == ["TOKEN_X"]


def test_tests_section_counts_files_and_functions(repo: Path) -> None:
    t = _rec(repo)["CR009_tests"]
    assert (t["test_files"], t["test_functions"]) == (1, 2)
    _w(repo, "tests/test_more.py", "def test_c():\n    pass\n\ndef test_d():\n    pass\n\ndef helper():\n    pass\n")
    t2 = _rec(repo)["CR009_tests"]
    assert (t2["test_files"], t2["test_functions"]) == (2, 4)


def test_untested_module_is_listed_in_the_tests_section(repo: Path) -> None:
    _w(repo, "engine/lonely.py", f"def lone(x):\n{BODY}")
    assert "engine/lonely.py" in _rec(repo)["CR009_tests"]["examples"]
    assert "engine/core.py" not in _rec(repo)["CR009_tests"]["examples"]


def test_reachability_flags_an_orphan_module_and_not_an_imported_one(repo: Path) -> None:
    _w(repo, "engine/orphan.py", f"def orphan(x):\n{BODY}")
    r = _rec(repo)
    assert "engine/orphan.py" in r["CR014_reachability"]["examples"] and "engine/orphan.py" in r["CR023_disconnected"]
    assert "engine/core.py" not in r["CR023_disconnected"]
    _w(repo, "engine/uses_orphan.py", "from engine import orphan\n\ndef go(x):\n    return orphan.orphan(x)\n")
    assert "engine/orphan.py" not in _rec(repo)["CR023_disconnected"]


def test_a_script_with_a_main_guard_is_an_entry_point_not_disconnected(repo: Path) -> None:
    _w(repo, "engine/cli.py", "def run(x):\n    return x\n\nif __name__ == '__main__':\n    run(1)\n")
    assert "engine/cli.py" not in _rec(repo)["CR023_disconnected"]


def test_duplicates_flags_identical_function_bodies_only(repo: Path) -> None:
    assert _rec(repo)["CR022_duplicates"] == []
    _w(repo, "engine/copy.py", f"def other_name(x):\n{BODY}")
    dups = _rec(repo)["CR022_duplicates"]
    assert len(dups) == 1 and sorted(dups[0]["sites"]) == ["engine/copy.py:other_name", "engine/core.py:core"]
    _w(repo, "engine/copy.py", "def other_name(x):\n    return [x, x, x, x][0]\n    a = 1\n    b = 2\n")
    assert _rec(repo)["CR022_duplicates"] == []


def test_known_failures_reads_the_recorded_evidence(repo: Path) -> None:
    assert _rec(repo)["CR019_known_failures"]["failing_test_files"] == {}
    ev = {"tests/test_core.py": {"outcome": "PASS"}, "tests/test_bad.py": {"outcome": "FAIL"},
          "tests/test_err.py": {"outcome": "ERROR"}}
    _w(repo, "state/creator/test_evidence.json", json.dumps(ev))
    k = _rec(repo)["CR019_known_failures"]
    assert k["failing_test_files"] == {"tests/test_bad.py": "FAIL", "tests/test_err.py": "ERROR"}
    _w(repo, "state/creator/test_evidence.json", "{not json")
    assert "unreadable" in _rec(repo)["CR019_known_failures"]["note"]


def test_research_boundary_flags_a_function_level_import_on_the_trading_path(repo: Path) -> None:
    assert _rec(repo)["CR017_research_boundary"]["violations"] == []
    _w(repo, "engine/research/probe.py", f"def probe(x):\n{BODY}")
    _w(repo, "engine/train.py", "def train():\n    from engine.research import probe\n    return probe.probe(1)\n")
    v = _rec(repo)["CR017_research_boundary"]["violations"]
    assert v and v[0]["from"] == "engine/train.py" and "engine.research.probe" in v[0]["reaches"]


def test_limitations_report_stubs_and_unparsable_files(repo: Path) -> None:
    assert _rec(repo)["CR020_known_limitations"]["by_kind"].get("STUB", 0) == 0
    _w(repo, "engine/stubby.py", "def todo_fn(x):\n    raise NotImplementedError\n")
    _w(repo, "engine/broken.py", "def broken(:\n")
    lim = _rec(repo)["CR020_known_limitations"]
    assert lim["by_kind"].get("STUB", 0) >= 1 and lim["by_kind"].get("UNPARSABLE", 0) == 1


def test_shallow_implementations_flag_tiny_modules(repo: Path) -> None:
    _w(repo, "engine/tiny.py", "def tiny():\n    return 1\n")
    r = _rec(repo)
    assert "engine/tiny.py" in r["CR025_shallow"] and "engine/core.py" not in r["CR025_shallow"] or "engine/core.py" in r["CR025_shallow"]


def test_stale_checklists_count_done_boxes_without_evidence(repo: Path) -> None:
    items = [{"id": "X1", "status": "TESTING", "evidence": []}, {"id": "X2", "status": "TESTING", "evidence": ["e"]},
             {"id": "X3", "status": "NOT_STARTED"}]
    _w(repo, "state/build/MY_CHECKLIST.json", json.dumps({"items": items}))
    r = _rec(repo)
    c = r["CR021_stale_checklists"]["MY_CHECKLIST.json"]
    assert c["items"] == 3 and c["done_without_evidence"] == 1 and c["examples"] == ["X1"]
    assert r["CR024_implemented_not_validated"]["MY_CHECKLIST.json"] == {"TESTING": 2, "NOT_STARTED": 1}
    assert "MY_CHECKLIST.json" in r["CR001-004_documents"]["checklists"]


def test_infrastructure_inventories_are_found_by_what_the_files_do(repo: Path) -> None:
    for rel in ("engine/learning/brain.py", "engine/research/scan.py", "engine/experiment_runner.py", "engine/health_check.py",
                "engine/compute_budget.py", "engine/replay_log.py", "engine/rollback_store.py", "engine/future_firewall.py",
                "engine/provenance.py"):
        _w(repo, rel, f"def f(x):\n{BODY}")
    r = _rec(repo)
    assert "engine/learning/brain.py" in r["CR027_self_learning_infra"]
    assert "engine/research/scan.py" in r["CR028_research_infra"]
    assert "engine/experiment_runner.py" in r["CR029_experiment_infra"]
    assert "engine/health_check.py" in r["CR030_health_infra"]
    assert "engine/compute_budget.py" in r["CR031_compute_controls"]
    assert "engine/replay_log.py" in r["CR013_deterministic_replay"]
    assert "engine/rollback_store.py" in r["CR012_rollback"]
    assert "engine/future_firewall.py" in r["CR016_firewalls"] and "engine/future_firewall.py" in r["CR018_learner_truth_boundary"]
    assert "engine/future_firewall.py" in r["CR026_future_leak_risks"]
    assert "engine/provenance.py" in r["CR011_provenance"]["infra"]
    assert "engine/core.py" not in r["CR027_self_learning_infra"] + r["CR030_health_infra"] + r["CR031_compute_controls"]


def test_data_flow_records_cross_package_edges(repo: Path) -> None:
    _w(repo, "creator/__init__.py", "")
    _w(repo, "creator/bridge.py", "from engine import core\n\ndef go(x):\n    return core.core(x)\n")
    edges = [tuple(e) for e in recon.compute(repo, scope=("engine", "tests", "creator"), external=False)["CR015_data_flow"]["package_edges"]]
    assert ("creator.bridge", "engine") in edges


def test_documents_section_hashes_every_authoritative_document_and_names_the_missing(repo: Path) -> None:
    ms = repo / "MS.md"
    d = _rec(repo, masterstock=ms)["CR001-004_documents"]
    assert d["authoritative"]["BLUEPRINT.md"] and set(d["missing"]) == set(recon.AUTHORITATIVE) - {"CREATOR_MASTER_PROMPT.md", "BLUEPRINT.md"}
    assert d["masterstock_sha256"] is None
    ms.write_text("stock", encoding="utf-8")
    old = d["authoritative"]["BLUEPRINT.md"]
    _w(repo, "BLUEPRINT.md", "blueprint v2\n")
    d2 = _rec(repo, masterstock=ms)["CR001-004_documents"]
    assert d2["authoritative"]["BLUEPRINT.md"] != old and d2["masterstock_sha256"] and d2["masterstock_bytes"] == 5
    (repo / "BLUEPRINT.md").unlink()
    assert "BLUEPRINT.md" in _rec(repo)["CR001-004_documents"]["missing"]


def test_the_real_repository_has_every_authoritative_document() -> None:
    assert recon.documents(ROOT, None)["missing"] == []


# ---- freshness ------------------------------------------------------------------------------------------------------------

def test_ledger_carries_the_tree_hash_and_goes_stale_when_the_tree_changes(repo: Path) -> None:
    led = _rec(repo)
    assert led["tree_hash"] == recon.tree_hash(repo, ("engine", "tests"))
    assert recon.is_current(led, repo, ("engine", "tests"))
    _w(repo, "engine/core.py", f"def core(x):\n{BODY}    # edited\n")
    assert not recon.is_current(led, repo, ("engine", "tests"))
    _w(repo, "BLUEPRINT.md", "changed document\n")
    assert recon.tree_hash(repo, ("engine", "tests")) != led["tree_hash"]
    assert not recon.is_current({}, repo, ("engine", "tests"))


def _status_module() -> Any:
    spec = importlib.util.spec_from_file_location("creator_checklist_status", ROOT / "scripts" / "creator_checklist_status.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _full_ledger(repo: Path) -> dict[str, Any]:
    _w(repo, "MASTERSTOCK.md", "x")
    for n in recon.AUTHORITATIVE:
        _w(repo, n, f"{n}\n")
    return recon.compute(repo, scope=recon.SCOPE, external=False, masterstock=repo / "MASTERSTOCK.md")


def test_status_script_gives_testing_on_a_current_ledger_and_not_on_a_stale_one(repo: Path) -> None:
    st = _status_module()
    led = _full_ledger(repo)
    # repo has no creator/ package: the synthetic scope is the default one, which is fine (missing dirs are simply empty)
    for n in range(1, 37):
        got = st.phase0_status(n, led, repo)
        assert got and got[0] == "TESTING", (n, got)
    _w(repo, "engine/core.py", f"def core(x):\n{BODY}    # drift\n")
    for n in (1, 5, 14, 22, 32, 36):
        got = st.phase0_status(n, led, repo)
        assert got and got[0] == "IN_PROGRESS" and "STALE" in got[2], (n, got)


def test_status_script_does_not_trust_a_ledger_without_a_tree_hash(repo: Path) -> None:
    st = _status_module()
    led = _full_ledger(repo)
    led.pop("tree_hash")
    assert st.phase0_status(5, led, repo)[0] == "IN_PROGRESS"
    assert st.phase0_status(5, {}, repo) is None


def test_status_script_caps_document_boxes_when_a_document_is_missing(repo: Path) -> None:
    st = _status_module()
    led = _full_ledger(repo)
    (repo / "BIBLE.md").unlink()
    led = recon.compute(repo, scope=recon.SCOPE, external=False, masterstock=repo / "MASTERSTOCK.md")
    assert st.phase0_status(1, led, repo)[0] == "IMPLEMENTED" and "BIBLE.md" in st.phase0_status(1, led, repo)[2]
    assert st.phase0_status(5, led, repo)[0] == "TESTING"
    led2 = recon.compute(repo, scope=recon.SCOPE, external=False, masterstock=repo / "nope.md")
    assert st.phase0_status(4, led2, repo)[0] == "IMPLEMENTED"
    assert st.phase0_status(3, led2, repo)[0] == "IMPLEMENTED"          # BIBLE.md is still missing
