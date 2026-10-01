"""scripts/backfill_lessons.py on a synthetic git repo: one patch, ledger-like records, a kernel log."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backfill_lessons as B  # noqa: E402


def _git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def _setup(tmp: Path, crlf: bool = False) -> tuple[Path, Path]:
    repo, state = tmp / "repo", tmp / "repo" / "state" / "creator"
    (repo / "creator").mkdir(parents=True)
    state.mkdir(parents=True)
    (repo / "creator" / "m.py").write_text("import os\n\n\ndef f():\n    return 1\n", encoding="utf-8", newline="\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "creator")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "creator" / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8", newline="\n")
    patch = _git(repo, "diff") + "\n"
    _git(repo, "checkout", "-q", "--", "creator")
    (repo / "creator" / "m.py").write_text("import os\n\n\ndef f():\n    return 2\n", encoding="utf-8", newline="\n")   # main moved on
    _git(repo, "commit", "-qam", "later")
    (state / "pending").mkdir()
    (state / "pending" / "CP0001_20261001T101010-abcd.patch").write_bytes((patch.replace("\n", "\r\n") if crlf else patch).encode())
    rows = [
        {"id": "REQ-1", "rtype": "Requirement", "data": {"key": "EFF.size", "parents": []}, "provenance": {"timestamp": "2026-10-01T10:00:00+00:00"}},
        {"id": "GAP-1", "rtype": "Gap", "data": {"parents": ["REQ-1"]}, "provenance": {"timestamp": "2026-10-01T10:00:00+00:00"}},
        {"id": "WP-1", "rtype": "WorkPackage", "data": {"package_id": "CP0001", "objective": "shrink creator/m.py", "why_it_exists": "w",
                                                       "parents": ["GAP-1"], "implementation_requirements": ["remove unused imports"]},
         "provenance": {"timestamp": "2026-10-01T10:05:00+00:00"}},
        {"id": "CHG-1", "rtype": "ChangeProposal", "data": {"parents": ["WP-1"]}, "provenance": {"timestamp": "2026-10-01T10:05:00+00:00"}},
        {"id": "EXP-1", "rtype": "Experiment", "data": {"parents": ["CHG-1"], "baseline_ref": base + "+dirty"},
         "provenance": {"timestamp": "2026-10-01T10:05:00+00:00"}},
    ]
    (state / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (state / "kernel_log.jsonl").write_text(json.dumps({"package": "CP0001", "outcome": "ADOPTED", "reason": "merged x", "verdict": "IMPROVEMENT",
                                                        "details": {"worker": {"claimed_done": True, "notes": "dropped the unused os import"}}}) + "\n",
                                            encoding="utf-8")
    return repo, state


def test_lesson_from_patch_and_ledger(tmp_path: Path) -> None:
    repo, state = _setup(tmp_path)
    lessons, skipped = B.build(repo, state)
    assert not skipped and len(lessons) == 1
    les = lessons[0]
    assert (les.package_id, les.component, les.task_kind, les.solver) == ("CP0001", "EFF", "shrink", "claude")
    assert les.files_before == {"creator/m.py": "import os\n\n\ndef f():\n    return 1\n"}     # the base, not main's later text
    assert les.files_after == {"creator/m.py": "def f():\n    return 1\n"}
    assert les.adopted is True and "ADOPTED" in les.verdict and "IMPROVEMENT" in les.verdict
    assert les.reasoning == "dropped the unused os import"            # the teacher's own notes are kept unprefixed
    assert "shrink creator/m.py" in les.context and "remove unused imports" in les.context
    assert les.at == "2026-10-01T10:10:10"


def test_unknown_outcome_reconstructed_reasoning_and_crlf_patch(tmp_path: Path) -> None:
    repo, state = _setup(tmp_path, crlf=True)
    (state / "kernel_log.jsonl").write_text("", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert B.main([str(out), "--repo", str(repo), "--state", str(state)]) == 0
    les = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert les["adopted"] is None and les["verdict"] == ""
    assert les["reasoning"].startswith("[reconstructed] ") and "imports removed os" in les["reasoning"]
    assert les["files_after"]["creator/m.py"] == "def f():\n    return 1\n"


def test_skips_empty_patch_and_unplaceable_patch(tmp_path: Path) -> None:
    repo, state = _setup(tmp_path)
    (state / "pending" / "CP0002_20261001T101111-ffff.patch").write_bytes(b"")
    (state / "pending" / "CP0003_20261001T101212-eeee.patch").write_text(
        "diff --git a/creator/zzz.py b/creator/zzz.py\n--- a/creator/zzz.py\n+++ b/creator/zzz.py\n@@ -1 +1 @@\n-nothing here\n+x\n", encoding="utf-8")
    lessons, skipped = B.build(repo, state)
    assert [les.package_id for les in lessons] == ["CP0001"]
    assert dict(skipped)["CP0002"].startswith("empty patch")
    assert "CP0003" in dict(skipped)
