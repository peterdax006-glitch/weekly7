"""CR03: the Creator's sandbox (C77 secs 22-25, 28, 29, 44-46, 59-61). Every test uses a TEMPORARY git repository."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import build as B
from creator import sandbox as S
from creator import testrun as T

NO_TYPES = B.BuildConfig(run_typecheck=False, require_typecheck=False)


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "pkg" / "mathx.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (r / "pkg" / "other.py").write_text("def name():\n    return 'other'\n", encoding="utf-8")
    (r / "tests" / "test_mathx.py").write_text(
        "from pkg.mathx import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    (r / "tests" / "test_other.py").write_text(
        "from pkg.other import name\n\ndef test_name():\n    assert name() == 'other'\n", encoding="utf-8")
    (r / "tests" / "test_known_broken.py").write_text(
        "from pkg.mathx import add\n\ndef test_already_broken():\n    assert add(1, 1) == 3\n", encoding="utf-8")
    (r / "canon").mkdir()
    (r / "canon" / "CANON.md").write_text("canon\n", encoding="utf-8")
    sh(r, "init", "-q")
    sh(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    sh(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    return r


@pytest.fixture()
def scratch(tmp_path: Path) -> Path:
    return tmp_path / "scratch"


def adopt_decision() -> SimpleNamespace:
    return SimpleNamespace(verdict=SimpleNamespace(value="ADOPT"))


def test_open_and_close_leave_the_main_tree_untouched(repo: Path, scratch: Path) -> None:
    before = (S.head(repo), sh(repo, "status", "--porcelain"))
    sb = S.Sandbox.open(repo, scratch=scratch)
    assert sb.path.is_dir() and sb.path.parent == scratch.resolve() and sb.base == before[0]
    sb.write("pkg/new.py", "X = 1\n")
    assert not (repo / "pkg" / "new.py").exists()
    sb.close(delete_branch=True)
    assert (S.head(repo), sh(repo, "status", "--porcelain")) == before and not sb.path.exists()


def test_unknown_base_and_scratch_inside_the_tree_are_refused(repo: Path) -> None:
    with pytest.raises(S.SandboxError, match="not a known commit"):
        S.Sandbox.open(repo, base="0" * 40)
    with pytest.raises(S.SandboxError, match="OUTSIDE"):
        S.Sandbox.open(repo, scratch=repo / "inside")


@pytest.mark.parametrize("path", ["canon/CANON.md", "canon/canon.lock.json", "x.lock.json", "CREATOR_MASTER_PROMPT.md",
                                  ".github/workflows/ci.yml", "creator/devbench/sealed/keys.json", "creator/audit/rules.py",
                                  "creator/evaluate_thresholds.json", "state/livesim/w01a/result.json",
                                  "creator/capabilities.json", "creator/devbench.py", "creator/model.py", "creator/ledger.py",
                                  "creator/sandbox.py", "state/creator/ledger.jsonl"])
def test_protected_paths_are_refused(repo: Path, scratch: Path, path: str) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        with pytest.raises(S.ProtectedPathError):
            sb.write(path, "tampered\n")
        with pytest.raises(S.ProtectedPathError):
            sb.apply({"pkg/ok.py": "Y = 1\n", path: "tampered\n"})
        assert not (sb.path / "pkg" / "ok.py").exists()                  # nothing applied when any path is refused


def test_protected_change_sneaked_in_another_way_is_still_refused(repo: Path, scratch: Path) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        (sb.path / "canon" / "CANON.md").write_text("edited behind the API\n", encoding="utf-8")
        with pytest.raises(S.ProtectedPathError):
            sb.changes()


def test_patch_touching_a_protected_path_is_refused(repo: Path, scratch: Path) -> None:
    patch = "--- a/canon/CANON.md\n+++ b/canon/CANON.md\n@@ -1 +1 @@\n-canon\n+tampered\n"
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        with pytest.raises(S.ProtectedPathError):
            sb.apply_patch(patch)


def test_path_escape_is_refused(repo: Path, scratch: Path) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        with pytest.raises(S.SandboxError):
            sb.write("../outside.py", "x\n")


def test_build_failure_is_classified_and_tests_do_not_run(repo: Path, scratch: Path) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        sb.write("pkg/mathx.py", "def add(a, b)\n    return a + b\n")                   # syntax error
        ev = sb.evaluate(build_config=NO_TYPES)
        assert not ev.builds and ev.candidate_run is None and ev.report is None
        kinds = {i.kind for step in ev.build.steps for i in step.issues}
        assert B.BuildFailureKind.SYNTAX in kinds
        sb.cleanup_base()


def test_affected_selection_and_regression_vs_a_preexisting_failure(repo: Path, scratch: Path) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        sb.write("pkg/mathx.py", "def add(a, b):\n    return a - b\n")                  # breaks test_add
        ev = sb.evaluate(build_config=NO_TYPES)
        assert ev.builds
        assert "tests/test_mathx.py" in ev.selection.tests and "tests/test_other.py" not in ev.selection.tests
        assert ev.report is not None and ev.report.verdict is T.Verdict.REGRESSED
        assert any("test_add" in k for k in ev.report.blocking)
        assert not any("test_already_broken" in k for k in ev.report.blocking)     # failing at base is not a regression
        sb.cleanup_base()


def test_clean_change_reports_clean(repo: Path, scratch: Path) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        sb.write("pkg/other.py", "def name():\n    return 'other'\n\n\ndef shout():\n    return name().upper()\n")
        ev = sb.evaluate(build_config=NO_TYPES)
        assert ev.builds and ev.report is not None and ev.report.verdict is T.Verdict.CLEAN and ev.clean
        sb.cleanup_base()


def test_adopt_needs_an_adopt_decision_then_merges(repo: Path, scratch: Path) -> None:
    sb = S.Sandbox.open(repo, scratch=scratch)
    sb.write("pkg/other.py", "def name():\n    return 'renamed'\n")
    with pytest.raises(S.SandboxError, match="ADOPT decision"):
        S.adopt(sb, SimpleNamespace(verdict=SimpleNamespace(value="REJECT")), "should not merge")
    res = S.adopt(sb, adopt_decision(), "rename")
    assert (repo / "pkg" / "other.py").read_text(encoding="utf-8").endswith("'renamed'\n")
    assert S.head(repo) == res.merge_commit != res.main_before
    sb.close()
    revert = S.rollback(repo, res.merge_commit, "regressed later")
    assert (repo / "pkg" / "other.py").read_text(encoding="utf-8").endswith("'other'\n") and revert != res.merge_commit


def test_conflicting_adopt_is_refused_and_main_is_left_intact(repo: Path, scratch: Path) -> None:
    sb = S.Sandbox.open(repo, scratch=scratch)
    sb.write("pkg/other.py", "def name():\n    return 'from sandbox'\n")
    (repo / "pkg" / "other.py").write_text("def name():\n    return 'from main'\n", encoding="utf-8")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "main moved on")
    before = S.head(repo)
    with pytest.raises(S.SandboxError, match="conflicts"):
        S.adopt(sb, adopt_decision(), "conflicting")
    assert S.head(repo) == before and sh(repo, "status", "--porcelain") == ""
    assert (repo / "pkg" / "other.py").read_text(encoding="utf-8").endswith("'from main'\n")
    S.discard(sb)


def test_adopt_refuses_when_main_has_uncommitted_edits_to_the_same_files(repo: Path, scratch: Path) -> None:
    sb = S.Sandbox.open(repo, scratch=scratch)
    sb.write("pkg/other.py", "def name():\n    return 'x'\n")
    (repo / "pkg" / "other.py").write_text("# someone is editing\n", encoding="utf-8")
    with pytest.raises(S.SandboxError, match="uncommitted"):
        S.adopt(sb, adopt_decision(), "x")
    S.discard(sb)


def test_discard_removes_worktree_and_branch(repo: Path, scratch: Path) -> None:
    sb = S.Sandbox.open(repo, scratch=scratch)
    sb.write("pkg/tmp.py", "Z = 0\n")
    S.discard(sb)
    assert not sb.path.exists() and S.SANDBOX_PREFIX not in sh(repo, "branch", "--list")


def test_interrupted_sandboxes_are_recovered_and_never_assumed_adopted(repo: Path, scratch: Path) -> None:
    sb = S.Sandbox.open(repo, scratch=scratch)
    sb.write("pkg/half.py", "W = 1\n")                                             # the process "crashes" here: no close
    found = S.recover(repo, scratch)
    me = [f for f in found if f["id"] == sb.id]
    assert me and me[0]["state"] == "INTERRUPTED" and not me[0]["adopted"]
    S.discard(sb)


def test_nothing_changed_cannot_be_evaluated(repo: Path, scratch: Path) -> None:
    with S.Sandbox.open(repo, scratch=scratch) as sb:
        with pytest.raises(S.SandboxError, match="nothing changed"):
            sb.evaluate(build_config=NO_TYPES)


def test_hidden_paths_are_absent_but_kept_in_history(repo: Path, scratch: Path) -> None:
    """A worker's sandbox must not contain the sealed answer keys; hiding them must not delete them from commits."""
    (repo / "secret").mkdir()
    (repo / "secret" / "answers.txt").write_text("42\n", encoding="utf-8")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "answers")
    sb = S.Sandbox.open(repo, scratch=scratch, hide=("secret",))
    assert not (sb.path / "secret").exists() and (sb.path / "pkg" / "mathx.py").is_file()
    assert not sb.changes().paths                                                        # hiding is not a deletion
    sb.write("pkg/other.py", "def name():\n    return 'changed'\n")
    res = S.adopt(sb, adopt_decision(), "change with hidden paths")
    assert (repo / "secret" / "answers.txt").read_text(encoding="utf-8") == "42\n"           # not deleted by the merge
    assert "secret/answers.txt" in sh(repo, "ls-tree", "-r", "--name-only", res.merge_commit)
    main_sparse = subprocess.run(["git", "config", "--get", "core.sparseCheckout"], cwd=repo, capture_output=True, text=True)
    assert main_sparse.stdout.strip() != "true"                                               # the main worktree stays full
    sb.close()


def test_typecheck_command_must_invoke_mypy(tmp_path: Path) -> None:
    """Regression (1 Oct): the CI line `pip install -r requirements.txt pytest mypy` was taken as the type check."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text("steps:\n  - run: pip install -r requirements.txt pytest mypy\n  - run: python -m mypy --strict\n",
                               encoding="utf-8")
    assert B.ci_typecheck_command(tmp_path, python="PY") == ["PY", "-m", "mypy", "--strict"]
    (wf / "ci.yml").write_text("steps:\n  - run: pip install mypy\n", encoding="utf-8")
    assert B.ci_typecheck_command(tmp_path, python="PY") == ["PY", "-m", "mypy"]
