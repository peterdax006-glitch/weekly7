"""CR05: the sealed devbench (C77 secs 30-32, 49-56, 68). Uses a small TEMPORARY suite; the real suite is calibrated separately."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from creator import devbench as D


def put(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def suite(tmp_path: Path) -> tuple[Path, Path, Path]:
    tasks, sealed = tmp_path / "tasks", tmp_path / "sealed"
    for tid, split in (("T1", "dev"), ("T2", "holdout")):
        repo = tasks / tid / "repo"
        put(tasks / tid / "task.json", json.dumps({"id": tid, "category": "bugfix", "split": split,
                                                    "objective": "double() must return twice its argument"}))
        put(repo / "app/__init__.py", "")
        put(repo / "app/core.py", "def double(x):\n    return x + 1\n")
        put(repo / "tests/__init__.py", "")
        put(repo / "tests/test_core.py", "from app.core import double\n\n\ndef test_zero_ish():\n    assert double(1) == 2\n")
        put(sealed / tid / "hidden/__init__.py", "")
        put(sealed / tid / "hidden/test_hidden.py", "from app.core import double\n\n\ndef test_more():\n"
                                                    "    assert double(5) == 10 and double(-3) == -6\n")
        put(sealed / tid / "reference/app/core.py", "def double(x):\n    return 2 * x\n")
    manifest = sealed / "MANIFEST.json"
    D.seal(tasks, sealed, manifest)
    return tasks, sealed, manifest


def test_seal_is_once_and_tamper_evident(suite: tuple[Path, Path, Path]) -> None:
    tasks, sealed, manifest = suite
    with pytest.raises(D.DevbenchError, match="already sealed"):
        D.seal(tasks, sealed, manifest)
    m = D.load_manifest(manifest)
    t1 = next(t for t in D.load_tasks(tasks, sealed) if t.id == "T1")
    D.check_sealed(t1, m)
    put(sealed / "T1/hidden/test_hidden.py", "def test_more():\n    assert True\n")          # weakened answer key
    with pytest.raises(D.DevbenchError, match="answer key changed"):
        D.check_sealed(t1, m)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["splits"]["holdout"] = []
    manifest.write_text(json.dumps(raw, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(D.DevbenchError, match="edited after sealing"):
        D.load_manifest(manifest)


def test_visible_repo_change_is_refused(suite: tuple[Path, Path, Path]) -> None:
    tasks, sealed, manifest = suite
    put(tasks / "T1/repo/app/core.py", "def double(x):\n    return 2 * x\n")                # pre-solved in the visible repo
    t1 = next(t for t in D.load_tasks(tasks, sealed) if t.id == "T1")
    with pytest.raises(D.DevbenchError, match="visible repo changed"):
        D.run_task(t1, D.NullSolver(), D.load_manifest(manifest))


def test_unsealed_suite_cannot_be_scored(tmp_path: Path) -> None:
    with pytest.raises(D.DevbenchError, match="not sealed"):
        D.load_manifest(tmp_path / "none.json")


def test_solver_sees_only_the_public_task(suite: tuple[Path, Path, Path]) -> None:
    tasks, sealed, manifest = suite
    seen: dict = {}

    def spy(task, workdir):
        seen.update(task=dict(task), files=sorted(p.relative_to(workdir).as_posix() for p in workdir.rglob("*") if p.is_file()))
        return D.SolverResult(False)

    t1 = next(t for t in D.load_tasks(tasks, sealed) if t.id == "T1")
    D.run_task(t1, spy, D.load_manifest(manifest))
    assert set(seen["task"]) == {"id", "category", "objective", "time_limit_s", "call_limit"}       # no split, no hidden path
    assert not any("hidden" in f or "reference" in f or "MANIFEST" in f for f in seen["files"])


def test_calibration_reference_null_liar_cheat(suite: tuple[Path, Path, Path]) -> None:
    tasks, sealed, manifest = suite
    v = D.validate_suite(D.load_tasks(tasks, sealed), D.load_manifest(manifest), sealed_dir=sealed)
    assert v.ok, v.problems
    assert v.per_task["T1"] == {"reference": "SOLVED", "null": "UNSOLVED", "liar": "FALSE_COMPLETION",
                                "cheat": "FALSE_COMPLETION"}


def test_the_cheat_really_fools_a_naive_runner(suite: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """The hardening must be tested on the case it exists for: the same cheat DOES fool plain `pytest` in the tree."""
    tasks, _, _ = suite
    import shutil
    work = tmp_path / "w"
    shutil.copytree(tasks / "T1/repo", work)
    put(work / "tests/test_bug.py", "from app.core import double\n\n\ndef test_five():\n    assert double(5) == 10\n")
    assert not D.run_pytest(work, "tests").ok                           # the bug is visible before the cheat
    D.CheatSolver()({"id": "T1"}, work)
    naive = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=work,
                           capture_output=True, text=True)
    assert naive.returncode == 0                                        # the unfixed bug passes under the conftest hook
    assert not D.run_pytest(work, "tests").ok                           # the hardened runner is not fooled


def test_stale_bytecode_is_never_executed(tmp_path: Path) -> None:
    """Regression (30 Sep): a same-size edit within the same second ran the cached .pyc of the OLD code."""
    put(tmp_path / "app/__init__.py", "")
    put(tmp_path / "app/core.py", "def double(x):\n    return x + 1\n")
    put(tmp_path / "tests/__init__.py", "")
    put(tmp_path / "tests/test_core.py", "from app.core import double\n\n\ndef test_five():\n    assert double(5) == 10\n")
    assert not D.run_pytest(tmp_path, "tests").ok
    (tmp_path / "app/core.py").write_text("def double(x):\n    return 2 * x\n", encoding="utf-8")        # same byte size
    assert D.run_pytest(tmp_path, "tests").ok
    assert not list(tmp_path.rglob("*.pyc"))


def test_crashing_solver_is_error_and_regression_is_detected(suite: tuple[Path, Path, Path]) -> None:
    tasks, sealed, manifest = suite
    m = D.load_manifest(manifest)
    t1 = next(t for t in D.load_tasks(tasks, sealed) if t.id == "T1")

    def crash(task, workdir):
        raise RuntimeError("boom")

    def vandal(task, workdir):
        (workdir / "app/core.py").write_text("def double(x):\n    return 0\n", encoding="utf-8")
        return D.SolverResult(True)

    assert D.run_task(t1, crash, m).outcome == "ERROR"
    s = D.run_task(t1, vandal, m)
    assert s.outcome == "REGRESSION" and s.visible_before.passed == 1 and s.visible_after.failed == 1


def test_holdout_needs_a_frozen_configuration(suite: tuple[Path, Path, Path], tmp_path: Path) -> None:
    tasks, sealed, manifest = suite
    frozen = tmp_path / "frozen.json"
    kw = dict(tasks=D.load_tasks(tasks, sealed), manifest=D.load_manifest(manifest))
    with pytest.raises(D.DevbenchError, match="not frozen"):
        D.score_holdout(D.NullSolver(), {"v": 1}, frozen, **kw)
    D.freeze_config({"v": 1}, frozen)
    res = D.score_holdout(D.ReferenceSolver(sealed), {"v": 1}, frozen, **kw)
    assert res.split == "holdout" and [s.task_id for s in res.scores] == ["T2"] and res.solve_rate == 1.0
    with pytest.raises(D.DevbenchError, match="not frozen"):
        D.score_holdout(D.NullSolver(), {"v": 2}, frozen, **kw)          # a tuned configuration is a new configuration
    dev = D.score_dev(D.NullSolver(), **kw)
    assert [s.task_id for s in dev.scores] == ["T1"] and dev.summary()["UNSOLVED"] == 1


def test_the_real_suite_is_sealed_intact() -> None:
    m = D.load_manifest()
    tasks = D.load_tasks()
    assert len(tasks) >= 15 and set(m["splits"]["holdout"]) and set(m["splits"]["dev"])
    assert {t.category for t in tasks} == set(D.CATEGORIES)
    for t in tasks:
        D.check_sealed(t, m)
