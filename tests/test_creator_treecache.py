"""Evaluation-cost cut: passing results of a byte-identical clean git tree are served, never re-run - and every decision is the
one a fresh run gives. All tests use a TEMPORARY git repository."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from creator import build as B
from creator import sandbox as S
from creator import selfmodel as SM
from creator import testrun as T
from creator import treecache as TC

NO_TYPES = B.BuildConfig(run_typecheck=False, require_typecheck=False)
PY = T.PytestConfig().python


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "pkg" / "mathx.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (r / "tests" / "test_mathx.py").write_text(
        "from pkg.mathx import add\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_add0():\n    assert add(0, 0) == 0\n",
        encoding="utf-8")
    (r / "tests" / "test_more.py").write_text(
        "from pkg.mathx import add\n\nclass TestC:\n    def test_neg(self):\n        assert add(-1, 1) == 0\n", encoding="utf-8")
    (r / "tests" / "test_known_broken.py").write_text(
        "from pkg.mathx import add\n\ndef test_already_broken():\n    assert add(1, 1) == 3\n", encoding="utf-8")
    (r / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    sh(r, "init", "-q")
    sh(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    sh(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    return r


def per_file_cache(repo: Path, cache: TC.TreeCache) -> dict[str, SM.TestEvidence]:
    """What main's assessment records: one pytest per test file, stored under the tree key."""
    ev = SM.collect_test_evidence(repo, sorted(f"tests/{p.name}" for p in (repo / "tests").glob("test_*.py")),
                                  repo.parent / "ev" / "t.json", junit_dir=repo.parent / "ev" / "junit")
    cache.store(TC.tree_key(repo, PY), ev)
    return ev


def test_tree_key_needs_a_clean_tree_and_follows_content(repo: Path) -> None:
    k = TC.tree_key(repo, PY)
    assert k and k == TC.tree_key(repo, PY)
    (repo / "pkg" / "mathx.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
    assert TC.tree_key(repo, PY) is None                                           # dirty: never vouched for
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    sh(repo, "checkout", "--", "pkg/mathx.py")
    assert TC.tree_key(repo, PY) is None                                           # an untracked file is dirty too
    (repo / "stray.txt").unlink()
    assert TC.tree_key(repo, PY) == k
    (repo / "state").mkdir()                                                       # the kernel's own evidence is not the tree
    (repo / "state" / "e.json").write_text("{}", encoding="utf-8")
    assert TC.tree_key(repo, PY) == k
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "state only")
    assert TC.tree_key(repo, PY) == k
    assert TC.tree_key(repo, "other-python") != k
    (repo / "pkg" / "mathx.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "edit")
    assert TC.tree_key(repo, PY) != k
    assert TC.tree_key(repo, PY, rev="HEAD~2", require_clean=False) == k           # the base's key, for the sandbox lookup


def test_only_passing_files_are_cached_and_entries_expire(repo: Path, tmp_path: Path) -> None:
    cache = TC.TreeCache(tmp_path / "tc")
    ev = per_file_cache(repo, cache)
    assert ev["tests/test_known_broken.py"].outcome == "FAIL"
    got = cache.lookup(TC.tree_key(repo, PY))
    assert set(got) == {"tests/test_mathx.py", "tests/test_more.py"}                # the failing file is never served
    assert cache.lookup("tree:unknown") == {} and cache.lookup(None) == {}
    assert cache.lookup(TC.tree_key(repo, PY), now=time.time() + TC.MAX_AGE_S + 5) == {}
    for i in range(TC.KEEP_TREES + 2):                                              # old trees are pruned
        cache.store(f"k{i}", ev)
    assert cache.lookup("k0") == {} and cache.lookup(f"k{TC.KEEP_TREES + 1}")


def test_assessment_serves_a_matching_file_and_runs_a_mismatched_one(repo: Path, tmp_path: Path, monkeypatch: Any) -> None:
    cache = TC.TreeCache(tmp_path / "tc")
    per_file_cache(repo, cache)
    served = cache.lookup(TC.tree_key(repo, PY))
    calls: list[str] = []
    real = T.run_pytest

    def spy(root: Any, targets: Any, *a: Any, **k: Any) -> Any:
        calls.extend(targets)
        return real(root, targets, *a, **k)
    monkeypatch.setattr(T, "run_pytest", spy)
    tests = ["tests/test_mathx.py", "tests/test_more.py", "tests/test_known_broken.py"]
    ev = SM.collect_test_evidence(repo, tests, tmp_path / "ev2" / "t.json", junit_dir=tmp_path / "ev2" / "j", reuse=served)
    assert calls == ["tests/test_known_broken.py"]                                  # only the file with no vouched pass ran
    assert [ev[t].outcome for t in tests] == ["PASS", "PASS", "FAIL"]
    assert all(Path(ev[t].where).is_file() for t in tests)
    calls.clear()
    lying = {t: ("0" * 64, x) for t, (_, x) in served.items()}                      # a digest that does not match this tree
    ev = SM.collect_test_evidence(repo, tests[:2], tmp_path / "ev3" / "t.json", junit_dir=tmp_path / "ev3" / "j", reuse=lying)
    assert sorted(calls) == tests[:2] and [ev[t].outcome for t in tests[:2]] == ["PASS", "PASS"]


CHANGES = {
    "clean": "def add(a, b):\n    return b + a\n",                                   # no behaviour change
    "regressing": "def add(a, b):\n    return a - b\n",                             # breaks test_add / test_neg
    "fixing": "def add(a, b):\n    return a + b + (1 if (a, b) == (1, 1) else 0)\n",   # fixes the base's known failure
    "broken-syntax": "def add(a, b)\n    return a + b\n",
}


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_serving_the_base_changes_no_decision(repo: Path, tmp_path: Path, name: str) -> None:
    cache = TC.TreeCache(tmp_path / "tc")
    per_file_cache(repo, cache)
    served = {t: x for t, (_, x) in cache.lookup(TC.tree_key(repo, PY, rev=S.head(repo), require_clean=False)).items()}
    assert served
    out = []
    for i, reuse in enumerate((None, served)):
        with S.Sandbox.open(repo, scratch=tmp_path / f"scratch{i}") as sb:
            sb.write("pkg/mathx.py", CHANGES[name])
            ev = sb.evaluate(build_config=NO_TYPES, base_reuse=reuse)
            out.append(ev)
            sb.cleanup_base()
    fresh, cheap = out
    assert fresh.clean == cheap.clean and fresh.builds == cheap.builds
    if fresh.report is None:
        assert cheap.report is None
        return
    assert cheap.report is not None
    assert fresh.report.verdict == cheap.report.verdict
    assert fresh.report.classes == cheap.report.classes
    assert fresh.report.blocking == cheap.report.blocking and fresh.report.reasons == cheap.report.reasons
    assert fresh.base_run is not None and cheap.base_run is not None
    assert {k: v.outcome for k, v in fresh.base_run.cases.items()} == {k: v.outcome for k, v in cheap.base_run.cases.items()}
    assert fresh.base_run.selection == cheap.base_run.selection


def test_a_fully_served_base_runs_no_pytest(repo: Path, tmp_path: Path) -> None:
    cache = TC.TreeCache(tmp_path / "tc")
    per_file_cache(repo, cache)
    served = {t: x for t, (_, x) in cache.lookup(TC.tree_key(repo, PY)).items()}
    run = T.run_base(repo, ["tests/test_mathx.py", "tests/test_more.py"], tmp_path / "b.xml", tree="t", reuse=served)
    assert run.status is T.RunStatus.PASSED and run.proc is None and not (tmp_path / "b.xml").exists()
    assert set(run.cases) == {"tests/test_mathx.py::test_add", "tests/test_mathx.py::test_add0", "tests/test_more.py::TestC::test_neg"}
    mixed = T.run_base(repo, ["tests/test_mathx.py", "tests/test_known_broken.py"], tmp_path / "m.xml", tree="t", reuse=served)
    assert mixed.status is T.RunStatus.FAILED and "tests/test_mathx.py::test_add" in mixed.cases
    assert mixed.cases["tests/test_known_broken.py::test_already_broken"].outcome.bad
    (tmp_path / "bad.xml").write_text("not xml", encoding="utf-8")                  # an unreadable record is run, not trusted
    run = T.run_base(repo, ["tests/test_mathx.py"], tmp_path / "c.xml", tree="t", reuse={"tests/test_mathx.py": str(tmp_path / "bad.xml")})
    assert run.status is T.RunStatus.PASSED and run.proc is not None
