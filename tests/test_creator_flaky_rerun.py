"""CR03 flaky re-run wiring: Sandbox.evaluate re-runs blocking cases and a coin-flip case is FLAKY, never a regression."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import build as B
from creator import kernel as K
from creator import sandbox as S
from creator import testrun as T

NO_TYPES = B.BuildConfig(run_typecheck=False, require_typecheck=False)
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]

# Alternates pass/fail by a counter file kept OUTSIDE the tree (shared by base and candidate runs): deterministic "50%".
FLAKY_TEST = (
    "import os\nfrom pkg.mathx import add\n\n"
    "def test_flip():\n"
    "    p = os.environ['FLIP_COUNTER']\n"
    "    n = int(open(p).read()) if os.path.exists(p) else 0\n"
    "    open(p, 'w').write(str(n + 1))\n"
    "    assert n % 2 == 0 and add(1, 1) == 2\n")
STEADY_TEST = "from pkg.mathx import add\n\ndef test_steady():\n    assert add(2, 3) == 5\n"


def sh(repo: Path, *args: str) -> None:
    subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True)


def make_repo(tmp: Path, test_src: str, name: str) -> Path:
    r = tmp / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "pkg" / "mathx.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (r / "tests" / name).write_text(test_src, encoding="utf-8")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    return r


def cfg(tmp: Path) -> T.PytestConfig:
    return T.PytestConfig(env={"FLIP_COUNTER": str(tmp / "counter.txt")})


def test_flaky_case_is_flaky_not_regressed(tmp_path: Path) -> None:
    (tmp_path / "counter.txt").write_text("1")           # the candidate's first run (n=1) fails, the re-run (n=2) passes
    repo = make_repo(tmp_path, "def test_flip():\n    assert True\n", "test_flip.py")   # passes at base
    with S.Sandbox.open(repo, scratch=tmp_path / "scratch") as sb:
        sb.write("pkg/mathx.py", "def add(a, b):\n    return a + b  # harmless\n")
        sb.write("tests/test_flip.py", FLAKY_TEST)
        ev = sb.evaluate(build_config=NO_TYPES, pytest_config=cfg(tmp_path), flaky_reruns=3)
        assert ev.report is not None
        assert ev.report.classes["tests/test_flip.py::test_flip"] is T.CaseClass.FLAKY
        assert ev.report.verdict is T.Verdict.FLAKY and not ev.clean
        assert K._pass_fraction(ev.candidate_run, 1.0, ev) == 1.0      # flaky cases do not move the guard
        sb.cleanup_base()


def test_deterministic_regression_survives_reruns(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, STEADY_TEST, "test_steady.py")
    with S.Sandbox.open(repo, scratch=tmp_path / "scratch") as sb:
        sb.write("pkg/mathx.py", "def add(a, b):\n    return a - b\n")
        ev = sb.evaluate(build_config=NO_TYPES, pytest_config=cfg(tmp_path), flaky_reruns=2)
        assert ev.report is not None and ev.report.verdict is T.Verdict.REGRESSED
        assert ev.report.classes["tests/test_steady.py::test_steady"] is T.CaseClass.REGRESSION
        assert ev.report.flaky_history["tests/test_steady.py::test_steady"]["candidate"] == ("FAILED",) * 3
        sb.cleanup_base()


def test_reruns_off_keeps_single_run_behaviour(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, STEADY_TEST, "test_steady.py")
    with S.Sandbox.open(repo, scratch=tmp_path / "scratch") as sb:
        sb.write("pkg/mathx.py", "def add(a, b):\n    return a - b\n")
        ev = sb.evaluate(build_config=NO_TYPES, pytest_config=cfg(tmp_path), flaky_reruns=0)
        assert ev.report is not None and not ev.report.flaky_history
        sb.cleanup_base()


def test_pass_fraction_excludes_flaky_cases() -> None:
    mk = lambda o: T.CaseResult("x", o)  # noqa: E731
    run = SimpleNamespace(cases={"a::t1": mk(T.Outcome.PASSED), "a::t2": mk(T.Outcome.FAILED)})
    ev = SimpleNamespace(report=SimpleNamespace(ids=lambda c: ["a::t2"]))
    assert K._pass_fraction(run, 1.0, ev) == 1.0  # type: ignore[arg-type]
    assert K._pass_fraction(run, 1.0) == 0.5  # type: ignore[arg-type]
