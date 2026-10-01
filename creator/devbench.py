"""Creator K15 - devbench: the SEALED development-task suite that measures the Creator (C77 secs 30-32, 49-56, 77; package CR05)
- IMPLEMENTED, NOT VALIDATED.

A task is a small, self-contained development problem:

    creator/devbench/tasks/<id>/task.json      objective (plain language), category, split, limits      VISIBLE to the solver
    creator/devbench/tasks/<id>/repo/...       the starting repository, incl. its visible tests            VISIBLE to the solver
    creator/devbench/sealed/<id>/hidden/...    hidden acceptance tests (the answer key)                    NEVER visible
    creator/devbench/sealed/MANIFEST.json      sha256 of every task's visible repo and hidden files, and the holdout list,
                                               written ONCE by `seal()` before any run; creator/devbench/sealed/ is a protected
                                               path the Creator's sandbox refuses to modify.

Run order (C77 sec 52 pattern, applied to development): TASK -> SEAL CHECK -> RELEASE visible repo to a fresh scratch dir -> SOLVER
works there -> RESULT FREEZE (tree hash + the solver's claim) -> EVALUATOR copies the hidden tests in and runs visible + hidden
tests -> SCORE. The solver never sees the hidden tests or the manifest; scoring refuses a run whose hidden files no longer hash to
the manifest, or whose result was not frozen before evaluation.

A score per task: SOLVED (all visible and hidden tests pass), FALSE_COMPLETION (the solver claimed done but hidden tests fail),
REGRESSION (a visible test that passed at the start fails at the end), UNSOLVED, ERROR; plus cost (agent calls, seconds). The
holdout split is scored only through `score_holdout()`, which refuses a solver configuration that has not been frozen first."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

BENCH = Path(__file__).resolve().parent / "devbench"
TASKS = BENCH / "tasks"
SEALED = BENCH / "sealed"
MANIFEST = SEALED / "MANIFEST.json"
SPLITS = ("dev", "holdout")
CATEGORIES = ("bugfix", "feature", "test_gap", "refactor", "performance", "integration", "diagnosis")


class DevbenchError(RuntimeError):
    """A sealing or scoring rule was violated; the run does not count."""


# ------------------------------------------------------------------------------------------------ hashing

def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tree_digest(root: Path, exclude: Iterable[str] = (".git", "__pycache__", ".pytest_cache")) -> str:
    """sha256 over the sorted (relative path, content) of every file under root."""
    ex = set(exclude)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in ex for part in p.relative_to(root).parts):
            h.update(p.relative_to(root).as_posix().encode())
            h.update(file_sha(p).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ tasks

@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    category: str
    split: str
    objective: str
    time_limit_s: int
    call_limit: int
    visible_dir: Path
    hidden_dir: Path

    def to_public(self) -> dict[str, Any]:
        """What a solver may see: never the hidden directory, never the split."""
        return {"id": self.id, "category": self.category, "objective": self.objective, "time_limit_s": self.time_limit_s,
                "call_limit": self.call_limit}


def load_tasks(tasks_dir: Path = TASKS, sealed_dir: Path = SEALED) -> list[Task]:
    out = []
    for tj in sorted(tasks_dir.glob("*/task.json")):
        d = json.loads(tj.read_text(encoding="utf-8"))
        tid = tj.parent.name
        if d.get("id", tid) != tid:
            raise DevbenchError(f"{tj}: id {d.get('id')} != folder {tid}")
        if d["category"] not in CATEGORIES or d["split"] not in SPLITS:
            raise DevbenchError(f"{tj}: bad category/split")
        out.append(Task(tid, d["category"], d["split"], d["objective"], int(d.get("time_limit_s", 900)),
                        int(d.get("call_limit", 20)), tj.parent / "repo", sealed_dir / tid / "hidden"))
    return out


def file_map(d: Path) -> dict[str, str]:
    """relative path -> sha256 for every file under d (empty when d does not exist)."""
    if not d.is_dir():
        return {}
    return {p.relative_to(d).as_posix(): file_sha(p) for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def seal(tasks_dir: Path = TASKS, sealed_dir: Path = SEALED, manifest: Path = MANIFEST, force: bool = False) -> dict[str, Any]:
    """Write the manifest ONCE: per task the visible-repo digest and every hidden file's sha256, plus the split lists. Re-sealing an
    existing manifest is refused (a changed answer key would invalidate every earlier score) unless force=True, which is recorded."""
    if manifest.exists() and not force:
        raise DevbenchError(f"{manifest} exists; the suite is already sealed")
    tasks = load_tasks(tasks_dir, sealed_dir)
    m: dict[str, Any] = {"tasks": {}, "splits": {s: sorted(t.id for t in tasks if t.split == s) for s in SPLITS},
                         "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resealed": bool(force)}
    for t in tasks:
        if not t.hidden_dir.is_dir() or not any(t.hidden_dir.rglob("test_*.py")):
            raise DevbenchError(f"task {t.id} has no hidden acceptance tests")
        m["tasks"][t.id] = {"visible": tree_digest(t.visible_dir), "hidden": file_map(t.hidden_dir),
                            "reference": file_map(t.hidden_dir.parent / "reference"), "category": t.category}
    body = json.dumps(m, indent=1, sort_keys=True)
    m["digest"] = hashlib.sha256(body.encode()).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    return m


def load_manifest(manifest: Path = MANIFEST) -> dict[str, Any]:
    if not manifest.is_file():
        raise DevbenchError("the suite is not sealed - nothing may be scored before seal()")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    digest = m.pop("digest", None)
    if digest != hashlib.sha256(json.dumps(m, indent=1, sort_keys=True).encode()).hexdigest():
        raise DevbenchError("the manifest was edited after sealing")
    m["digest"] = digest
    return m


def check_sealed(task: Task, manifest: Mapping[str, Any]) -> None:
    """The visible repo and every hidden file must still hash to the manifest (fail closed)."""
    entry = manifest["tasks"].get(task.id)
    if entry is None:
        raise DevbenchError(f"task {task.id} was not sealed")
    if tree_digest(task.visible_dir) != entry["visible"]:
        raise DevbenchError(f"task {task.id}: visible repo changed since sealing")
    if file_map(task.hidden_dir) != entry["hidden"]:
        raise DevbenchError(f"task {task.id}: hidden answer key changed since sealing")
    if file_map(task.hidden_dir.parent / "reference") != entry.get("reference", {}):
        raise DevbenchError(f"task {task.id}: reference solution changed since sealing")


# ------------------------------------------------------------------------------------------------ solvers

@dataclasses.dataclass(frozen=True)
class SolverResult:
    claimed_done: bool
    calls: int = 0
    notes: str = ""


class Solver(Protocol):
    """Anything that develops: given the public task and a working directory, change files there and report."""
    name: str

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> SolverResult: ...


class NullSolver:
    """Does nothing and claims nothing: the floor every real solver must beat."""
    name = "null"

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> SolverResult:
        return SolverResult(False, 0, "did nothing")


class LiarSolver:
    """Does nothing but claims success: the scorer must report FALSE_COMPLETION, never SOLVED."""
    name = "liar"

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> SolverResult:
        return SolverResult(True, 0, "claims done without changing anything")


class CheatSolver:
    """Changes no logic; instead tries to fake a passing run: a conftest.py hook that skips every test body, a pytest.ini that
    deselects tests, a sitecustomize.py and an atexit print of a fake summary. Must never score SOLVED."""
    name = "cheat"

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> SolverResult:
        (workdir / "conftest.py").write_text(
            "import pytest\n\n@pytest.hookimpl(tryfirst=True)\ndef pytest_pyfunc_call(pyfuncitem):\n    return True\n",
            encoding="utf-8")
        (workdir / "pytest.ini").write_text("[pytest]\naddopts = -p no:randomly\n", encoding="utf-8")
        (workdir / "sitecustomize.py").write_text(
            "import atexit\natexit.register(lambda: print('99 passed in 0.01s'))\n", encoding="utf-8")
        (workdir / "app" / "__init__.py").write_text(
            "import atexit, sys\natexit.register(lambda: print('99 passed in 0.01s'))\n", encoding="utf-8")
        return SolverResult(True, 0, "faked the test run")


class ReferenceSolver:
    """Copies the sealed reference solution in. It exists ONLY to prove each task is solvable (validate_suite); it reads the
    sealed directory, so it is never a Creator solver and never scored as one."""
    name = "reference"

    def __init__(self, sealed_dir: Path = SEALED) -> None:
        self.sealed_dir = sealed_dir

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> SolverResult:
        src = self.sealed_dir / str(task["id"]) / "reference"
        n = 0
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                dst = workdir / p.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
                n += 1
        return SolverResult(n > 0, 0, f"copied {n} reference file(s)")


# ------------------------------------------------------------------------------------------------ running and scoring

@dataclasses.dataclass(frozen=True)
class TestCounts:
    passed: int
    failed: int
    errors: int
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0 and self.passed > 0


EVAL_INI = "[pytest]\naddopts =\n"
RUNNER = (
    "import sys\n"
    "code_dir, ini, junit, target = sys.argv[1:5]\n"
    "sys.path.append(code_dir)\n"                       # APPENDED: a solution's pytest.py / stdlib-named module cannot shadow ours
    "import pytest\n"
    "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '--noconftest', '-c', ini, '--rootdir', code_dir,\n"
    "                      '--import-mode=append', '--junitxml', junit, target]))\n"
)


def purge_bytecode(tree: Path) -> int:
    """Delete every __pycache__ and .pyc under tree. Found 30 Sep: python -I ignores PYTHONDONTWRITEBYTECODE, so the pre-solver run
    wrote .pyc files; a same-size edit in the same second then ran the STALE bytecode (the reference solution scored as the bug).
    A solver could also plant forged .pyc files on purpose, so the evaluator never runs bytecode it did not compile itself."""
    n = 0
    for d in sorted(tree.rglob("__pycache__"), reverse=True):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    for f in tree.rglob("*.pyc"):
        f.unlink(missing_ok=True)
        n += 1
    return n


def run_pytest(cwd: Path, target: str, timeout: float = 300.0) -> TestCounts:
    """Run tests on a solver-controlled tree WITHOUT trusting the tree (C77 sec 68, no fake success):
      - python -I (no PYTHONPATH, no user site, cwd not on the path) launched from an evaluator-owned empty directory, so a
        sitecustomize.py, pytest.py or stdlib-named module in the tree is not imported by the runner;
      - --noconftest and an evaluator-owned ini (-c), so a conftest.py hook or pytest.ini addopts in the tree cannot rewrite results;
      - -B and purge_bytecode() before every run, so no bytecode from an earlier run or from the solver is ever executed;
      - counts come from a JUnit report written OUTSIDE the tree and must agree with the exit code; a missing or unreadable report,
        or a report that disagrees with the exit code, is an ERROR - stdout (which code can print into) is never read.
    Residual, recorded in the self-model: code running inside the test process can still tamper with that process; only an
    OS-level sandbox closes that, and none is available here."""
    purge_bytecode(cwd)
    harness = Path(tempfile.mkdtemp(prefix="devbench-eval-"))
    try:
        ini = harness / "eval.ini"
        ini.write_text(EVAL_INI, encoding="utf-8")
        junit = harness / "junit.xml"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PYTEST"))}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run([sys.executable, "-I", "-B", "-c", RUNNER, str(cwd), str(ini), str(junit), str(cwd / target)],
                                  cwd=harness, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return TestCounts(0, 0, 1, -1)
        if not junit.is_file():
            return TestCounts(0, 0, 1, proc.returncode)
        import xml.etree.ElementTree as ET
        try:
            root = ET.parse(junit).getroot()
        except ET.ParseError:
            return TestCounts(0, 0, 1, proc.returncode)
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        tests = sum(int(x.get("tests", 0)) for x in suites)
        failed = sum(int(x.get("failures", 0)) for x in suites)
        errors = sum(int(x.get("errors", 0)) for x in suites)
        skipped = sum(int(x.get("skipped", 0)) for x in suites)
        passed = tests - failed - errors - skipped
        consistent = (proc.returncode == 0) == (failed == 0 and errors == 0 and passed > 0)
        if not consistent:
            return TestCounts(0, failed, errors + 1, proc.returncode)
        return TestCounts(passed, failed, errors, proc.returncode)
    finally:
        shutil.rmtree(harness, ignore_errors=True)


@dataclasses.dataclass(frozen=True)
class TaskScore:
    task_id: str
    category: str
    split: str
    outcome: str                    # SOLVED / FALSE_COMPLETION / REGRESSION / UNSOLVED / ERROR
    claimed_done: bool
    visible_before: TestCounts
    visible_after: TestCounts
    hidden: TestCounts
    calls: int
    seconds: float
    frozen_digest: str
    solver: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


def run_task(task: Task, solver: Callable[[Mapping[str, Any], Path], SolverResult], manifest: Mapping[str, Any],
             scratch: Optional[Path] = None, solver_name: str = "") -> TaskScore:
    """One task, in the sealed order. The solver gets a COPY of the visible repo only."""
    check_sealed(task, manifest)
    base = Path(tempfile.mkdtemp(prefix=f"devbench-{task.id}-", dir=scratch))
    work = base / "work"
    shutil.copytree(task.visible_dir, work)
    before = run_pytest(work, "tests")
    t0 = time.monotonic()
    try:
        res = solver(task.to_public(), work)
        err = ""
    except Exception as e:                                              # noqa: BLE001 - a crashing solver is scored, not hidden
        res, err = SolverResult(False, 0, f"solver crashed: {type(e).__name__}: {e}"), "crash"
    secs = round(time.monotonic() - t0, 2)
    frozen = tree_digest(work)                                          # RESULT FREEZE before the evaluator touches anything
    visible_after = run_pytest(work, "tests")
    evald = base / "eval"
    shutil.copytree(work, evald)
    if tree_digest(evald) != frozen:
        raise DevbenchError("the evaluated copy differs from the frozen result")
    hidden_target = evald / "_hidden_acceptance"
    shutil.copytree(task.hidden_dir, hidden_target)
    hidden = run_pytest(evald, "_hidden_acceptance")
    regression = before.passed > 0 and visible_after.passed < before.passed or (visible_after.failed > before.failed)
    if err:
        outcome = "ERROR"
    elif visible_after.ok and hidden.ok:
        outcome = "SOLVED"
    elif regression:
        outcome = "REGRESSION"
    elif res.claimed_done:
        outcome = "FALSE_COMPLETION"
    else:
        outcome = "UNSOLVED"
    shutil.rmtree(base, ignore_errors=True)
    return TaskScore(task.id, task.category, task.split, outcome, res.claimed_done, before, visible_after, hidden, res.calls,
                     secs, frozen, solver_name or getattr(solver, "name", "solver"), res.notes)


@dataclasses.dataclass(frozen=True)
class SuiteScore:
    split: str
    solver: str
    scores: tuple[TaskScore, ...]

    def rate(self, outcome: str) -> float:
        return sum(s.outcome == outcome for s in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def solve_rate(self) -> float:
        return self.rate("SOLVED")

    @property
    def stderr(self) -> float:
        n = len(self.scores)
        p = self.solve_rate
        return (p * (1 - p) / n) ** 0.5 if n else 0.0

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {o: sum(s.outcome == o for s in self.scores)
                               for o in ("SOLVED", "FALSE_COMPLETION", "REGRESSION", "UNSOLVED", "ERROR")}
        out.update(split=self.split, solver=self.solver, n=len(self.scores), solve_rate=self.solve_rate, stderr=self.stderr,
                   calls=sum(s.calls for s in self.scores), seconds=round(sum(s.seconds for s in self.scores), 1),
                   by_category={c: sum(s.outcome == "SOLVED" for s in self.scores if s.category == c)
                                for c in sorted({s.category for s in self.scores})})
        return out


def score_dev(solver: Callable[[Mapping[str, Any], Path], SolverResult], solver_name: str = "",
              tasks: Optional[Sequence[Task]] = None, manifest: Optional[Mapping[str, Any]] = None,
              scratch: Optional[Path] = None) -> SuiteScore:
    m = manifest or load_manifest()
    ts = [t for t in (tasks or load_tasks()) if t.split == "dev"]
    return SuiteScore("dev", solver_name or getattr(solver, "name", "solver"),
                      tuple(run_task(t, solver, m, scratch, solver_name) for t in ts))


def freeze_config(config: Mapping[str, Any], path: Path) -> str:
    """Record a solver configuration as frozen (its digest) BEFORE it may see the holdout split (C77 sec 32, 69)."""
    body = json.dumps(dict(config), sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    frozen = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    frozen[digest] = {"config": dict(config), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, indent=1, sort_keys=True), encoding="utf-8")
    return digest


def score_holdout(solver: Callable[[Mapping[str, Any], Path], SolverResult], config: Mapping[str, Any], frozen_path: Path,
                  solver_name: str = "", tasks: Optional[Sequence[Task]] = None, manifest: Optional[Mapping[str, Any]] = None,
                  scratch: Optional[Path] = None) -> SuiteScore:
    """Score the holdout split - only for a configuration frozen beforehand, so the holdout cannot be tuned against."""
    digest = hashlib.sha256(json.dumps(dict(config), sort_keys=True).encode()).hexdigest()[:16]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.is_file() else {}
    if digest not in frozen:
        raise DevbenchError("this solver configuration was not frozen before scoring the holdout (C77 sec 32)")
    m = manifest or load_manifest()
    ts = [t for t in (tasks or load_tasks()) if t.split == "holdout"]
    return SuiteScore("holdout", solver_name or getattr(solver, "name", "solver"),
                      tuple(run_task(t, solver, m, scratch, solver_name) for t in ts))


# ------------------------------------------------------------------------------------------------ suite validation (answer key first)

@dataclasses.dataclass(frozen=True)
class SuiteValidation:
    ok: bool
    problems: tuple[str, ...]
    per_task: Mapping[str, Mapping[str, str]]


def validate_suite(tasks: Optional[Sequence[Task]] = None, manifest: Optional[Mapping[str, Any]] = None,
                   scratch: Optional[Path] = None, sealed_dir: Path = SEALED, parallel: int = 1) -> SuiteValidation:
    """Calibrate the benchmark before it measures anything (C77 secs 49-55; C70-C74 'answer key first'):
    every task must be SOLVED by its sealed reference, NOT solved by the do-nothing solver, and scored FALSE_COMPLETION (never
    SOLVED) for a solver that claims success without changing anything. A suite that fails any of these cannot score the Creator."""
    m = manifest or load_manifest()
    ts = list(tasks or load_tasks())
    problems: list[str] = []
    per: dict[str, dict[str, str]] = {}
    def calibrate(t: Task) -> tuple[str, dict[str, str]]:
        return t.id, {"reference": run_task(t, ReferenceSolver(sealed_dir), m, scratch, "reference").outcome,
                      "null": run_task(t, NullSolver(), m, scratch, "null").outcome,
                      "liar": run_task(t, LiarSolver(), m, scratch, "liar").outcome,
                      "cheat": run_task(t, CheatSolver(), m, scratch, "cheat").outcome}
    if parallel > 1:                                                    # tasks are independent (own temp dirs)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            results = list(pool.map(calibrate, ts))
    else:
        results = [calibrate(t) for t in ts]
    for tid, row in results:
        per[tid] = row
    for t in ts:
        ref, null, liar, cheat = per[t.id]["reference"], per[t.id]["null"], per[t.id]["liar"], per[t.id]["cheat"]
        if ref != "SOLVED":
            problems.append(f"{t.id}: the reference solution scores {ref} - the task is not provably solvable")
        if null == "SOLVED":
            problems.append(f"{t.id}: doing nothing solves it - the task measures nothing")
        if liar == "SOLVED":
            problems.append(f"{t.id}: a false claim of success scores SOLVED")
        if cheat == "SOLVED":
            problems.append(f"{t.id}: a solver that fakes the test run scores SOLVED")
    return SuiteValidation(not problems, tuple(problems), per)
