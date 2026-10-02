"""K06 test stage (CR03; canon C77 sec 24-25 test execution, sec 29 version comparison, sec 46 regression protection).

Selects the tests a change can affect from the static import graph (changed module -> every module that imports it, transitively,
including lazy imports inside functions and importlib.import_module("literal") calls) plus an always-run smoke set; runs them with
pytest in the tree; parses junit xml into per-case results; compares the candidate run against the SAME selection on the base
commit. A regression is "failing now, passing at base"; a test that already failed at base is PREEXISTING, never a regression;
a regression that flips on re-run is FLAKY (reported, still not CLEAN). Anything that prevents a trustworthy comparison
(timeout, crash, unparsable results, nothing ran) makes the verdict INCONCLUSIVE - never CLEAN.
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from creator import diskcache as DC
from creator.build import ProcResult, clean_env, module_name_for, run_cmd

SKIP_DIRS = frozenset({".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache",
                       "build", "dist", ".tox"})
# Skipped only at the tree root: state/ holds run artefacts (logs, ledgers, snapshots of agent runs, ~22k directories) and no
# test or module imports from it (checked 1 Oct 2026: no `state.` import, no sys.path entry into it).
ROOT_SKIP_DIRS = frozenset({"state"})
CONFIG_FILES = frozenset({"pyproject.toml", "setup.cfg", "setup.py", "pytest.ini", "tox.ini", "requirements.txt",
                          "requirements-dev.txt", "conftest.py"})
DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})


# --------------------------------------------------------------------------------------------------------- import graph
def is_test_file(rel: str) -> bool:
    name = rel.replace("\\", "/").rsplit("/", 1)[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


_IMPORTS_SALT: list[str] = []


def _imports_salt() -> str:
    """Code-version salt: hash of the source of everything the per-file result depends on."""
    if not _IMPORTS_SALT:
        _IMPORTS_SALT.append(DC.salt_of((ImportGraph._add_file, _imports_of, module_name_for), (sys.version.encode(),)))
    return _IMPORTS_SALT[0]


_IMPORTS_CACHE: dict[tuple[str, str, bool, bytes], tuple[str, Any]] = {}      # content-keyed, see ImportGraph._add_file


@dataclass
class ImportGraph:
    """Static import graph of a tree. `imports[m]` are the dotted names module m may import (resolved against the tree when
    possible, kept raw otherwise so that a DELETED module's former importers are still found)."""
    root: str
    modules: dict[str, str] = field(default_factory=dict)            # module name -> relative path
    imports: dict[str, set[str]] = field(default_factory=dict)       # module name -> imported dotted names
    unparsable: dict[str, str] = field(default_factory=dict)         # relative path -> error
    nonmodule_tests: dict[str, set[str]] = field(default_factory=dict)  # test path not importable by name -> imports

    @classmethod
    def build(cls, root: str | Path, skip_dirs: Iterable[str] = SKIP_DIRS,
              root_skip_dirs: Iterable[str] = ROOT_SKIP_DIRS) -> "ImportGraph":
        g = cls(str(root))
        skip = set(skip_dirs)
        root_skip = set(root_skip_dirs)
        rootp = Path(root)
        for dirpath, dirnames, filenames in os.walk(rootp):
            at_root = Path(dirpath) == rootp
            dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith(".")
                                 and not (at_root and d in root_skip))
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(rootp).as_posix()
                g._add_file(rel, full)
        return g

    def _add_file(self, rel: str, full: Path) -> None:
        name = module_name_for(rel)
        is_pkg = rel.endswith("__init__.py")
        raw = full.read_bytes()
        # Parsing + walking the AST is the cost; the result depends only on (rel, module name, content), so it is cached by
        # CONTENT hash (no staleness possible; identical files in main and candidate trees share one entry).
        ckey = (rel, name or "", is_pkg, hashlib.sha256(raw).digest())
        hit = _IMPORTS_CACHE.get(ckey)
        if hit is None:
            dkey = DC.key_of(ckey[0], ckey[1], str(is_pkg), ckey[3])            # persisted across processes (creator/diskcache.py)
            hit = DC.get("imports", _imports_salt(), dkey)
            if hit is None:
                try:
                    hit = ("ok", frozenset(_imports_of(ast.parse(raw, filename=rel), name or "", is_pkg)))
                except (SyntaxError, ValueError) as e:
                    hit = ("err", f"{type(e).__name__}: {e}")
                DC.put("imports", _imports_salt(), dkey, hit)
            _IMPORTS_CACHE[ckey] = hit
        if hit[0] == "err":
            self.unparsable[rel] = str(hit[1])
            if name:
                self.modules[name] = rel
                self.imports.setdefault(name, set())
            return
        names = set(hit[1])
        if name:
            self.modules[name] = rel
            self.imports[name] = names
        elif is_test_file(rel) or rel.endswith("conftest.py"):
            self.nonmodule_tests[rel] = names

    def resolve(self, dotted: str) -> set[str]:
        """Tree modules an import of `dotted` executes: the module itself and every parent package's __init__."""
        out: set[str] = set()
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            cand = ".".join(parts[:i])
            if cand in self.modules:
                out.add(cand)
        return out

    def importers_index(self) -> dict[str, set[str]]:
        """Reverse index: dotted name -> module keys (module names or non-module test paths) importing it or a child of it."""
        rev: dict[str, set[str]] = defaultdict(set)
        sources: list[tuple[str, set[str]]] = list(self.imports.items()) + list(self.nonmodule_tests.items())
        for src, names in sources:
            for n in names:
                parts = n.split(".")
                for i in range(1, len(parts) + 1):
                    rev[".".join(parts[:i])].add(src)
        return rev

    def dependents(self, changed_modules: Iterable[str]) -> set[str]:
        """Transitive importers of the changed modules (module names and non-module test paths). Importing pkg.sub executes
        pkg/__init__ too, so a change to a package __init__ reaches everyone importing anything under it."""
        rev = self.importers_index()
        seen: set[str] = set()
        frontier = [m for m in changed_modules if m]
        while frontier:
            cur = frontier.pop()
            for h in rev.get(cur, set()):         # rev is keyed by every prefix, so importers of cur.child are included
                if h not in seen:
                    seen.add(h)
                    if h in self.modules:
                        frontier.append(h)
        return seen

    def path_of(self, key: str) -> str:
        return self.modules.get(key, key)


def _imports_of(tree: ast.AST, modname: str, is_pkg: bool) -> set[str]:
    out: set[str] = set()
    pkg_parts = modname.split(".") if modname else []
    if not is_pkg and pkg_parts:
        pkg_parts = pkg_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(pkg_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base = ".".join(pkg_parts[:keep] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if not base:
                continue
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
        elif isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                out.add(node.args[0].value)
    return {n for n in out if n}


# ------------------------------------------------------------------------------------------------------------ selection
@dataclass(frozen=True)
class TestSelection:
    tests: tuple[str, ...]                       # relative test file paths (sorted)
    reasons: Mapping[str, str]                   # test path -> why it was selected
    select_all: bool = False
    why_all: str = ""
    unresolved: tuple[str, ...] = ()             # changed paths whose effect could not be traced

    @property
    def empty(self) -> bool:
        return not self.tests

    def digest(self) -> str:
        return hashlib.sha256("\n".join(self.tests).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"tests": list(self.tests), "reasons": dict(self.reasons), "select_all": self.select_all,
                "why_all": self.why_all, "unresolved": list(self.unresolved), "digest": self.digest()}


def all_tests(graph: ImportGraph, under: str = "") -> list[str]:
    prefix = under.rstrip("/") + "/" if under else ""
    paths = set(graph.modules.values()) | set(graph.nonmodule_tests) | set(graph.unparsable)
    return sorted(p for p in paths if is_test_file(p) and p.startswith(prefix))


def select_tests(graph: ImportGraph, changed_paths: Iterable[str], smoke: Iterable[str] = ()) -> TestSelection:
    """Affected tests of a change. Conservative where the graph cannot see: configuration or conftest changes select every test
    under their directory; non-Python, non-doc files (data a module may read) select everything; docs select only smoke."""
    reasons: dict[str, str] = {}
    unresolved: list[str] = []
    select_all, why_all = False, ""
    changed_mods: list[str] = []
    for raw in sorted(set(changed_paths)):
        rel = raw.replace("\\", "/")
        name = rel.rsplit("/", 1)[-1]
        if name == "conftest.py":
            d = rel.rsplit("/", 1)[0] if "/" in rel else ""
            for t in all_tests(graph, d):
                reasons.setdefault(t, f"conftest changed: {rel}")
            continue
        if name in CONFIG_FILES:
            select_all, why_all = True, f"configuration changed: {rel}"
            continue
        if not rel.endswith(".py"):
            if Path(rel).suffix.lower() in DOC_SUFFIXES:
                continue
            select_all, why_all = True, f"non-Python file changed (readers unknown): {rel}"
            continue
        if is_test_file(rel):
            if rel in graph.modules.values() or rel in graph.nonmodule_tests or rel in graph.unparsable:
                reasons.setdefault(rel, "test file changed")
            continue                                 # a deleted test is caught by the base comparison (REMOVED)
        mod = module_name_for(rel)
        if mod is None:
            unresolved.append(rel)
            select_all, why_all = True, f"changed Python file is not importable by name: {rel}"
            continue
        changed_mods.append(mod)
    for dep in sorted(graph.dependents(changed_mods)):
        path = graph.path_of(dep)
        if is_test_file(path):
            reasons.setdefault(path, "imports a changed module (transitively)")
        elif path.endswith("conftest.py"):
            d = path.rsplit("/", 1)[0] if "/" in path else ""
            for t in all_tests(graph, d):
                reasons.setdefault(t, f"conftest {path} imports a changed module")
    if select_all:
        for t in all_tests(graph):
            reasons.setdefault(t, why_all)
    for s in smoke:
        s = s.replace("\\", "/")
        if s in reasons:
            continue
        if (Path(graph.root) / s).is_file():
            reasons[s] = "smoke set (always run)"
    return TestSelection(tuple(sorted(reasons)), reasons, select_all, why_all, tuple(unresolved))


# -------------------------------------------------------------------------------------------------------------- results
class Outcome(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"

    @property
    def bad(self) -> bool:
        return self in (Outcome.FAILED, Outcome.ERROR)


class RunStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NO_TESTS = "NO_TESTS"
    TIMEOUT = "TIMEOUT"
    CRASHED = "CRASHED"          # pytest internal/usage error, or results missing/unparsable/inconsistent


@dataclass(frozen=True)
class CaseResult:
    case_id: str                 # pytest node id when the file is known, else classname::name
    outcome: Outcome
    seconds: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class TestRun:
    """One pytest invocation on one tree. `trustworthy` is False whenever the per-case results cannot be relied on."""
    label: str                   # candidate | baseline | rerun-candidate | rerun-baseline
    tree: str
    selection: tuple[str, ...]
    status: RunStatus
    cases: Mapping[str, CaseResult]
    exit_code: int | None
    seconds: float
    proc: ProcResult | None = None
    junit_path: str = ""
    problems: tuple[str, ...] = ()

    @property
    def trustworthy(self) -> bool:
        return self.status in (RunStatus.PASSED, RunStatus.FAILED)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {o.value: 0 for o in Outcome}
        for r in self.cases.values():
            c[r.outcome.value] += 1
        return c

    def to_record(self) -> dict[str, Any]:
        return {"label": self.label, "tree": self.tree, "status": self.status.value, "exit_code": self.exit_code,
                "seconds": round(self.seconds, 3), "selection": list(self.selection), "counts": self.counts(),
                "problems": list(self.problems), "junit": self.junit_path,
                "cases": {k: {"outcome": v.outcome.value, "seconds": round(v.seconds, 4), "message": v.message[:500]}
                          for k, v in sorted(self.cases.items())},
                "proc": self.proc.evidence() if self.proc else None}


def _case_id(classname: str, name: str, file_of: Mapping[str, str]) -> str:
    """junit classname 'tests.test_x.TestC' + name -> 'tests/test_x.py::TestC::name' when the module is one we selected."""
    parts = classname.split(".") if classname else []
    for i in range(len(parts), 0, -1):
        mod = ".".join(parts[:i])
        if mod in file_of:
            rest = parts[i:]
            return "::".join([file_of[mod], *rest, name])
    return f"{classname}::{name}" if classname else name


def parse_junit(path: str | Path, file_of: Mapping[str, str] | None = None) -> dict[str, CaseResult]:
    """Per-case outcomes from a pytest junit xml file. Raises ValueError if the file is missing or malformed: the caller
    must treat that as CRASHED, never as 'no failures'."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"junit file missing: {p}")
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as e:
        raise ValueError(f"junit file unparsable: {e}") from e
    fmap = dict(file_of or {})
    out: dict[str, CaseResult] = {}
    for tc in root.iter("testcase"):
        cid = _case_id(tc.get("classname", ""), tc.get("name", ""), fmap)
        try:
            secs = float(tc.get("time", "0") or 0)
        except ValueError:
            secs = 0.0
        outcome, msg = Outcome.PASSED, ""
        for child in tc:
            tag = child.tag
            if tag == "failure":
                outcome, msg = Outcome.FAILED, child.get("message", "") or (child.text or "")[:500]
                break
            if tag == "error":
                outcome, msg = Outcome.ERROR, child.get("message", "") or (child.text or "")[:500]
                break
            if tag == "skipped":
                outcome, msg = Outcome.SKIPPED, child.get("message", "")
        if cid in out and out[cid].outcome.bad and not outcome.bad:
            continue                      # a test that errors in teardown appears twice; the bad entry wins
        out[cid] = CaseResult(cid, outcome, secs, msg)
    return out


@dataclass
class PytestConfig:
    python: str = sys.executable
    timeout: float = 600.0
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def run_pytest(root: str | Path, targets: Sequence[str], junit_path: str | Path, *, label: str, tree: str = "",
               config: PytestConfig | None = None, file_of: Mapping[str, str] | None = None) -> TestRun:
    """Run pytest on explicit targets (test files or node ids) inside `root`. Never runs with an empty target list (pytest
    would collect the whole tree): an empty selection is reported as NO_TESTS without launching anything."""
    cfg = config or PytestConfig()
    sel = tuple(targets)
    if not sel:
        return TestRun(label, tree, sel, RunStatus.NO_TESTS, {}, None, 0.0, problems=("empty selection",))
    jp = Path(junit_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    if jp.exists():
        jp.unlink()                        # a stale file from an earlier run must never be read as this run's result
    argv = [cfg.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(root), f"--junitxml={jp}",
            "-o", "junit_family=xunit2", *cfg.extra_args, "--", *sel]
    proc = run_cmd(argv, root, timeout=cfg.timeout, env=clean_env(root, cfg.env))
    fmap = dict(file_of or {})
    for t in sel:
        f = t.split("::", 1)[0]
        m = module_name_for(f)
        if m:
            fmap.setdefault(m, f)
    problems: list[str] = []
    if proc.timed_out:
        return TestRun(label, tree, sel, RunStatus.TIMEOUT, {}, None, proc.seconds, proc, str(jp),
                       (f"pytest exceeded {cfg.timeout}s",))
    if proc.launch_error:
        return TestRun(label, tree, sel, RunStatus.CRASHED, {}, None, proc.seconds, proc, str(jp), (proc.launch_error,))
    if proc.returncode == 5:
        return TestRun(label, tree, sel, RunStatus.NO_TESTS, {}, 5, proc.seconds, proc, str(jp), ("no tests collected",))
    try:
        cases = parse_junit(jp, fmap)
    except ValueError as e:
        return TestRun(label, tree, sel, RunStatus.CRASHED, {}, proc.returncode, proc.seconds, proc, str(jp), (str(e),))
    any_bad = any(c.outcome.bad for c in cases.values())
    if proc.returncode not in (0, 1):
        status = RunStatus.CRASHED
        problems.append(f"pytest exit code {proc.returncode} (interrupted, internal or usage error)")
    elif proc.returncode == 0 and any_bad:
        status = RunStatus.CRASHED
        problems.append("exit code 0 but failing cases in junit: results inconsistent")
    elif proc.returncode == 1 and not any_bad:
        status = RunStatus.CRASHED
        problems.append("exit code 1 but no failing case in junit: results inconsistent")
    elif not cases:
        status = RunStatus.NO_TESTS
    else:
        status = RunStatus.FAILED if any_bad else RunStatus.PASSED
    return TestRun(label, tree, sel, status, cases, proc.returncode, proc.seconds, proc, str(jp), tuple(problems))


# ---------------------------------------------------------------------------------------------------------- comparison
class CaseClass(str, Enum):
    PASS = "PASS"
    FIXED = "FIXED"
    REGRESSION = "REGRESSION"            # passing at base, failing now
    PREEXISTING = "PREEXISTING"          # failing at base and now: not caused by the change
    NEW_FAILING = "NEW_FAILING"          # did not exist at base, failing now
    NEW_PASSING = "NEW_PASSING"
    REMOVED = "REMOVED"                  # existed at base, gone now (deleting a failing test must not hide it)
    SKIPPED = "SKIPPED"
    FLAKY = "FLAKY"                      # outcome flipped across re-runs at base or candidate


class Verdict(str, Enum):
    CLEAN = "CLEAN"
    REGRESSED = "REGRESSED"
    FLAKY = "FLAKY"
    INCONCLUSIVE = "INCONCLUSIVE"


BLOCKING = frozenset({CaseClass.REGRESSION, CaseClass.NEW_FAILING, CaseClass.REMOVED})


@dataclass(frozen=True)
class RegressionReport:
    verdict: Verdict
    classes: Mapping[str, CaseClass]
    reasons: tuple[str, ...]
    flaky_history: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    timing: Mapping[str, float] = field(default_factory=dict)

    def ids(self, cls: CaseClass) -> list[str]:
        return sorted(k for k, v in self.classes.items() if v is cls)

    @property
    def blocking(self) -> list[str]:
        return sorted(k for k, v in self.classes.items() if v in BLOCKING)

    def summary(self) -> dict[str, int]:
        s: dict[str, int] = {c.value: 0 for c in CaseClass}
        for v in self.classes.values():
            s[v.value] += 1
        return s

    def to_record(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "summary": self.summary(), "reasons": list(self.reasons),
                "blocking": self.blocking, "classes": {k: v.value for k, v in sorted(self.classes.items())},
                "flaky_history": {k: {lab: list(seq) for lab, seq in v.items()} for k, v in self.flaky_history.items()},
                "timing": dict(self.timing)}


def classify_case(base: CaseResult | None, cand: CaseResult | None) -> CaseClass:
    if cand is None:
        return CaseClass.REMOVED if base is not None and base.outcome is not Outcome.SKIPPED else CaseClass.SKIPPED
    if cand.outcome is Outcome.SKIPPED:
        if base is not None and base.outcome is Outcome.PASSED:
            return CaseClass.REGRESSION      # a test that passed and is now skipped no longer protects anything
        return CaseClass.SKIPPED
    if base is None:
        return CaseClass.NEW_FAILING if cand.outcome.bad else CaseClass.NEW_PASSING
    if cand.outcome.bad:
        return CaseClass.PREEXISTING if base.outcome.bad else (
            CaseClass.REGRESSION if base.outcome is Outcome.PASSED else CaseClass.NEW_FAILING)
    return CaseClass.FIXED if base.outcome.bad else CaseClass.PASS


def compare_runs(baseline: TestRun, candidate: TestRun, *, base_files: Iterable[str] | None = None) -> RegressionReport:
    """Candidate vs base on the same selection. Only test files that exist at base are expected at base: a case present only
    in the candidate is NEW_*, a base case missing from the candidate is REMOVED. Untrustworthy runs -> INCONCLUSIVE."""
    reasons: list[str] = []
    if not candidate.trustworthy:
        reasons.append(f"candidate run {candidate.status.value}: {'; '.join(candidate.problems)}")
    base_ok = baseline.trustworthy or (baseline.status is RunStatus.NO_TESTS and not baseline.selection)
    if not base_ok:
        reasons.append(f"baseline run {baseline.status.value}: {'; '.join(baseline.problems)}")
    classes: dict[str, CaseClass] = {}
    if candidate.trustworthy and base_ok:
        cand_files = {t.split("::", 1)[0] for t in candidate.selection}
        for cid in sorted(set(baseline.cases) | set(candidate.cases)):
            b, c = baseline.cases.get(cid), candidate.cases.get(cid)
            if c is None and cid.split("::", 1)[0] not in cand_files and "::" in cid:
                continue                   # the file was not in this selection at all; absence says nothing
            classes[cid] = classify_case(b, c)
    if not classes and not reasons:
        reasons.append("no test cases were compared")
    if reasons:
        verdict = Verdict.INCONCLUSIVE
    elif any(v in BLOCKING for v in classes.values()):
        verdict = Verdict.REGRESSED
    else:
        verdict = Verdict.CLEAN
    return RegressionReport(verdict, classes, tuple(reasons),
                            timing={"baseline_s": round(baseline.seconds, 3), "candidate_s": round(candidate.seconds, 3)})


def apply_flaky_reruns(report: RegressionReport, rerun: Callable[[str, list[str]], TestRun], *, reruns: int,
                       baseline: TestRun, candidate: TestRun) -> RegressionReport:
    """Re-run every blocking-or-fixed case `reruns` times on each side. A case whose outcome differs across runs on either
    side is FLAKY: not counted as a regression, but the verdict can then be at best FLAKY, never CLEAN (a flaky suite
    proves nothing). A re-run that cannot be trusted keeps the original class and makes the verdict INCONCLUSIVE."""
    suspects = sorted(k for k, v in report.classes.items()
                      if v in (CaseClass.REGRESSION, CaseClass.NEW_FAILING, CaseClass.FIXED) and "::" in k)
    if reruns <= 0 or not suspects or report.verdict is Verdict.INCONCLUSIVE:
        return report
    hist: dict[str, dict[str, list[str]]] = {k: {"baseline": [], "candidate": []} for k in suspects}
    for k in suspects:
        b, c = baseline.cases.get(k), candidate.cases.get(k)
        if b is not None:
            hist[k]["baseline"].append(b.outcome.value)
        if c is not None:
            hist[k]["candidate"].append(c.outcome.value)
    problems: list[str] = []
    for i in range(reruns):
        for side in ("candidate", "baseline"):
            ids = [k for k in suspects if side == "candidate" or k in baseline.cases]
            if not ids:
                continue
            run = rerun(side, ids)
            if not run.trustworthy:
                problems.append(f"re-run {i + 1} on {side} {run.status.value}")
                continue
            for k in ids:
                r = run.cases.get(k)
                hist[k][side].append(r.outcome.value if r else "MISSING")
    classes = dict(report.classes)
    for k in suspects:
        if any(len({o for o in seq if o != Outcome.SKIPPED.value}) > 1 for seq in hist[k].values()):
            classes[k] = CaseClass.FLAKY
    reasons = list(report.reasons) + problems
    if problems:
        verdict = Verdict.INCONCLUSIVE
    elif any(v in BLOCKING for v in classes.values()):
        verdict = Verdict.REGRESSED
    elif any(v is CaseClass.FLAKY for v in classes.values()):
        verdict = Verdict.FLAKY
    else:
        verdict = Verdict.CLEAN
    return RegressionReport(verdict, classes, tuple(reasons),
                            {k: {s: tuple(seq) for s, seq in v.items()} for k, v in hist.items()}, report.timing)


def comparison_record(baseline: TestRun, candidate: TestRun, report: RegressionReport) -> dict[str, Any]:
    """C77 sec 29 BASELINE vs MODIFIED summary: target outcome, regressions, stability (flaky), resource use (time)."""
    bc, cc = baseline.counts(), candidate.counts()
    return {"baseline": {"tree": baseline.tree, "status": baseline.status.value, "counts": bc,
                         "seconds": round(baseline.seconds, 3)},
            "candidate": {"tree": candidate.tree, "status": candidate.status.value, "counts": cc,
                          "seconds": round(candidate.seconds, 3)},
            "verdict": report.verdict.value, "summary": report.summary(),
            "time_delta_s": round(candidate.seconds - baseline.seconds, 3)}

