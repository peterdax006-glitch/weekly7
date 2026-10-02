"""Pre-screen for student candidates: cheap, deterministic checks BEFORE a change reaches the kernel's full cycle (2 Oct 2026).

Measured cause: a kernel cycle costs tens of CPU-minutes and every student shrink was a lazy_import whose size delta was >= 0, so
the cycle could only be cancelled or rejected. Here the student checks its own candidate first, in this order (cheapest first):
  1. METRIC (in-process): the AST-size delta of the target (shrink) or the kernel-start activation delta (activation), computed with
     creator.efficiency's own measures. A candidate that cannot improve the targeted metric is rejected.
  2. IMPORT (subprocess): the changed module still imports.
  3. DIRECT TESTS (creator.testrun): only the test files that import the changed module; failures that already fail on the base
     do not count against the candidate.
The pre-screen can only REJECT early. It never adopts, never writes a ledger and never touches the kernel path: the kernel's own
measurement stays the only adoption authority. `check_in_tree` restores the file it temporarily rewrites."""
from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from creator import efficiency as E
from creator import testrun as TR
from creator.build import module_name_for

MAX_TEST_FILES = 6
TEST_TIMEOUT_S = 120.0
IMPORT_TIMEOUT_S = 60.0
ENTRY = "creator/kernel.py"


@dataclasses.dataclass
class Verdict:
    ok: bool
    reason: str = ""
    stage: str = ""                      # metric | import | tests | ok
    size_delta: Optional[int] = None     # candidate - base, AST nodes of the target (negative = smaller)
    act_delta: Optional[int] = None      # candidate - base, AST nodes loaded when the kernel starts (None: not a Creator tree)
    static_delta: Optional[int] = None
    tests: tuple[str, ...] = ()
    tests_pass: Optional[bool] = None    # None = not run
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def metric_for(plan: Any) -> str:
    """'activation' for the activation requirement, else 'size' (creator.curriculum.task_kind: shrink / activation)."""
    key = str(getattr(plan, "requirement_key", ""))
    return "activation" if key.endswith("activation") else "size"


# ------------------------------------------------------------------------------------------------ metric (in-process)

def _eager_deps(src: str, mod_of: dict[str, str]) -> Optional[set[str]]:
    """The Creator module files `src` imports at MODULE level (the logic of efficiency.eager_graph, on text)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    deps: set[str] = set()
    for n in tree.body:
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
            ([f"{n.module}.{a.name}" for a in n.names] + [n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
        deps.update(mod_of[x] for x in names if x in mod_of)
    return deps


class Footing:
    """Sizes and the eager-import graph of one tree, read once; candidate deltas are then pure arithmetic."""

    def __init__(self, root: Path, texts: Optional[dict[str, str]] = None) -> None:
        self.root = Path(root)
        files = E.production_files(self.root) if (self.root / "creator").is_dir() else []
        self.texts = dict(texts) if texts is not None else {
            f: (self.root / f).read_text(encoding="utf-8", errors="replace") for f in files}
        self.mod_of = {f[:-3].replace("/", "."): f for f in self.texts}
        self.mod_of.update({f[:-12].replace("/", "."): f for f in self.texts if f.endswith("/__init__.py")})
        self.sizes = {f: max(E.ast_size(t), 0) for f, t in self.texts.items()}
        self.eager: dict[str, set[str]] = {f: _eager_deps(t, self.mod_of) or set() for f, t in self.texts.items()}
        self.is_creator = ENTRY in self.texts

    def _loaded(self, eager: dict[str, set[str]], sz: dict[str, int]) -> tuple[int, int]:
        act = sum(sz.get(m, 0) for m in E._closure(eager, ENTRY)) if self.is_creator else 0
        static = sum(sum(sz.get(d, 0) for d in E._closure(eager, f) - {f}) for f in eager)
        return act, static

    def load(self, rel: str) -> None:
        """Read a target outside the Creator package (or in a plain tree) on demand."""
        if rel not in self.texts and (self.root / rel).is_file():
            self.texts[rel] = (self.root / rel).read_text(encoding="utf-8", errors="replace")
            self.sizes[rel] = max(E.ast_size(self.texts[rel]), 0)

    def deltas(self, rel: str, new_src: str) -> tuple[int, Optional[int], Optional[int]]:
        """(size delta of `rel`, kernel-start activation delta, static eager-load delta) if `rel` became `new_src`."""
        new_size = E.ast_size(new_src)
        self.load(rel)
        old_size = self.sizes.get(rel, 0)
        if new_size < 0:
            return 10 ** 9, None, None                      # does not parse: never an improvement
        if not self.is_creator or rel not in self.eager:
            return new_size - old_size, None, None
        deps = _eager_deps(new_src, self.mod_of)
        eager = dict(self.eager)
        eager[rel] = deps or set()
        sz = dict(self.sizes)
        sz[rel] = new_size
        a0, s0 = self.base()
        a1, s1 = self._loaded(eager, sz)
        return new_size - old_size, a1 - a0, s1 - s0

    def base(self) -> tuple[int, int]:
        if not hasattr(self, "_base"):
            self._base = self._loaded(self.eager, self.sizes)
        return self._base                                    # type: ignore[no-any-return]


def improves(size_delta: int, act_delta: Optional[int], metric: str) -> bool:
    """Can this candidate improve the targeted metric? shrink: the target got smaller; activation: less code loads at start."""
    if metric == "activation":
        return act_delta is not None and act_delta < 0
    return size_delta < 0


def predicted_gain(size_delta: int, act_delta: Optional[int], metric: str) -> int:
    return -(act_delta or 0) if metric == "activation" else -size_delta


def metric_verdict(foot: Footing, rel: str, new_src: str, metric: str) -> Verdict:
    sd, ad, st = foot.deltas(rel, new_src)
    if improves(sd, ad, metric):
        return Verdict(True, "", "metric", sd, ad, st)
    what = (f"activation delta {ad}" if metric == "activation" else f"size delta {sd}")
    return Verdict(False, f"cannot improve {metric}: {what} (>= 0)", "metric", sd, ad, st)


# ------------------------------------------------------------------------------------------------ runtime (subprocess)

def direct_tests(graph: TR.ImportGraph, rel: str) -> list[str]:
    """Test files that DIRECTLY import the module (no transitive closure), sorted."""
    mod = module_name_for(rel)
    if not mod:
        return []
    rev = graph.importers_index()
    out = {graph.path_of(k) for k in rev.get(mod, set())}
    return sorted(p for p in out if TR.is_test_file(p))


def import_ok(root: Path, rel: str, python: str = sys.executable, timeout: float = IMPORT_TIMEOUT_S) -> tuple[bool, str]:
    mod = module_name_for(rel)
    if not mod:
        return True, ""
    code = f"import sys; sys.path.insert(0, {str(root)!r}); import importlib; importlib.import_module({mod!r})"
    try:
        p = subprocess.run([python, "-c", code], cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"import of {mod} timed out"
    return p.returncode == 0, p.stderr.strip()[-300:]


def _bad(run: TR.TestRun) -> set[str]:
    return {k for k, c in run.cases.items() if c.outcome.bad}


def run_direct_tests(root: Path, rel: str, graph: Optional[TR.ImportGraph] = None, python: str = sys.executable,
                     timeout: float = TEST_TIMEOUT_S, original: Optional[str] = None) -> tuple[Optional[bool], tuple[str, ...], str]:
    """(passed?, test files, reason). None = nothing to run. A failure is compared with the base (`original` source, written back
    temporarily by the caller via `base_runner`) - only cases that fail now but passed before count."""
    g = graph or TR.ImportGraph.build(root)
    tests = direct_tests(g, rel)[:MAX_TEST_FILES]
    if not tests:
        return None, (), "no test imports this module"
    cfg = TR.PytestConfig(python=python, timeout=timeout)
    with tempfile.TemporaryDirectory() as td:
        run = TR.run_pytest(root, tests, Path(td) / "c.xml", label="candidate", tree=str(root), config=cfg)
        if run.status in (TR.RunStatus.PASSED, TR.RunStatus.NO_TESTS):
            return True, tuple(tests), ""
        failing = _bad(run)
        if run.trustworthy and original is not None and failing:
            p = Path(root) / rel
            new_bytes = p.read_bytes()
            try:
                p.write_bytes(original.encode("utf-8") if isinstance(original, str) else original)
                base = TR.run_pytest(root, tests, Path(td) / "b.xml", label="baseline", tree=str(root), config=cfg)
            finally:
                p.write_bytes(new_bytes)
            if base.trustworthy and failing <= _bad(base):
                return True, tuple(tests), "failures already present on the base"
        why = f"{run.status.value}: {sorted(failing)[:3] or run.problems}"
        if not run.trustworthy:                              # timeout / crash: no verdict either way (never a rejection)
            return None, tuple(tests), f"inconclusive: {why}"[:300]
        return False, tuple(tests), why[:300]


def check_in_tree(root: Path, rel: str, new_src: str, graph: Optional[TR.ImportGraph] = None, run_tests: bool = True,
                  python: str = sys.executable, test_timeout: float = TEST_TIMEOUT_S) -> Verdict:
    """Import + direct tests of `new_src` placed at root/rel. The file is restored byte-for-byte afterwards."""
    t0 = time.monotonic()
    p = Path(root) / rel
    old_bytes = p.read_bytes()
    old_src = old_bytes.decode("utf-8", errors="replace")
    eol = "\r\n" if b"\r\n" in old_bytes else "\n"
    try:
        p.write_bytes(new_src.replace("\r\n", "\n").replace("\n", eol).encode("utf-8"))
        ok, err = import_ok(Path(root), rel, python)
        if not ok:
            return Verdict(False, f"module no longer imports: {err}", "import", seconds=time.monotonic() - t0)
        if not run_tests:
            return Verdict(True, "", "ok", seconds=time.monotonic() - t0)
        passed, tests, why = run_direct_tests(Path(root), rel, graph, python, test_timeout, original=old_src)
        if passed is False:
            return Verdict(False, f"direct test failed: {why}", "tests", tests=tests, tests_pass=False, seconds=time.monotonic() - t0)
        return Verdict(True, why, "ok", tests=tests, tests_pass=passed, seconds=time.monotonic() - t0)
    finally:
        p.write_bytes(old_bytes)


def prescreen(root: Path, rel: str, new_src: str, metric: str = "size", foot: Optional[Footing] = None,
              graph: Optional[TR.ImportGraph] = None, run_tests: bool = True, python: str = sys.executable) -> Verdict:
    """Full pre-screen of one candidate: metric first (free), then import and direct tests. Never adopts anything."""
    t0 = time.monotonic()
    foot = foot or Footing(Path(root))
    v = metric_verdict(foot, rel, new_src, metric)
    if not v.ok:
        v.seconds = time.monotonic() - t0
        return v
    r = check_in_tree(Path(root), rel, new_src, graph, run_tests, python)
    r.size_delta, r.act_delta, r.static_delta = v.size_delta, v.act_delta, v.static_delta
    r.seconds = time.monotonic() - t0
    return r


def rank(foot: Footing, rel: str, applied: list[tuple[Any, str]], metric: str) -> list[tuple[int, Any, str, Verdict]]:
    """[(predicted gain, item, new source, metric verdict)] best gain first (stable). `applied` = (item, new source) pairs."""
    out = []
    for item, new in applied:
        v = metric_verdict(foot, rel, new, metric)
        out.append((predicted_gain(v.size_delta if v.size_delta is not None else 0, v.act_delta, metric), item, new, v))
    out.sort(key=lambda x: -x[0])
    return out
