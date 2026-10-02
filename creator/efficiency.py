"""Creator K17 - self-efficiency: shrink its own code and memory (owner, 1 Oct 2026: "constantly develop itself to shrink its own
code down so that it can run more code at a time and run on less memory"; ruling: capability, not lines) - IMPLEMENTED, NOT VALIDATED.

SIZE is counted in AST nodes of production code (docstrings excluded), so reformatting, joining statements on one line or deleting
comments gains nothing. MEMORY is the peak working set of a fixed workload (import every Creator module and build the self-model
of the tree) run in a fresh interpreter, replicated so noise has a spread.

An efficiency package targets the largest module the Creator may edit (protected measuring sticks excluded) or, in memory mode,
the workload's peak. It is adopted only when (creator.kernel decides, creator.model computes):
    primary  (DEV)      the targeted module's size (or the workload's peak memory) went down, reproduced in two measurements
    guards   (DEV)      affected tests pass, every base-met requirement kept, test-case count not lower, peak memory not
                        significantly higher (size mode) / package size not higher (memory mode)
    holdout  (HOLDOUT)  the WHOLE creator package's size went down - moving code to another file is not shrinking
plus the kernel's existing audits: no weakened tests, no planted answers, no protected path."""
from __future__ import annotations

import ast
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from creator import sandbox as S

SCOPE = "creator"
EXCLUDE = ("creator/devbench/",)                         # benchmark material, not the Creator's code
WORKLOAD = ("import importlib, json, pkgutil, sys, psutil\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import creator\n"
            "for m in pkgutil.walk_packages(creator.__path__, 'creator.'):\n"
            "    if '.devbench' not in m.name:\n"
            "        importlib.import_module(m.name)\n"
            "from creator import selfmodel as SM\n"
            "SM.build(sys.argv[1], scope=('creator', 'tests'), include_versions=False)\n"
            "mi = psutil.Process().memory_info()\n"
            "print(json.dumps({'peak_mb': getattr(mi, 'peak_wset', mi.rss) / 1e6}))\n")


def ast_size(source: str) -> int:
    """Number of AST nodes, docstrings excluded (a module that does not parse counts as infinitely large: -1 signals it)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc_ids.update(id(n) for n in ast.walk(first))
    return sum(1 for n in ast.walk(tree) if id(n) not in doc_ids)


def production_files(root: Path, scope: str = SCOPE) -> list[str]:
    out = []
    for p in sorted((root / scope).rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if any(rel.startswith(x) for x in EXCLUDE) or "__pycache__" in rel:
            continue
        out.append(rel)
    return out


def sizes(root: Path, files: Optional[Iterable[str]] = None) -> dict[str, int]:
    return {f: ast_size((root / f).read_text(encoding="utf-8", errors="replace"))
            for f in (files if files is not None else production_files(root)) if (root / f).is_file()}


def package_size(root: Path) -> int:
    s = sizes(root)
    if any(v < 0 for v in s.values()):
        raise ValueError(f"unparsable module(s): {[k for k, v in s.items() if v < 0]}")
    return sum(s.values())


def editable(rel: str) -> bool:
    return not S.is_protected(rel) and not rel.endswith("__init__.py")


def pick_target(root: Path, avoid: Sequence[str] = ()) -> Optional[tuple[str, int]]:
    """The largest module the Creator may edit, skipping ones that recently failed to shrink (`avoid`)."""
    cands = [(f, n) for f, n in sizes(root).items() if editable(f) and f not in avoid and n > 0]
    return max(cands, key=lambda x: (x[1], x[0])) if cands else None


def peak_memory_mb(root: Path, python: str = sys.executable, replicates: int = 2, timeout: float = 600.0) -> list[float]:
    """Peak working set (MB) of the fixed workload, once per replicate in a fresh interpreter."""
    out = []
    for _ in range(replicates):
        p = subprocess.run([python, "-c", WORKLOAD, str(root)], cwd=root, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"memory workload failed: {p.stderr.strip()[-500:]}")
        out.append(float(json.loads(p.stdout.strip().splitlines()[-1])["peak_mb"]))
    return out


ACTIVATION_WORKLOAD = ("import json, sys\n"
                       "sys.path.insert(0, sys.argv[1])\n"
                       "import creator.kernel\n"
                       "print(json.dumps(sorted(m.__file__ for n, m in list(sys.modules.items())\n"
                       "                        if n.startswith('creator') and getattr(m, '__file__', None))))\n")


def activation(root: Path, python: str = sys.executable, timeout: float = 300.0) -> dict[str, Any]:
    """Owner, 1 Oct 2026: 'it can have a billion lines of code but it isnt using its entire capablity 24/7 ... never run any more
    code than absolutely necessarry'. The ACTIVATION FOOTPRINT is the AST size of the Creator modules that starting the kernel
    loads, as a share of the whole package: total capability may grow, what is loaded to run must not."""
    p = subprocess.run([python, "-c", ACTIVATION_WORKLOAD, str(root)], cwd=root, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"activation workload failed: {p.stderr.strip()[-500:]}")
    loaded = []
    for f in json.loads(p.stdout.strip().splitlines()[-1]):
        try:
            loaded.append(Path(f).resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            continue
    sz = sizes(root)
    active = sum(sz.get(f, 0) for f in loaded)
    total = sum(sz.values())
    return {"loaded_modules": sorted(f for f in loaded if f in sz), "active_nodes": active, "package_nodes": total,
            "fraction": round(active / total, 4) if total else 0.0}


def test_count(root: Path, scope: str = "tests") -> int:
    """Test functions in the tree (a shrink may never make the suite smaller)."""
    n = 0
    for p in sorted((root / scope).rglob("test_*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        n += sum(1 for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test"))
    return n


def _test_text(root: Path, scope: str = "tests") -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted((root / scope).rglob("test_*.py")))


def public_names(src: str) -> list[str]:
    """Top-level public functions and classes of a module's source ([] when it does not parse)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not n.name.startswith("_")]


def uncovered_public(root: Path, rel: str, tests: Optional[str] = None) -> list[str]:
    """Public functions/classes of `rel` that no test file names (a whole-word match): the measurable test gap. Coverage work adds
    tests until this list shrinks; it never touches production code."""
    tests = _test_text(root) if tests is None else tests
    src = (root / rel).read_text(encoding="utf-8", errors="replace") if (root / rel).is_file() else ""
    return [n for n in public_names(src) if not re.search(rf"\b{re.escape(n)}\b", tests)]


def uncovered_total(root: Path) -> int:
    """The test gap over every module the Creator may edit (the coverage holdout: tests for one module do not hide another's gap)."""
    tests = _test_text(root)
    return sum(len(uncovered_public(root, f, tests)) for f in production_files(root) if editable(f))


@dataclasses.dataclass(frozen=True)
class Footprint:
    target: str
    target_size: int
    package_size: int
    memory_mb: tuple[float, ...]
    test_cases: int
    active_nodes: int = -1                      # AST nodes the kernel loads at start (-1 = not measured)
    static_load: int = 0                        # sum over modules of the Creator code each one imports eagerly
    uncovered: int = 0                          # public names of the target no test names / the same over the whole package
    uncovered_package: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def eager_graph(root: Path) -> dict[str, set[str]]:
    """module file -> the Creator module files it imports at MODULE level (what loading it loads)."""
    files = production_files(root)
    mod_of = {f[:-3].replace("/", "."): f for f in files}
    mod_of.update({f[:-12].replace("/", "."): f for f in files if f.endswith("/__init__.py")})
    eager: dict[str, set[str]] = {}
    for f in files:
        tree = ast.parse((root / f).read_text(encoding="utf-8", errors="replace"))
        deps: set[str] = set()
        for n in tree.body:
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                ([f"{n.module}.{a.name}" for a in n.names] + [n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
            deps.update(mod_of[name] for name in names if name in mod_of)
        eager[f] = deps
    return eager


def _closure(eager: dict[str, set[str]], start: str, cut: Optional[str] = None) -> set[str]:
    """Everything loading `start` loads; with `cut`, as if that module's own Creator imports were lazy."""
    seen, todo = {start}, [start]
    while todo:
        x = todo.pop()
        for d in (() if x == cut else eager.get(x, ())):
            if d not in seen:
                seen.add(d)
                todo.append(d)
    return seen


def static_load(root: Path) -> int:
    """For every production module, the AST size of all Creator modules it loads EAGERLY (module-level imports, transitively);
    summed. Lazy imports lower it; it does not depend on any one entry point (the activation holdout)."""
    eager = eager_graph(root)
    sz = sizes(root, list(eager))
    return sum(sum(sz.get(d, 0) for d in _closure(eager, f) - {f}) for f in eager)


def activation_gains(root: Path, entry: str = "creator/kernel.py") -> dict[str, int]:
    """For each module the entry loads: how many AST nodes would no longer load at start if THAT module imported its Creator
    components lazily (2 Oct: activation packages were planned for modules whose imports the kernel loads anyway - no gain was
    possible and they were rejected). Only modules with a positive gain are worth a package."""
    eager = eager_graph(root)
    if entry not in eager:
        return {}
    sz = sizes(root, list(eager))
    loaded = _closure(eager, entry)
    base = sum(sz.get(m, 0) for m in loaded)
    return {m: base - sum(sz.get(x, 0) for x in _closure(eager, entry, cut=m)) for m in sorted(loaded)}


def footprint(root: Path, target: str, memory: bool = True, activation_: Optional[bool] = None) -> Footprint:
    s = sizes(root, [target]).get(target, -1)
    want = memory if activation_ is None else activation_
    act = activation(root)["active_nodes"] if want and (root / "creator" / "kernel.py").is_file() else -1
    return Footprint(target, s, package_size(root), tuple(peak_memory_mb(root)) if memory else (), test_count(root), act,
                     static_load(root), len(uncovered_public(root, target)), uncovered_total(root))
