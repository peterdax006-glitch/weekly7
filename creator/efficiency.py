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


@dataclasses.dataclass(frozen=True)
class Footprint:
    target: str
    target_size: int
    package_size: int
    memory_mb: tuple[float, ...]
    test_cases: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def footprint(root: Path, target: str, memory: bool = True) -> Footprint:
    s = sizes(root, [target]).get(target, -1)
    return Footprint(target, s, package_size(root), tuple(peak_memory_mb(root)) if memory else (), test_count(root))
