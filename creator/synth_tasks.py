"""Creator K24c - a cheap, deterministic, lever-sensitive workload for the recursion (C77 CR191-198) - IMPLEMENTED, NOT VALIDATED.

The devbench dev tasks cost minutes each on a busy machine (three pytest runs plus a search that runs pytest per candidate), so a
process A/B on them cannot reach statistical power in an evening. These tasks run IN PROCESS:

    corpus   the Creator's own source files (read as text, never imported): top-level functions that are pure by construction -
             every name they load is a parameter, a local, themselves or a whitelisted builtin, no dunder attributes. They run in a
             namespace holding only those builtins, so nothing can touch a file, the network or another module.
    task     "S<n>": function and bug are fixed by a hash of (n). The function's behaviour on edge-value calls (the testgen
             argument generator) is the truth; the calls are split into VISIBLE (even index) and HIDDEN (odd index). The bug is one
             or two edits (generator mutation operators) chosen so a visible call changes.
    solver   the process in force: the repair search of creator.generator (targeted + generic single edits, then pair edits when
             pair_width > 0) with the levers of creator.process_levers - candidate budget per attempt, candidates kept per family,
             pair width, and `retries` further attempts that each look at wider families and only count NEW candidates.
    outcome  SOLVED = visible and hidden calls all match the original;  harm = it passed the visible calls but not the hidden ones
             (a false completion);  otherwise unsolved.
Nothing here reads the devbench, its holdout or any sealed material, and no number is typed in: the effect of a lever is whatever
the search does on these tasks."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import itertools
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from creator import generator as G
from creator import process_levers as PL
from creator import recursion as R

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = ("devbench", "audit", "__pycache__", "sealed")
SCAN = ("creator",)
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "len", "range", "sorted", "min", "max", "sum", "abs", "int", "str", "float", "bool", "list", "dict", "set", "tuple", "enumerate",
    "zip", "any", "all", "isinstance", "reversed", "round", "repr", "map", "filter", "divmod", "pow", "bytes", "frozenset",
    "True", "False", "None", "ValueError", "KeyError", "IndexError", "TypeError", "Exception", "next", "iter", "chr", "ord", "hex",
    "NotImplementedError", "RuntimeError", "ZeroDivisionError", "AssertionError", "StopIteration", "OverflowError")}
SAFE_NAMES = frozenset(SAFE_BUILTINS)
MIN_CALLS = 4
POOL = 100000                                       # task ids S0..S{POOL-1} map onto the corpus by hash


def _h(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def strip(fn: ast.FunctionDef) -> ast.FunctionDef:
    """A copy without annotations (they name types the pure namespace does not have; the originals still drive the call generator)."""
    c = copy.deepcopy(fn)
    c.returns = None
    for n in ast.walk(c):
        if isinstance(n, ast.arg):
            n.annotation = None
    return c


def pure_function(fn: ast.FunctionDef) -> bool:
    """True when fn can only compute on its arguments: every loaded name is bound inside it or whitelisted."""
    fn = strip(fn)
    if fn.decorator_list or fn.args.kwarg or fn.args.vararg:
        return False
    bound = {fn.name} | {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.While, ast.With, ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await,
                            ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            return False
        elif isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            return False
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound and n.id not in SAFE_NAMES:
            return False
    return True


def corpus(root: Path = ROOT) -> list[tuple[str, ast.FunctionDef]]:
    """('<file>:<function>', node) for every small top-level function of the Creator's own files, sorted (purity is checked per task)."""
    out: list[tuple[str, ast.FunctionDef]] = []
    for p in sorted(q for d in SCAN for q in (root / d).rglob("*.py")):
        if any(s in p.relative_to(root).parts for s in SKIP_DIRS):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and 14 <= sum(1 for _ in ast.walk(n)) <= 90:
                out.append((f"{p.name}:{n.name}", n))
    return out


def _compile(fn: ast.FunctionDef) -> Optional[Callable[..., Any]]:
    mod = ast.Module(body=[fn], type_ignores=[])
    ns: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS)}
    try:
        exec(compile(ast.fix_missing_locations(mod), "<synth>", "exec"), ns)     # noqa: S102 - pure namespace, see pure_function
    except Exception:                                                         # noqa: BLE001
        return None
    f = ns.get(fn.name)
    return f if callable(f) else None


def _outcome(f: Optional[Callable[..., Any]], args: Sequence[Any]) -> str:
    if f is None:
        return "nocompile"
    try:
        return "ok:" + repr(f(*[a.copy() if isinstance(a, (list, dict, set)) else a for a in args]))[:200]
    except Exception as e:                                                    # noqa: BLE001 - the exception type is the outcome
        return "exc:" + type(e).__name__


class Task:
    def __init__(self, tid: str, name: str, fn: ast.FunctionDef, calls: list[tuple[Any, ...]], truth: list[str],
                 buggy: ast.Module, bugs: int) -> None:
        self.id, self.name, self.fn, self.calls, self.truth, self.buggy, self.bugs = tid, name, fn, calls, truth, buggy, bugs
        self.visible = list(range(0, len(calls), 2))
        self.hidden = list(range(1, len(calls), 2))

    def check(self, tree: ast.Module, idx: Sequence[int]) -> bool:
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == self.fn.name), None)
        f = _compile(node) if node is not None else None
        return all(_outcome(f, self.calls[i]) == self.truth[i] for i in idx)


def build_task(tid: str, funcs: Sequence[tuple[str, ast.FunctionDef]]) -> Optional[Task]:
    """The task for id `tid`, or None when this hash picks a function that cannot be turned into a (visible-failing) bug."""
    from creator import testgen as TG
    n = int(tid[1:])
    name, fn = funcs[_h("fn", n) % len(funcs)]
    if not pure_function(fn):
        return None
    calls = TG.candidate_calls(fn)
    if len(calls) < MIN_CALLS:
        return None
    f0 = _compile(strip(fn))
    truth = [_outcome(f0, c) for c in calls]
    if len(set(truth)) < 2:
        return None
    fn = strip(fn)
    task = Task(tid, name, fn, calls, truth, ast.Module(body=[fn], type_ignores=[]), 1)
    bugs = 1 if _h("k", n) % 10 < 7 else 2
    tree = ast.Module(body=[fn], type_ignores=[])
    for b in range(bugs):
        opts = [m for m in G.generic_mutations(tree)]
        if not opts:
            return None
        start = _h("bug", n, b) % len(opts)
        for j in range(len(opts)):
            cand = opts[(start + j) % len(opts)]
            if ast.dump(cand) != ast.dump(tree):
                tree = cand
                break
    if task.check(tree, task.visible) or ast.dump(tree) == ast.dump(task.buggy):
        return None                                                   # the bug must make a visible call fail
    task.buggy, task.bugs = tree, bugs
    return task


def candidates(tree: ast.Module, lv: PL.Levers, attempt: int) -> Iterator[ast.Module]:
    legacy = list(G.generic_mutations(tree))
    singles = list(G.targeted_mutations(tree, lv.per_family * (attempt + 1))) + legacy
    if lv.pair_width > 0:
        yield from itertools.chain(singles, (m2 for m1 in legacy[:lv.pair_width] for m2 in itertools.islice(G.mutations(m1), lv.pair_width)))
    else:
        yield from singles


def repair(task: Task, p: R.ProcessConfig) -> tuple[Optional[ast.Module], int]:
    """The process's search on one task: (first candidate passing the visible calls or None, new candidates tried)."""
    lv = PL.levers(p)
    seen: set[str] = set()
    tried = 0
    for attempt in range(lv.retries + 1):
        fresh = 0
        for cand in candidates(task.buggy, lv, attempt):
            if fresh >= lv.search_budget:
                break
            src = ast.dump(cand)
            if src in seen:
                continue
            seen.add(src)
            fresh += 1
            tried += 1
            if task.check(cand, task.visible):
                return cand, tried
    return None, tried


class SynthBench:
    """Task source for DevWorkload: ids come from a seed, tasks are built lazily and cached."""

    def __init__(self, root: Path = ROOT) -> None:
        self.funcs = corpus(root)
        self._tasks: dict[str, Optional[Task]] = {}

    def ids(self, seed: int, n: int) -> list[str]:
        """n distinct, buildable task ids drawn by `seed` (a different seed gives different tasks)."""
        out: list[str] = []
        k = 0
        while len(out) < n and k < n * 400:
            tid = f"S{_h('id', seed, k) % POOL}"
            k += 1
            if tid not in out and self.task(tid) is not None:
                out.append(tid)
        return out

    def task(self, tid: str) -> Optional[Task]:
        if tid not in self._tasks:
            self._tasks[tid] = build_task(tid, self.funcs)
        return self._tasks[tid]

    def evaluate(self, p: R.ProcessConfig, tid: str) -> tuple[bool, bool]:
        t = self.task(tid)
        assert t is not None
        cand, _ = repair(t, p)
        if cand is None:
            return False, True
        if t.check(cand, t.hidden):
            return True, True
        return False, False                                             # passed the visible calls, wrong elsewhere: a false completion


def workload(seed: int = 1, chunk: int = 40, reps: int = 3, confirm: int = 60, cache_path: Optional[Path] = None,
             bench: Optional[SynthBench] = None) -> PL.DevWorkload:
    """A DevWorkload over synthetic tasks: `reps` disjoint chunks (the choice arm) and a further disjoint `confirm` arm."""
    b = bench or SynthBench()
    wl = PL.DevWorkload(seed=seed, chunk=chunk, reps=reps, confirm=confirm, evaluate=b.evaluate,
                        dev_ids=b.ids(seed, chunk * reps + confirm), cache_path=cache_path)
    return wl
