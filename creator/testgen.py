"""Creator CR093/CR096/CR098 - deterministic test generation, no AI (owner 1 Oct 2026: the Creator works on itself).

For a module it reads the AST (signatures, defaults, annotations) and writes a pytest file with
  (a) smoke tests: the module imports and its public functions/classes exist and are callable;
  (b) edge-case CHARACTERIZATION tests: each eligible function is called with empty containers, 0, negatives, None for Optional,
      boundary ints, one parameter varied at a time. The call is RUN in a subprocess against the current code and the observed
      return value (when it is a plain literal) or its type, or the exception type, is pinned - so a later behaviour change fails;
  (c) anything with side effects or non-determinism is skipped: file/network/subprocess/os use, print/input/exec, global writes,
      non-allowlisted imports, calls to such functions in the same module (transitively), random/time/uuid; and any call whose
      outcome differs between two runs (including different PYTHONHASHSEED) or that prints.
Characterization pins what the code does today, not what it should do: a failing generated test says "behaviour changed"."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Optional

from creator import kernel as K
from creator import model as M
from creator import planner as P

PURE_MODULES = frozenset({"math", "re", "json", "ast", "dataclasses", "typing", "itertools", "functools", "collections", "string",
                          "textwrap", "hashlib", "enum", "__future__", "operator", "statistics", "fractions", "decimal", "abc",
                          "bisect", "heapq", "copy", "numbers", "types"})
SIDE_NAMES = frozenset({"open", "print", "input", "exec", "eval", "compile", "__import__", "globals", "locals", "exit", "quit",
                        "breakpoint", "setattr", "delattr", "id", "hash", "vars", "Path", "random", "time", "uuid", "datetime",
                        "secrets", "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib", "sqlite3", "logging"})
MAX_CALLS_PER_FUNCTION = 10
MAX_FUNCTIONS = 60
TIMEOUT_S = 10


# ------------------------------------------------------------------------------------------------ analysis

@dataclass
class FuncInfo:
    name: str
    node: ast.FunctionDef
    skip: str = ""                                      # why it is not tested ("" = eligible)


def _imports(tree: ast.Module) -> dict[str, str]:
    """bound name -> root module it comes from (module-level and nested imports)."""
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[(a.asname or a.name).split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out[a.asname or a.name] = (n.module or "").split(".")[0] if n.level == 0 else "."
    return out


def _direct_problem(fn: ast.FunctionDef, imports: dict[str, str]) -> str:
    if fn.decorator_list:
        return "decorated"
    for n in ast.walk(fn):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            return "writes global state"
        if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)):
            return "generator/async"
        if isinstance(n, ast.Name):
            if n.id in SIDE_NAMES:
                return f"uses {n.id} (side effect or non-deterministic)"
            if n.id in imports and imports[n.id] not in PURE_MODULES:
                return f"uses {n.id} from {imports[n.id] or '.'}"
    return ""


def analyse(src: str) -> list[FuncInfo]:
    """Public module-level functions, each marked eligible or skipped with the reason."""
    tree = ast.parse(src)
    imports = _imports(tree)
    funcs: dict[str, ast.FunctionDef] = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems = {name: _direct_problem(fn, imports) for name, fn in funcs.items()}
    changed = True
    while changed:                                                  # transitive: calling a flagged function is flagged
        changed = False
        for name, fn in funcs.items():
            if problems[name]:
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and problems.get(n.id) and n.id != name:
                    problems[name], changed = f"calls {n.id}: {problems[n.id]}", True
                    break
    out = [FuncInfo(n, fn, problems[n]) for n, fn in funcs.items() if not n.startswith("_")]
    out += [FuncInfo(n.name, n, "async") for n in tree.body  # type: ignore[arg-type]
            if isinstance(n, ast.AsyncFunctionDef) and not n.name.startswith("_")]
    return out


# ------------------------------------------------------------------------------------------------ argument candidates

_UNKNOWN: list[Any] = [0, "", None, []]


def _edge_values(ann: Optional[ast.expr]) -> list[Any]:
    """Edge values for a parameter annotation (first one is the baseline)."""
    if ann is None:
        return list(_UNKNOWN)
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            ann = ast.parse(ann.value, mode="eval").body
        except SyntaxError:
            return list(_UNKNOWN)
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        out = _edge_values(ann.left)
        for v in _edge_values(ann.right):
            if not any(type(v) is type(o) and v == o for o in out):
                out.append(v)
        return out
    if isinstance(ann, ast.Constant) and ann.value is None:
        return [None]
    base = ann.value if isinstance(ann, ast.Subscript) else ann
    name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
    if name == "Optional" and isinstance(ann, ast.Subscript):
        return _edge_values(ann.slice) + [None]
    table: dict[str, list[Any]] = {
        "int": [0, -1, 1, 100], "float": [0.0, -1.0, 1.5], "str": ["", "a", " x "], "bool": [False, True],
        "bytes": [b"", b"a"], "list": [[], [0]], "List": [[], [0]], "Sequence": [[], [0]], "Iterable": [[], [0]],
        "dict": [{}, {"a": 1}], "Dict": [{}, {"a": 1}], "Mapping": [{}, {"a": 1}],
        "tuple": [(), (0,)], "Tuple": [(), (0,)], "set": [set()], "Set": [set()], "frozenset": [frozenset()]}
    return list(table.get(name, _UNKNOWN))


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def candidate_calls(fn: ast.FunctionDef) -> list[tuple[Any, ...]]:
    """Positional-argument tuples: the baseline plus each parameter varied alone over its edge values.
    Empty when the signature cannot be reproduced (non-literal default, keyword-only parameter without default)."""
    a = fn.args
    if any(kd is None for kd in a.kw_defaults):
        return []
    pos = list(a.posonlyargs) + list(a.args)
    defaults: list[Optional[ast.expr]] = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    params: list[tuple[list[Any], Any]] = []
    for arg, d in zip(pos, defaults):
        edges = _edge_values(arg.annotation)
        base = edges[0]
        if d is not None:
            try:
                base = ast.literal_eval(d)
            except (ValueError, SyntaxError, TypeError):
                return []
            if base is None and not any(_same(None, e) for e in edges):
                edges.append(None)
        params.append((edges, base))
    calls: list[tuple[Any, ...]] = [tuple(p[1] for p in params)]
    for i, (edges, base) in enumerate(params):
        for v in edges:
            if _same(v, base):
                continue
            c = tuple(v if j == i else p[1] for j, p in enumerate(params))
            if not any(len(c) == len(o) and all(_same(x, y) for x, y in zip(c, o)) for o in calls):
                calls.append(c)
    return calls[:MAX_CALLS_PER_FUNCTION]


# ------------------------------------------------------------------------------------------------ characterization (runs the code)

RUNNER = '''
import contextlib, importlib, io, json, sys
sys.path.insert(0, ROOT)
mod = importlib.import_module(MODNAME)
def outcome(fn, args):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            v = fn(*[eval(s) for s in args])
    except Exception as e:
        return ["exc", type(e).__name__, ""]
    if buf.getvalue():
        return ["noisy", "", ""]
    try:
        r = repr(v)
    except Exception:
        r = ""
    return ["ok", type(v).__name__, r]
for name, calls in PLAN:
    fn = getattr(mod, name)
    print(json.dumps({"start": name}), flush=True)
    for i, c in enumerate(calls):
        o1 = outcome(fn, c)
        o2 = outcome(fn, c)
        print(json.dumps({"f": name, "i": i, "o": o1 if o1 == o2 else ["nondet", "", ""]}), flush=True)
'''

Plan = list[tuple[str, list[list[str]]]]


def _run_plan(root: Path, modname: str, plan: Plan, seed: str) -> tuple[dict[tuple[str, int], list[str]], str]:
    """Run the calls in a subprocess; (results, name of the function that hung or crashed, else '')."""
    code = f"ROOT={str(root)!r}\nMODNAME={modname!r}\nPLAN={plan!r}\n" + RUNNER
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
    out, bad = "", False
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=TIMEOUT_S, env=env,
                           encoding="utf-8", errors="replace")
        out, bad = r.stdout, r.returncode != 0
    except subprocess.TimeoutExpired as e:
        raw = e.stdout or ""
        out, bad = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw), True
    res: dict[tuple[str, int], list[str]] = {}
    last = ""
    for ln in out.splitlines():
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if "start" in d:
            last = d["start"]
        elif "f" in d:
            res[(d["f"], d["i"])] = d["o"]
    return res, (last or "*") if bad else ""


def characterize(root: Path, modname: str, plan: Plan) -> dict[tuple[str, int], list[str]]:
    """Outcome per (function, call index), keeping only outcomes identical under two hash seeds; hung/crashing functions are
    dropped. {} when the module itself cannot be imported or runs nothing."""
    dropped: set[str] = set()
    for _ in range(len(plan) + 1):
        cur = [(n, cs) for n, cs in plan if n not in dropped]
        if not cur:
            return {}
        a, bad = _run_plan(root, modname, cur, "1")             # the hang check runs once; the second seed only on survivors
        if bad:
            if bad == "*" or bad in dropped:
                return {}
            dropped.add(bad)
            continue
        b, bad = _run_plan(root, modname, cur, "2")
        if bad:
            if bad == "*" or bad in dropped:
                return {}
            dropped.add(bad)
            continue
        return {k: v for k, v in a.items() if b.get(k) == v and v[0] in ("ok", "exc")}
    return {}


# ------------------------------------------------------------------------------------------------ test source

def _pin(outcome: list[str]) -> str:
    kind, name, rep = outcome
    if kind == "exc":
        return f"assert r == (\"exc\", {name!r})"
    try:
        v = ast.literal_eval(rep)
        pinned = repr(v) == rep and v == v
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pinned = False
    if pinned and len(rep) <= 300:
        return f"assert r == (\"ok\", {rep})"
    return f"assert r[0] == \"ok\" and type(r[1]).__name__ == {name!r}"


@dataclass
class Generated:
    source: str
    tested: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    cases: int = 0


def generate(root: Path, rel: str, modname: Optional[str] = None, only: Optional[Collection[str]] = None) -> Generated:
    """Test source for the module at root/rel (a .py path relative to root); runs the eligible calls to pin their outcomes.
    `only` restricts it to those public names (coverage work: the names no existing test names)."""
    src = (root / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mod = modname or Path(rel).with_suffix("").as_posix().replace("/", ".")
    infos = [i for i in analyse(src) if only is None or i.name in only]
    skipped = {i.name: i.skip for i in infos if i.skip}
    callmap: dict[str, list[tuple[Any, ...]]] = {}
    for i in infos:
        if i.skip:
            continue
        calls = candidate_calls(i.node)
        if not calls:
            skipped[i.name] = "no drivable signature"
        elif len(callmap) >= MAX_FUNCTIONS:
            skipped[i.name] = "function budget"
        else:
            callmap[i.name] = calls
    plan: Plan = [(n, [[repr(x) for x in c] for c in cs]) for n, cs in callmap.items()]
    res = characterize(root, mod, plan) if plan else {}
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
               and (only is None or n.name in only)]
    out = ['"""Generated by creator.testgen (deterministic, no AI): characterization tests pin what the code does today."""',
           "from __future__ import annotations", "", "import importlib", "", f"M = importlib.import_module({mod!r})", "", "",
           "def _outcome(fn, *a):", "    try:", "        return (\"ok\", fn(*a))", "    except Exception as e:",
           "        return (\"exc\", type(e).__name__)", "", "",
           "def test_module_imports():", f"    assert M.__name__ == {mod!r}", "", "",
           "def test_public_names_exist():",
           *[f"    assert callable(M.{n}), {n!r}" for n in sorted([i.name for i in infos] + classes)], ""]
    cases, tested = 0, []
    for name, calls in callmap.items():
        got = [(i, res[(name, i)]) for i in range(len(calls)) if (name, i) in res]
        if not got:
            skipped[name] = "no stable outcome"
            continue
        tested.append(name)
        for k, o in got:
            args = ", ".join(repr(x) for x in calls[k])
            out += ["", f"def test_{name}_edge_{k}():",
                    f"    r = _outcome(M.{name}{', ' if args else ''}{args})", f"    {_pin(o)}", ""]
            cases += 1
    return Generated("\n".join(out).rstrip("\n") + "\n", tested, skipped, cases)


# ------------------------------------------------------------------------------------------------ the worker

def _has_tests(workdir: Path, stem: str) -> bool:
    d = workdir / "tests"
    return d.is_dir() and any(stem in p.name for p in d.glob("test_*.py"))


class TestGenWorker:
    """Writes tests/test_gen_<module>.py for the package's untested modules; keeps one only if it passes on the code it came from."""
    __test__ = False                                                # not a pytest class
    name = "self-testgen-v1"
    steps = ("tested", "coverage")

    def targets(self, package: M.WorkPackage, workdir: Path) -> list[str]:
        return [x for x in package.outputs if x.endswith(".py") and not Path(x).name.startswith("test_")
                and (workdir / x).is_file()]

    def _cover(self, package: M.WorkPackage, workdir: Path) -> K.WorkResult:
        """Coverage package: tests for the public names no test names yet, kept only if they pass and shrink that gap."""
        from creator import efficiency as E
        made: list[str] = []
        for rel in self.targets(package, workdir):
            missing = E.uncovered_public(workdir, rel)
            if not missing:
                continue
            try:
                g = generate(workdir, rel, only=missing)
            except (SyntaxError, OSError, ValueError):
                continue
            if g.cases == 0:
                continue
            dest = workdir / "tests" / f"test_gen_{Path(rel).stem}.py"
            if dest.exists():
                dest = dest.with_name(f"test_gen_{Path(rel).stem}_cov.py")
            dest.write_text(g.source, encoding="utf-8")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(dest)], cwd=workdir,
                               capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONPATH": str(workdir)})
            if r.returncode == 0 and len(E.uncovered_public(workdir, rel)) < len(missing):
                made.append(f"{dest.relative_to(workdir).as_posix()} ({g.cases} cases)")
            else:
                dest.unlink()
        if not made:
            return K.WorkResult(False, "no coverage test generated")
        return K.WorkResult(True, f"generated {len(made)} coverage test files: {', '.join(made)}", by=self.name)

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> K.WorkResult:
        if plan is not None and plan.step == "coverage":
            return self._cover(package, workdir)
        made: list[str] = []
        for rel in self.targets(package, workdir):
            stem = Path(rel).stem
            if _has_tests(workdir, stem):
                continue
            try:
                g = generate(workdir, rel)
            except (SyntaxError, OSError, ValueError):
                continue
            if g.cases == 0:
                continue
            dest = workdir / "tests" / f"test_gen_{stem}.py"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(g.source, encoding="utf-8")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(dest)], cwd=workdir,
                               capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONPATH": str(workdir)})
            if r.returncode == 0:
                made.append(f"{dest.relative_to(workdir).as_posix()} ({g.cases} cases)")
            else:
                dest.unlink()
        if not made:
            return K.WorkResult(False, "no test generated")
        return K.WorkResult(True, f"generated {len(made)} test files: {', '.join(made)}", by=self.name)
