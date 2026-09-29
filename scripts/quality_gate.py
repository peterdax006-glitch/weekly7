"""The single code-quality gate: exits nonzero when any check fails (Bible PHASE 31 code quality firewall, PHASE 32 test
pyramid; canons: C6 regular hours only / paper-only, C11 sealed windows stay sealed, no research path may reach a broker).

Static, offline and deterministic: every check parses source with ast/tokenize (nothing is imported or executed, so a
broken module cannot hide from the gate by failing to import). Checks, each a function ctx -> [Finding]:

  syntax               every .py under engine/, scripts/, tests/ parses
  import_boundaries    research never imports the broker order paths or the alpaca SDK; engine never imports scripts/tests;
                       no top-level import cycle inside engine (lazy imports excluded - they are how cycles are broken)
  bare_except          `except:` in engine is an error; `except Exception: pass` (a swallowed failure) is a warning
  print_debug          print()/breakpoint()/pdb in engine, outside modules whose job is progress output to a terminal
  network_in_tests     tests import no network client and never open a socket/URL
  tests_touch_data     tests never read the caches or the sealed livesim windows
  randomness           unseeded / global-state randomness in engine and tests (PHASE 31 "no accidental randomness")
  wall_clock           engine reads the wall clock (warning: no-look-ahead code takes an explicit `now`)
  test_hygiene         a test with nothing that can fail, `assert True`, unconditional skips
  mutable_defaults     def f(x=[]) / def f(x={}) - hidden state shared across calls
  hidden_global_state  `global` statements in engine
  giant_functions      warning above `giant_warn` lines, error above `giant_error`
  duplicates           identical function bodies in different places (warning)
  magic_constants      module-level numeric constants with no comment saying why (warning)
  module_docstrings    engine modules carry a docstring naming their phase/canon; test files carry one
  inventory            scripts/test_inventory.py: untested Bible phases / checklist items / inventory config drift
  tests_pass           (--run-tests) each test file passes and finishes inside its budget

Severity: error fails the gate; warning fails only with --strict. Known findings can be accepted with a baseline file
(--write-baseline / --baseline): baselined findings are reported but do not fail, a NEW finding always does, and a
baseline entry that no longer matches anything is reported as stale so the baseline cannot rot into a blanket waiver.
A check that raises is itself a failure (fail closed). Thresholds live in pyproject.toml [tool.quality_gate].

Usage: python scripts/quality_gate.py [--root DIR] [--strict] [--run-tests] [--baseline F | --write-baseline F]
                                      [--only check,check] [--json OUT] [--quiet]"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import time
import tokenize
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Documented reason for every threshold (PHASE 31: "every configurable value must have a documented reason").
DEFAULTS = {
    "packages": ["engine"],                   # the importable source packages under root
    "giant_warn": 120,                        # a function this long rarely fits in one head; review it
    "giant_error": 300,                       # beyond this it cannot be unit-tested piecewise
    "duplicate_min_lines": 8,                 # shorter bodies collide by accident (getters, thin wrappers)
    "test_seconds": 90.0,                     # per test file budget stated in the builder rules
    "print_allowed": ["data", "edgar", "data_sources"],   # downloaders that report progress to a terminal
    "live_modules": ["live", "broker"],       # own the broker order path
    "broker_modules": ["broker"],             # anything that can place an order
    "known_live_couplings": {                 # reviewed research modules that read live paths/constants only
        "improve": ["live"], "scoring": ["live"], "shadows": ["live"], "site_data": ["live"], "tick": ["live"]},
    "live_scripts": ["livesim_cycle", "livesim_loop2", "collect_intraday", "smoke"],   # scripts that legitimately drive live
    "wall_clock_allowed": ["live", "broker", "tick", "health", "data", "edgar", "checkpoint", "baseline", "champion",
                           "registry", "experiment_memory", "provenance", "improve", "site_data", "isolation",
                           "livesim", "run_report", "data_sources", "scoring", "shadows"],
    "network_modules": ["requests", "urllib.request", "urllib3", "http.client", "socket", "yfinance", "httpx", "aiohttp",
                        "alpaca", "ftplib", "smtplib", "websocket", "websockets"],
    "data_tokens": ["data/cache", "livesim", "K.CACHE", "CACHE /"],
    "test_allowed_data": [],                  # test files with a reviewed reason to name the caches
    "inventory_fail": ["items_untested", "phases_untested", "config_problems"],   # gap kinds that fail the gate
}


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str            # "error" | "warning"
    path: str
    line: int
    msg: str

    def key(self) -> str:
        """Stable identity for baselining: line numbers move, the check/path/message do not."""
        return f"{self.check}|{self.path}|{self.msg}"

    def show(self) -> str:
        return f"{self.severity.upper():7s} {self.check:18s} {self.path}:{self.line}  {self.msg}"


@dataclass
class Source:
    path: Path
    rel: str
    text: str
    tree: ast.AST | None
    error: str = ""

    @property
    def module(self) -> str:
        return self.path.stem


@dataclass
class Context:
    root: Path
    cfg: dict
    files: dict = field(default_factory=dict)          # group ('engine'|'scripts'|'tests') -> [Source]

    def group(self, name: str) -> list[Source]:
        return self.files.get(name, [])


def load_config(root: Path) -> dict:
    """DEFAULTS overlaid by [tool.quality_gate] in root/pyproject.toml (tomllib is stdlib on 3.11)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    pp = root / "pyproject.toml"
    if pp.exists():
        try:
            import tomllib
            cfg.update(tomllib.loads(pp.read_text(encoding="utf-8")).get("tool", {}).get("quality_gate", {}))
        except Exception as e:                           # a malformed config must not silently mean "defaults"
            raise SystemExit(f"quality_gate: cannot read [tool.quality_gate] in {pp}: {e}")
    return cfg


def _parse(path: Path, root: Path) -> Source:
    rel = path.relative_to(root).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return Source(path, rel, "", None, f"unreadable: {e}")
    try:
        return Source(path, rel, text, ast.parse(text, filename=rel))
    except SyntaxError as e:
        return Source(path, rel, text, None, f"{e.msg} (line {e.lineno})")


def build_context(root: Path, cfg: dict | None = None) -> Context:
    cfg = cfg or load_config(root)
    ctx = Context(root, cfg)
    for pkg in cfg["packages"]:
        ctx.files[pkg] = [_parse(p, root) for p in sorted((root / pkg).rglob("*.py")) if "__pycache__" not in p.parts]
    ctx.files["scripts"] = [_parse(p, root) for p in sorted((root / "scripts").glob("*.py"))]
    ctx.files["tests"] = [_parse(p, root) for p in sorted((root / "tests").rglob("*.py")) if "__pycache__" not in p.parts]
    return ctx


def engine_sources(ctx: Context) -> list[Source]:
    return [s for pkg in ctx.cfg["packages"] for s in ctx.group(pkg)]


# ------------------------------------------------------------------ AST helpers
def _dotted(node) -> str:
    """'np.random.default_rng' for an Attribute/Name chain, '' when the base is not a plain name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _imports(tree: ast.AST, pkg: str) -> list[tuple[str, int, bool]]:
    """(engine module short name, line, lazy) for every import of `pkg.x`, `from pkg import x`, `from . import x`.
    lazy=True when the import sits inside a function body (executed on call, not at import time)."""
    out = []

    def visit(node, lazy):
        for ch in ast.iter_child_nodes(node):
            in_fn = lazy or isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(ch, ast.ImportFrom):
                if ch.level == 1 and ch.module:                        # from .broker import X
                    out.append((ch.module.split(".")[0], ch.lineno, lazy))
                elif ch.level == 1:                                    # from . import broker
                    out.extend((a.name, ch.lineno, lazy) for a in ch.names)
                elif ch.module and ch.module.split(".")[0] == pkg:
                    parts = ch.module.split(".")
                    if len(parts) > 1:
                        out.append((parts[1], ch.lineno, lazy))
                    else:
                        out.extend((a.name, ch.lineno, lazy) for a in ch.names)
            elif isinstance(ch, ast.Import):
                for a in ch.names:
                    p = a.name.split(".")
                    if p[0] == pkg and len(p) > 1:
                        out.append((p[1], ch.lineno, lazy))
            visit(ch, in_fn)

    visit(tree, False)
    return out


def _external_imports(tree: ast.AST) -> list[tuple[str, int]]:
    """(dotted module, line) for absolute imports; `from a.b import c` also yields a.b.c so submodules match."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno) for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.append((n.module, n.lineno))
            out += [(f"{n.module}.{a.name}", n.lineno) for a in n.names]
    return out


def _matches(mod: str, banned: list[str]) -> str | None:
    for b in banned:
        if mod == b or mod.startswith(b + "."):
            return b
    return None


def _functions(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _end(fn) -> int:
    return getattr(fn, "end_lineno", fn.lineno) or fn.lineno


# ------------------------------------------------------------------ checks
def check_syntax(ctx: Context) -> list[Finding]:
    out = []
    for grp in ctx.files.values():
        out += [Finding("syntax", "error", s.rel, 1, s.error) for s in grp if s.error]
    return out


def check_import_boundaries(ctx: Context) -> list[Finding]:
    """PHASE 28 in static form: research code cannot reach the order paths; engine does not depend on its own callers."""
    cfg, out = ctx.cfg, []
    live, brokers = set(cfg["live_modules"]), set(cfg["broker_modules"])
    couplings = {k: set(v) for k, v in cfg["known_live_couplings"].items()}
    graph = defaultdict(set)
    for s in engine_sources(ctx):
        if s.tree is None:
            continue
        for mod, line, lazy in _imports(s.tree, "engine"):
            if not lazy and mod != s.module:
                graph[s.module].add(mod)
            if s.module in live or s.module == "isolation":
                continue                                    # live owns the order path; isolation is the guard itself
            if mod in brokers:
                out.append(Finding("import_boundaries", "error", s.rel, line,
                                   f"research module '{s.module}' imports broker order path 'engine.{mod}'"))
            elif mod in live and mod not in couplings.get(s.module, set()):
                out.append(Finding("import_boundaries", "error", s.rel, line,
                                   f"research module '{s.module}' imports live module 'engine.{mod}' "
                                   f"(not in known_live_couplings)"))
        order_sdk_lines = set()
        for mod, line in _external_imports(s.tree):
            top = mod.split(".")[0]
            if top in ("scripts", "tests"):
                out.append(Finding("import_boundaries", "error", s.rel, line, f"engine imports '{top}' (a caller of engine)"))
            if _is_order_sdk(mod) and s.module not in brokers and line not in order_sdk_lines:
                order_sdk_lines.add(line)                       # alpaca.data (market data) is fine; alpaca.trading is not
                out.append(Finding("import_boundaries", "error", s.rel, line, "alpaca trading SDK imported outside the broker module"))
    for s in ctx.group("scripts"):
        if s.tree is None or s.module in cfg["live_scripts"] or s.module.startswith("live"):
            continue
        for mod, line, _ in _imports(s.tree, "engine"):
            if mod in brokers | live:
                out.append(Finding("import_boundaries", "error", s.rel, line,
                                   f"research script '{s.module}' imports 'engine.{mod}' (live path)"))
        for mod, line in _external_imports(s.tree):
            if _is_order_sdk(mod):
                out.append(Finding("import_boundaries", "error", s.rel, line, "research script imports the alpaca trading SDK"))
    for cyc in _cycles(graph):
        first = next((s for s in engine_sources(ctx) if s.module == cyc[0]), None)
        out.append(Finding("import_boundaries", "error", first.rel if first else cyc[0], 1,
                           "top-level import cycle: " + " -> ".join(cyc + [cyc[0]])))
    return out


def _is_order_sdk(mod: str) -> bool:
    """The Alpaca SDK surface that can place orders: the bare package or alpaca.trading.*."""
    return mod == "alpaca" or mod.startswith("alpaca.trading")


def _cycles(graph: dict) -> list[list[str]]:
    """Strongly connected components with more than one node (iterative Tarjan), each reported once, sorted."""
    index, low, on, stack, res, n = {}, {}, set(), [], [], [0]
    nodes = sorted(set(graph) | {m for v in graph.values() for m in v})

    def strong(v0):
        work = [(v0, iter(sorted(graph.get(v0, ()))))]
        index[v0] = low[v0] = n[0]; n[0] += 1; stack.append(v0); on.add(v0)
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = n[0]; n[0] += 1; stack.append(w); on.add(w)
                    work.append((w, iter(sorted(graph.get(w, ())))))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop(); on.discard(w); comp.append(w)
                        if w == v:
                            break
                    if len(comp) > 1:
                        res.append(sorted(comp))

    for v in nodes:
        if v not in index:
            strong(v)
    return sorted(res)


def check_bare_except(ctx: Context) -> list[Finding]:
    out = []
    for s in engine_sources(ctx):
        if s.tree is None:
            continue
        for n in ast.walk(s.tree):
            if not isinstance(n, ast.ExceptHandler):
                continue
            if n.type is None:
                out.append(Finding("bare_except", "error", s.rel, n.lineno, "bare 'except:' also swallows KeyboardInterrupt/SystemExit"))
                continue
            broad = _dotted(n.type) in ("Exception", "BaseException")
            silent = all(isinstance(b, (ast.Pass, ast.Continue)) or
                         (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)) for b in n.body)
            if broad and silent:
                out.append(Finding("bare_except", "warning", s.rel, n.lineno,
                                   f"'except {_dotted(n.type)}' with an empty body swallows every failure silently"))
    return out


def check_print_debug(ctx: Context) -> list[Finding]:
    out, allowed = [], set(ctx.cfg["print_allowed"])
    for s in engine_sources(ctx):
        if s.tree is None:
            continue
        for n in ast.walk(s.tree):
            if isinstance(n, ast.Call):
                name = _dotted(n.func)
                if name == "print" and s.module not in allowed:
                    out.append(Finding("print_debug", "error", s.rel, n.lineno, "print() in engine (use logging or return the value)"))
                elif name in ("breakpoint", "pdb.set_trace", "ipdb.set_trace"):
                    out.append(Finding("print_debug", "error", s.rel, n.lineno, f"{name}() left in engine"))
            elif isinstance(n, ast.Import) and any(a.name in ("pdb", "ipdb") for a in n.names):
                out.append(Finding("print_debug", "error", s.rel, n.lineno, "debugger import left in engine"))
    return out


def check_network_in_tests(ctx: Context) -> list[Finding]:
    out, banned, seen = [], ctx.cfg["network_modules"], set()
    for s in ctx.group("tests"):
        if s.tree is None:
            continue
        for mod, line in _external_imports(s.tree):
            hit = _matches(mod, banned)
            if hit and (s.rel, line, hit) not in seen:          # `from a.b import c` yields a.b and a.b.c: report once
                seen.add((s.rel, line, hit))
                out.append(Finding("network_in_tests", "error", s.rel, line, f"test imports network module '{hit}'"))
        for n in ast.walk(s.tree):
            if isinstance(n, ast.Call) and _dotted(n.func).split(".")[-1] in ("urlopen", "create_connection", "getaddrinfo"):
                out.append(Finding("network_in_tests", "error", s.rel, n.lineno, f"test calls {_dotted(n.func)}()"))
    return out


def check_tests_touch_data(ctx: Context) -> list[Finding]:
    """Builder rule 7: unit tests never load the caches and never touch the sealed test windows."""
    out, allowed = [], set(ctx.cfg["test_allowed_data"])
    for s in ctx.group("tests"):
        if s.tree is None or s.rel in allowed or s.path.name == "_smoke.py":
            continue
        for n in ast.walk(s.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 200:
                v = n.value.replace("\\", "/")
                if "data/cache" in v or "state/livesim" in v:
                    out.append(Finding("tests_touch_data", "error", s.rel, n.lineno, f"test path names a protected location: {v[:60]!r}"))
            elif isinstance(n, ast.Attribute) and n.attr in ("CACHE",) and _dotted(n.value) in ("K", "config"):
                out.append(Finding("tests_touch_data", "error", s.rel, n.lineno, "test reads config.CACHE (the real caches)"))
    return out


_LEGACY_NP = {"rand", "randn", "randint", "random", "random_sample", "choice", "shuffle", "permutation", "normal", "uniform",
              "seed", "standard_normal", "beta", "binomial", "poisson", "exponential", "sample"}
_PY_RANDOM = {"random", "randint", "choice", "choices", "shuffle", "sample", "uniform", "gauss", "seed", "randrange"}
_SEEDED_CTORS = {"RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
                 "LGBMClassifier", "LGBMRegressor", "KMeans", "GradientBoostingClassifier", "GradientBoostingRegressor",
                 "SGDClassifier", "MLPClassifier", "train_test_split", "KFold", "ShuffleSplit"}


def check_randomness(ctx: Context) -> list[Finding]:
    """Global-state or unseeded draws make a run irreproducible (PHASE 33). Applies to engine and to tests."""
    out = []
    for grp, sev in (("engine", "error"), ("tests", "error")):
        srcs = engine_sources(ctx) if grp == "engine" else ctx.group("tests")
        for s in srcs:
            if s.tree is None:
                continue
            py_random = any(isinstance(n, ast.Import) and any(a.name == "random" for a in n.names) for n in ast.walk(s.tree))
            for n in ast.walk(s.tree):
                if not isinstance(n, ast.Call):
                    continue
                name = _dotted(n.func)
                tail = name.split(".")[-1]
                if name.startswith(("np.random.", "numpy.random.")) and tail in _LEGACY_NP:
                    out.append(Finding("randomness", sev, s.rel, n.lineno, f"legacy global numpy RNG call {name}()"))
                elif tail in ("default_rng", "RandomState") and not n.args and not n.keywords:
                    out.append(Finding("randomness", sev, s.rel, n.lineno, f"{name}() with no seed"))
                elif tail == "default_rng" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value is None:
                    out.append(Finding("randomness", sev, s.rel, n.lineno, f"{name}(None) is unseeded"))
                elif py_random and name.startswith("random.") and tail in _PY_RANDOM:
                    out.append(Finding("randomness", sev, s.rel, n.lineno, f"stdlib global RNG call {name}()"))
                elif tail == "Random" and name.startswith("random.") and not n.args:
                    out.append(Finding("randomness", sev, s.rel, n.lineno, "random.Random() with no seed"))
                elif tail in _SEEDED_CTORS and grp == "engine":
                    kw = {k.arg for k in n.keywords}
                    if not kw & {"random_state", "seed", "random_seed", "params", "shuffle"} and not any(k.arg is None for k in n.keywords):
                        out.append(Finding("randomness", "warning", s.rel, n.lineno, f"{tail}(...) without random_state/seed"))
    return out


def check_wall_clock(ctx: Context) -> list[Finding]:
    out, allowed = [], set(ctx.cfg["wall_clock_allowed"])
    clock = {"datetime.now", "datetime.utcnow", "datetime.datetime.now", "datetime.datetime.utcnow", "date.today",
             "datetime.date.today", "time.time", "pd.Timestamp.now", "pd.Timestamp.today", "pd.Timestamp.utcnow"}
    for s in engine_sources(ctx):
        if s.tree is None or s.module in allowed:
            continue
        for n in ast.walk(s.tree):
            if isinstance(n, ast.Call) and _dotted(n.func) in clock:
                out.append(Finding("wall_clock", "warning", s.rel, n.lineno,
                                   f"{_dotted(n.func)}() in research code: take an explicit now/as_of instead"))
    return out


_CAN_FAIL_CALLS = ("raises", "approx", "warns", "fail", "assert_", "allclose", "array_equal", "equal", "check")


def _can_fail(fn) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            return True
        if isinstance(n, ast.Call):
            tail = _dotted(n.func).split(".")[-1].lower()
            if any(tail.startswith(c) or tail == c for c in _CAN_FAIL_CALLS) or tail.startswith(("assert", "must_", "expect", "verify")):
                return True
        if isinstance(n, ast.With):                       # `with pytest.raises(...)`
            for it in n.items:
                if "raises" in _dotted(getattr(it.context_expr, "func", it.context_expr)) or \
                        "warns" in _dotted(getattr(it.context_expr, "func", it.context_expr)):
                    return True
        if isinstance(n, ast.Raise):
            return True
    return False


def check_test_hygiene(ctx: Context) -> list[Finding]:
    """A check that cannot fail is worthless: every test needs an assert (or raises/approx/...) that can trip."""
    out = []
    for s in ctx.group("tests"):
        if s.tree is None or not s.path.name.startswith("test_"):
            continue
        for fn in _functions(s.tree):
            if not fn.name.startswith("test"):
                continue
            body = fn.body[1:] if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) else fn.body
            if not body or all(isinstance(b, ast.Pass) for b in body):
                out.append(Finding("test_hygiene", "error", s.rel, fn.lineno, f"{fn.name}: empty test body"))
            elif not _can_fail(fn):
                out.append(Finding("test_hygiene", "error", s.rel, fn.lineno, f"{fn.name}: nothing in it can fail (no assert/raises/approx)"))
            for n in ast.walk(fn):
                if isinstance(n, ast.Assert) and isinstance(n.test, ast.Constant) and n.test.value:
                    out.append(Finding("test_hygiene", "error", s.rel, n.lineno, f"{fn.name}: assert on a truthy constant"))
                if isinstance(n, ast.Call) and _dotted(n.func) in ("pytest.skip", "pytest.xfail") and n is fn.body[0]:
                    out.append(Finding("test_hygiene", "warning", s.rel, n.lineno, f"{fn.name}: unconditional skip"))
            for d in fn.decorator_list:
                if _dotted(getattr(d, "func", d)) in ("pytest.mark.skip",):
                    out.append(Finding("test_hygiene", "warning", s.rel, fn.lineno, f"{fn.name}: @pytest.mark.skip"))
    return out


def check_mutable_defaults(ctx: Context) -> list[Finding]:
    out = []
    for s in engine_sources(ctx) + ctx.group("scripts"):
        if s.tree is None:
            continue
        for fn in _functions(s.tree):
            for d in fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)) or (isinstance(d, ast.Call) and _dotted(d.func) in ("list", "dict", "set")):
                    out.append(Finding("mutable_defaults", "error", s.rel, fn.lineno, f"{fn.name}: mutable default argument"))
    return out


def check_hidden_global_state(ctx: Context) -> list[Finding]:
    out = []
    for s in engine_sources(ctx):
        if s.tree is None:
            continue
        for n in ast.walk(s.tree):
            if isinstance(n, ast.Global):
                out.append(Finding("hidden_global_state", "warning", s.rel, n.lineno, f"'global {', '.join(n.names)}' mutates module state"))
    return out


def check_giant_functions(ctx: Context) -> list[Finding]:
    out, warn, err = [], ctx.cfg["giant_warn"], ctx.cfg["giant_error"]
    for s in engine_sources(ctx) + ctx.group("scripts"):
        if s.tree is None:
            continue
        for fn in _functions(s.tree):
            n = _end(fn) - fn.lineno + 1
            if n > err:
                out.append(Finding("giant_functions", "error", s.rel, fn.lineno, f"{fn.name} is {n} lines (limit {err})"))
            elif n > warn:
                out.append(Finding("giant_functions", "warning", s.rel, fn.lineno, f"{fn.name} is {n} lines (review above {warn})"))
    return out


class _Normalise(ast.NodeTransformer):
    """Erase names, constants' identity of identifiers and docstrings so renamed copies hash equal."""

    def visit_Name(self, n):
        return ast.copy_location(ast.Name(id="_", ctx=n.ctx), n)

    def visit_arg(self, n):
        return ast.copy_location(ast.arg(arg="_", annotation=None), n)

    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        n.name = "_"
        n.decorator_list = []
        n.returns = None
        if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
        return n


def _fingerprint(fn) -> str:
    import copy
    tree = _Normalise().visit(copy.deepcopy(fn))
    return hashlib.sha1(ast.dump(tree, annotate_fields=False).encode()).hexdigest()


def check_duplicates(ctx: Context) -> list[Finding]:
    """Two functions whose bodies are the same modulo names. Nested/method duplicates inside one class are ignored."""
    seen, out, min_lines = defaultdict(list), [], ctx.cfg["duplicate_min_lines"]
    for s in engine_sources(ctx) + ctx.group("scripts"):
        if s.tree is None:
            continue
        for fn in _functions(s.tree):
            if _end(fn) - fn.lineno + 1 >= min_lines and not fn.name.startswith("test"):
                seen[_fingerprint(fn)].append((s.rel, fn.lineno, fn.name))
    for grp in seen.values():
        if len(grp) > 1 and len({g[0] for g in grp}) > 1:
            first = grp[0]
            for rel, line, name in grp[1:]:
                out.append(Finding("duplicates", "warning", rel, line, f"{name} duplicates {first[2]} in {first[0]}:{first[1]}"))
    return out


def _comment_lines(text: str) -> set[int]:
    lines = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type == tokenize.COMMENT:
                lines.add(t.start[0])
    except (tokenize.TokenError, IndentationError):
        pass
    return lines


def _numeric(node) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value not in (0, 1, -1)
               for n in ast.walk(node))


def check_magic_constants(ctx: Context) -> list[Finding]:
    """PHASE 31: 'every configurable value must have a documented reason'. A module-level UPPER_CASE numeric constant
    counts as documented when a comment sits on any of its lines or on the line above."""
    out = []
    for s in engine_sources(ctx):
        if s.tree is None:
            continue
        comments = _comment_lines(s.text)
        for n in s.tree.body:
            if not isinstance(n, ast.Assign) or len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name):
                continue
            name = n.targets[0].id
            if name.upper() != name or name.startswith("_") or not _numeric(n.value):
                continue
            lo, hi = n.lineno, getattr(n, "end_lineno", n.lineno)
            if not (set(range(lo - 1, hi + 1)) & comments):
                out.append(Finding("magic_constants", "warning", s.rel, lo, f"{name} has no comment saying why it has this value"))
    return out


def check_module_docstrings(ctx: Context) -> list[Finding]:
    out = []
    for s in engine_sources(ctx):
        if s.tree is None or s.path.name == "__init__.py":
            continue
        doc = ast.get_docstring(s.tree)
        if not doc:
            out.append(Finding("module_docstrings", "error", s.rel, 1, "engine module has no docstring"))
        elif not any(w in doc.lower() for w in ("phase", "canon", "bible", "blueprint", "part ", "checklist")):
            out.append(Finding("module_docstrings", "warning", s.rel, 1, "docstring does not say which Bible phase/canon it serves"))
    for s in ctx.group("tests"):
        if s.tree is not None and s.path.name.startswith("test_") and not ast.get_docstring(s.tree):
            out.append(Finding("module_docstrings", "warning", s.rel, 1, "test file has no docstring saying what it proves"))
    return out


def _load_inventory(root: Path):
    for cand in (root / "scripts" / "test_inventory.py", ROOT / "scripts" / "test_inventory.py"):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("_gate_test_inventory", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod                 # dataclasses resolve annotations through sys.modules
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("scripts/test_inventory.py not found")


def check_inventory(ctx: Context) -> list[Finding]:
    """Every Bible phase and checklist item must map to at least one test; the mapping itself must match the Bible."""
    bible = ctx.root / "BIBLE.md"
    if not bible.exists():
        return [Finding("inventory", "error", "BIBLE.md", 1, "BIBLE.md not found: nothing to inventory against")]
    inv_mod = _load_inventory(ctx.root)
    inv = inv_mod.run(ctx.root, write=False)
    g, fail, out = inv["gaps"], set(ctx.cfg["inventory_fail"]), []
    for p in g["phases_untested"]:
        out.append(Finding("inventory", "error" if "phases_untested" in fail else "warning", "BIBLE.md", inv["phases"][p]["title"] and 1,
                           f"phase {p} ({inv['phases'][p]['title'].title()}) has engine modules {inv['phases'][p]['modules']} but no test"))
    for i in g["items_untested"]:
        row = next(r for r in inv["items"] if r["id"] == i)
        out.append(Finding("inventory", "error" if "items_untested" in fail else "warning", "BIBLE.md", 1,
                           f"checklist item {i} ({row['title']}) has no test"))
    for c in g["config_problems"]:
        out.append(Finding("inventory", "error" if "config_problems" in fail else "warning", "scripts/test_inventory.py", 1, c))
    for p, lv in g["level_gaps"].items():
        names = ", ".join(inv_mod.LEVEL_NAMES[int(l)] for l in lv)
        out.append(Finding("inventory", "warning", "BIBLE.md", 1, f"phase {p} lacks pyramid level(s): {names}"))
    for o in g["orphan_test_files"]:
        out.append(Finding("inventory", "warning", o, 1, "test file maps to no Bible phase or checklist item"))
    return out


def check_tests_pass(ctx: Context) -> list[Finding]:
    """Run each test file in its own process; failing or over-budget files are errors."""
    out, budget = [], ctx.cfg["test_seconds"]
    for s in ctx.group("tests"):
        if not s.path.name.startswith("test_"):
            continue
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(s.path)],
                               cwd=ctx.root, capture_output=True, text=True, timeout=budget * 3)
        except subprocess.TimeoutExpired:
            out.append(Finding("tests_pass", "error", s.rel, 1, f"timed out after {budget * 3:.0f}s"))
            continue
        dt = time.time() - t0
        if r.returncode != 0:
            tail = (r.stdout.strip().splitlines() or ["no output"])[-1][:120]
            out.append(Finding("tests_pass", "error", s.rel, 1, f"pytest exit {r.returncode}: {tail}"))
        elif dt > budget:
            out.append(Finding("tests_pass", "warning", s.rel, 1, f"took {dt:.0f}s (budget {budget:.0f}s)"))
    return out


CHECKS = {f.__name__[len("check_"):]: f for f in (
    check_syntax, check_import_boundaries, check_bare_except, check_print_debug, check_network_in_tests,
    check_tests_touch_data, check_randomness, check_wall_clock, check_test_hygiene, check_mutable_defaults,
    check_hidden_global_state, check_giant_functions, check_duplicates, check_magic_constants,
    check_module_docstrings, check_inventory, check_tests_pass)}
OPT_IN = {"tests_pass"}


# ------------------------------------------------------------------ run / baseline / report
def run_checks(ctx: Context, only: list[str] | None = None, run_tests: bool = False) -> tuple[list[Finding], dict]:
    """(findings, per-check timing/count). A check that raises becomes an error finding: the gate fails closed."""
    names = only or [n for n in CHECKS if n not in OPT_IN or run_tests]
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        raise SystemExit(f"quality_gate: unknown check(s) {unknown}; known: {sorted(CHECKS)}")
    findings, stats = [], {}
    for n in names:
        t0 = time.time()
        try:
            got = CHECKS[n](ctx)
        except Exception as e:
            got = [Finding(n, "error", "<gate>", 0, f"check crashed: {type(e).__name__}: {e}")]
        findings += got
        stats[n] = {"seconds": round(time.time() - t0, 3), "errors": sum(f.severity == "error" for f in got),
                    "warnings": sum(f.severity == "warning" for f in got)}
    return sorted(findings, key=lambda f: (f.check, f.path, f.line, f.msg)), stats


def read_baseline(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("accepted", []))


def write_baseline(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"note": "known findings accepted at this date; a NEW finding still fails the gate",
                                "accepted": sorted({f.key() for f in findings})}, indent=1), encoding="utf-8")


def decide(findings: list[Finding], baseline: set[str], strict: bool) -> dict:
    """Split findings into failing / baselined and list stale baseline entries."""
    failing, accepted = [], []
    for f in findings:
        if f.key() in baseline:
            accepted.append(f)
        elif f.severity == "error" or strict:
            failing.append(f)
    stale = sorted(baseline - {f.key() for f in findings})
    return {"failing": failing, "baselined": accepted, "stale_baseline": stale, "exit_code": 1 if failing else 0}


def summarize(findings, stats, verdict, quiet=False) -> str:
    lines = []
    if not quiet:
        lines += [f.show() for f in findings]
    lines.append("")
    lines.append(f"{'check':20s} {'errors':>6s} {'warnings':>8s} {'seconds':>8s}")
    for n, st in stats.items():
        lines.append(f"{n:20s} {st['errors']:6d} {st['warnings']:8d} {st['seconds']:8.2f}")
    lines.append(f"failing {len(verdict['failing'])}, baselined {len(verdict['baselined'])}, "
                 f"stale baseline entries {len(verdict['stale_baseline'])} -> "
                 f"{'FAIL' if verdict['exit_code'] else 'PASS'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Weekly7 code-quality gate (exits nonzero on any failure)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--strict", action="store_true", help="warnings fail too")
    ap.add_argument("--run-tests", action="store_true", help="also run every test file (slow)")
    ap.add_argument("--only", help="comma-separated check names")
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--write-baseline", type=Path)
    ap.add_argument("--json", type=Path, help="write the machine-readable result here")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    a = ap.parse_args(argv)
    ctx = build_context(a.root.resolve())
    findings, stats = run_checks(ctx, a.only.split(",") if a.only else None, a.run_tests)
    if a.write_baseline:
        write_baseline(a.write_baseline, findings)
        print(f"baseline of {len({f.key() for f in findings})} findings written to {a.write_baseline}")
        return 0
    verdict = decide(findings, read_baseline(a.baseline), a.strict)
    print(summarize(findings, stats, verdict, a.quiet))
    out = a.json or (a.root / "state" / "quality" / "quality_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"exit_code": verdict["exit_code"], "stats": stats, "findings": [asdict(f) for f in findings],
                               "failing": [f.key() for f in verdict["failing"]], "stale_baseline": verdict["stale_baseline"]},
                              indent=1), encoding="utf-8")
    return verdict["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
