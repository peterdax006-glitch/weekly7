"""Creator K21 - the system's OWN workers: it works on itself with no AI at all (owner, 1 Oct 2026: "it should work toward the level
where it works on itself more than a third party works on it") - IMPLEMENTED, NOT VALIDATED.

    RuleWorker    deterministic code transforms the Creator applies to its own modules:
                    unused_imports   drop module-level imports nothing uses
                    lazy_imports     move a module-level import of a Creator component into the functions that use it, so the
                                     component is loaded only when needed (owner: "never run any more code than absolutely
                                     necessarry")
                    dead_private     delete private top-level functions/classes nothing in the repository references
    SearchWorker  test-guided mutation repair (creator.generator.solve_with_search) for packages whose tests fail
    SelfFirst     tries the system's own workers first, each in a clean sandbox state, and hands the package to the next
                  worker - finally the Claude session - only when its own workers produced nothing that passes the tests
                  that reach the change. The worker that made the change is recorded (WorkResult.by), so
                  creator.objective.self_share counts who really did the work.

The kernel still measures and decides; a transform that breaks anything is rejected like any other change."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import kernel as K
from creator import model as M
from creator import planner as P

SELF_PREFIX = "self-"
COMPONENT = re.compile(r"^creator(\.|$)")


# ------------------------------------------------------------------------------------------------ helpers

def _used_names(tree: ast.AST, skip: Optional[ast.AST] = None) -> set[str]:
    out: set[str] = set()
    skipped = set(ast.walk(skip)) if skip is not None else set()          # the node AND everything inside it
    for n in ast.walk(tree):
        if n in skipped:
            continue
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            base = n
            while isinstance(base, ast.Attribute):
                base = base.value                                   # type: ignore[assignment]
            if isinstance(base, ast.Name):
                out.add(base.id)
    for n in ast.walk(tree):                                        # names in string annotations / __all__ count as used
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.update(re.findall(r"[A-Za-z_]\w*", n.value))
    return out


def _bound(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".")[0]


def _replace_lines(src: str, start: int, end: int, new: Sequence[str]) -> str:
    lines = src.split("\n")
    return "\n".join(lines[:start - 1] + list(new) + lines[end:])


def _parse(src: str) -> Optional[ast.Module]:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


# ------------------------------------------------------------------------------------------------ rules (pure: source -> source)

def unused_imports(src: str) -> Optional[str]:
    """Drop module-level import names nothing in the module uses. None when there is nothing to drop."""
    tree = _parse(src)
    if tree is None:
        return None
    for node in reversed([n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        used = _used_names(tree, skip=node)
        keep = [a for a in node.names if _bound(a) in used or a.name == "*"]
        if len(keep) == len(node.names):
            continue
        indent = " " * node.col_offset
        new = [] if not keep else [indent + ast.unparse(type(node)(**{**{f: getattr(node, f) for f in node._fields},
                                                                     "names": keep}))]
        return _replace_lines(src, node.lineno, node.end_lineno or node.lineno, new)
    return None


def lazy_imports(src: str) -> Optional[str]:
    """Move ONE module-level import of a Creator component, used only inside functions (never at module level, in a decorator,
    a default value, a class body or an annotation), into those functions."""
    tree = _parse(src)
    if tree is None:
        return None
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    top_funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for node in [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]:
        mod = node.module if isinstance(node, ast.ImportFrom) else (node.names[0].name if node.names else "")
        if not mod or not COMPONENT.match(mod) or len(node.names) != 1:
            continue
        name = _bound(node.names[0])
        outside = [s for s in tree.body if s is not node and not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if any(name in _used_names(s) for s in outside):
            continue                                                # used at module level or in a class body
        if any(name in _used_names(d) for f in funcs for d in (*f.decorator_list, *f.args.defaults, *f.args.kw_defaults)
               if d is not None):
            continue
        if any(name in _used_names(a) for f in funcs for a in (f.returns, *[x.annotation for x in f.args.args])
               if a is not None):
            continue                                                # annotations may be evaluated (get_type_hints)
        users = [f for f in top_funcs if name in _used_names(f)]
        if not users or len(users) > 6:
            continue
        stmt = ast.unparse(node)
        out = src.split("\n")
        for f in sorted(users, key=lambda f: f.lineno, reverse=True):
            first = f.body[0]
            at = first.end_lineno if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                                      and isinstance(first.value.value, str) and len(f.body) > 1) else first.lineno - 1
            body_indent = " " * (f.body[1].col_offset if at == first.end_lineno else first.col_offset)
            out.insert(at or 0, f"{body_indent}{stmt}")
        new_src = "\n".join(out)
        new_tree = _parse(new_src)
        if new_tree is None:
            continue
        lines = new_src.split("\n")
        top = next(n for n in new_tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.unparse(n) == stmt)
        return "\n".join(lines[:top.lineno - 1] + lines[top.end_lineno or top.lineno:])
    return None


def dead_private(src: str, referenced: Callable[[str], bool]) -> Optional[str]:
    """Delete ONE private top-level function/class whose name `referenced` reports as unused anywhere else."""
    tree = _parse(src)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith("_") \
                and not node.name.startswith("__") and node.name not in _used_names(tree, skip=node) \
                and not referenced(node.name):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return _replace_lines(src, start, node.end_lineno or node.lineno, [])
    return None


RULES: tuple[str, ...] = ("unused_imports", "lazy_imports", "dead_private")


# ------------------------------------------------------------------------------------------------ workers

def _run_tests(workdir: Path, changed: Sequence[str]) -> bool:
    from creator import localworker as LW                          # test selection only (no model is loaded)
    tests = LW.tests_for(list(changed), workdir)
    ok, _ = LW.run_tests(workdir, tests)
    return ok


def _referenced_in(workdir: Path, exclude: str) -> Callable[[str], bool]:
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in workdir.rglob("*.py")
             if "__pycache__" not in p.parts and p.relative_to(workdir).as_posix() != exclude]
    return lambda name: any(re.search(rf"\b{re.escape(name)}\b", t) for t in texts)


class RuleWorker:
    """Applies its transforms to the package's target modules until one passes the tests that reach it."""
    name = SELF_PREFIX + "rules-v1"
    steps = ("efficiency", "no_stubs", "integrated")

    def __init__(self, rules: Sequence[str] = RULES, max_rounds: int = 25) -> None:
        self.rules, self.max_rounds = tuple(rules), max_rounds

    def targets(self, plan: P.Plan, package: M.WorkPackage) -> list[str]:
        return [x for x in package.outputs if x.endswith(".py") and not Path(x).name.startswith("test_")]

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> K.WorkResult:
        done: list[str] = []
        for rel in self.targets(plan, package):
            p = workdir / rel
            if not p.is_file():
                continue
            for rule in self.rules:
                def apply_once(src: str) -> Optional[str]:
                    if rule == "unused_imports":
                        return unused_imports(src)
                    if rule == "lazy_imports":
                        return lazy_imports(src)
                    return dead_private(src, _referenced_in(workdir, rel))
                start = p.read_text(encoding="utf-8")
                batch, n = start, 0                                  # BATCH first: every application, then ONE test run
                for _ in range(self.max_rounds):                    # (2 Oct: one test run per application timed out on
                    nxt = apply_once(batch)                          # kernel.py under a full swarm)
                    if nxt is None or nxt == batch:
                        break
                    batch, n = nxt, n + 1
                if n == 0:
                    continue
                p.write_text(batch, encoding="utf-8")
                if _run_tests(workdir, [rel]):
                    done.extend(f"{rule}:{rel}" for _ in range(n))
                    continue
                p.write_text(start, encoding="utf-8")                # the batch broke something: fall back one at a time
                for _ in range(min(n, self.max_rounds)):
                    src = p.read_text(encoding="utf-8")
                    new = apply_once(src)
                    if new is None or new == src:
                        break
                    p.write_text(new, encoding="utf-8")
                    if not _run_tests(workdir, [rel]):
                        p.write_text(src, encoding="utf-8")         # this application broke something: undo, try next rule
                        break
                    done.append(f"{rule}:{rel}")
        if not done:
            return K.WorkResult(False, "no rule applied")
        return K.WorkResult(True, f"{len(done)} transforms: {', '.join(done[:12])}", by=self.name)


class SearchWorker:
    """Test-guided mutation repair over the package's modules (ONLY for packages whose tests fail: on passing code a mutation
    that keeps the tests green is just an untested change - 1 Oct: it grew a module and was rejected as a regression)."""
    name = SELF_PREFIX + "search-v1"
    steps = ("tested",)

    def __init__(self, budget: int = 150) -> None:
        self.budget = budget

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> K.WorkResult:
        from creator import generator as G
        from creator import localworker as LW
        mods = [x for x in package.outputs if x.endswith(".py") and not Path(x).name.startswith("test_") and (workdir / x).is_file()]
        tests = LW.tests_for(mods, workdir)
        if not mods or not tests:
            return K.WorkResult(False, "nothing to search")
        for rel in mods:
            p = workdir / rel
            original = p.read_text(encoding="utf-8")
            tree = ast.parse(original)
            for i, cand in enumerate(G.mutations(tree)):
                if i >= self.budget:
                    break
                p.write_text(ast.unparse(cand) + "\n", encoding="utf-8")
                if LW.run_tests(workdir, tests)[0]:
                    return K.WorkResult(True, f"mutation {i} of {rel} passes {len(tests)} test files", by=self.name)
            p.write_text(original, encoding="utf-8")
        return K.WorkResult(False, "no mutation passes")


def reset(workdir: Path) -> None:
    """Back to the sandbox's committed state between workers (inside the sandbox worktree only)."""
    subprocess.run(["git", "checkout", "--", "."], cwd=workdir, capture_output=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=workdir, capture_output=True)


class SelfFirst:
    """The system's own workers first; the next worker (finally the Claude session) only for what they could not do."""
    name = "self-first"

    def __init__(self, own: Sequence[Any], fallback: Optional[Any] = None) -> None:
        self.own, self.fallback = list(own), fallback

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> K.WorkResult:
        tried = []
        for w in self.own:
            handles = getattr(w, "steps", None)
            if plan is not None and handles is not None and plan.step not in handles:
                continue                                            # this worker does not do this kind of package
            r = w(plan, package, workdir)
            if r.claimed_done:
                return r
            tried.append(f"{getattr(w, 'name', 'worker')}: {r.notes}")
            reset(workdir)
        if self.fallback is not None:
            r = self.fallback(plan, package, workdir)
            return K.WorkResult(r.claimed_done, "; ".join(tried + [r.notes])[:2000], r.calls, r.usd, r.refused, r.contaminated,
                                by=getattr(self.fallback, "name", "fallback"), reasoning=r.reasoning, deferred=r.deferred)
        return K.WorkResult(False, "; ".join(tried) or "no worker")
