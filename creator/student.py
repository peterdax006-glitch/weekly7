"""Creator student: an apprentice that LEARNS rewrite templates from Claude's adopted solved work packages (lessons) - no model.

Learning = transformation inference. For every adopted lesson the before/after sources of each Python file are compared at the AST
level; a recognised change is abstracted into a template with HOLES (module names, function names, parameter names, constants), and
templates of the same kind from different lessons are anti-unified (a field on which examples disagree becomes a hole). Templates keep
their support (distinct lessons) and the task_kinds they came from, plus per-task_kind usefulness.

    lazy_import     module-level `import m` used only inside functions  ->  `import m` as first statement of those functions
    remove_unused   top-level function nothing references                 ->  deleted
    inline_temp     `t = expr; <stmt using t once>`                       ->  expr substituted at the use
    empty_guard     `for x in p: ...; return C`                           ->  `if not p: return C` first

Applying a template is purely syntactic and conservative (it refuses anything it cannot prove local). The kernel measures the result;
the student never judges itself. A model-backed student can later sit behind the same protocol (`__call__`, `can_attempt`, `name`)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from creator.kernel import WorkResult

KINDS = ("lazy_import", "remove_unused", "inline_temp", "empty_guard")
FuncT = (ast.FunctionDef, ast.AsyncFunctionDef)


def _parse(src: str) -> Optional[ast.Module]:
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError):
        return None


def _is_doc(s: ast.stmt) -> bool:
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def _bound(a: ast.alias) -> str:
    return a.asname or a.name.split(".")[0]


# ------------------------------------------------------------------------------------------------ rewrite rules (source -> edits)

def _scopes(tree: ast.Module, names: set[str]) -> tuple[bool, dict[int, ast.stmt]]:
    """(used outside function bodies?, outermost-function-by-id that uses any of `names` in its body)."""
    outside = False
    users: dict[int, ast.stmt] = {}

    def visit(n: ast.AST, fn: Optional[ast.stmt]) -> None:
        nonlocal outside
        if isinstance(n, FuncT):
            for d in (*n.decorator_list, *n.args.defaults, *[x for x in n.args.kw_defaults if x is not None]):
                visit(d, fn)
            for a in (*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs, n.args.vararg, n.args.kwarg):
                if a is not None and a.annotation is not None:
                    visit(a.annotation, fn)
            if n.returns is not None:
                visit(n.returns, fn)
            for s in n.body:
                visit(s, fn or n)
            return
        if isinstance(n, ast.Name) and n.id in names:
            if fn is None:
                outside = True
            else:
                users[id(fn)] = fn
        for c in ast.iter_child_nodes(n):
            visit(c, fn)

    for s in tree.body:
        if not (isinstance(s, (ast.Import, ast.ImportFrom)) and {_bound(a) for a in s.names} & names):
            visit(s, None)
    return outside, users


def rule_lazy_import(src: str, params: dict[str, Any]) -> Optional[str]:
    tree = _parse(src)
    if tree is None:
        return None
    strings = {c.value for c in ast.walk(tree) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.ImportFrom) and (node.module == "__future__" or node.level)):
            continue
        names = {_bound(a) for a in node.names}
        if any(a.name == "*" for a in node.names) or names & strings:
            continue
        outside, users = _scopes(tree, names)
        if outside or not users or len(users) > 8:
            continue
        funcs = list(users.values())
        if any(isinstance(g, ast.Global) and names & set(g.names) for f in funcs for g in ast.walk(f)):
            continue
        if any(a.arg in names for f in funcs if isinstance(f, FuncT) for a in (*f.args.args, *f.args.kwonlyargs)):
            continue
        lines = src.split("\n")
        for f in sorted(funcs, key=lambda f: f.lineno, reverse=True):
            assert isinstance(f, FuncT)
            used = {n.id for n in ast.walk(f) if isinstance(n, ast.Name)}
            keep = [a for a in node.names if _bound(a) in used]
            stmt = ast.unparse(ast.Import(names=keep) if isinstance(node, ast.Import)
                               else ast.ImportFrom(module=node.module, names=keep, level=0))
            first = f.body[0]
            at = (first.end_lineno or first.lineno) if (_is_doc(first) and len(f.body) > 1) else first.lineno - 1
            ref = f.body[1] if (_is_doc(first) and len(f.body) > 1) else first
            if f.lineno == ref.lineno:
                break                                               # one-line def: leave it
            lines.insert(at, " " * ref.col_offset + stmt)
        else:
            end = node.end_lineno or node.lineno
            start = node.lineno
            out = "\n".join(lines[:start - 1] + lines[end:])
            return out if _parse(out) is not None else None
    return None


def _refs(text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\b", text))


def rule_remove_unused(src: str, params: dict[str, Any], elsewhere: Optional[Callable[[str], bool]] = None) -> Optional[str]:
    tree = _parse(src)
    if tree is None:
        return None
    for node in tree.body:
        if not isinstance(node, FuncT) or node.decorator_list:
            continue
        if params.get("private_only") and not node.name.startswith("_"):
            continue
        if node.name.startswith("__") or node.name == "main":
            continue
        start, end = node.lineno, node.end_lineno or node.lineno
        lines = src.split("\n")
        rest = "\n".join(lines[:start - 1] + lines[end:])
        if _refs(rest, node.name) or (elsewhere is not None and elsewhere(node.name)):
            continue
        out = "\n".join(lines[:start - 1] + lines[end:])
        out = re.sub(r"\n{4,}", "\n\n\n", out)
        return out if _parse(out) is not None else None
    return None


def _has_call(n: ast.AST) -> bool:
    return any(isinstance(x, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)) for x in ast.walk(n))


def _stmt_lists(tree: ast.Module) -> list[list[ast.stmt]]:
    out: list[list[ast.stmt]] = []
    for n in ast.walk(tree):
        if isinstance(n, FuncT):
            out.append(n.body)
    return out


def rule_inline_temp(src: str, params: dict[str, Any]) -> Optional[str]:
    tree = _parse(src)
    if tree is None:
        return None
    lines = src.split("\n")
    for fn in ast.walk(tree):
        if not isinstance(fn, FuncT):
            continue
        loads: dict[str, int] = {}
        stores: dict[str, int] = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                (loads if isinstance(n.ctx, ast.Load) else stores)[n.id] = (loads if isinstance(n.ctx, ast.Load) else stores).get(n.id, 0) + 1
        if any(isinstance(g, (ast.Global, ast.Nonlocal)) for g in ast.walk(fn)):
            continue
        for body in [fn.body]:
            for i in range(len(body) - 1):
                a, b = body[i], body[i + 1]
                if not (isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name)):
                    continue
                t = a.targets[0].id
                if loads.get(t) != 1 or stores.get(t) != 1 or not isinstance(b, (ast.Return, ast.Assign, ast.Expr)):
                    continue
                if (a.end_lineno or 0) != a.lineno or b.lineno != (b.end_lineno or 0):
                    continue
                uses = [n for n in ast.walk(b) if isinstance(n, ast.Name) and n.id == t and isinstance(n.ctx, ast.Load)]
                if len(uses) != 1 or any(isinstance(n, ast.Name) and n.id == t and isinstance(n.ctx, ast.Store) for n in ast.walk(b)):
                    continue
                if _has_call(b) and not (isinstance(b, ast.Return) and isinstance(b.value, ast.Name)):
                    continue                                        # evaluation order of calls would change
                if isinstance(a.value, (ast.Yield, ast.Await, ast.YieldFrom, ast.NamedExpr)):
                    continue
                u = uses[0]
                if u.lineno != u.end_lineno or a.lineno + 1 != b.lineno:
                    continue
                seg = ast.get_source_segment(src, a.value)
                if seg is None or "\n" in seg:
                    continue
                if not isinstance(a.value, (ast.Name, ast.Attribute, ast.Constant, ast.Call, ast.Subscript)):
                    seg = f"({seg})"
                line = lines[u.lineno - 1].encode("utf-8")
                new_line = (line[:u.col_offset] + seg.encode("utf-8") + line[u.end_col_offset or u.col_offset:]).decode("utf-8")
                out = "\n".join(lines[:a.lineno - 1] + [new_line] + lines[b.lineno:])
                if _parse(out) is not None:
                    return out
    return None


def _guard_const(p: str, c: ast.Constant) -> str:
    return f"if not {p}:\n    return {c.value!r}"


def rule_empty_guard(src: str, params: dict[str, Any]) -> Optional[str]:
    tree = _parse(src)
    if tree is None:
        return None
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = fn.body[1:] if _is_doc(fn.body[0]) else fn.body
        if len(body) != 2:
            continue
        loop, ret = body
        if not isinstance(loop, ast.For) or not isinstance(ret, ast.Return):
            continue
        if not isinstance(loop.iter, ast.Name) or loop.orelse or not isinstance(ret.value, ast.Constant):
            continue
        argn = [a.arg for a in fn.args.args if a.arg != "self"]
        p = loop.iter.id
        if p not in argn or (params.get("param_index") is not None and argn.index(p) != params["param_index"]):
            continue
        if params.get("const") is not None and repr(ret.value.value) != params["const"]:
            continue
        if fn.lineno == loop.lineno:
            continue
        lines = src.split("\n")
        ind = " " * loop.col_offset
        g = [ind + "if not " + p + ":", ind + "    return " + repr(ret.value.value)]
        out = "\n".join(lines[:loop.lineno - 1] + g + lines[loop.lineno - 1:])
        if _parse(out) is not None:
            return out
    return None


RULES: dict[str, Callable[..., Optional[str]]] = {
    "lazy_import": rule_lazy_import, "remove_unused": rule_remove_unused, "inline_temp": rule_inline_temp,
    "empty_guard": rule_empty_guard,
}


# ------------------------------------------------------------------------------------------------ inference (before, after) -> examples

def _dump(n: ast.AST) -> str:
    return ast.dump(n, include_attributes=False)


def infer(before: str, after: str) -> list[tuple[str, dict[str, Any], str]]:
    """Recognise (kind, concrete params, skeleton) for every abstractable change from `before` to `after`."""
    tb, ta = _parse(before), _parse(after)
    if tb is None or ta is None:
        return []
    out: list[tuple[str, dict[str, Any], str]] = []
    top_b = [n for n in tb.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    top_a = {_dump(n) for n in ta.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    inner_a = {_dump(n) for f in ast.walk(ta) if isinstance(f, FuncT) for n in f.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    inner_b = {_dump(n) for f in ast.walk(tb) if isinstance(f, FuncT) for n in f.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    for n in top_b:
        names = {_bound(a) for a in n.names}
        moved = {_dump(ast.Import(names=[a]) if isinstance(n, ast.Import) else ast.ImportFrom(module=n.module, names=[a], level=n.level))
                 for a in n.names}
        if _dump(n) not in top_a and (_dump(n) in inner_a or moved & inner_a) and not (_dump(n) in inner_b):
            out.append(("lazy_import", {}, f"import $M  ->  def $F: import $M   [{', '.join(sorted(names))}]"))
    fa = {n.name: n for n in ta.body if isinstance(n, FuncT)}
    fb = {n.name: n for n in tb.body if isinstance(n, FuncT)}
    for name, node in fb.items():
        if name not in fa and not any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(ta)):
            out.append(("remove_unused", {"private_only": name.startswith("_")}, "def $F(...): ... (unreferenced)  ->  (deleted)"))
    for name, f in fb.items():
        g = fa.get(name)
        if g is None or _dump(f) == _dump(g):
            continue
        if len(g.body) == len(f.body) - 1 and rule_inline_temp(ast.unparse(f), {}) is not None:
            if _dump(_parse(rule_inline_temp(ast.unparse(f), {}) or "") or f) == _dump(ast.Module(body=[g], type_ignores=[])):
                out.append(("inline_temp", {}, "$T = $E; <stmt using $T once>  ->  <stmt using $E>"))
        if len(g.body) == len(f.body) + 1:
            first = g.body[1] if _is_doc(g.body[0]) else g.body[0]
            if isinstance(first, ast.If) and isinstance(first.test, ast.UnaryOp) and isinstance(first.test.op, ast.Not) \
                    and isinstance(first.test.operand, ast.Name) and len(first.body) == 1 and isinstance(first.body[0], ast.Return) \
                    and isinstance(first.body[0].value, ast.Constant):
                argn = [a.arg for a in g.args.args if a.arg != "self"]
                p = first.test.operand.id
                if p in argn:
                    out.append(("empty_guard", {"param_index": argn.index(p), "const": repr(first.body[0].value.value)},
                                "def $F($P, ...): <body>  ->  if not $P: return $C; <body>"))
    return out


# ------------------------------------------------------------------------------------------------ the student

class LessonStudent:
    name = "self-student-v1"
    steps = ("efficiency", "no_stubs", "integrated")

    def __init__(self, lessons_path: Path, min_support: int = 1, max_templates: int = 6, max_files: int = 12) -> None:
        self.lessons_path, self.min_support = Path(lessons_path), min_support
        self.max_templates, self.max_files = max_templates, max_files
        self.templates: dict[str, dict[str, Any]] = {}
        self.skipped = 0                                            # corrupt or non-adopted lines
        self.lessons_seen = 0
        self.reload()

    # -- learning
    def reload(self) -> int:
        self.templates, self.skipped, self.lessons_seen = {}, 0, 0
        if not self.lessons_path.is_file():
            return 0
        rows: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for line in self.lessons_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                les = json.loads(line)
                if not isinstance(les, dict):
                    raise ValueError("not an object")
            except ValueError:
                self.skipped += 1
                continue
            if les.get("event") == "outcome":                       # creator.curriculum.LessonLog: the verdict arrives later
                tgt = by_id.get(str(les.get("lesson_id")))
                if tgt is not None:
                    tgt["adopted"] = les.get("adopted")
                continue
            rows.append(les)
            if les.get("lesson_id") is not None:
                by_id[str(les["lesson_id"])] = les
        for les in rows:
            if les.get("adopted") is not True:
                self.skipped += 1
                continue
            self.learn(les)
        return len(self.templates)

    def learn(self, les: dict[str, Any]) -> int:
        before, after = les.get("files_before") or {}, les.get("files_after") or {}
        if not isinstance(before, dict) or not isinstance(after, dict):
            return 0
        self.lessons_seen += 1
        lid = str(les.get("lesson_id") or les.get("package_id") or self.lessons_seen)
        tk = str(les.get("task_kind") or "")
        n = 0
        for path, b in before.items():
            a = after.get(path)
            if not (isinstance(b, str) and isinstance(a, str)) or a == b or not str(path).endswith(".py"):
                continue
            for kind, params, skeleton in infer(b, a):
                self._merge(kind, params, skeleton, lid, tk)
                n += 1
        return n

    def _merge(self, kind: str, params: dict[str, Any], skeleton: str, lid: str, tk: str) -> None:
        t = self.templates.get(kind)
        if t is None:
            self.templates[kind] = {"kind": kind, "params": dict(params), "skeleton": skeleton, "lessons": [lid],
                                    "task_kinds": [tk] if tk else [], "useful": {tk: 1} if tk else {}}
            return
        for k, v in params.items():                                 # anti-unify: disagreement becomes a hole (None)
            if t["params"].get(k) != v:
                t["params"][k] = None
        if lid not in t["lessons"]:
            t["lessons"].append(lid)
        if tk:
            if tk not in t["task_kinds"]:
                t["task_kinds"].append(tk)
            t["useful"][tk] = t["useful"].get(tk, 0) + 1

    def support(self, kind: str) -> int:
        return len(self.templates[kind]["lessons"]) if kind in self.templates else 0

    def learned(self) -> list[dict[str, Any]]:
        return sorted(self.templates.values(), key=lambda t: -len(t["lessons"]))

    # -- use
    def can_attempt(self, task: Any) -> bool:
        tk = task if isinstance(task, str) else str(getattr(task, "task_kind", None) or (task.get("task_kind") if isinstance(task, dict) else "") or "")
        return any(len(t["lessons"]) >= self.min_support and (not tk or tk in t["task_kinds"]) for t in self.templates.values())

    def _candidates(self, package: Any, workdir: Path) -> list[Path]:
        outs = [x for x in getattr(package, "outputs", ()) or () if str(x).endswith(".py") and not Path(str(x)).name.startswith("test_")]
        return [workdir / x for x in outs if (workdir / x).is_file()][: self.max_files]

    def apply_source(self, src: str, tk: str = "", elsewhere: Optional[Callable[[str], bool]] = None) -> tuple[str, list[str]]:
        """Apply the best-supported matching templates to one source text; returns (new source, applied kinds)."""
        done: list[str] = []
        ranked = [t for t in self.learned() if len(t["lessons"]) >= self.min_support][: self.max_templates]
        if tk:
            ranked.sort(key=lambda t: (-t["useful"].get(tk, 0), -len(t["lessons"])))
        for t in ranked:
            for _ in range(20):
                rule = RULES[t["kind"]]
                new = rule(src, t["params"], elsewhere) if t["kind"] == "remove_unused" else rule(src, t["params"])
                if new is None or new == src or _parse(new) is None:
                    break
                src = new
                done.append(t["kind"])
        return src, done

    def __call__(self, plan: Any, package: Any, workdir: Path) -> "WorkResult":
        from creator.kernel import WorkResult
        from creator.curriculum import task_kind
        tk = str(getattr(package, "task_kind", "") or (task_kind(plan) if plan is not None else ""))   # packages carry no kind
        workdir = Path(workdir)
        applied: list[str] = []
        texts: Optional[list[tuple[str, str]]] = None
        for p in self._candidates(package, workdir):
            rel = p.relative_to(workdir).as_posix()
            if texts is None:
                texts = [(q.relative_to(workdir).as_posix(), q.read_text(encoding="utf-8", errors="replace"))
                         for q in workdir.rglob("*.py") if "__pycache__" not in q.parts and ".venv" not in q.parts][:3000]
            others = [t for r, t in texts if r != rel]
            start = p.read_text(encoding="utf-8")
            new, done = self.apply_source(start, tk, lambda name: any(_refs(t, name) for t in others))
            if done and new != start:
                eol = "\r\n" if b"\r\n" in p.read_bytes() else "\n"        # keep the file's line endings: no whole-file diff
                p.write_text(new, encoding="utf-8", newline=eol)
                applied.extend(f"{k}:{rel}" for k in done)
        if not applied:
            return WorkResult(False, "no learned template applies")
        return WorkResult(True, f"{len(applied)} learned rewrites: {', '.join(applied[:12])}", by=self.name)
