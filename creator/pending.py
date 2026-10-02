"""Resume: finished work that was never measured gets measured (Nupen's #1 constraint is learning signal; 2 Oct 2026).

Two sources of unmeasured finished work: state/creator/pending/*.patch (sandbox diffs saved when a cycle was pulled back for RAM,
cancelled by a pause or hit an error) and lessons whose cycle never produced a verdict (teacher OR student). `replay_student` only
replays a teacher lesson on byte-identical files; files drift. ResumeStudent ('nupen-resume-v1') re-applies such work to the CURRENT
tree for a package whose objective / files match, in three steps:

  1. exact:    `git apply --check` (a lesson: the files equal what the solver started from) - then apply;
  2. function: for each function / class the old change touched, when the current version equals the old 'before' version (AST
               equality, by qualified name) replace it with the old 'after' version; otherwise that hunk is skipped;
  3. the result must parse; the hunks applied and skipped are recorded in the reasoning.

It judges nothing: the kernel measures. Credit goes to the ORIGINAL solver (WorkResult.by = original solver; teacher work stays
curriculum.TEACHER work, a student's resumed work counts to that student). A pending item is retired once measured (pending/measured/
with the verdict) or after MAX_FAILS failed re-applications (pending/stale/ with why)."""
from __future__ import annotations

import ast
import copy
import dataclasses
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from creator import curriculum as C

NAME = "nupen-resume-v1"
MAX_FAILS = 3
TRIES_PER_CALL = 3


@dataclasses.dataclass
class Item:
    item_id: str                                    # "patch:<file name>" or "lesson:<lesson_id>"
    package_id: str
    solver: str                                     # the original solver (credit)
    objective: str = ""
    files: tuple[str, ...] = ()
    patch: str = ""                                 # unified diff (patch items)
    before: dict[str, str] = dataclasses.field(default_factory=dict)   # lesson items: path -> text before ("" when new)
    after: dict[str, str] = dataclasses.field(default_factory=dict)
    reasoning: str = ""
    source: Optional[Path] = None


@dataclasses.dataclass
class Result:
    ok: bool
    mode: str                                       # exact | function | none
    applied: list[str] = dataclasses.field(default_factory=list)
    skipped: list[str] = dataclasses.field(default_factory=list)
    why: str = ""
    files: dict[str, str] = dataclasses.field(default_factory=dict)    # path -> new text (function level; written by the caller)


def _git(cwd: Path, *args: str, stdin: Optional[str] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _norm(t: str) -> str:
    return t.replace("\r\n", "\n")


def _split_patch(patch: str) -> dict[str, str]:
    """path -> that file's part of the diff (the b/ path)."""
    parts: dict[str, str] = {}
    cur: list[str] = []
    for ln in patch.splitlines(keepends=True):
        if ln.startswith("diff --git "):
            if cur:
                parts[_part_path(cur[0])] = "".join(cur)
            cur = []
        cur.append(ln)
    if cur and cur[0].startswith("diff --git "):
        parts[_part_path(cur[0])] = "".join(cur)
    return parts


def _part_path(header: str) -> str:
    return header.rstrip("\n").split(" b/", 1)[-1]


# ------------------------------------------------------------------------------------------------ items

def _lesson_unmeasured(les: C.Lesson) -> bool:
    return bool(les.claimed_done and les.files_after and (les.adopted is None or not C.is_skill_signal(les)))


def _state(pending: Path, sub: str) -> Path:
    return pending / sub


def _retired(pending: Path) -> set[str]:
    out: set[str] = set()
    for sub in ("measured", "stale"):
        d = _state(pending, sub)
        if d.is_dir():
            out.update(p.name for p in d.iterdir())
    return out


def _retired_id(item_id: str) -> str:
    return item_id.split(":", 1)[1] + (".json" if item_id.startswith("lesson:") else "")


def load_items(pending: Path, lessons_path: Path) -> list[Item]:
    """Every pending patch and unmeasured lesson not yet retired, newest lessons first."""
    log = C.LessonLog(Path(lessons_path))
    lessons = log.lessons()
    by_pkg = {les.package_id: les for les in lessons}
    done = _retired(pending)
    items: list[Item] = []
    if pending.is_dir():
        for p in sorted(pending.glob("*.patch")):
            if p.name in done:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            pkg = p.name.split("_", 1)[0]
            orig = by_pkg.get(pkg)
            items.append(Item(f"patch:{p.name}", pkg, orig.solver if orig else C.CLAUDE, orig.objective if orig else "",
                              tuple(_split_patch(text)), patch=text, source=p))
    for les in reversed(lessons):
        if _lesson_unmeasured(les) and f"{les.lesson_id}.json" not in done:
            items.append(Item(f"lesson:{les.lesson_id}", les.package_id, les.solver, les.objective, tuple(les.files_after),
                              before=les.files_before, after=les.files_after, reasoning=les.reasoning))
    return items


def matches(item: Item, objective: str, package_id: str = "", targets: tuple[str, ...] = ()) -> bool:
    """The item answers this package: same objective, same package id, or it touches only files the package works on."""
    if objective and item.objective == objective:
        return True
    if package_id and item.package_id == package_id:
        return True
    code = [f for f in item.files if not f.startswith("tests/")]
    return bool(targets and code and set(code) <= set(targets))


def related(item: Item, objective: str, component: str = "") -> bool:
    """Cheap pre-check for Student.can_attempt (no package files known yet)."""
    if objective and item.objective == objective:
        return True
    return bool(component and any(component in f or f.endswith(component) for f in item.files))


# ------------------------------------------------------------------------------------------------ patch reconstruction

def _blob(repos: list[Path], ref: str) -> Optional[str]:
    if not ref.strip("0"):
        return ""
    for r in repos:
        cp = subprocess.run(["git", "cat-file", "blob", ref], cwd=r, capture_output=True)
        if cp.returncode == 0:
            return cp.stdout.decode("utf-8", errors="replace")
    return None


def _before_after(part: str, repos: list[Path]) -> Optional[tuple[str, str]]:
    """(before, after) text of one file's diff: the before blob named in the 'index' line, the after by applying the diff to it."""
    ref = next((ln.split()[1].split("..")[0] for ln in part.splitlines() if ln.startswith("index ")), "")
    if not ref or "Binary files" in part or "\ndeleted file" in part:
        return None
    before = _blob(repos, ref)
    if before is None:
        return None
    rel = _part_path(part.splitlines()[0])
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        if before:
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            (t / rel).write_bytes(before.encode("utf-8"))
        cp = subprocess.run(["git", "apply", "-"], input=part.encode("utf-8"), capture_output=True, cwd=td)
        if cp.returncode != 0 or not (t / rel).is_file():
            return None
        return before, (t / rel).read_bytes().decode("utf-8", errors="replace")


# ------------------------------------------------------------------------------------------------ function-level re-application

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _dump(n: ast.AST) -> str:
    return ast.dump(n, include_attributes=False)


def _scope_defs(body: list[ast.stmt]) -> dict[str, ast.AST]:
    return {s.name: s for s in body if isinstance(s, _DEFS)}


def _shell(n: ast.AST) -> str:
    """A class without its method/class children (the part a function-level replace cannot reach)."""
    if isinstance(n, ast.ClassDef):
        c = copy.copy(n)
        c.body = [s for s in n.body if not isinstance(s, _DEFS)]
        return _dump(c)
    return _dump(n)


def _diff_units(b: list[ast.stmt], a: list[ast.stmt], prefix: str, out: list[tuple[str, Optional[ast.AST], Optional[ast.AST]]]) -> None:
    bd, ad = _scope_defs(b), _scope_defs(a)
    for name in list(dict.fromkeys([*bd, *ad])):
        bn, an = bd.get(name), ad.get(name)
        q = f"{prefix}{name}"
        if bn is None or an is None or type(bn) is not type(an):
            out.append((q, bn, an))
        elif _dump(bn) != _dump(an):
            if isinstance(bn, ast.ClassDef) and isinstance(an, ast.ClassDef) and _shell(bn) == _shell(an):
                _diff_units(bn.body, an.body, q + ".", out)
            else:
                out.append((q, bn, an))


def _find(body: list[ast.stmt], qname: str) -> tuple[Optional[ast.AST], Optional[ast.AST]]:
    """(node, parent scope node or None for module) for a qualified name."""
    parent: Optional[ast.AST] = None
    cur_body, node = body, None
    for part in qname.split("."):
        node = _scope_defs(cur_body).get(part)
        if node is None:
            return None, parent
        parent = node
        cur_body = node.body if isinstance(node, ast.ClassDef) else []
    return node, None


def _span(n: ast.AST) -> tuple[int, int]:
    decos = getattr(n, "decorator_list", [])
    start = min([n.lineno, *[d.lineno for d in decos]])           # type: ignore[attr-defined]
    return start, n.end_lineno or n.lineno                         # type: ignore[attr-defined]


def _text(src_lines: list[str], n: ast.AST) -> list[str]:
    s, e = _span(n)
    return src_lines[s - 1:e]


def _reindent(lines: list[str], delta: int) -> list[str]:
    if delta == 0:
        return lines
    return [(" " * delta + ln) if delta > 0 and ln.strip() else (ln[-delta:] if delta < 0 and ln[:-delta].strip() == "" else ln)
            for ln in lines]


def _imports(tree: ast.Module) -> list[str]:
    return [ast.unparse(s) for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))]


def function_level(current: str, before: str, after: str, rel: str = "") -> tuple[Optional[str], list[str], list[str]]:
    """Re-apply the old before->after change function by function. -> (new text or None, applied hunks, skipped hunks)."""
    applied: list[str] = []
    skipped: list[str] = []
    try:
        tb, ta, tc = ast.parse(before), ast.parse(after), ast.parse(current)
    except SyntaxError as e:
        return None, [], [f"{rel}: does not parse ({e.msg})"]
    units: list[tuple[str, Optional[ast.AST], Optional[ast.AST]]] = []
    _diff_units(tb.body, ta.body, "", units)
    cur = _norm(current)
    for q, bn, an in units:
        tc = ast.parse(cur)
        lines = cur.split("\n")
        node, _ = _find(tc.body, q)
        tag = f"{rel}::{q}"
        if bn is None and an is not None:                                    # added by the old change
            if node is not None:
                skipped.append(f"{tag} (added: a definition of that name exists now)")
                continue
            if "." in q:
                par, _ = _find(tc.body, q.rsplit(".", 1)[0])
                if not isinstance(par, ast.ClassDef):
                    skipped.append(f"{tag} (added: enclosing class missing)")
                    continue
                at, ind = (par.end_lineno or par.lineno), par.body[0].col_offset
            else:
                at, ind = len(lines), 0
                while at > 0 and not lines[at - 1].strip():
                    at -= 1
            seg = _reindent(_text(_norm(after).split("\n"), an), ind - an.col_offset)       # type: ignore[attr-defined]
            gap = [""] if ind == 0 else [""]
            lines[at:at] = [*gap, *([""] if ind == 0 else []), *seg]
            cur = "\n".join(lines)
        elif an is None and bn is not None:                                  # removed by the old change
            if node is None or _dump(node) != _dump(bn):
                skipped.append(f"{tag} (removed: current differs from the old before)")
                continue
            s, en = _span(node)
            del lines[s - 1:en]
            cur = "\n".join(lines)
        elif bn is not None and an is not None:
            if node is None or type(node) is not type(bn) or _dump(node) != _dump(bn):
                skipped.append(f"{tag} (current version is not the old before version)")
                continue
            s, en = _span(node)
            seg = _reindent(_text(_norm(after).split("\n"), an), node.col_offset - an.col_offset)    # type: ignore[attr-defined]
            lines[s - 1:en] = seg
            cur = "\n".join(lines)
        else:
            continue
        applied.append(tag)
    new_imports = [i for i in _imports(ta) if i not in set(_imports(tb))]          # imports the old change added
    if new_imports and applied:
        tc = ast.parse(cur)
        have = set(_imports(tc))
        add = [i for i in new_imports if i not in have]
        if add:
            lines = cur.split("\n")
            last = max([s.end_lineno or s.lineno for s in tc.body if isinstance(s, (ast.Import, ast.ImportFrom))] or [0])
            if not last:
                doc = tc.body[0] if tc.body and isinstance(tc.body[0], ast.Expr) else None
                last = (doc.end_lineno or doc.lineno) if doc else 0
            lines[last:last] = add
            cur = "\n".join(lines)
            applied.append(f"{rel}::imports(+{len(add)})")
    try:
        ast.parse(cur)
    except SyntaxError as e:
        return None, [], [*skipped, f"{rel}: result does not parse ({e.msg})"]
    return (cur if applied else None), applied, skipped


# ------------------------------------------------------------------------------------------------ applying one item to a tree

def try_apply(item: Item, workdir: Path, write: bool = True, repos: Optional[list[Path]] = None) -> Result:
    """Exact, then function-level. With write=False nothing in workdir changes (the report path)."""
    workdir = Path(workdir)
    root = workdir.resolve()
    for rel in item.files:
        if Path(rel).is_absolute() or not (root / rel).resolve().is_relative_to(root):
            return Result(False, "none", why=f"names a path outside the work directory: {rel!r}")
    if item.item_id.startswith("patch:"):
        res = _apply_patch(item, workdir, write, repos or [workdir])
    else:
        res = _apply_lesson(item, workdir, write)
    return res


def _read(workdir: Path, rel: str) -> Optional[str]:
    p = workdir / rel
    return _norm(p.read_bytes().decode("utf-8", errors="replace")) if p.is_file() else None


def _write(workdir: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))


def _parses(files: dict[str, str]) -> Optional[str]:
    for rel, text in files.items():
        if rel.endswith(".py"):
            try:
                ast.parse(text)
            except SyntaxError as e:
                return f"{rel} does not parse ({e.msg})"
    return None


def _apply_patch(item: Item, workdir: Path, write: bool, repos: list[Path]) -> Result:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8", newline="\n") as fh:
        fh.write(item.patch)
        pf = fh.name
    try:
        if not item.patch.strip() or not item.files:
            return Result(False, "none", why="empty diff")
        if _git(workdir, "apply", "--check", "-R", pf).returncode == 0:
            return Result(False, "none", why="already in the current tree")
        if _git(workdir, "apply", "--check", pf).returncode == 0:
            if write:
                cp = _git(workdir, "apply", pf)
                if cp.returncode != 0:
                    return Result(False, "none", why=f"git apply failed: {cp.stderr[:200]}")
            return Result(True, "exact", applied=list(item.files))
    finally:
        Path(pf).unlink(missing_ok=True)
    out: dict[str, str] = {}
    applied: list[str] = []
    skipped: list[str] = []
    for rel, part in _split_patch(item.patch).items():
        if not rel.endswith(".py"):
            skipped.append(f"{rel} (not python: exact apply only)")
            continue
        ba = _before_after(part, repos)
        cur = _read(workdir, rel)
        if ba is None or cur is None:
            skipped.append(f"{rel} ({'old blob unavailable' if ba is None else 'file is gone'})")
            continue
        new, ap, sk = function_level(cur, ba[0], ba[1], rel)
        applied += ap
        skipped += sk
        if new is not None:
            out[rel] = new
    return _finish(out, applied, skipped, workdir, write)


def _apply_lesson(item: Item, workdir: Path, write: bool) -> Result:
    if item.after and all(_read(workdir, r) == _norm(t) for r, t in item.after.items()):
        return Result(False, "none", why="already in the current tree")
    if all((_read(workdir, r) or "") == _norm(t) and (t != "" or _read(workdir, r) is None) for r, t in item.before.items()):
        if _parses(item.after) is None:
            if write:
                _write(workdir, item.after)
            return Result(True, "exact", applied=list(item.after))
    out: dict[str, str] = {}
    applied: list[str] = []
    skipped: list[str] = []
    for rel, after in item.after.items():
        before, cur = item.before.get(rel, ""), _read(workdir, rel)
        if cur is None and before == "":
            out[rel] = after                                                  # a new file that is still absent
            applied.append(f"{rel} (new file)")
        elif not rel.endswith(".py") or cur is None or before == "":
            skipped.append(f"{rel} (cannot re-apply: exists now / not python)")
        else:
            new, ap, sk = function_level(cur, before, after, rel)
            applied += ap
            skipped += sk
            if new is not None:
                out[rel] = new
    return _finish(out, applied, skipped, workdir, write)


def _finish(out: dict[str, str], applied: list[str], skipped: list[str], workdir: Path, write: bool) -> Result:
    bad = _parses(out)
    if bad:
        return Result(False, "none", skipped=skipped, why=bad)
    if not out:
        return Result(False, "none", skipped=skipped, why="no hunk applies to the current tree")
    if write:
        _write(workdir, out)
    return Result(True, "function", applied=applied, skipped=skipped, files=out)


# ------------------------------------------------------------------------------------------------ retiring

def _attempts_path(pending: Path) -> Path:
    return pending / "attempts.json"


def _load_attempts(pending: Path) -> dict[str, Any]:
    try:
        return dict(json.loads(_attempts_path(pending).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def retire(pending: Path, item: Item, sub: str, why: str) -> None:
    """Move the item to pending/<sub>/ (measured or stale) with the verdict / reason beside it."""
    d = pending / sub
    d.mkdir(parents=True, exist_ok=True)
    meta = {"item": item.item_id, "package": item.package_id, "solver": item.solver, "why": why}
    if item.source is not None and item.source.is_file():
        shutil.move(str(item.source), str(d / item.source.name))
        (d / (item.source.name + ".json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    else:
        (d / _retired_id(item.item_id)).write_text(json.dumps(meta, indent=1), encoding="utf-8")


def record_failure(pending: Path, item: Item, why: str) -> bool:
    """Count a failed re-application; at MAX_FAILS the item goes to stale/. -> True when retired."""
    pending.mkdir(parents=True, exist_ok=True)
    att = _load_attempts(pending)
    rec = att.setdefault(item.item_id, {"fails": 0, "why": []})
    rec["fails"] += 1
    rec["why"] = [*rec["why"], why[:300]][-MAX_FAILS:]
    _attempts_path(pending).write_text(json.dumps(att, indent=1), encoding="utf-8")
    if rec["fails"] >= MAX_FAILS:
        retire(pending, item, "stale", f"{rec['fails']} failed re-applications: " + " | ".join(rec["why"]))
        return True
    return False


# ------------------------------------------------------------------------------------------------ the student

class ResumeStudent:
    """nupen-resume-v1. Its work is credited to the original solver (credits_original), see curriculum._StudentStep."""
    name = NAME
    credits_original = True

    def __init__(self, lessons_path: Path, pending_dir: Optional[Path] = None, repos: Optional[list[Path]] = None) -> None:
        self.lessons_path = Path(lessons_path)
        self.pending = Path(pending_dir) if pending_dir is not None else self.lessons_path.parent / "pending"
        self.repos = repos or []
        self._open: dict[str, Item] = {}                       # package_id -> the item being measured

    def can_attempt(self, task: Any) -> bool:
        obj, comp = str(getattr(task, "objective", "")), str(getattr(task, "component", ""))
        return any(related(i, obj, comp) for i in load_items(self.pending, self.lessons_path))

    def __call__(self, plan: Any, package: Any, workdir: Path) -> Any:
        from creator.kernel import WorkResult
        pid = str(getattr(plan, "package_id", ""))
        obj = str(getattr(package, "objective", ""))
        targets = tuple(str(x) for x in (*getattr(package, "outputs", ()), *getattr(package, "inputs", ())))
        cands = [i for i in load_items(self.pending, self.lessons_path) if matches(i, obj, pid, targets)]
        cands.sort(key=lambda i: (i.objective != obj, i.package_id != pid))          # the same objective / package first (stable: newest first)
        if not cands:
            return WorkResult(False, "no pending or unmeasured solution matches this package")
        why = []
        for item in cands[:TRIES_PER_CALL]:
            res = try_apply(item, Path(workdir), True, [Path(workdir), *self.repos])
            if res.ok:
                self._open[pid] = item
                note = (f"resumed {item.item_id} (original package {item.package_id}, solver {item.solver}) as {res.mode}: "
                        f"applied {len(res.applied)} [{'; '.join(res.applied)[:600]}], skipped {len(res.skipped)} "
                        f"[{'; '.join(res.skipped)[:400]}]")
                by = C.REPLAY if item.solver in C.TEACHER else item.solver       # credit: the original solver
                return WorkResult(True, note, by=by, reasoning=f"{note}\n{item.reasoning}"[:4000])
            record_failure(self.pending, item, res.why)
            why.append(f"{item.item_id}: {res.why}")
        return WorkResult(False, "no pending solution re-applies: " + " | ".join(why))

    def resolved(self, package_id: str, adopted: Optional[bool], verdict: str) -> None:
        """The kernel judged the cycle: the resumed item is measured (adopted or rejected) and retired. No verdict: stays."""
        item = self._open.pop(package_id, None)
        if item is not None and adopted is not None:
            retire(self.pending, item, "measured", f"{'ADOPTED' if adopted else 'REJECTED'}: {verdict}"[:600])
