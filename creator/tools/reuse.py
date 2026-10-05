"""Reuse: retrieve proven code before the CODER writes a function (R3, lever "tokens not generated at all").

An index of module-level functions (signature, docstring, body hash, the tests that mention it) over this repo and the permissively licensed
public code under ~/creator_runtime/public_repos (source repo + licence recorded; a repo without a permissive licence is skipped). Query = the
task's signature + docstring -> top-k candidates with a similarity score in 0..1. Two uses:
  adapt  - the best candidate, renamed to the task's signature, becomes the starting code the CODER edits (a diff, not a rewrite)
  direct - the unmodified-but-renamed candidate runs the hidden-test-free checks (static + the task's visible examples) first; a pass
           means no model call was needed.
LEAKAGE GUARD: LeakGuard excludes every file the held-out tasks were derived from (by source path), every function whose normalized body or
signature+docstring matches one of them (copies, other versions) and every file touched by a codetrust held-out commit. The index builder
applies it while building and `assert_clean` re-checks a finished index; a test asserts both.
Lazy: nothing is imported or built until a query; the index is incremental per file (sha) and lives in the usual index cache dir."""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

SCHEMA = 2
MIN_LINES, MAX_LINES = 3, 80
MAX_FILE_BYTES = 300_000
PER_REPO_CAP = 25_000
SKIP_DIRS = frozenset("tests test testing docs doc examples example benchmarks bench build dist node_modules migrations vendor vendored "
                      "_vendor third_party site-packages .git .venv venv __pycache__ ".split())
PERMISSIVE = (("PSF", r"python software foundation license|psf license agreement"), ("MIT", r"permission is hereby granted, free of charge"), ("BSD", r"redistribution and use in source and binary forms"),
              ("Apache-2.0", r"apache license\s*,?\s*version 2\.0|apache license\s+version 2"), ("ISC", r"permission to use, copy, modify, and/or distribute"),
              ("PSF", r"python software foundation license|psf license agreement"), ("Unlicense", r"this is free and unencumbered software"),
              ("Zlib", r"this software is provided 'as-is'"))
COPYLEFT = re.compile(r"gnu (affero |lesser )?general public license|mozilla public license", re.I)
_STOP = frozenset("a an and the of to for in on with from by is are be it this that as at or not into if its then than each given "
                  "return returns returned function value values any all".split())
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BUILTINS = frozenset(dir(builtins)) | {"__name__", "__file__"}


# ------------------------------------------------------------------------------------------------------------ text helpers
def _split(name: str) -> list[str]:
    words: list[str] = []
    for p in re.split(r"[._]+", name):
        words += [w.lower() for w in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", p)]
    return [w for w in words if w]


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text):
        out += [_stem(x) for x in _split(w) if len(x) > 1 and x not in _STOP]
    return out


def _strip_doc(node: ast.AST) -> list[ast.stmt]:
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def body_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Normalized body: docstring, function name, parameter names and annotations dropped, so a renamed or re-documented copy hashes the same."""
    import copy
    ren = {a.arg: f"_a{i}" for i, a in enumerate(node.args.posonlyargs + node.args.args + node.args.kwonlyargs)}
    mod = ast.Module(body=[copy.deepcopy(s) for s in _strip_doc(node)], type_ignores=[])
    for n in ast.walk(mod):
        if isinstance(n, ast.Name) and n.id in ren:
            n.id = ren[n.id]
    return hashlib.sha1((str(len(ren)) + "|" + ast.dump(mod, annotate_fields=False, include_attributes=False)).encode()).hexdigest()[:20]


def sigdoc_hash(name: str, args: Sequence[str], doc: str) -> str:
    return hashlib.sha1((name + "|" + ",".join(args) + "|" + " ".join(doc.split())).encode()).hexdigest()[:20]


def _args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = node.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs if x.arg not in ("self", "cls")]


def licence_of(repo: Path) -> Optional[str]:
    """Permissive licence name from the repo's licence file, or None (no file, copyleft or unrecognised -> the repo is skipped)."""
    for p in sorted(repo.glob("LICEN*")) + sorted(repo.glob("COPYING*")):
        if p.is_file():
            try:
                t = p.read_text(encoding="utf-8", errors="replace")[:20000]
            except OSError:
                continue
            low = " ".join(t.lower().split())
            if COPYLEFT.search(low[:400]) and "apache" not in low[:400]:
                return None
            for name, pat in PERMISSIVE:
                if re.search(pat, low):
                    return name
    return None


# ------------------------------------------------------------------------------------------------------------ leakage guard
@dataclass
class LeakGuard:
    """What the held-out suite was derived from. `files` are (source, path) pairs, `names` are (source, path, name) of the derived functions."""
    files: set[tuple[str, str]] = field(default_factory=set)
    hashes: set[str] = field(default_factory=set)
    sigdocs: set[str] = field(default_factory=set)
    heldout_files: set[str] = field(default_factory=set)
    task_names: set[str] = field(default_factory=set)

    def file_blocked(self, source: str, path: str) -> bool:
        return (source, path) in self.files or (source == "self" and path in self.heldout_files) or path.startswith("minishop/")

    def func_blocked(self, h: str, sd: str) -> bool:
        return h in self.hashes or sd in self.sigdocs

    # -- construction
    @classmethod
    def from_sources(cls, rl_task_files: Iterable[Path], suite_dir: Optional[Path] = None, heldout_json: Optional[Path] = None,
                     repo_root: Optional[Path] = None, public_root: Optional[Path] = None) -> "LeakGuard":
        g = cls()
        ids: set[str] = set()
        for f in rl_task_files:
            if not Path(f).is_file():
                continue
            for ln in Path(f).read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r.get("split") != "eval":                       # the eval split is the pool the suite was drawn from: all of it is excluded
                    continue
                ids.add(r["id"])
                g.task_names.add(r["name"])
        if suite_dir and (Path(suite_dir) / "tasks.json").is_file():
            for t in json.loads((Path(suite_dir) / "tasks.json").read_text(encoding="utf-8"))["tasks"]:
                if t.get("family") == "fn":
                    g.task_names.add(t["name"])
                    try:
                        fn = next(n for n in ast.walk(ast.parse(t["stub"])) if isinstance(n, ast.FunctionDef) and n.name == t["name"])
                        g.sigdocs.add(sigdoc_hash(fn.name, _args(fn), ast.get_docstring(fn) or ""))
                    except (StopIteration, SyntaxError):
                        pass
        for i in ids:
            if i.startswith("pub:"):
                _, repo, rest = i.split(":", 2)
                g.files.add((repo, rest.rsplit(":", 1)[0]))
            elif ":" in i:
                g.files.add(("self", i.split(":", 1)[0].replace(".", "/") + ".py"))
        if heldout_json and repo_root and Path(heldout_json).is_file():
            shas = json.loads(Path(heldout_json).read_text(encoding="utf-8")).get("held_out_commits") or []
            for k in range(0, len(shas), 40):
                out = subprocess.run(["git", "-C", str(repo_root), "show", "--name-only", "--format=", *shas[k:k + 40]], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=120)
                g.heldout_files |= {x.strip() for x in out.stdout.splitlines() if x.strip().endswith(".py")}
        # hashes of every function inside the blocked source files: copies elsewhere (vendored, other versions) are blocked too
        for source, path in list(g.files) + [("self", p) for p in g.heldout_files]:
            root = repo_root if source == "self" else (Path(public_root) / source if public_root else None)
            p = Path(root) / path if root else None
            if p is None or not p.is_file():
                continue
            try:
                tree = ast.parse(p.read_bytes())
            except (SyntaxError, ValueError):
                continue
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    g.hashes.add(body_hash(n))
                    g.sigdocs.add(sigdoc_hash(n.name, _args(n), ast.get_docstring(n) or ""))
        return g


# ------------------------------------------------------------------------------------------------------------ index
@dataclass(frozen=True)
class Cand:
    source: str
    licence: str
    path: str
    line: int
    end: int
    name: str
    sig: str
    doc: str
    tests: tuple[str, ...]
    score: float
    code: str


def _db_path() -> Path:
    base = os.environ.get("WEEKLY7_INDEX_DIR")
    if base:
        d = Path(base)
    else:
        try:
            from creator import diskcache
            c = diskcache.cache_dir()
        except Exception:
            c = None
        d = (Path(c) if c else Path.home() / ".cache" / "weekly7") / "index"
    d.mkdir(parents=True, exist_ok=True)
    return d / "reuse.sqlite"


def _code_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "*.py"], capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            return [p for p in out.stdout.splitlines() if p]
    except (OSError, subprocess.TimeoutExpired):
        pass
    res: list[str] = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
        res += [os.path.relpath(os.path.join(dp, f), root).replace("\\", "/") for f in fns if f.endswith(".py")]
    return res


def _skippable(rel: str) -> bool:
    parts = rel.split("/")
    return any(p in SKIP_DIRS or p.startswith("test_") or p.endswith("_test.py") or p == "conftest.py" for p in parts[:-1]) or parts[-1].startswith("test_") \
        or parts[-1].endswith("_test.py") or parts[-1] in ("conftest.py", "setup.py")


class ReuseIndex:
    def __init__(self, db: Optional[str | Path] = None, guard: Optional[LeakGuard] = None) -> None:
        self.db_path = Path(db) if db else _db_path()
        self.guard = guard or LeakGuard()
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        if self.db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA:
            self.db.executescript("""
                DROP TABLE IF EXISTS fn; DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS words; DROP TABLE IF EXISTS repos;
                CREATE TABLE repos(source TEXT PRIMARY KEY, root TEXT, licence TEXT, skipped TEXT);
                CREATE TABLE files(source TEXT, path TEXT, sha TEXT, PRIMARY KEY(source, path));
                CREATE TABLE fn(id INTEGER PRIMARY KEY, source TEXT, path TEXT, line INT, end INT, name TEXT, sig TEXT, doc TEXT, nargs INT,
                                args TEXT, bhash TEXT, sdhash TEXT, tests TEXT);
                CREATE VIRTUAL TABLE words USING fts5(txt, tokenize='unicode61');
                CREATE INDEX fn_path ON fn(source, path); CREATE INDEX fn_bhash ON fn(bhash);
            """)
            self.db.execute(f"PRAGMA user_version={SCHEMA}")
            self.db.commit()

    # -------------------------------------------------------------------------------------------------- building
    def add_repo(self, source: str, root: str | Path, licence: Optional[str] = None, cap: int = PER_REPO_CAP) -> dict[str, Any]:
        """Index one source. A public repo with no permissive licence is recorded as skipped and contributes nothing."""
        root = Path(root)
        t0 = time.perf_counter()
        lic = licence or ("self" if source == "self" else licence_of(root))
        self.db.execute("INSERT OR REPLACE INTO repos VALUES (?,?,?,?)", (source, str(root), lic or "", "" if lic else "no permissive licence"))
        if not lic:
            self.db.commit()
            return {"source": source, "skipped": "no permissive licence"}
        files = [f for f in _code_files(root) if not _skippable(f)]
        testwords = self._test_words(root)
        known = {p: s for p, s in self.db.execute("SELECT path, sha FROM files WHERE source=?", (source,))}
        n_fn = self.db.execute("SELECT COUNT(*) FROM fn WHERE source=?", (source,)).fetchone()[0]
        parsed = blocked = 0
        for rel in files:
            if n_fn >= cap:
                break
            if self.guard.file_blocked(source, rel):
                blocked += 1
                self._drop(source, rel)
                continue
            p = root / rel
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = p.read_bytes()
            except OSError:
                continue
            sha = hashlib.sha1(data).hexdigest()
            if known.get(rel) == sha:
                continue
            self._drop(source, rel)
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?)", (source, rel, sha))
            try:
                tree = ast.parse(data)
            except (SyntaxError, ValueError, RecursionError):
                continue
            parsed += 1
            lines = data.decode("utf-8", "replace").splitlines()
            for n in tree.body:
                if not isinstance(n, ast.FunctionDef):
                    continue
                end = getattr(n, "end_lineno", n.lineno) or n.lineno
                doc = ast.get_docstring(n) or ""
                if not doc or not (MIN_LINES <= end - n.lineno + 1 <= MAX_LINES) or n.name.startswith("test"):
                    continue
                args = _args(n)
                h, sd = body_hash(n), sigdoc_hash(n.name, args, doc)
                if self.guard.func_blocked(h, sd):
                    blocked += 1
                    continue
                sig = " ".join(l.strip() for l in lines[n.lineno - 1:_sig_end(n)])
                tw = ",".join(sorted(testwords.get(n.name, ()))[:3])
                cur = self.db.execute("INSERT INTO fn(source,path,line,end,name,sig,doc,nargs,args,bhash,sdhash,tests) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                      (source, rel, n.lineno, end, n.name, sig[:300], doc[:400], len(args), ",".join(args), h, sd, tw))
                self.db.execute("INSERT INTO words(rowid, txt) VALUES (?,?)",
                                (cur.lastrowid, " ".join(tokens(n.name) + tokens(" ".join(args)) + tokens(doc[:400]))))
                n_fn += 1
        self.db.commit()
        return {"source": source, "licence": lic, "files_parsed": parsed, "blocked": blocked, "functions": n_fn, "seconds": round(time.perf_counter() - t0, 2)}

    @staticmethod
    def _test_words(root: Path, cap: int = 300) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        seen = 0
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "node_modules", "__pycache__") and not d.startswith(".")]
            for f in fns:
                if f.startswith("test_") and f.endswith(".py") and seen < cap:
                    seen += 1
                    p = Path(dp) / f
                    try:
                        for w in set(_WORD.findall(p.read_text(encoding="utf-8", errors="replace")[:60000])):
                            out.setdefault(w, set())
                            if len(out[w]) < 3:
                                out[w].add(os.path.relpath(p, root).replace("\\", "/"))
                    except OSError:
                        pass
        return out

    def _drop(self, source: str, rel: str) -> None:
        ids = [r[0] for r in self.db.execute("SELECT id FROM fn WHERE source=? AND path=?", (source, rel))]
        for i in ids:
            self.db.execute("DELETE FROM words WHERE rowid=?", (i,))
        self.db.execute("DELETE FROM fn WHERE source=? AND path=?", (source, rel))
        self.db.execute("DELETE FROM files WHERE source=? AND path=?", (source, rel))

    def build(self, repo_root: str | Path, public_root: Optional[str | Path] = None, repos: Optional[Iterable[str]] = None, cap: int = PER_REPO_CAP) -> list[dict[str, Any]]:
        res = [self.add_repo("self", repo_root, cap=max(cap, 30000))]
        if public_root and Path(public_root).is_dir():
            for d in sorted(Path(public_root).iterdir()):
                if d.is_dir() and (repos is None or d.name in repos):
                    res.append(self.add_repo(d.name, d, cap=cap))
        return res

    def stats(self) -> dict[str, Any]:
        n = self.db.execute("SELECT COUNT(*) FROM fn").fetchone()[0]
        by = dict(self.db.execute("SELECT source, COUNT(*) FROM fn GROUP BY source"))
        lic = {s: l for s, l, k in self.db.execute("SELECT source, licence, skipped FROM repos")}
        return {"functions": n, "sources": len(by), "by_source": by, "licences": lic, "db_bytes": self.db_path.stat().st_size,
                "skipped_repos": [s for s, k in self.db.execute("SELECT source, skipped FROM repos") if k]}

    def assert_clean(self) -> None:
        """Re-check a finished index against the guard (the builder already applied it); raises AssertionError on any leaked row."""
        g = self.guard
        for source, path, h, sd, name in self.db.execute("SELECT source, path, bhash, sdhash, name FROM fn"):
            assert not g.file_blocked(source, path), f"leaked file {source}:{path}"
            assert not g.func_blocked(h, sd), f"leaked function {source}:{path}:{name}"

    # -------------------------------------------------------------------------------------------------- querying
    def query(self, query: str, k: int = 5, name: str = "", args: Sequence[str] = (), doc: str = "", pool: int = 150) -> list[Cand]:
        """Task signature + docstring -> top-k candidates (score 0..1: name 0.30, argument names 0.20, docstring coverage 0.40, arity 0.10)."""
        qt = tokens(query + " " + name)
        if not qt:
            return []
        uniq = list(dict.fromkeys(qt))[:14]
        fts = " OR ".join(f'"{t}"' for t in uniq)
        try:
            rows = self.db.execute("SELECT f.id, f.source, f.path, f.line, f.end, f.name, f.sig, f.doc, f.nargs, f.args, f.tests, f.bhash, f.sdhash, r.licence "
                                   "FROM words w JOIN fn f ON f.id=w.rowid JOIN repos r ON r.source=f.source WHERE words MATCH ? "
                                   "ORDER BY bm25(words) LIMIT ?", (fts, pool)).fetchall()
        except sqlite3.OperationalError:
            return []
        want_name, want_args = set(tokens(name)), {a for x in args for a in tokens(x)}
        doc_t = set(tokens(doc or query))
        out: list[Cand] = []
        for _i, source, path, line, end, nm, sig, cdoc, nargs, cargs, tests, h, sd, lic in rows:
            if self.guard.func_blocked(h, sd) or self.guard.file_blocked(source, path):
                continue                                                     # defence in depth: also at query time
            ct = set(tokens(nm + " " + cdoc))
            ns = len(want_name & set(tokens(nm))) / max(1, len(want_name | set(tokens(nm)))) if want_name else 0.0
            cs = set(tokens(cargs.replace(",", " ")))
            ar = len(want_args & cs) / max(1, len(want_args | cs)) if want_args else 0.0
            dc = len(doc_t & ct) / max(1, len(doc_t)) if doc_t else 0.0
            ok = 1.0 if (not args or len(args) == nargs) else 0.0
            score = 0.30 * ns + 0.20 * ar + 0.40 * min(1.0, dc) + 0.10 * ok
            root = Path(self.db.execute("SELECT root FROM repos WHERE source=?", (source,)).fetchone()[0])
            code = _read_lines(root / path, line, end)
            out.append(Cand(source, lic, path, line, end, nm, sig, cdoc, tuple(t for t in tests.split(",") if t), round(score, 4), code))
        out.sort(key=lambda c: -c.score)
        return out[:k]


def _sig_end(n: ast.FunctionDef) -> int:
    return n.body[0].lineno - 1 if n.body and n.body[0].lineno > n.lineno else n.lineno


def _read_lines(p: Path, a: int, b: int) -> str:
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[a - 1:b])
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------------------------ adapt and direct
def parse_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """name, args, doc, header of the function a task asks for (from the stub when it has one); empty for non-function tasks."""
    stub = task.get("stub") or ""
    if task.get("family") != "fn" or not stub:
        return {}
    try:
        fn = next(n for n in ast.walk(ast.parse(stub)) if isinstance(n, ast.FunctionDef) and n.name == task.get("name"))
    except (StopIteration, SyntaxError):
        return {}
    return {"name": fn.name, "args": _args(fn), "doc": ast.get_docstring(fn) or "", "node": fn, "stub": stub}


def _free_names(fn: ast.FunctionDef) -> set[str]:
    bound = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    loads: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else loads).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn:
            bound.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
    return loads - bound - BUILTINS


def adapt(cand: Cand, task: Mapping[str, Any], repo_root: Optional[Path] = None) -> dict[str, Any]:
    """The candidate re-headed with the task's own signature (name, parameter names, annotations) and docstring, on top of the stub's imports.
    {'code': module text, 'ok': bool, 'free': names the candidate needs that the module does not provide, 'renamed': bool}"""
    pt = parse_task(task)
    if not pt:
        return {"ok": False, "code": "", "free": [], "why": "not a function task"}
    try:
        src = ast.parse("\n".join(l for l in cand.code.splitlines()) if not cand.code.startswith((" ", "\t")) else _dedent(cand.code))
        cfn = next(n for n in src.body if isinstance(n, ast.FunctionDef))
    except (SyntaxError, StopIteration):
        return {"ok": False, "code": "", "free": [], "why": "candidate does not parse"}
    stub_fn: ast.FunctionDef = pt["node"]
    cargs, targs = _args(cfn), pt["args"]
    if len(cargs) != len(targs):
        return {"ok": False, "code": "", "free": sorted(_free_names(cfn)), "why": "arity differs"}
    mapping = {a: b for a, b in zip(cargs, targs) if a != b}

    class Ren(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            return ast.copy_location(ast.Name(id=mapping.get(n.id, n.id), ctx=n.ctx), n)

        def visit_arg(self, n: ast.arg) -> ast.AST:
            return n

        def visit_keyword(self, n: ast.keyword) -> ast.AST:
            self.generic_visit(n)
            return n

    body = [Ren().visit(s) for s in cfn.body]
    # own recursion calls follow the new name
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and n.id == cfn.name:
                n.id = stub_fn.name
    new = ast.FunctionDef(name=stub_fn.name, args=stub_fn.args, body=body, decorator_list=[], returns=stub_fn.returns, type_comment=None)
    if hasattr(new, "type_params"):
        new.type_params = []
    ast.fix_missing_locations(new)
    # imports the candidate's own module used (resolved from its file), appended to the stub's preamble
    pre = pt["stub"].split("def " + stub_fn.name)[0].rstrip()
    free = _free_names(new) - {n for n in re.findall(r"^(?:from\s+\S+\s+)?import\s+(.+)$", pre, re.M) for n in re.split(r"[ ,]+", n)}
    extra: list[str] = []
    if repo_root is not None and free:
        try:
            mod = ast.parse((repo_root / cand.path).read_text(encoding="utf-8", errors="replace"))
            for n in mod.body:
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    names = {(a.asname or a.name).split(".")[0] for a in n.names}
                    if names & free and not (isinstance(n, ast.ImportFrom) and (n.level or 0) > 0):
                        extra.append(ast.unparse(n))
                        free -= names
        except (OSError, SyntaxError):
            pass
    free -= {n for e in extra for n in re.findall(r"\b([A-Za-z_]\w*)\b", e)}
    text = (pre + "\n" + "\n".join(extra) + "\n\n" if extra else pre + "\n\n") + ast.unparse(new)
    # keep the task's docstring (the contract), not the candidate's
    d = ast.get_docstring(stub_fn)
    if d:
        text = text.replace("def " + stub_fn.name, "def " + stub_fn.name, 1)
        new.body = [ast.Expr(ast.Constant(d))] + [s for s in body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant) and isinstance(s.value.value, str))]
        text = (pre + "\n" + "\n".join(extra) + "\n\n" if extra else pre + "\n\n") + ast.unparse(new)
    return {"ok": not free, "code": text + "\n", "free": sorted(free), "renamed": bool(mapping) or cfn.name != stub_fn.name}


def _dedent(text: str) -> str:
    import textwrap
    return textwrap.dedent(text)


CHECK = r'''
import json, math, sys
sys.path.insert(0, ".")
cases = json.loads(open(sys.argv[1], encoding="utf-8").read()); name = sys.argv[2]
def norm(x): return json.loads(json.dumps(x, default=list))
def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try: return (a is None) == (b is None) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except Exception: return False
    if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b
try:
    import solution
    f = getattr(solution, name)
except BaseException as e:
    print(json.dumps({"ok": False, "why": "import: " + repr(e)[:150]})); sys.exit(0)
for c in cases:
    try: got = norm(f(*c["args"]))
    except BaseException as e: got = "raised " + type(e).__name__
    if not close(got, c["expect"]):
        print(json.dumps({"ok": False, "why": f"{c['args']!r:.50} -> {got!r:.40}"})); sys.exit(0)
print(json.dumps({"ok": True, "why": ""}))
'''


def direct_check(code: str, name: str, visible: Sequence[Mapping[str, Any]], python: str = "", timeout: float = 15.0) -> dict[str, Any]:
    """Hidden-test-free checks of an adapted candidate: it parses, is not a stub, and passes the task's visible examples (a subprocess)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "why": f"static: {e.msg}"}
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None or any(isinstance(s, ast.Raise) and "NotImplemented" in ast.dump(s) for s in ast.walk(fn)):
        return {"ok": False, "why": "static: no real definition"}
    if not visible:
        return {"ok": False, "why": "no visible examples to check"}
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "solution.py").write_text(code, encoding="utf-8")
        (dd / "cases.json").write_text(json.dumps(list(visible)), encoding="utf-8")
        (dd / "chk.py").write_text(CHECK, encoding="utf-8")
        try:
            p = subprocess.run([python or sys.executable, "chk.py", "cases.json", name], cwd=d, capture_output=True, text=True, timeout=timeout)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as e:
            return {"ok": False, "why": f"run: {type(e).__name__}"}


ADAPT_MIN = 0.5      # below this a candidate is noise: the CODER starts from the stub instead


def prepare(index: ReuseIndex, task: Mapping[str, Any], repo_root: Path, visible_n: int = 2, direct_min: float = ADAPT_MIN, k: int = 5,
            python: str = "") -> dict[str, Any]:
    """Best use of the index for one task: mode 'direct' (the adapted candidate passes the visible examples), 'adapt' (best adaptable
    candidate, as the starting code) or 'none'. Returns the mode, the candidate's provenance and the starting code."""
    pt = parse_task(task)
    if not pt:
        return {"mode": "none", "why": "not a function task"}
    t0 = time.perf_counter()
    cands = index.query(task.get("query") or task.get("request", ""), k=k, name=pt["name"], args=pt["args"], doc=pt["doc"])
    vis = list(task.get("tests") or [])[:visible_n]
    best: Optional[dict[str, Any]] = None
    for c in cands:
        root = Path(index.db.execute("SELECT root FROM repos WHERE source=?", (c.source,)).fetchone()[0])
        a = adapt(c, task, root)
        if not a["code"] or c.score < ADAPT_MIN:
            continue
        info = {"source": c.source, "licence": c.licence, "path": c.path, "name": c.name, "score": c.score, "tests": list(c.tests), "free": a["free"]}
        if a["ok"] and c.score >= direct_min:
            r = direct_check(a["code"], pt["name"], vis, python)
            if r["ok"]:
                return {"mode": "direct", "code": a["code"], "cand": info, "seconds": round(time.perf_counter() - t0, 3), "checked": len(vis)}
        if best is None and a["ok"]:
            best = {"mode": "adapt", "code": a["code"], "cand": info}
    if best:
        best["seconds"] = round(time.perf_counter() - t0, 3)
        return best
    return {"mode": "none", "why": "no adaptable candidate", "seconds": round(time.perf_counter() - t0, 3)}


def default_guard(repo_root: str | Path, public_root: Optional[str | Path] = None, runtime: Optional[Path] = None) -> LeakGuard:
    """The guard for the real suite: the eval split of the rl-task exports, the suite's own tasks and the codetrust held-out commits."""
    rt = runtime or Path.home() / "creator_runtime"
    gd = rt / "gpuday"
    return LeakGuard.from_sources([gd / "export" / "rl_tasks.jsonl", gd / "export_plus" / "rl_tasks.jsonl"], suite_dir=gd / "baseline_suite",
                                  heldout_json=Path(repo_root) / "creator" / "codetrust_heldout.json", repo_root=Path(repo_root),
                                  public_root=public_root or rt / "public_repos")


def main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="reuse index: build / query")
    ap.add_argument("cmd", choices=["build", "stats", "query", "check"])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--public", default=str(Path.home() / "creator_runtime" / "public_repos"))
    ap.add_argument("--db", default="")
    ap.add_argument("--cap", type=int, default=PER_REPO_CAP)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("text", nargs="*")
    a = ap.parse_args(list(argv))
    guard = default_guard(a.repo, a.public)
    ix = ReuseIndex(a.db or None, guard)
    if a.cmd == "build":
        for r in ix.build(a.repo, a.public, repos=set(a.only) if a.only else None, cap=a.cap):
            print(json.dumps(r))
    elif a.cmd == "query":
        for c in ix.query(" ".join(a.text)):
            print(f"{c.score:.3f} {c.source}[{c.licence}] {c.path}:{c.line} {c.sig}")
    elif a.cmd == "check":
        ix.assert_clean()
        print("clean")
    print(json.dumps(ix.stats()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
