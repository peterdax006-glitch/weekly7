"""Locate index + context packer (MASTER_BLUEPRINT P0.3, owner 4 Oct 2026: "code first ... absolutely crazy efficent").

Finding the code a change needs used to cost a model call that read ~2000 tokens (~60 s on the home CPU). This index answers the same
question from a SQLite file in milliseconds: definitions, call sites and ripgrep text hits, ranked; the packer then returns only the
exact line ranges, under a hard token budget per role, instead of whole files.

    ix = Index(root); ix.update()                 # incremental: only files whose bytes changed are re-parsed
    ix.defs("select_tests")                       # [Hit(path, line, end, kind, qual)]
    ix.callers("select_tests")                    # call sites, with the enclosing function
    ix.locate("select tests for changed files")   # ranked hits for a free-text or symbol query
    ix.pack(hits, max_tokens=1500)                # "### path:line-end (qual)\n<code>" blocks within the budget

Stored outside the repo (diskcache.cache_dir()/index/<root hash>.sqlite, or WEEKLY7_INDEX_DIR); loaded on demand, imports nothing heavy.
CLI: python -m creator.tools.index [root] update | defs NAME | callers NAME | locate QUERY | bench
"""
from __future__ import annotations

import ast
import math
import functools
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

CHARS_PER_TOKEN = 3.6
SCHEMA = 6
FULL_EVERY_S = 600.0     # the directory short-circuit is healed by a full git listing at least this often
_TOK = re.compile(r"[a-z0-9]+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[a-z][A-Z]")
RG_BELOW = 3             # ripgrep fallback only when fewer than this many strong index hits
_STOP = frozenset("a an and the of to for in on with from by is are be it this that as at or not into".split())


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    end: int
    kind: str          # def | class | call | text
    qual: str          # dotted name (defs) / enclosing function (calls) / matched text (text)
    score: float = 0.0


def _db_path(root: Path) -> Path:
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
    return d / (hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16] + ".sqlite")


def _py_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "*.py"], capture_output=True, text=True,
                             timeout=30)
        if out.returncode == 0:
            return [p for p in out.stdout.splitlines() if p]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.py") if ".git" not in p.parts]


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.defs: list[tuple[str, str, str, int, int]] = []    # name, qual, kind, line, end
        self.calls: list[tuple[str, int, str]] = []               # callee name, line, caller qual

    def _def(self, node: ast.AST, kind: str) -> None:
        name = getattr(node, "name")
        qual = ".".join(self.stack + [name])
        self.defs.append((name, qual, kind, node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno))
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._def(node, "def")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._def(node, "def")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._def(node, "class")

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name:
            self.calls.append((name, node.lineno, ".".join(self.stack)))
        self.generic_visit(node)


class _Mem:
    """In-memory copy of the symbol and word tables: name -> definitions, term -> postings, for millisecond-free lookups."""

    def __init__(self, db: sqlite3.Connection, dv: int) -> None:
        self.dv = dv
        self._sub: dict[str, list[tuple]] = {}
        self.by_name: dict[str, list[Hit]] = {}
        self.sym_rows: list[tuple[str, str, str, str, int, int, str]] = []
        for name, qual, kind, path, line, end in db.execute("SELECT name, qual, kind, path, line, end FROM syms ORDER BY rowid"):
            self.sym_rows.append((name.lower(), name, kind, path, line, end, qual))
            self.by_name.setdefault(name, []).append(Hit(path, line, end, kind, qual))
        for hs in self.by_name.values():
            hs.sort(key=lambda h: (h.path, h.line))
        self.rows: list[tuple[tuple[str, ...], str, str, int, int, str]] = []
        self.post: dict[str, list[tuple[int, int]]] = {}
        total = 0
        for txt, qual, path, line, end, kind in db.execute("SELECT txt, qual, path, line, end, kind FROM words ORDER BY rowid"):
            toks = _TOK.findall(txt.lower())
            i = len(self.rows)
            self.rows.append((tuple(txt.split()), qual, path, int(line), int(end), kind))
            total += len(toks)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.post.setdefault(t, []).append((i, c))
            self.rows[i] = self.rows[i] + (len(toks),)                  # type: ignore[assignment]
        self.avgdl = (total / len(self.rows)) if self.rows else 1.0

    def substr(self, t: str) -> list[tuple]:
        """Symbols whose name contains `t` (first 80 in table order; module/class names always, function names only when they hold no
        underscore - snake_case names are already word-matched), as word rows. Memoized per term."""
        r = self._sub.get(t)
        if r is None:
            r = []
            for nl, name, kind, p, a, b, qual in self.sym_rows:
                if t in nl and (kind != "def" or "_" not in name):
                    r.append((_split_t(qual), qual, p, a, b, kind, 0.0))
                    if len(r) >= 80:
                        break
            r = self._sub[t] = r
        return r

    def bm25(self, terms: Sequence[str], limit: int) -> list[tuple]:
        """Top `limit` word rows matching ANY term, best first, as (words, qual, path, line, end, kind, bm25>0) (FTS5's scoring formula)."""
        n = len(self.rows)
        acc: dict[int, float] = {}
        for t in terms:
            plist = self.post.get(t)
            if not plist:
                continue
            df = len(plist)
            idf = max(1e-6, math.log((n - df + 0.5) / (df + 0.5) + 1.0)) if n else 0.0
            for i, tf in plist:
                dl = self.rows[i][6]
                acc[i] = acc.get(i, 0.0) + idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / self.avgdl))
        top = sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:limit]
        return [self.rows[i][:6] + (sc,) for i, sc in top]


class Index:
    def __init__(self, root: str | Path, db: Optional[str | Path] = None) -> None:
        self.root = Path(root).resolve()
        self.db_path = Path(db) if db else _db_path(self.root)
        self.db = sqlite3.connect(str(self.db_path))
        self._mem: Optional[_Mem] = None
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        cur = self.db.execute("PRAGMA user_version").fetchone()[0]
        if cur != SCHEMA:
            self.db.executescript("""
                DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS syms; DROP TABLE IF EXISTS calls; DROP TABLE IF EXISTS words; DROP TABLE IF EXISTS dirs; DROP TABLE IF EXISTS meta;
                CREATE TABLE files(path TEXT PRIMARY KEY, sha TEXT, nlines INT, mtime INT, size INT);
                CREATE TABLE dirs(path TEXT PRIMARY KEY, mtime INT, haspy INT);
                CREATE TABLE meta(k TEXT PRIMARY KEY, v REAL);
                CREATE TABLE syms(name TEXT, qual TEXT, kind TEXT, path TEXT, line INT, end INT);
                CREATE TABLE calls(name TEXT, path TEXT, line INT, caller TEXT);
                CREATE VIRTUAL TABLE words USING fts5(txt, qual UNINDEXED, path UNINDEXED, line UNINDEXED, end UNINDEXED,
                                                      kind UNINDEXED);
                CREATE INDEX syms_name ON syms(name); CREATE INDEX syms_path ON syms(path);
                CREATE INDEX calls_name ON calls(name); CREATE INDEX calls_path ON calls(path);
            """)
            self.db.execute(f"PRAGMA user_version={SCHEMA}")
            self.db.commit()

    # ------------------------------------------------------------------------------------------------------------------ building
    def update(self, paths: Optional[Iterable[str]] = None) -> dict[str, float]:
        """Re-parse only files whose bytes changed; drop deleted ones. Returns counts and seconds.
        Whole-tree updates first try a stat-only pass over the known directories (a file is re-read only when its mtime/size moved; a
        directory whose own mtime moved - something was added or removed - or a new .py file sends it to the full git listing)."""
        t0 = time.perf_counter()
        known = {r[0]: (r[1], r[2], r[3]) for r in self.db.execute("SELECT path, sha, mtime, size FROM files")}
        parsed = failed = 0
        if paths is None:
            quick = self._quick_changes(known)
            if quick is not None:
                for rel, st in quick.items():
                    r = self._index_file(rel, known, st)
                    parsed += r == 1
                    failed += r == 2
                self.db.commit()
                if parsed or failed:
                    self._mem = None
                return {"files": float(len(known)), "parsed": float(parsed), "failed": float(failed),
                        "seconds": round(time.perf_counter() - t0, 3)}
        files = list(paths) if paths is not None else _py_files(self.root)
        seen: set[str] = set()
        for rel in files:
            try:
                st = os.stat(self.root / rel)
            except OSError:
                continue
            seen.add(rel)
            r = self._index_file(rel, known, (st.st_mtime_ns, st.st_size))
            parsed += r == 1
            failed += r == 2
        if paths is None:
            for rel in set(known) - seen:
                self._mem = None
                self._drop(rel)
                self.db.execute("DELETE FROM files WHERE path=?", (rel,))
            self._save_dirs(seen)
        self.db.commit()
        if parsed or failed:
            self._mem = None
        return {"files": float(len(seen)), "parsed": float(parsed), "failed": float(failed), "seconds": round(time.perf_counter() - t0, 3)}

    def _index_file(self, rel: str, known: dict, st: tuple[int, int]) -> int:
        """0 = unchanged, 1 = parsed, 2 = syntax error (recorded, no symbols)."""
        old = known.get(rel)
        if old is not None and old[1] == st[0] and old[2] == st[1]:
            return 0                                                  # stat unchanged: not even read
        try:
            data = (self.root / rel).read_bytes()
        except OSError:
            return 0
        sha = hashlib.sha1(data).hexdigest()
        if old is not None and old[0] == sha:
            self.db.execute("UPDATE files SET mtime=?, size=? WHERE path=?", (st[0], st[1], rel))
            return 0
        self._drop(rel)
        nl = data.count(b"\n") + 1
        try:
            tree = ast.parse(data, filename=rel)
        except (SyntaxError, ValueError):
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?)", (rel, sha, nl, st[0], st[1]))
            return 2
        v = _Visitor()
        v.visit(tree)
        mod = rel[:-3].replace("/", ".")                        # the module itself is a symbol: "slow path" finds creator/slowpath.py
        doc = (ast.get_docstring(tree) or "").split("\n", 1)[0][:200]
        v.defs.insert(0, (mod.rsplit(".", 1)[-1], mod, "module", 1, nl))
        if doc:
            self.db.execute("INSERT INTO words VALUES (?,?,?,?,?,?)", (" ".join(_split(mod)) + " " + doc.lower(), mod, rel, 1, nl, "module"))
        self.db.executemany("INSERT INTO syms VALUES (?,?,?,?,?,?)", [(n, q, k, rel, a, b) for n, q, k, a, b in v.defs])
        self.db.executemany("INSERT INTO words VALUES (?,?,?,?,?,?)",
                            [(" ".join(_split(q)), q, rel, a, b, k) for _n, q, k, a, b in v.defs if k != "module"])
        self.db.executemany("INSERT INTO calls VALUES (?,?,?,?)", [(n, rel, ln, c) for n, ln, c in v.calls])
        self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?)", (rel, sha, nl, st[0], st[1]))
        return 1

    def _save_dirs(self, files: Iterable[str]) -> None:
        """Remember every directory that holds an indexed file, and its ancestors, with its mtime (the next update's short-circuit)."""
        haspy = {os.path.dirname(f) for f in files}
        alld = set(haspy)
        for d in haspy:
            while d:
                d = os.path.dirname(d)
                alld.add(d)
        rows = []
        for d in alld:
            try:
                rows.append((d, os.stat(self.root / d).st_mtime_ns, d in haspy))
            except OSError:
                pass
        self.db.execute("DELETE FROM dirs")
        self.db.executemany("INSERT INTO dirs VALUES (?,?,?)", rows)
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('full', ?)", (time.time(),))

    def _quick_changes(self, known: dict) -> Optional[dict[str, tuple[int, int]]]:
        """{rel: (mtime, size)} of the .py files that moved since the last update, or None when only a full listing can tell
        (no dirs yet, a directory changed, a new/removed file, or the last full listing is older than FULL_EVERY_S)."""
        row = self.db.execute("SELECT v FROM meta WHERE k='full'").fetchone()
        if row is None or time.time() - row[0] > FULL_EVERY_S:
            return None
        root = str(self.root)
        changed: dict[str, tuple[int, int]] = {}
        seen = 0
        for d, mt, haspy in self.db.execute("SELECT path, mtime, haspy FROM dirs").fetchall():
            full = root + "/" + d if d else root
            try:
                if os.stat(full).st_mtime_ns != mt:
                    return None
                if not haspy:
                    continue
                with os.scandir(full) as it:
                    for e in it:
                        n = e.name
                        if n[-3:] != ".py" or not e.is_file():
                            continue
                        rel = d + "/" + n if d else n
                        old = known.get(rel)
                        if old is None:
                            return None
                        seen += 1
                        st = e.stat()
                        if old[1] != st.st_mtime_ns or old[2] != st.st_size:
                            changed[rel] = (st.st_mtime_ns, st.st_size)
            except OSError:
                return None
        return changed if seen == len(known) else None

    def _drop(self, rel: str) -> None:
        for t in ("syms", "calls", "words"):
            self.db.execute(f"DELETE FROM {t} WHERE path=?", (rel,))

    # ------------------------------------------------------------------------------------------------------------------ queries
    def defs(self, name: str) -> list[Hit]:
        rows = self.db.execute("SELECT path, line, end, kind, qual FROM syms WHERE name=? ORDER BY path, line", (name,))
        return [Hit(p, a, b, k, q) for p, a, b, k, q in rows]

    def callers(self, name: str, limit: int = 200) -> list[Hit]:
        rows = self.db.execute("SELECT path, line, caller FROM calls WHERE name=? ORDER BY path, line LIMIT ?", (name, limit))
        return [Hit(p, ln, ln, "call", c) for p, ln, c in rows]

    def text(self, pattern: str, limit: int = 50, fixed: bool = True) -> list[Hit]:
        """ripgrep over the tracked Python files (literal by default)."""
        args = ["rg", "-n", "--no-heading", "--color", "never", "-g", "*.py", "-m", "20"] + (["-F"] if fixed else []) + ["--", pattern, "."]
        try:
            out = subprocess.run(args, cwd=self.root, capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired):
            return []
        hits = []
        for line in out.stdout.splitlines()[:limit]:
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[1].isdigit():
                hits.append(Hit(parts[0].lstrip("./").replace("\\", "/"), int(parts[1]), int(parts[1]), "text", parts[2].strip()[:120]))
        return hits

    def enclosing(self, path: str, line: int) -> Optional[Hit]:
        r = self.db.execute("SELECT path, line, end, kind, qual FROM syms WHERE path=? AND line<=? AND end>=? ORDER BY line DESC LIMIT 1",
                            (path, line, line)).fetchone()
        return Hit(*r[:4], r[4]) if r else None

    def _memory(self) -> "_Mem":
        """The words/symbols tables as in-process dicts, loaded once (lazily) and rebuilt only when the database changed."""
        dv = self.db.execute("PRAGMA data_version").fetchone()[0]
        m = self._mem
        if m is None or m.dv != dv:
            m = self._mem = _Mem(self.db, dv)
        return m

    def locate(self, query: str, k: int = 10) -> list[Hit]:
        """Ranked definitions for a symbol or free-text query: exact name > identifier words in the query > name-word match (BM25 over
        the name words, in memory) > ripgrep hits mapped to their enclosing definition (only when the index found almost nothing)."""
        mem = self._memory()
        scores: dict[tuple[str, int], tuple[float, Hit]] = {}

        def add(h: Hit, s: float) -> None:
            key = (h.path, h.line)
            old = scores.get(key)
            scores[key] = (s + (old[0] if old else 0.0), h)

        q = query.strip()
        idents = [w for w in _WORD.findall(q) if len(w) > 2 and w.lower() not in _STOP]
        specific = [w for w in idents if "_" in w or _CAMEL.search(w)]     # select_tests / ImportGraph: a real name
        if _WORD.fullmatch(q):
            specific = specific or [q]
        for w in specific:
            for h in mem.by_name.get(w, ()):
                add(h, 10.0)
        terms = sorted({t for w in idents for t in _split(w) if len(t) > 2 and t not in _STOP})
        if terms:
            want = set(terms)
            seen_rows: set[tuple[str, int]] = set()
            test_q = any(t.startswith("test") for t in want)
            nwant = len(want)

            def score_rows(rs: Iterable[tuple]) -> None:
                for words, qual, p, a, b, kind, bm in rs:
                    if (p, a) in seen_rows:
                        continue
                    seen_rows.add((p, a))
                    name_words = _split_t(qual.rsplit(".", 1)[-1])
                    cov_last = sum([_cov(t, name_words) for t in want]) / nwant       # how much of the query the NAME covers
                    cov_full = sum([_cov(t, words) for t in want]) / nwant            # ... or with its class/module context
                    extra = sum(1 for w in name_words if not any([_cov(t, (w,)) for t in want])) / max(1, len(name_words))
                    pen = 1.5 if (not test_q and (p.startswith("tests/") or "/test_" in p or p.startswith("test_"))) else 0.0
                    add(Hit(p, a, b, kind, qual), 6.0 * cov_last + 3.0 * cov_full - 1.0 * extra - pen + min(1.0, bm / 10))

            score_rows(mem.bm25(terms, 120))
            if sum(1 for sc, _h in scores.values() if sc >= 6.0) < 3:   # compound names (slowpath, modelpool) are one token:
                extra_rows = []                                       # substring scan only when the token match found too little
                for t in (t for t in terms if len(t) >= 4):
                    extra_rows += mem.substr(t)
                score_rows(extra_rows)
        strong = sum(1 for sc, _h in scores.values() if sc >= 3.0)
        for w in (idents[:3] if strong < min(k, RG_BELOW) else []):   # ripgrep (a subprocess, ~20-30 ms) only when the index found almost nothing
            for t in self.text(w, limit=20):
                e = self.enclosing(t.path, t.line)
                if e:
                    add(e, 0.5)
        ranked = sorted(scores.values(), key=lambda x: (-x[0], x[1].path, x[1].line))
        return [Hit(h.path, h.line, h.end, h.kind, h.qual, round(s, 2)) for s, h in ranked[:k]]

    # ------------------------------------------------------------------------------------------------------------------ packing
    def pack(self, hits: Sequence[Hit], max_tokens: int = 1500, max_lines_per_hit: int = 80) -> str:
        """Exact line ranges for the hits, in order, until the token budget is spent. A def longer than max_lines_per_hit is cut with a
        marker. Overlapping ranges in the same file are emitted once."""
        budget = int(max_tokens * CHARS_PER_TOKEN)
        out, used, done = [], 0, set()
        cache: dict[str, list[str]] = {}
        for h in hits:
            lines = cache.get(h.path)
            if lines is None:
                try:
                    lines = (self.root / h.path).read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                cache[h.path] = lines
            a, b = h.line, max(h.line, h.end)
            if h.kind in ("call", "text"):
                a, b = max(1, h.line - 3), min(len(lines), h.line + 3)
            if any(p == h.path and x <= a and b <= y for p, x, y in done):
                continue
            cut = b - a + 1 > max_lines_per_hit
            body = lines[a - 1:(a - 1 + max_lines_per_hit) if cut else b]
            block = f"### {h.path}:{a}-{b} ({h.qual})\n" + "\n".join(body) + ("\n# ... cut" if cut else "") + "\n"
            if used + len(block) > budget:
                if not out:
                    out.append(block[:budget])
                break
            out.append(block)
            used += len(block)
            done.add((h.path, a, b))
        return "".join(out)


def _split(name: str) -> list[str]:
    """snake_case / CamelCase / dotted -> lower-case words."""
    parts = re.split(r"[._]+", name)
    words: list[str] = []
    for p in parts:
        words += [w.lower() for w in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", p)]
    return [w for w in words if w]


@functools.lru_cache(maxsize=65536)
def _split_t(name: str) -> tuple[str, ...]:
    return tuple(_split(name))


@functools.lru_cache(maxsize=1 << 18)
def _cov(term: str, words: tuple[str, ...]) -> bool:
    return any(term == w or (len(term) >= 4 and term in w) for w in words)


def _covered(term: str, words: Sequence[str]) -> bool:
    """A query term is covered by a name word when equal, or (4+ letters) inside a compound word: slow -> slowpath."""
    return any(term == w or (len(term) >= 4 and term in w) for w in words)


def _unsplit(words: str, path: str, line: int, ix: "Index") -> str:
    r = ix.db.execute("SELECT qual FROM syms WHERE path=? AND line=? LIMIT 1", (path, line)).fetchone()
    return r[0] if r else words


def _bench(ix: Index, queries: Sequence[str]) -> dict[str, float]:
    times = []
    for q in queries:
        t0 = time.perf_counter()
        ix.locate(q)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"n": float(len(times)), "p50_ms": round(1000 * times[len(times) // 2], 1), "p95_ms": round(1000 * times[int(len(times) * 0.95) - 1], 1)}


def main(argv: Sequence[str]) -> int:
    root = Path(argv[0]) if argv and not argv[0] in ("update", "defs", "callers", "locate", "bench") else Path.cwd()
    rest = list(argv[1:] if argv and Path(argv[0]) == root else argv)
    ix = Index(root)
    cmd = rest[0] if rest else "update"
    if cmd == "update":
        print(ix.update())
    elif cmd == "defs":
        for h in ix.defs(rest[1]):
            print(f"{h.path}:{h.line}-{h.end} {h.kind} {h.qual}")
    elif cmd == "callers":
        for h in ix.callers(rest[1]):
            print(f"{h.path}:{h.line} in {h.qual}")
    elif cmd == "locate":
        ix.update()
        hits = ix.locate(" ".join(rest[1:]))
        for h in hits:
            print(f"{h.score:6.2f} {h.path}:{h.line}-{h.end} {h.qual}")
    elif cmd == "bench":
        print(ix.update())
        names = [r[0] for r in ix.db.execute("SELECT name FROM syms ORDER BY random() LIMIT 100")]
        print(_bench(ix, names + [n.replace("_", " ") for n in names]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
