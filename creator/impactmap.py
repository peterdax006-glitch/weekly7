"""Function-level test impact map: which tests execute which functions, so an edit to one function runs only the tests that touch it.

Standalone (stdlib only; the warm worker loads this file by path). Two halves:

Recorder (a pytest plugin, active only in recording runs): sys.settrace on 'call' events only (no line tracing), per test: the code objects
executed in non-test repo files (their first line), files opened for reading, and whether a python subprocess was launched (opaque: its
coverage is invisible). Calls made during collection and by module/session-scoped fixtures are attributed to every test that could depend
on them; calls made from module-level code (import time) are recorded globally.

ImpactMap (the store + the decision): per test file a shard {nodeid: {opaque, reads, files: {rel: [module-residue hash, {unit: hash}]}}} and a
baseline {rel: [module-residue hash, {unit: hash}]} of the last verified version of each file. A "unit" is an outermost def (method or
function); its hash is the ast dump of the whole def, the residue hash is the module dump with every def body elided (signatures, decorators,
defaults, constants, class attributes stay in). Selection for a change to file F:
  * residue changed, F unknown, or an import-time unit changed  -> no narrowing (every import-graph test runs)
  * otherwise a test is selected iff it is opaque, unknown, its test file changed, it reads F, or a unit it recorded changed/vanished.
Anything the map cannot vouch for is selected: the map can only remove tests it has evidence for."""
from __future__ import annotations

import ast
import bisect
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

Table = tuple[str, dict[str, tuple[int, int, str]]]          # residue hash, {unit: (start, end, hash)}


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:12]


def unit_table(src: bytes | str) -> Optional[Table]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    units: dict[str, tuple[int, int, str]] = {}
    nodes: list[Any] = []

    def walk(node: Any, prefix: str) -> None:
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
                n = 1
                while q in units:
                    n += 1
                    q = f"{prefix}{ch.name}#{n}"
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                units[q] = (start, ch.end_lineno or ch.lineno, _h(ast.dump(ch)))
                nodes.append(ch)
            elif isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
            elif isinstance(ch, ast.stmt):
                walk(ch, prefix)
    walk(tree, "")
    saved = [(n, n.body) for n in nodes]
    for n in nodes:
        n.body = [ast.Pass()]
    try:
        mod = _h(ast.dump(tree))
    finally:
        for n, b in saved:
            n.body = b
    return mod, units


def unit_at(table: Table, lineno: int) -> Optional[str]:
    items = sorted(((s, e, q) for q, (s, e, _h2) in table[1].items()))
    i = bisect.bisect_right([s for s, _e, _q in items], lineno) - 1
    return items[i][2] if i >= 0 and items[i][0] <= lineno <= items[i][1] else None


# ------------------------------------------------------------------------------------------------------------- recorder
_ACTIVE: list["Recorder"] = []
_HOOKED = [False]


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _ACTIVE:
        return
    r = _ACTIVE[-1]
    if r.cur is None:
        return
    try:
        if event == "open":
            p = args[0]
            if isinstance(p, str) and p.endswith(".py"):
                rel = r.rel(p)
                if rel:
                    r.reads.add(rel)
        elif event in ("_winapi.CreateProcess", "os.system", "subprocess.Popen", "os.exec", "os.spawn", "os.posix_spawn"):
            if "python" in " ".join(str(a) for a in args[:2]).lower():
                r.opaque = True
    except Exception:                                              # noqa: BLE001
        pass


class Recorder:
    def __init__(self, root: str, out: str) -> None:
        self.root = os.path.normcase(os.path.abspath(root)) + os.sep
        self.out = out
        self.scope: dict[str, Optional[str]] = {}
        self.cur: Optional[dict[str, set[int]]] = None
        self.reads: set[str] = set()
        self.opaque = False
        self.tests: dict[str, dict[str, Any]] = {}
        self.collect: dict[str, dict[str, set[int]]] = {}          # test module rel path -> calls during its collection ("*" = other)
        self.fix: dict[str, dict[str, set[int]]] = {}              # non-function-scoped fixture name -> calls during its setup
        self.import_time: dict[str, set[int]] = {}
        self._self = os.path.normcase(os.path.abspath(__file__))

    # -- scope
    def rel(self, filename: str) -> Optional[str]:
        hit = self.scope.get(filename, "?")
        if hit != "?":
            return hit
        n = os.path.normcase(os.path.abspath(filename))
        rel = None
        if n.startswith(self.root) and n.endswith(".py") and n != self._self:
            r = n[len(self.root):].replace(os.sep, "/")
            base = r.rsplit("/", 1)[-1]
            if not (r.startswith(("tests/", ".venv/", "venv/", "state/")) or "site-packages" in r or base.startswith("test_")
                    or base.endswith("_test.py") or base == "conftest.py"):
                rel = r
        self.scope[filename] = rel
        return rel

    def _trace(self, frame: Any, event: str, arg: Any) -> None:
        cur = self.cur
        if cur is None or event != "call":
            return None
        code = frame.f_code
        rel = self.rel(code.co_filename)
        if rel is None:
            return None
        cur.setdefault(rel, set()).add(code.co_firstlineno)
        f = frame.f_back
        for _ in range(3):                                         # called from module-level code (possibly via a class body)?
            if f is None:
                break
            if f.f_code.co_name == "<module>":
                if self.rel(f.f_code.co_filename):
                    self.import_time.setdefault(rel, set()).add(code.co_firstlineno)
                break
            f = f.f_back
        return None

    # -- pytest hooks (called by the plugin object)
    def pytest_configure(self, config: Any) -> None:
        _ACTIVE.append(self)
        if not _HOOKED[0]:
            sys.addaudithook(_audit)
            _HOOKED[0] = True
        sys.settrace(self._trace)
        threading.settrace(self._trace)

    def pytest_unconfigure(self, config: Any) -> None:
        sys.settrace(None)
        threading.settrace(None)                                    # type: ignore[arg-type]
        if self in _ACTIVE:
            _ACTIVE.remove(self)

    def _wrap_collect(self, collector: Any):                        # type: ignore[no-untyped-def]
        key = "*"
        try:
            if type(collector).__name__ == "Module":
                key = self.rel_test(str(collector.path))
        except Exception:                                          # noqa: BLE001
            pass
        prev = self.cur
        self.cur = self.collect.setdefault(key, {})
        return prev

    def rel_test(self, p: str) -> str:
        n = os.path.normcase(os.path.abspath(p))
        return n[len(self.root):].replace(os.sep, "/") if n.startswith(self.root) else p


def make_plugin(root: str, out: str) -> Any:
    """The pytest plugin object (hook wrappers need the pytest import, so it is built lazily)."""
    import pytest
    r = Recorder(root, out)

    class Plugin:
        recorder = r

        def pytest_configure(self, config: Any) -> None:
            r.pytest_configure(config)

        def pytest_unconfigure(self, config: Any) -> None:
            r.pytest_unconfigure(config)

        @pytest.hookimpl(hookwrapper=True)
        def pytest_make_collect_report(self, collector: Any):      # type: ignore[no-untyped-def]
            prev = r._wrap_collect(collector)
            try:
                yield
            finally:
                r.cur = prev

        @pytest.hookimpl(hookwrapper=True)
        def pytest_fixture_setup(self, fixturedef: Any, request: Any):   # type: ignore[no-untyped-def]
            if getattr(fixturedef, "scope", "function") == "function" or r.cur is None:
                yield
                return
            outer = r.cur
            mine: dict[str, set[int]] = {}
            r.cur = mine
            try:
                yield
            finally:
                r.cur = outer
                have = r.fix.setdefault(fixturedef.argname, {})
                for k, v in mine.items():
                    have.setdefault(k, set()).update(v)
                    outer.setdefault(k, set()).update(v)

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_protocol(self, item: Any, nextitem: Any):     # type: ignore[no-untyped-def]
            r.cur, r.reads, r.opaque = {}, set(), False
            try:
                yield
            finally:
                calls = r.cur or {}
                r.cur = None
                try:
                    names = list(getattr(item, "fixturenames", []))
                except Exception:                                  # noqa: BLE001
                    names = []
                merged: list[dict[str, set[int]]] = [calls, r.collect.get("*", {}), r.collect.get(item.nodeid.split("::", 1)[0], {})]
                merged += [r.fix[n] for n in names if n in r.fix]
                out: dict[str, set[int]] = {}
                for d in merged:
                    for k, v in d.items():
                        out.setdefault(k, set()).update(v)
                r.tests[item.nodeid] = {"calls": {k: sorted(v) for k, v in out.items()}, "reads": sorted(r.reads), "opaque": r.opaque}

        def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
            if r.out:
                data = {"tests": r.tests, "import_time": {k: sorted(v) for k, v in r.import_time.items()}}
                tmp = r.out + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp, r.out)

    return Plugin()


# ---------------------------------------------------------------------------------------------------------------- store
class ImpactMap:
    def __init__(self, directory: str | Path, root: str | Path) -> None:
        self.dir, self.root = Path(directory), Path(root)
        self._tables: dict[str, tuple[tuple[int, int], Optional[Table]]] = {}
        self._base: Optional[dict[str, Any]] = None

    # -- files
    def _shard_path(self, test_file: str) -> Path:
        return self.dir / "shards" / (hashlib.sha1(test_file.encode()).hexdigest()[:16] + ".json")

    def _read(self, p: Path) -> Any:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, p: Path, obj: Any) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, p)

    def baseline(self) -> dict[str, Any]:
        if self._base is None:
            self._base = self._read(self.dir / "baseline.json") or {"files": {}, "import_time": {}}
        return self._base

    def table(self, rel: str) -> Optional[Table]:
        p = self.root / rel
        try:
            st = p.stat()
        except OSError:
            return None
        sig = (st.st_mtime_ns, st.st_size)
        hit = self._tables.get(rel)
        if hit and hit[0] == sig:
            return hit[1]
        t = unit_table(p.read_bytes())
        self._tables[rel] = (sig, t)
        return t

    def _file_hash(self, rel: str) -> str:
        try:
            return hashlib.sha1((self.root / rel).read_bytes()).hexdigest()[:12]
        except OSError:
            return ""

    # -- decision
    def narrow(self, changed: Iterable[str], tests: dict[str, list[str]]) -> dict[str, Optional[list[str]]]:
        """`tests`: test file -> [] (to be narrowed). Returns test file -> None (run the whole file) or the node ids to run (maybe [])."""
        files = [c for c in changed]
        base = self.baseline()
        stale: dict[str, dict[str, str]] = {}                       # changed file -> {unit: current hash} (current table units)
        whole = False
        for rel in files:
            cur, old = self.table(rel), base["files"].get(rel)
            if cur is None or old is None or old[0] != cur[0]:
                whole = True
                break
            if set(self._changed_units(cur, old)) & set(base["import_time"].get(rel, [])):
                whole = True                                        # a function that runs at import time changed: every importer is affected
                break
            stale[rel] = {q: v[2] for q, v in cur[1].items()}
        res: dict[str, Optional[list[str]]] = {}
        for tf in tests:
            shard = None if whole else self._read(self._shard_path(tf))
            if shard is None or shard.get("th") != self._file_hash(tf):
                res[tf] = None
                continue
            sel: list[str] = []
            for nodeid, rec in shard["tests"].items():
                if rec["o"] or any(rel in rec["r"] for rel in files):
                    sel.append(nodeid)
                    continue
                for rel in files:
                    ent = rec["f"].get(rel)
                    if ent and any(stale[rel].get(q) != h for q, h in ent[1].items()):
                        sel.append(nodeid)
                        break
            res[tf] = sel
        return res

    @staticmethod
    def _changed_units(cur: Table, old: Any) -> list[str]:
        return [q for q in set(cur[1]) | set(old[1]) if (cur[1][q][2] if q in cur[1] else None) != old[1].get(q)]

    # -- record
    def update(self, raw_path: str | Path, commit: Iterable[str] = ()) -> int:
        """Merge a recorder output into the shards. `commit`: changed files whose affected tests have all just been re-recorded: their baseline
        becomes the current version. Returns the number of tests recorded."""
        data = (self._read(Path(raw_path)) if str(raw_path) else None) or {"tests": {}, "import_time": {}}
        by_file: dict[str, dict[str, Any]] = {}
        for nodeid, rec in data["tests"].items():
            ent: dict[str, Any] = {}
            for rel, lines in rec["calls"].items():
                t = self.table(rel)
                if t is None:
                    continue
                units = {u for u in (unit_at(t, ln) for ln in lines) if u}
                ent[rel] = [t[0], {u: t[1][u][2] for u in units}]
            by_file.setdefault(nodeid.split("::", 1)[0], {})[nodeid] = {"o": bool(rec["opaque"]), "r": rec["reads"], "f": ent}
        for tf, nodes in by_file.items():
            p = self._shard_path(tf)
            old = self._read(p)
            th = self._file_hash(tf)
            tests = old["tests"] if old and old.get("th") == th else {}
            tests.update(nodes)
            self._write(p, {"tf": tf, "th": th, "tests": tests})
        base = self.baseline()
        for rel, lines in data.get("import_time", {}).items():
            t = self.table(rel)
            if t:
                have = set(base["import_time"].get(rel, []))
                have |= {u for u in (unit_at(t, ln) for ln in lines) if u}
                base["import_time"][rel] = sorted(have)
        for rel in commit:
            t = self.table(rel)
            if t:
                base["files"][rel] = [t[0], {q: v[2] for q, v in t[1].items()}]
        self._write(self.dir / "baseline.json", base)
        return len(data["tests"])
