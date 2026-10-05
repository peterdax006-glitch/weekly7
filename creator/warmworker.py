"""Warm test worker: a long-lived python process with the heavy third-party imports (pytest, numpy, pandas ...) already loaded, driven
over a JSON-lines pipe. Each request runs pytest.main() in that process on explicit targets and writes a junit xml that
creator.testrun.parse_junit reads unchanged. Cold pytest start-up (interpreter + plugins + numpy/pandas) is paid once, not per run.

Isolation rules (a stale module would be a false verdict, so they are strict):
  * before every run, every module whose file lives under the tree root (and the old root) is dropped from sys.modules, and the
    import caches are invalidated: edited code is always re-imported; only third-party/stdlib modules stay warm;
  * cwd, sys.path[0], os.environ and sys.argv are restored/set per run;
  * the worker is recycled after MAX_RUNS runs or when it crashes; any doubt -> the caller falls back to a cold run_pytest.
Machine budget: the CLIENT holds a creator.testslots.Slot for the duration of each run (the worker itself idles at ~0 CPU), and the pool
never keeps more workers than there are slots. The child runs with HELD_ENV set so nested test launches run under that slot."""
from __future__ import annotations

import atexit
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

MAX_RUNS = 200
DEFAULT_PRELOAD = ("pytest", "numpy", "pandas")


# ------------------------------------------------------------------------------------------------------------------- child
_SIGS: dict[str, tuple[int, int]] = {}                  # module file -> (mtime_ns, size) it was imported with


def _sig(f: str) -> tuple[int, int]:
    try:
        st = os.stat(f)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def _under(f: str, norm: list[str]) -> bool:
    n = os.path.normcase(os.path.realpath(f))
    return any(n.startswith(r) for r in norm)


_IMPS: dict[tuple[str, tuple[int, int]], frozenset[str]] = {}


def _static_imports(f: str, modname: str) -> frozenset[str]:
    """Dotted names a module may import (any depth, relative imports resolved); cached by (file, signature)."""
    import ast
    key = (f, _sig(f))
    got = _IMPS.get(key)
    if got is not None:
        return got
    out: set[str] = set()
    try:
        tree = ast.parse(open(f, "rb").read())
    except (OSError, SyntaxError, ValueError):
        return frozenset(["*"])
    pkg = modname.split(".") if os.path.basename(f) == "__init__.py" else modname.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(pkg) - (node.level - 1)
                if keep < 0:
                    continue
                base = ".".join(pkg[:keep] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                out.add(base)
                out.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
    _IMPS[key] = res = frozenset(out)
    return res


def _purge(roots: list[str], keep_unchanged: bool = True) -> int:
    """Drop from sys.modules everything that may be stale. Always: test modules and conftests, and any same-named package that belongs
    to ANOTHER tree. With keep_unchanged, a module of the tree stays warm unless its file changed since import or it holds a module-level
    reference (a module, or an object/function/class defined in a dropped module) to a dropped module - computed to a fixpoint."""
    norm = [os.path.normcase(os.path.realpath(r)) + os.sep for r in roots if r]
    mine: dict[str, str] = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and _under(str(f), norm):
            mine[name] = str(f)
    tops: set[str] = set()
    for r in roots:
        try:
            tops |= {e[:-3] if e.endswith(".py") else e for e in os.listdir(r) if e.endswith(".py") or os.path.isdir(os.path.join(r, e))}
        except OSError:
            pass
    drop: set[str] = set()
    for n, m in list(sys.modules.items()):
        if n in mine or n.split(".", 1)[0] not in tops or n.split(".", 1)[0] in sys.stdlib_module_names or _is_site(m):
            continue
        drop.add(n)                                       # a same-named module from another tree
    for n, f in mine.items():
        base = os.path.basename(f)
        if (not keep_unchanged or base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"
                or _SIGS.get(f) != _sig(f)):
            drop.add(n)
    deps = {n: _static_imports(f, n) for n, f in mine.items() if n not in drop}
    while True:                                            # fixpoint: whoever imports a dropped module (statically, or by a module-level reference)
        add = set()
        for n, imps in deps.items():
            if n in drop:
                continue
            if any(i == d or i.startswith(d + ".") for d in drop for i in imps):
                add.add(n)
                continue
            mod = sys.modules.get(n)
            for v in list(vars(mod).values()) if mod is not None else ():
                vm = v.__name__ if isinstance(v, type(sys)) else getattr(v, "__module__", None) or getattr(type(v), "__module__", None)
                if vm in drop:
                    add.add(n)
                    break
        if not add:
            break
        drop |= add
    for n in drop:
        sys.modules.pop(n, None)
        _SIGS.pop(mine.get(n, ""), None)
    return len(drop)


def _record(roots: list[str]) -> None:
    """After a run: remember the file signature every tree module was imported with (the file cannot have been edited since)."""
    norm = [os.path.normcase(os.path.realpath(r)) + os.sep for r in roots if r]
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and _under(str(f), norm):
            _SIGS.setdefault(str(f), _sig(str(f)))


def _is_site(mod: Any) -> bool:
    f = str(getattr(mod, "__file__", "") or "")
    return "site-packages" in f or "dist-packages" in f


class _OnlyTargets:
    """Collection shortcut: pytest stats every entry of tests/ (300+ files, ~0.4 s) just to ignore all but the requested files.
    Say so up front: only paths on the way to a target are looked at. Same tests run, same results."""

    def __init__(self, root: str, targets: list[str]) -> None:
        self.files = {os.path.normcase(os.path.realpath(os.path.join(root, t.split("::", 1)[0]))) for t in targets}
        self.dirs = {os.path.dirname(f) for f in self.files}
        self.anc = set()
        for d in self.dirs:
            while d and d not in self.anc:
                self.anc.add(d)
                nd = os.path.dirname(d)
                if nd == d:
                    break
                d = nd

    def pytest_ignore_collect(self, collection_path: Any, config: Any) -> Optional[bool]:
        p = os.path.normcase(str(collection_path))
        if p in self.files or p in self.anc:
            return None
        return True



def _serve() -> None:
    import contextlib
    proto = sys.stdout
    if sys.path and os.path.normcase(os.path.realpath(sys.path[0])) == os.path.normcase(os.path.dirname(os.path.realpath(__file__))):
        sys.path.pop(0)                                              # script dir (creator/) must not shadow the tree's own modules
    sys.dont_write_bytecode = True
    preload = [m for m in os.environ.get("NUPEN_WARM_PRELOAD", ",".join(DEFAULT_PRELOAD)).split(",") if m]
    loaded = []
    for m in preload:
        try:
            __import__(m)
            loaded.append(m)
        except Exception:                                              # noqa: BLE001 - an absent heavy module is just not warm
            pass
    import pytest
    _OnlyTargets.pytest_ignore_collect = pytest.hookimpl(tryfirst=True)(_OnlyTargets.pytest_ignore_collect)    # type: ignore[method-assign]
    base_path = list(sys.path)
    base_env = dict(os.environ)
    cur_root = ""
    proto.write(json.dumps({"ready": True, "preloaded": loaded, "pid": os.getpid()}) + "\n")
    proto.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        t0 = time.monotonic()
        root = str(req["root"])
        buf = io.StringIO()
        rc: Optional[int] = None
        purged = 0
        err = ""
        try:
            purged = _purge([root, cur_root], bool(req.get('keep', True)))
            cur_root = root
            importlib_invalidate()
            sys.path[:] = [root, *[p for p in base_path if os.path.normcase(os.path.realpath(p or ".")) != os.path.normcase(os.path.realpath(root))]]
            os.environ.clear()
            os.environ.update(base_env)
            os.environ.update({str(k): str(v) for k, v in (req.get("env") or {}).items()})
            os.chdir(root)
            argv = ["-q", "-p", "no:cacheprovider", "--rootdir", root, f"--junitxml={req['junit']}", "-o", "junit_family=xunit2",
                    "--basetemp", os.path.join(tempfile.gettempdir(), f"nupen_warm_{os.getpid()}"),
                    *req.get("extra", []), "--", *req["targets"]]
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = int(pytest.main(argv, plugins=[_OnlyTargets(root, req['targets']), *_impact_plugin(root, req)]))
            _record([root])
        except SystemExit as e:
            rc = int(e.code) if isinstance(e.code, int) else 2
        except BaseException as e:                                     # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        proto.write(json.dumps({"id": req.get("id"), "rc": rc, "out": buf.getvalue()[-20000:], "err": err, "purged": purged,
                                "seconds": round(time.monotonic() - t0, 4)}) + "\n")
        proto.flush()


def _impact_plugin(root: str, req: dict[str, Any]) -> list[Any]:
    """Recording plugin (creator/impactmap.py, loaded by path: it is stdlib-only) when the request asks for an impact recording."""
    if not req.get("impact"):
        return []
    import importlib.util
    spec = importlib.util.spec_from_file_location("_nupen_impactmap", os.path.join(os.path.dirname(os.path.abspath(__file__)), "impactmap.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_nupen_impactmap"] = mod
    spec.loader.exec_module(mod)
    return [mod.make_plugin(root, str(req["impact"]))]


def importlib_invalidate() -> None:
    import importlib
    importlib.invalidate_caches()


# ----------------------------------------------------------------------------------------------------------------- client
class WarmWorker:
    def __init__(self, python: str = sys.executable, preload: tuple[str, ...] = DEFAULT_PRELOAD) -> None:
        self.python, self.preload = python, preload
        self.proc: Optional[subprocess.Popen[str]] = None
        self.runs = 0
        self.startup_s = 0.0
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._n = 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 120.0) -> None:
        self.stop()
        t0 = time.monotonic()
        from creator.build import clean_env
        from creator import testslots
        env = clean_env(None, {"NUPEN_WARM_PRELOAD": ",".join(self.preload), testslots.HELD_ENV: "1"})
        self.proc = subprocess.Popen([self.python, "-u", str(Path(__file__).resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", cwd=str(Path(__file__).resolve().parents[1]),
                                     env=env)
        self._q = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._q), daemon=True).start()
        first = self._get(timeout)
        if not first or not json.loads(first).get("ready"):
            self.stop()
            raise RuntimeError("warm worker did not start")
        self.startup_s = time.monotonic() - t0
        self.runs = 0

    @staticmethod
    def _pump(p: "subprocess.Popen[str]", q: "queue.Queue[Optional[str]]") -> None:
        assert p.stdout is not None
        for ln in p.stdout:
            q.put(ln)
        q.put(None)

    def _get(self, timeout: float) -> Optional[str]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            from creator import testslots
            testslots._kill_tree(p.pid)
            p.wait(timeout=5)
        except Exception:                                              # noqa: BLE001
            pass

    def run(self, root: str | Path, targets: list[str], junit: str | Path, *, timeout: float = 600.0, extra: Optional[list[str]] = None,
            env: Optional[dict[str, str]] = None, impact: str = "") -> dict[str, Any]:
        """One pytest run in the warm process, under a machine test slot. Returns {rc, out, err, seconds, timed_out, launch_error}."""
        from creator import testslots
        with self._lock, testslots.Slot():
            t0 = time.monotonic()
            try:
                if not self.alive() or self.runs >= MAX_RUNS:
                    self.start()
                self._n += 1
                req = {"id": self._n, "root": str(root), "targets": list(targets), "junit": str(junit), "extra": extra or [],
                       "env": {k: v for k, v in (env or {}).items() if k not in ("PYTHONPATH",)}, "impact": str(impact)}
                assert self.proc is not None and self.proc.stdin is not None
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except (OSError, RuntimeError, AssertionError) as e:
                self.stop()
                return {"rc": None, "out": "", "err": "", "seconds": time.monotonic() - t0, "timed_out": False,
                        "launch_error": f"{type(e).__name__}: {e}"}
            line = self._get(timeout)
            if line is None:
                timed_out = self.alive()
                self.stop()
                return {"rc": None, "out": "", "err": "", "seconds": time.monotonic() - t0, "timed_out": timed_out,
                        "launch_error": "" if timed_out else "warm worker died"}
            self.runs += 1
            res = json.loads(line)
            res.update({"timed_out": False, "launch_error": res.get("err", "") if res.get("rc") is None else ""})
            res["seconds"] = time.monotonic() - t0
            return res


_POOL: "OrderedDict[tuple[str, str], WarmWorker]" = OrderedDict()
_POOL_LOCK = threading.Lock()


def get_worker(python: str = sys.executable, preload: tuple[str, ...] = DEFAULT_PRELOAD) -> WarmWorker:
    """The pooled worker for this python; never more workers than the machine has test slots (least recently used is stopped)."""
    from creator import testslots
    key = (python, ",".join(preload))
    with _POOL_LOCK:
        w = _POOL.pop(key, None) or WarmWorker(python, preload)
        _POOL[key] = w
        while len(_POOL) > max(1, testslots.slot_cap()):
            _k, old = _POOL.popitem(last=False)
            old.stop()
        return w


def shutdown() -> None:
    with _POOL_LOCK:
        for w in _POOL.values():
            w.stop()
        _POOL.clear()


atexit.register(shutdown)

if __name__ == "__main__":
    _serve()
