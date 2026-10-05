"""Warm pytest worker for creator.tools.pinpoint: ONE long-lived python process runs many traced pytest sessions back to back.

The expensive part of a planted-bug run is interpreter + third-party import start-up (~1 s). The worker keeps those imports and,
for each request, purges every module that lives under the tree (so a rewritten source file is re-imported, parent-package
attributes included), then calls pytest.main in-process with the settrace plugin. Protocol: one JSON request per stdin line
{"tests": [...], "out": path}; one JSON reply per stdout line {"rc": int}. The parent edits files on disk; the worker never does.

Parent side: Warm (one worker + one tree copy), WarmPool (n of them). Light imports (stdlib only until a worker is started).
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PLUGIN = "creator.tools.pinpoint_plugin"
_KEEP = ("creator.tools.pinpoint_plugin", "creator.tools.pinpoint_worker")


def _modname(rel: str) -> str:
    n = rel.replace("\\", "/")[:-3] if rel.endswith(".py") else rel
    return n.replace("/", ".").removesuffix(".__init__")


def _refs(mod: Any, names: set[str]) -> bool:
    """True when module `mod` holds a module, function, class or instance that belongs to one of `names`."""
    for v in list(vars(mod).values()):
        try:
            if isinstance(v, type(sys)):
                if v.__name__ in names:
                    return True
                continue
            if getattr(v, "__module__", None) in names or getattr(type(v), "__module__", None) in names:
                return True
        except Exception:                                              # noqa: BLE001
            continue
    return False


def _purge(root: str, changed: list[str] | None = None) -> None:
    """Drop from sys.modules what a rewritten source can have influenced. changed=None: every module under the tree.
    Otherwise: the changed modules, every repo module that (transitively) holds one of their objects, and all test/conftest modules;
    unrelated repo modules stay imported (their import cost is the bulk of a run). Parent-package attributes are removed too,
    because `from pkg import sub` would otherwise find the stale submodule there."""
    import importlib
    import linecache
    r = os.path.normcase(os.path.abspath(root)) + os.sep
    repo: dict[str, Any] = {}
    for name, mod in list(sys.modules.items()):
        if name in _KEEP:
            continue
        f = getattr(mod, "__file__", None)
        if f and os.path.normcase(os.path.abspath(f)).startswith(r):
            repo[name] = mod
    if changed is None:
        drop = set(repo)
    else:
        drop = {_modname(c) for c in changed}
        for name, mod in repo.items():
            f = os.path.normcase(os.path.abspath(mod.__file__)).replace(os.sep, "/")
            if "/tests/" in f or f.endswith("/conftest.py"):
                drop.add(name)
        grew = True
        while grew:
            grew = False
            for name, mod in repo.items():
                if name not in drop and _refs(mod, drop):
                    drop.add(name)
                    grew = True
    for name in drop:
        if name in repo:
            del sys.modules[name]
            parent, _, child = name.rpartition(".")
            pm = sys.modules.get(parent)
            if pm is not None and hasattr(pm, child):
                try:
                    delattr(pm, child)
                except AttributeError:
                    pass
    importlib.invalidate_caches()
    linecache.clearcache()


def _serve(root: str) -> int:
    import pytest
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    proto = sys.stdout
    prev: list[str] = []
    for line in sys.stdin:
        req = json.loads(line)
        c0 = time.process_time()
        cur = req.get("changed")
        _purge(root, None if cur is None else sorted(set(cur) | set(prev)))       # the previous edit was restored: purge it again
        prev = list(cur or [])
        os.environ["PINPOINT_OUT"] = req["out"]
        os.environ["PINPOINT_TRACE"] = "1" if req.get("trace", True) else "0"
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            rc = pytest.main(["-q", "-p", "no:cacheprovider", "-p", PLUGIN, "--tb=short", "--rootdir", root,
                              "-o", "addopts=", "--", *req["tests"]])
        proto.write(json.dumps({"rc": int(rc), "cpu": round(time.process_time() - c0, 3)}) + "\n")
        proto.flush()
    return 0


# ----------------------------------------------------------------------------------------------------------- parent side
class Warm:
    """One warm worker bound to one scratch tree. run() applies source edits, runs tests, restores the files."""

    RECYCLE = 120

    def __init__(self, tree: str | Path, python: str = sys.executable) -> None:
        self.tree = Path(tree)
        self.python = python
        self.proc: subprocess.Popen[str] | None = None
        self.uses = 0
        self.full = False                # True: purge every tree module on each run (slower, no dependency guess)
        self.last_cpu = 0.0
        self._lines: "queue.Queue[str]" = queue.Queue()

    def _start(self) -> None:
        from creator.build import clean_env
        env = clean_env(self.tree, {"PINPOINT_ROOT": str(self.tree), "PINPOINT_OUT": ""})
        self.proc = subprocess.Popen([self.python, "-m", "creator.tools.pinpoint_worker", str(self.tree)], cwd=str(self.tree),
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()
        self.uses = 0

    @staticmethod
    def _pump(proc: Any, q: "queue.Queue[str]") -> None:
        for ln in proc.stdout:
            q.put(ln)
        q.put("")                                                    # EOF marker

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:                                      # noqa: BLE001
                pass
            self.proc = None

    def run(self, tests: Sequence[str], sources: Mapping[str, str] | None = None, timeout: float = 120.0,
            trace: bool = True) -> dict[str, dict[str, Any]]:
        """Write `sources` (rel path -> text) into the tree, run `tests`, restore the originals; per-test result dict (see
        pinpoint.collect). RuntimeError on timeout/crash (the worker is then replaced)."""
        sources = dict(sources or {})
        orig = {rel: (self.tree / rel).read_text(encoding="utf-8") for rel in sources}
        fd, out = tempfile.mkstemp(prefix="pinpoint_", suffix=".json")
        os.close(fd)
        os.unlink(out)
        try:
            for rel, txt in sources.items():
                (self.tree / rel).write_text(txt, encoding="utf-8", newline="")
            if self.proc is None or self.proc.poll() is not None or self.uses >= self.RECYCLE:
                self.stop()
                self._start()
            self.uses += 1
            assert self.proc is not None and self.proc.stdin is not None
            try:
                req = {"tests": list(tests), "out": out, "changed": None if self.full else sorted(sources),
                       "trace": trace}
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                reply = self._lines.get(timeout=timeout)
                self.last_cpu = json.loads(reply).get("cpu", 0.0) if reply else 0.0
            except (queue.Empty, OSError):
                self.stop()
                raise RuntimeError("worker timed out") from None
            if not reply:
                self.stop()
                raise RuntimeError("worker died")
            try:
                with open(out, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError) as e:
                raise RuntimeError(f"no coverage result: {e}") from e
        finally:
            for rel, txt in orig.items():
                (self.tree / rel).write_text(txt, encoding="utf-8", newline="")
            try:
                os.unlink(out)
            except OSError:
                pass


class WarmPool:
    def __init__(self, root: str | Path, n: int, base_dir: str | Path, python: str = sys.executable) -> None:
        from creator.tools.pinpoint import make_copy
        self.base_dir = Path(base_dir)
        self.workers = [Warm(make_copy(root, self.base_dir / f"w{i}"), python) for i in range(max(1, n))]
        self._q: "queue.Queue[Warm]" = queue.Queue()
        for w in self.workers:
            self._q.put(w)

    def acquire(self) -> Warm:
        return self._q.get()

    def release(self, w: Warm) -> None:
        self._q.put(w)

    def close(self) -> None:
        for w in self.workers:
            w.stop()


if __name__ == "__main__":
    raise SystemExit(_serve(sys.argv[1]))
