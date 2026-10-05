"""pytest plugin for creator.tools.pinpoint: per-test line coverage via sys.settrace (coverage.py is not required).

Use:  python -m pytest -p creator.tools.pinpoint_plugin   with env PINPOINT_ROOT (tree root) and PINPOINT_OUT (json path).
Records, for every test, its outcome, the (relative file -> lines) it executed in non-test repo files, the failure text, and for
failing tests the last repo lines executed (a short tail, innermost activity just before the failure). Light imports only.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

_ROOT = os.path.normcase(os.path.abspath(os.environ.get("PINPOINT_ROOT", os.getcwd()))) + os.sep
_OUT = os.environ.get("PINPOINT_OUT", "")
_TAIL = 40

_scope: dict[str, str | None] = {}          # code filename -> relative path (posix) when in scope, else None
_cur: dict[str, set[int]] | None = None
_tail: list[tuple[str, int, int]] = []
_depth = 0                                   # repo-frame call depth (frames in scope only)
_data: dict[str, dict[str, Any]] = {}


def _rel(filename: str) -> str | None:
    hit = _scope.get(filename, "?")
    if hit != "?":
        return hit
    rel = None
    n = os.path.normcase(os.path.abspath(filename))
    if n.startswith(_ROOT):
        r = n[len(_ROOT):].replace(os.sep, "/")
        base = r.rsplit("/", 1)[-1]
        if (r.endswith(".py") and not r.startswith(("tests/", ".venv/", "venv/", "state/")) and "site-packages" not in r
                and not r.endswith("pinpoint_plugin.py") and not (base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py")):
            rel = r
    _scope[filename] = rel
    return rel


def _local(frame: Any, event: str, arg: Any) -> Any:
    global _depth
    if event == "line" and _cur is not None:
        rel = _scope[frame.f_code.co_filename]
        _cur.setdefault(rel, set()).add(frame.f_lineno)           # type: ignore[arg-type]
        _tail.append((rel, frame.f_lineno, _depth))               # type: ignore[arg-type]
        if len(_tail) > 4 * _TAIL:
            del _tail[:-_TAIL]
    elif event == "return":
        _depth -= 1
    return _local


def _glob(frame: Any, event: str, arg: Any) -> Any:
    global _depth
    if _cur is None:
        return None
    rel = _rel(frame.f_code.co_filename)
    if rel is None:
        return None
    if event == "call":
        _depth += 1
        _cur.setdefault(rel, set()).add(frame.f_code.co_firstlineno)
    return _local


import pytest  # noqa: E402


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: Any, nextitem: Any):
    global _cur, _depth
    _cur = {}
    _depth = 0
    _tail.clear()
    _data[item.nodeid] = {"outcome": "passed", "lines": {}, "text": "", "tail": []}
    t_cpu = time.process_time()
    tracing = os.environ.get("PINPOINT_TRACE", "1") != "0"          # 0: outcomes only (MBFL reruns), several times cheaper
    if tracing:
        sys.settrace(_glob)
        threading.settrace(_glob)
    try:
        yield
    finally:
        sys.settrace(None)
        threading.settrace(None)                                  # type: ignore[arg-type]
        rec = _data[item.nodeid]
        rec["cpu"] = round(time.process_time() - t_cpu, 4)           # CPU of this test (traced when tracing: a relative cost for the selection)
        rec["lines"] = {k: sorted(v) for k, v in (_cur or {}).items()}
        if rec["outcome"] != "passed":
            rec["tail"] = [[f, ln, d] for f, ln, d in _tail[-_TAIL:]]
        _cur = None


def pytest_runtest_logreport(report: Any) -> None:
    rec = _data.get(report.nodeid)
    if rec is None:
        return
    if report.failed:
        rec["outcome"] = "failed"
        rec["text"] = (rec["text"] + "\n" + (getattr(report, "longreprtext", "") or "")).strip()[:20000]
    elif report.skipped and rec["outcome"] == "passed":
        rec["outcome"] = "skipped"


def pytest_sessionstart(session: Any) -> None:
    _data.clear()


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    out = os.environ.get("PINPOINT_OUT", _OUT)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(_data, fh)
