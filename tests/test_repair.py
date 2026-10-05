"""R2 / R4 pure-text parts: loop guard, failure brief and score, suspect lines, one-line prefill, round planning, streamed abort."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from creator import modelpool as MP
from creator import repair as RP

LADDER = "".join(f"    if n == {i}:\n        return (0.0, 0.0)\n" for i in range(1, 7))


def test_looping_catches_numbered_ladder_and_plain_repeat():
    assert RP.looping("def f(n):\n" + LADDER)
    assert RP.looping("x = 1\n" + "    total = total + item.price\n" * 6)
    assert RP.looping("abcdefghijklmnop" * 5)


def test_looping_leaves_real_code_alone():
    code = "def f(xs):\n    out = []\n    for x in xs:\n        if x > 0:\n            out.append(x * 2)\n    return out\n"
    assert not RP.looping(code)
    assert not RP.looping("    if n == 1:\n        return 0\n    if n == 2:\n        return 1\n")


def test_brief_and_score():
    val = "FAIL FAILED: FAIL tests.test_examples.test_example[case0]: AssertionError: [1, 2] -> 3 != 4\nassert False"
    assert RP.brief(val) == "[1, 2] -> 3 != 4"
    assert RP.score("PASS") == 0
    assert RP.score("FAIL apply: SEARCH not found") == 1000
    assert RP.score(val) == 1
    two = val + "\nFAIL tests.test_examples.test_example[case1]: AssertionError: x"
    assert RP.score(two) == 2


def test_candidates_skip_tests_docstrings_and_tried(tmp_path: Path):
    pin = "solution.py:7:     return a + b\nsolution.py:5:     raise NotImplementedError\ntests/test_x.py:3: assert f(1)\nsolution.py:6:\nsolution.py:9:     y = 2"
    got = RP.candidates(pin, tmp_path, {("solution.py", 9)})
    assert got == [("solution.py", 7, "return a + b")]


def test_line_prefill_unique_and_duplicate(tmp_path: Path):
    (tmp_path / "solution.py").write_text("def f(a):\n    x = 1\n    return a\n    y = 1\n    x = 1\n", encoding="utf-8")
    pre, search = RP.line_prefill(tmp_path, "solution.py", 3, with_file=False)
    assert pre == "<<<<<<< SEARCH\n    return a\n=======\n" and search == "    return a"
    pre, search = RP.line_prefill(tmp_path, "solution.py", 5, with_file=True)
    assert pre.startswith("FILE: solution.py\n<<<<<<< SEARCH\n    y = 1\n    x = 1\n=======\n    y = 1\n") and search == "    y = 1\n    x = 1"


def test_plan_rounds_and_stop_early():
    assert RP.plan_rounds("none", 3) == []
    assert RP.plan_rounds("line", 2) == ["line", "line"]
    assert RP.plan_rounds("line+regen", 3) == ["line", "line", "regen"]
    assert RP.plan_rounds("regen+line", 1) == ["line"]
    assert not RP.stop_early([2, 1])
    assert RP.stop_early([2, 2, 3])
    assert not RP.stop_early([2, 2, 1])


def test_snapshot_restore(tmp_path: Path):
    (tmp_path / "solution.py").write_text("a = 1\n", encoding="utf-8")
    snap = RP.py_snapshot(tmp_path)
    (tmp_path / "solution.py").write_text("a = 2\n", encoding="utf-8")
    RP.py_restore(tmp_path, snap)
    assert (tmp_path / "solution.py").read_text(encoding="utf-8") == "a = 1\n"


class _Looper(BaseHTTPRequestHandler):
    closed = threading.Event()

    def log_message(self, *a: object) -> None:
        pass

    def do_POST(self) -> None:
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        assert body["stream"] is True and body["repeat_penalty"] == 1.15
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for i in range(200):
                piece = f"    if n == {i}:\n        return 0\n"
                self.wfile.write(b"data: " + json.dumps({"content": piece, "stop": False}).encode() + b"\n\n")
                self.wfile.flush()
        except OSError:
            _Looper.closed.set()


def test_slot_call_aborts_a_looping_stream():
    srv = HTTPServer(("127.0.0.1", 0), _Looper)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    spec = MP.SlotSpec("CODER", Path("x.gguf"))
    try:
        out = MP.slot_call(srv.server_address[1], spec, "prompt", None, 100, 20.0, guard=RP.looping, extra={"repeat_penalty": 1.15})
    finally:
        srv.shutdown()
    assert out["stop_type"] == "loop" and out["truncated"] and 4 <= out["tokens_predicted"] < 60


def test_regen_hint_and_attempt_sampling():
    h1 = RP.regen_hint("FAIL tests.test_examples.test_example[case0]: AssertionError: [1] -> 2 != 3")
    assert "[1] -> 2 != 3" in h1 and RP.attempt_extra(h1) is None
    h2 = RP.regen_hint("FAIL apply: SEARCH not found", 2)
    assert "short blocks" in h2 and RP.attempt_extra(h2)["temperature"] == 0.3


def test_looping_catches_growing_ladder_but_not_a_short_table():
    grow = "def f(ix):\n" + "".join(f"    if len(ix) == {i}:\n        return [{', '.join(f'ix[{j}]' for j in range(i))}]\n" for i in range(3, 8))
    assert RP.looping(grow)
    table = "    if op == 'add':\n        return a + b\n    if op == 'sub':\n        return a - b\n    if op == 'mul':\n        return a * b\n"
    assert not RP.looping(table)
