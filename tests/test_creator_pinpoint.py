"""P0.5 pinpoint: traceback parser, Ochiai ranking, mutation enumeration, and an end-to-end plant-and-locate on a toy package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from creator.tools import pinpoint as P

PY_TB = '''Traceback (most recent call last):
  File "/r/tests/test_a.py", line 5, in test_x
    f()
  File "/r/pkg/a.py", line 9, in f
    1/0
ZeroDivisionError: division by zero'''

PYTEST_TB = '''tests/test_a.py:7: in test_a
    assert f(1) == 2
pkg/a.py:3: in f
    return x - 1
E   assert 0 == 2
E    +  where 0 = f(1)'''


def test_parse_python_traceback_innermost_first():
    tb = P.parse_traceback(PY_TB, "/r")
    assert (tb.exc_type, tb.message) == ("ZeroDivisionError", "division by zero")
    assert [(f.file, f.line, f.func, f.is_test) for f in tb.frames] == [("pkg/a.py", 9, "f", False), ("tests/test_a.py", 5, "test_x", True)]
    assert tb.repo_frames()[0].file == "pkg/a.py"


def test_parse_pytest_short_with_assert_and_outside_frames_dropped():
    text = PYTEST_TB + '\n  File "C:/py/Lib/site-packages/pluggy/_callers.py", line 1, in x'
    tb = P.parse_traceback(text, None)
    assert tb.exc_type == "AssertionError" and tb.message.startswith("assert 0 == 2")
    assert tb.frames[0].file == "pkg/a.py" and tb.frames[0].line == 3
    assert all("site-packages" not in f.file for f in tb.frames)


def test_parse_assert_source_line_and_windows_paths():
    tb = P.parse_traceback('>       assert f(1) == 2\nE       assert 1 == 2\n\ntests\\test_a.py:7: AssertionError', None)
    assert tb.assert_line == "assert f(1) == 2" and tb.exc_type == "AssertionError"
    assert tb.frames[0].file == "tests/test_a.py"


def test_parse_diff_lines():
    d = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,4 @@\n a\n+b\n c\n-d\n+e\n"
    assert P.parse_diff_lines(d) == {"x.py": {2, 4}}


def test_ochiai_and_ranking_with_boosts(tmp_path):
    runs = {
        "t::fail": {"outcome": "failed", "lines": {"m.py": [1, 2, 3]}, "text": "m.py:3: in f\nE   assert 1 == 2", "tail": []},
        "t::pass1": {"outcome": "passed", "lines": {"m.py": [1, 2]}, "text": "", "tail": []},
        "t::pass2": {"outcome": "passed", "lines": {"m.py": [1]}, "text": "", "tail": []},
    }
    r = P.rank_lines(runs)
    assert r[0][:2] == ("m.py", 3)                      # only the failing test reaches line 3
    assert P.ochiai(1, 2, 1) < P.ochiai(1, 1, 1) < P.ochiai(1, 0, 1)
    # a diff on line 2 lifts it above line 1 (same spectrum) but the exclusive line 3 still leads or ties by traceback
    r2 = P.rank_lines(runs, diff={"m.py": [2]})
    pos = {ln: i for i, (_f, ln, _s) in enumerate(r2)}
    assert pos[2] < pos[1]
    assert P.rank_lines({"a": {"outcome": "passed", "lines": {}, "text": "", "tail": []}}) == []


def test_enumerate_mutants_kinds_and_in_place_edits():
    src = (
        "def f(a, b):\n"
        "    if a < b:\n"
        "        return a + 1\n"
        "    for i in range(3):\n"
        "        a = a * b\n"
        "    return a - b\n"
    )
    muts = P.enumerate_mutants(src)
    kinds = {m.kind for m in muts}
    assert {"cmp_flip", "off_by_one", "arith_flip", "drop_return", "negate_cond", "swap_var"} <= kinds
    for m in muts:
        out = P.apply_mutant(src, m)
        compile(out, "m", "exec")
        assert out.count("\n") == src.count("\n")                      # edits never shift line numbers
        a, b = src.splitlines(), out.splitlines()
        assert [i + 1 for i in range(len(a)) if a[i] != b[i]] == [m.line]
    only3 = P.enumerate_mutants(src, {3})
    assert only3 and all(m.line == 3 for m in only3)


def test_end_to_end_plant_and_locate(tmp_path):
    root = tmp_path / "proj"
    (root / "creator" / "tools").mkdir(parents=True)
    (root / "tests").mkdir()
    real = Path(__file__).resolve().parents[1]
    for rel in ("creator/__init__.py", "creator/tools/__init__.py", "creator/tools/pinpoint_plugin.py", "creator/build.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text((real / rel).read_text(encoding="utf-8"), encoding="utf-8")
    (root / "lib.py").write_text(
        "def clamp(x, lo, hi):\n"
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi\n"
        "    return x\n"
        "\n"
        "def total(xs):\n"
        "    s = 0\n"
        "    for v in xs:\n"
        "        s = s + v\n"
        "    return s\n", encoding="utf-8")
    (root / "tests" / "test_lib.py").write_text(
        "import lib\n"
        "def test_clamp():\n"
        "    assert lib.clamp(5, 0, 3) == 3\n"
        "    assert lib.clamp(-1, 0, 3) == 0\n"
        "    assert lib.clamp(2, 0, 3) == 2\n"
        "def test_total():\n"
        "    assert lib.total([1, 2, 3]) == 6\n", encoding="utf-8")
    runs = P.collect(root, ["tests/test_lib.py"], python=sys.executable)
    assert {r["outcome"] for r in runs.values()} == {"passed"}
    assert 11 in runs["tests/test_lib.py::test_total"]["lines"]["lib.py"]
    assert "tests/test_lib.py" not in {f for r in runs.values() for f in r["lines"]}
    (root / "lib.py").write_text((root / "lib.py").read_text(encoding="utf-8").replace("s = s + v", "s = s - v"), encoding="utf-8")
    ranked = P.pinpoint(root, ["tests/test_lib.py"], python=sys.executable)
    assert ranked[0][:2] == ("lib.py", 11)
    assert P.pinpoint(root, [], python=sys.executable) == []


def test_generate_rows_are_labelled_and_tree_unmutated(tmp_path):
    root = tmp_path / "proj"
    (root / "creator" / "tools").mkdir(parents=True)
    (root / "tests").mkdir()
    real = Path(__file__).resolve().parents[1]
    for rel in ("creator/__init__.py", "creator/tools/__init__.py", "creator/tools/pinpoint_plugin.py", "creator/build.py"):
        (root / rel).write_text((real / rel).read_text(encoding="utf-8"), encoding="utf-8")
    lib = ("def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n")
    (root / "creator" / "lib.py").write_text(lib, encoding="utf-8")
    (root / "tests" / "test_lib.py").write_text(
        "from creator import lib\n"
        "def test_clamp():\n"
        "    assert lib.clamp(5, 0, 3) == 3\n    assert lib.clamp(-1, 0, 3) == 0\n    assert lib.clamp(2, 0, 3) == 2\n", encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    st = P.generate(root, ["creator/lib.py"], out, tests=["tests/test_lib.py"], max_rows=5, python=sys.executable)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert st.kept == len(rows) >= 2
    assert (root / "creator" / "lib.py").read_text(encoding="utf-8") == lib       # source tree untouched (temp copies only)
    for r in rows:
        assert {"file", "true_line", "mutation", "failing_tests", "traceback", "original_snippet", "mutated_snippet"} <= set(r)
        assert r["failing_tests"] and r["original_snippet"] != r["mutated_snippet"]
    assert P.summarize(out)["rows"] == len(rows)
