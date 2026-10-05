"""R7 data generation: impact-selected planted-bug runs, mutation variants, cheap MBFL, the Phase 2 row builders and the leakage guards."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from creator.tools import phase2data as D
from creator.tools import pinpoint as P
from creator.tools import pinpoint_feat as PF

SRC = ("def clamp(x, lo, hi):\n"
       "    if x < lo:\n"
       "        return lo\n"
       "    if x > hi:\n"
       "        return hi\n"
       "    return x\n")


class FakeWorker:
    """Records every run; `fail` = test ids that fail whenever the source differs from SRC."""

    def __init__(self, fail=(), cpu=0.05):
        self.calls = []
        self.fail = set(fail)
        self.last_cpu = cpu
        self.tree = Path(".")

    def run(self, tests, sources=None, timeout=1.0, trace=True):
        self.calls.append((list(tests), dict(sources or {}), trace))
        return {t: {"outcome": "failed" if t in self.fail else "passed", "lines": {"m.py": [2, 3]}, "text": "m.py:2: in f\nE   assert 1 == 2" if t in self.fail else "",
                    "tail": []} for t in tests}


def test_variants_add_alternatives_and_stay_parsable():
    base = P.enumerate_mutants(SRC)
    var = P.enumerate_mutants(SRC, None, True)
    assert len(var) > len(base)
    for m in var:
        ast.parse(P.apply_mutant(SRC, m))
    kinds = {m.kind for m in P.enumerate_mutants("def f(a, b):\n    if not a and b:\n        x = True\n        a += 1\n    return a in [b]\n", None, True)}
    assert {"bool_flip", "drop_not", "const_flip", "aug_flip", "in_flip"} <= kinds
    assert {m.kind for m in P.enumerate_mutants("def f(a, b):\n    return a and b\n")} <= {"drop_return", "swap_var"}      # default mode unchanged


def test_traced_failing_runs_only_the_covering_nodes_and_keeps_baseline_for_the_rest():
    base = {f"tests/t.py::{n}": {"outcome": "passed", "lines": {"m.py": [2]}, "text": "", "tail": [], "cpu": 0.01} for n in ("a", "b", "c")}
    w = FakeWorker(fail={"tests/t.py::a"})
    res = P.traced_failing(w, "m.py", "mutated", ["tests/t.py"], ["tests/t.py::a"], base, 1.0)
    assert [c[0] for c in w.calls] == [["tests/t.py::a"]]                         # one traced run of the single covering node
    assert res["tests/t.py::a"]["outcome"] == "failed"
    assert res["tests/t.py::b"]["outcome"] == "passed" and res["tests/t.py::c"]["lines"] == {"m.py": [2]}     # untouched tests keep the baseline
    assert P.traced_failing(FakeWorker(), "m.py", "mutated", ["tests/t.py"], ["tests/t.py::a"], base, 1.0) is None      # survivor


def test_traced_failing_two_phase_when_many_nodes_cover_the_line():
    ids = [f"tests/t.py::t{i}" for i in range(20)]
    base = {k: {"outcome": "passed", "lines": {"m.py": [2]}, "text": "", "tail": [], "cpu": 0.5} for k in ids}
    w = FakeWorker(fail={ids[3]})
    res = P.traced_failing(w, "m.py", "mutated", ["tests/t.py"], ids, base, 1.0)
    assert [c[2] for c in w.calls] == [False, True] and w.calls[1][0] == [ids[3]]  # untraced pass over all nodes, traced re-run of the failing one
    assert res[ids[3]]["outcome"] == "failed"


def test_mbfl_fast_stops_on_the_first_fix_and_respects_the_budget():
    base = {"t::f1": {"outcome": "passed", "lines": {"m.py": [2]}, "cpu": 0.1}, "t::f2": {"outcome": "passed", "lines": {"m.py": [2]}, "cpu": 0.2}}

    class W(FakeWorker):
        def run(self, tests, sources=None, timeout=1.0, trace=True):
            self.calls.append((list(tests), dict(sources or {}), trace))
            fixed = "x < lo" in sources["m.py"]                                  # the mutant that restores the original comparison
            return {t: {"outcome": "passed" if fixed else "failed", "lines": {}, "text": "", "tail": []} for t in tests}
    buggy = SRC.replace("x < lo", "x <= lo")
    w = W()
    out, spent, runs = PF.mbfl_fast(w, ["t::f1", "t::f2"], base, "m.py", buggy, [("m.py", 4), ("m.py", 2), ("m.py", 5)], per_line=4)
    assert all(c[0] == ["t::f1", "t::f2"] and c[2] is False for c in w.calls)      # untraced, only the failing tests
    assert out[("m.py", 2)][4] == 1.0 and ("m.py", 5) not in out                 # fix on line 2 ends the search before line 5
    assert spent == pytest.approx(0.05 * runs)
    w2 = W(cpu=1.0)
    _o, spent2, runs2 = PF.mbfl_fast(w2, ["t::f1"], base, "m.py", buggy, [("m.py", 4), ("m.py", 2)], per_line=3, budget_cpu=1.5)
    assert runs2 <= 2 and spent2 <= 2.0


def test_wilson_interval_brackets_the_rate():
    lo, hi = PF.wilson(60, 100)
    assert lo < 0.6 < hi and 0.49 < lo and hi < 0.70
    assert PF.wilson(0, 0) == (0.0, 0.0)


def _row(tmp_snapshot_src=SRC, line=2):
    m = next(x for x in P.enumerate_mutants(tmp_snapshot_src) if x.line == line and x.kind == "cmp_flip")
    mutated = P.apply_mutant(tmp_snapshot_src, m)
    keys = [["creator/lib.py", 4], ["creator/lib.py", 2], ["creator/lib.py", 6]]
    xs = [[1, 0, 1, 3, 1, 1.0] + [0] * 19, [1, 0, 1, 3, 1, 0.9] + [0] * 19, [1, 0, 1, 3, 1, 0.5] + [0] * 19]
    row = {"file": "creator/lib.py", "true_line": line, "mutation": m.kind, "failing_tests": ["tests/test_lib.py::test_clamp"],
           "traceback": "tests/test_lib.py:5: in test_clamp\n    assert lib.clamp(2, 0, 3) == 2\ncreator/lib.py:2: in clamp\nE   assert 1 == 2",
           "mutant": {"kind": m.kind, "start": m.start, "end": m.end, "new": m.new_text}, "cand": {"cols": PF.BASE_FEATS, "keys": keys, "x": xs}}
    return row, mutated


def test_pinpoint_and_debug_fix_rows():
    row, mutated = _row()
    pr = D.pinpoint_row(row, mutated)
    assert pr["label"] == 2 and pr["messages"][2]["content"] == "2"
    user = pr["messages"][1]["content"]
    assert "[2] creator/lib.py:2" in user and ">>> 2|" in user and "Failing test: tests/test_lib.py::test_clamp" in user
    row["cand"]["keys"][1] = ["creator/lib.py", 99]
    assert D.pinpoint_row(row, mutated) is None                                   # true line not among the candidates
    row, mutated = _row()
    fx = D.debug_fix_row(row, SRC, mutated)
    ans = fx["messages"][2]["content"]
    assert "<<<<<<< SEARCH\n    if x <= lo:\n=======\n    if x < lo:\n>>>>>>> REPLACE" in ans
    assert ans.count("SEARCH") == 1 and "FILE: creator/lib.py" in ans
    assert mutated.count(ans.split("<<<<<<< SEARCH\n")[1].split("\n=======")[0]) == 1      # the SEARCH text is unique in the buggy file


def _suite(tmp_path):
    sd = tmp_path / "suite"
    (sd / "app_base" / "minishop").mkdir(parents=True)
    (sd / "app_base" / "minishop" / "cart.py").write_text("def total_with_shipping_and_tax(items, rate):\n    return sum(items) * (1 + rate) + shipping_cost_for_the_whole_order(items)\n", encoding="utf-8")
    (sd / "tasks.json").write_text(json.dumps({"tasks": [
        {"id": "fn.zorblax", "family": "fn", "name": "zorblax", "request": "Write the Python function `zorblax` that frobnicates the widget list",
         "stub": "def zorblax(widgets):\n    raise NotImplementedError\n", "tests": "[{'args': [[1]], 'expect': 2}]"}]}), encoding="utf-8")
    return sd


def test_guard_blocks_the_suite_and_anything_derived(tmp_path):
    g = D.Guard(_suite(tmp_path))
    assert g.violation("def zorblax(widgets):\n    return [w * 2 for w in widgets]") is not None             # a suite function name
    assert g.violation("import minishop.cart") is not None
    assert g.violation("x = 1\n    return sum(items) * (1 + rate) + shipping_cost_for_the_whole_order(items)\n") is not None                            # a source line of app_base
    assert g.violation("def add(a, b):\n    return a + b\n") is None
    rows = [{"id": "r1", "messages": [{"role": "user", "content": "def add(a, b):\n    return a + b"}], "meta": {"public": True}}]
    assert D.audit(rows, g)["clean"]
    rows.append({"id": "r2", "messages": [{"role": "user", "content": "please frobnicate: zorblax"}], "meta": {"public": True}})
    with pytest.raises(AssertionError):
        D.audit(rows, g)
    with pytest.raises(AssertionError):                                                                      # a tree row needs the pre-cut snapshot
        D.audit([{"id": "r3", "messages": [{"role": "user", "content": "def add(a, b): pass"}], "meta": {"tree": "precut", "commit": "abc"}}], g)


def test_build_planted_from_generator_rows(tmp_path):
    g = D.Guard(_suite(tmp_path))
    snap = tmp_path / "snap"
    (snap / "creator").mkdir(parents=True)
    (snap / "creator" / "lib.py").write_text(SRC, encoding="utf-8")
    (snap / "snapshot_meta.json").write_text(json.dumps({"commit": "abc123", "committed_unix": 1.0}), encoding="utf-8")
    row, _mut = _row()
    row["src_sha"] = hashlib.sha1(SRC.encode("utf-8")).hexdigest()[:12]
    stale = dict(row, src_sha="deadbeef0000")
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps(row) + "\n" + json.dumps(stale) + "\n", encoding="utf-8")
    res = D.build_planted([ds], snap, tmp_path / "out", g)
    assert res["counts"]["pinpoint"] == 1 and res["counts"]["debug_fix"] == 1 and res["drops"]["source differs from the snapshot"] == 1
    rows = list(D.read_rows(sorted((tmp_path / "out").glob("*.jsonl"))))
    assert {r["kind"] for r in rows} == {"pinpoint", "debug_fix"}
    assert D.audit(rows, g, {"commit": "abc123", "committed_unix": 1.0})["rows"] == 2


def test_code_candidates_are_verified_by_running_them():
    assert D.run_program("def f(x):\n    return x + 1\nassert f(1) == 2\n")
    assert not D.run_program("def f(x):\n    return x + 1\nassert f(1) == 3\n")
    assert not D.run_program("while True:\n    pass\n", timeout=1.0)
    prog = D._cases_program("f", [{"args": [1], "expect": 2}], "def f(x):\n    return x + 1")
    assert D.run_program(prog)
    assert not D.run_program(D._cases_program("f", [{"args": [1], "expect": 3}], "def f(x):\n    return x + 1"))
