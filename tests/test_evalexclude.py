"""R13: the eval-suite exclusion list (creator.tools.evalexclude) and the three guards that read it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import trainmix as TM
from creator.tools import evalexclude as EX
from creator.tools import phase2data as D
from creator.tools import reuse as RU

REF = ("def frobnicate_widgets(items, factor=2):\n"
       "    out = []\n"
       "    for it in items:\n"
       "        if it is not None and it > 0:\n"
       "            out.append(it * factor + len(out))\n"
       "    return out\n")
RENAMED = REF.replace("frobnicate_widgets", "scale_things").replace("items", "xs")      # parameters are renamed away by the body hash


def _exclude(tmp: Path) -> Path:
    fn = [{"id": "fn.frobnicate_widgets", "name": "frobnicate_widgets", "_src": "pub:sympy:sympy/utilities/widgets.py:frobnicate_widgets", "_ref": REF,
           "request": "Write the Python function `frobnicate_widgets` that scales every positive widget and adds its position", "stub": "def frobnicate_widgets(items, factor=2):\n    raise NotImplementedError\n",
           "tests": [{"args": [[1, 2]], "expect": [2, 5]}]}]
    app = [{"id": "app2.zzz", "request": "Add a Cart.zzz_method that does something with the quantity of every line in the cart", "accept": "def test_zzz():\n    assert Cart(demo_catalog()).zzz_method() == 12345\n"}]
    p = tmp / "EXCLUDE.json"
    n = EX.write_exclude(p, fn, app)
    assert n["ids"] == 3 and n["names"] == 1 and n["source_paths"] == 1 and n["body_hashes"] >= 1
    return p


def _suite(tmp: Path) -> Path:
    sd = tmp / "suite"
    sd.mkdir()
    (sd / "tasks.json").write_text(json.dumps({"tasks": [{"id": "fn.other", "family": "fn", "name": "other_fn", "request": "r", "stub": "def other_fn():\n    raise NotImplementedError\n"}]}),
                                   encoding="utf-8")
    return sd


def test_load_missing_file_is_empty(tmp_path):
    ex = EX.load(tmp_path / "nope.json")
    assert not ex and ex.violation("anything at all") is None


def test_violation_kinds(tmp_path):
    ex = EX.load(_exclude(tmp_path))
    assert ex.violation("x", "pub:sympy:sympy/utilities/widgets.py:frobnicate_widgets")                    # id
    assert ex.violation("```python\n" + REF + "```")                                                       # same body
    assert ex.violation("```python\n" + RENAMED + "```")                                                  # renamed copy: same normalized body
    assert ex.violation("def tiny(a):\n    return a\n") is None                                            # trivial bodies are not hashed
    assert ex.violation("def frobnicate_widgets(a):\n    return a\n")                                      # def name
    assert ex.violation("Write the Python function `frobnicate_widgets` that scales every positive widget and adds its position\n"
                        "    assert Cart(demo_catalog()).zzz_method() == 12345\n"
                        "Add a Cart.zzz_method that does something with the quantity of every line in the cart")   # three suite lines
    assert ex.violation("Add a Cart.zzz_method that does something with the quantity of every line in the cart") is None   # one shared line alone is not a copy
    assert ex.violation("def add(a, b):\n    return a + b\n") is None


def test_phase2data_guard_reads_it(tmp_path):
    g = D.Guard(_suite(tmp_path), exclude=_exclude(tmp_path))
    assert g.violation("```python\n" + REF + "```") is not None
    assert g.violation("def frobnicate_widgets(a):\n    return a\n") is not None
    assert g.violation("def add(a, b):\n    return a + b\n") is None
    rows = [{"id": "r1", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "```python\n" + REF + "```"}], "meta": {"public": True}}]
    with pytest.raises(AssertionError):
        D.audit(rows, g)
    assert "pub:sympy:sympy/utilities/widgets.py:frobnicate_widgets" in g.excl.ids


def test_phase2data_build_code_drops_excluded_ids(tmp_path, monkeypatch):
    g = D.Guard(_suite(tmp_path), exclude=_exclude(tmp_path))
    cand = {"source": "rl_tasks_more", "id": "pub:sympy:sympy/utilities/widgets.py:frobnicate_widgets", "split": "train", "prompt": "write it please", "code": "def f():\n    return 1\n",
            "program": "def f():\n    return 1\nassert f() == 1\n", "licence": "BSD", "group": "g", "fname": "f"}
    monkeypatch.setattr(D, "code_candidates", lambda gd: [cand])
    res = D.build_code(tmp_path, tmp_path / "out", g, {})
    assert res["kept_total"] == 0 and any("eval-suite" in k for k in res["drops"])


def test_reuse_leakguard_reads_it(tmp_path):
    ex = _exclude(tmp_path)
    g = RU.LeakGuard.from_sources([], exclude_json=ex)
    assert "frobnicate_widgets" in g.task_names
    assert g.file_blocked("sympy", "sympy/utilities/widgets.py")
    assert not g.file_blocked("sympy", "sympy/utilities/other.py")
    import ast
    node = next(n for n in ast.walk(ast.parse(REF)) if isinstance(n, ast.FunctionDef))
    assert g.func_blocked(RU.body_hash(node), "nope")
    assert not RU.LeakGuard.from_sources([]).file_blocked("sympy", "sympy/utilities/widgets.py")


def test_trainmix_eval_set_and_screen(tmp_path, monkeypatch):
    monkeypatch.setenv(EX.ENV, str(_exclude(tmp_path)))
    e = EX.load().eval_set()
    assert "app2.zzz" in e["ids"] and e["fn_names"] == ["frobnicate_widgets"] and any(i == "app2.zzz" for i, _ in e["texts"])
    # trainmix.screen drops a row that defines the excluded function
    import collections
    row = TM.Row("r1", "coder", "g1", {"messages": [{"role": "user", "content": "write the widget function please"},
                                                {"role": "assistant", "content": "```python\ndef frobnicate_widgets(a):\n    return a\n```"}]})
    class Fz:
        @staticmethod
        def leak(**kw):
            return False
    drops: collections.Counter[str] = collections.Counter()
    assert TM.screen([row], {"eval_suite200": e}, Fz(), 4000, drops) == []
    assert drops["bake-off function"] == 1
