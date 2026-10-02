"""CR093/CR096/CR098: deterministic (no AI) generation of characterization tests."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from creator import testgen as TG

SAMPLE = '''"""sample"""
from __future__ import annotations

import os
import random
import time
from typing import Optional


def add(a: int, b: int = 2) -> int:
    return a + b


def first(xs: list) -> object:
    return xs[0]


def clip(x: Optional[int], lo: int = 0) -> int:
    if x is None:
        return lo
    return max(x, lo)


def label(s: str) -> str:
    return s.strip().upper() or "EMPTY"


def untyped(a, b):
    return a * b


def uses_files(p: str) -> bool:
    return os.path.exists(p)


def noisy(x: int) -> int:
    print(x)
    return x


def rolls() -> float:
    return random.random()


def clock() -> float:
    return time.time()


def via_side(p: str) -> bool:
    return uses_files(p)


def _private(x: int) -> int:
    return x


class Box:
    pass
'''


HANG = '''def quick(a: int):
    return a


def forever(n: int) -> int:
    while n >= 0:
        pass
    return n
'''


def put(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def pytest_run(root: Path, test: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], cwd=root,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={"PATH": "", "SYSTEMROOT": "C:\\Windows", "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"})


def sample_repo(tmp_path: Path) -> Path:
    put(tmp_path, "samplemod.py", SAMPLE)
    return tmp_path


# ------------------------------------------------------------------------------------------------ analysis

def test_side_effects_and_nondeterminism_are_flagged() -> None:
    skip = {i.name: i.skip for i in TG.analyse(SAMPLE)}
    for bad in ("uses_files", "noisy", "rolls", "clock", "via_side"):
        assert skip[bad], bad
    assert "calls uses_files" in skip["via_side"]                       # transitive
    for ok in ("add", "first", "clip", "label", "untyped", "untyped"):
        assert skip[ok] == "", ok
    assert "_private" not in skip


def test_global_writes_and_decorators_are_skipped() -> None:
    src = "X = 1\n\n\ndef bump():\n    global X\n    X += 1\n    return X\n\n\n@staticmethod\ndef d(a):\n    return a\n"
    assert {i.name: bool(i.skip) for i in TG.analyse(src)} == {"bump": True, "d": True}


def test_non_pure_imports_are_skipped_but_pure_ones_allowed() -> None:
    src = "import math\nimport subprocess as sp\n\n\ndef ok(x):\n    return math.floor(x)\n\n\ndef bad(x):\n    return sp.run(x)\n"
    assert {i.name: bool(i.skip) for i in TG.analyse(src)} == {"ok": False, "bad": True}


def fn(src: str) -> ast.FunctionDef:
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))


def test_candidates_cover_zero_negative_empty_none_and_defaults() -> None:
    calls = TG.candidate_calls(fn("def f(a: int, b: str = 'x', c: Optional[list] = None): pass"))
    assert (0, "x", None) in calls                                       # baseline
    assert (-1, "x", None) in calls and (100, "x", None) in calls        # ints: negative and boundary
    assert (0, "", None) in calls                                        # empty string
    assert (0, "x", []) in calls                                         # empty container for the Optional's inner type
    assert len(calls) <= TG.MAX_CALLS_PER_FUNCTION


def test_optional_gets_none_and_pipe_union_gets_none() -> None:
    assert (None,) in TG.candidate_calls(fn("def f(a: Optional[int]): pass"))
    assert (None,) in TG.candidate_calls(fn("def f(a: int | None): pass"))
    assert (None,) in TG.candidate_calls(fn("def f(a: 'Optional[int]'): pass"))


def test_unreproducible_signatures_give_no_calls() -> None:
    assert TG.candidate_calls(fn("def f(a, b=SOME_CONST): pass")) == []
    assert TG.candidate_calls(fn("def f(a, *, key): pass")) == []
    assert TG.candidate_calls(fn("def f(): pass")) == [()]


def test_bool_edge_values_are_not_merged_with_ints() -> None:
    calls = TG.candidate_calls(fn("def f(a: bool): pass"))
    assert (False,) in calls and (True,) in calls


# ------------------------------------------------------------------------------------------------ generation

def test_generated_source_is_valid_python_and_skips_what_it_must(tmp_path: Path) -> None:
    root = sample_repo(tmp_path)
    g = TG.generate(root, "samplemod.py")
    ast.parse(g.source)
    assert g.cases > 10 and "def test_add_edge_0" in g.source
    for name in ("uses_files", "noisy", "rolls", "clock", "via_side"):
        assert f"M.{name}" not in g.source, name                         # never called by any test
        assert name in g.skipped
    assert set(g.tested) == {"add", "first", "clip", "label", "untyped"}


def test_outcomes_pin_values_and_exception_types(tmp_path: Path) -> None:
    g = TG.generate(sample_repo(tmp_path), "samplemod.py")
    assert 'assert r == ("ok", 2)' in g.source                           # add(0, 2)
    assert 'assert r == ("ok", 1)' in g.source                           # add(-1, 2)
    assert "IndexError" in g.source                                      # first([])
    assert "assert r == (\"ok\", 'EMPTY')" in g.source                     # label("")


def test_generated_tests_pass_on_the_code_they_came_from(tmp_path: Path) -> None:
    root = sample_repo(tmp_path)
    g = TG.generate(root, "samplemod.py")
    t = put(root, "test_gen_samplemod.py", g.source)
    r = pytest_run(root, t.name)
    assert r.returncode == 0, r.stdout[-1500:]
    assert f"{g.cases + 2} passed" in r.stdout


def test_a_behaviour_change_makes_a_generated_test_fail(tmp_path: Path) -> None:
    root = sample_repo(tmp_path)
    t = put(root, "test_gen_samplemod.py", TG.generate(root, "samplemod.py").source)
    assert pytest_run(root, t.name).returncode == 0
    for old, new in (("return a + b", "return a - b"),                    # arithmetic
                     ("return max(x, lo)", "return min(x, lo)"),         # comparison direction
                     ("or \"EMPTY\"", "or \"NONE\""),                     # a constant
                     ("return xs[0]", "return xs[1]")):                 # an index changes: [0] now raises
        src = SAMPLE.replace(old, new)
        assert src != SAMPLE, old
        put(root, "samplemod.py", src)
        r = pytest_run(root, t.name)
        assert r.returncode != 0 and "failed" in r.stdout, old
        put(root, "samplemod.py", SAMPLE)
    assert pytest_run(root, t.name).returncode == 0                      # restored: green again


def test_a_changed_exception_type_fails_the_test(tmp_path: Path) -> None:
    root = sample_repo(tmp_path)
    t = put(root, "test_gen_samplemod.py", TG.generate(root, "samplemod.py").source)
    put(root, "samplemod.py", SAMPLE.replace("return a * b", "raise ValueError('x')"))
    assert pytest_run(root, t.name).returncode != 0


def test_a_hanging_function_is_dropped_not_fatal(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(TG, "TIMEOUT_S", 3)
    put(tmp_path, "hangmod.py", HANG)
    g = TG.generate(tmp_path, "hangmod.py")
    assert "forever" in g.skipped and "quick" in g.tested


def test_set_results_are_pinned_by_type_not_by_unstable_repr(tmp_path: Path) -> None:
    put(tmp_path, "setmod.py", "def uniq(xs: list):\n    return {str(x) + 'k' for x in xs}\n")
    g = TG.generate(tmp_path, "setmod.py")
    t = put(tmp_path, "test_gen_setmod.py", g.source)
    assert pytest_run(tmp_path, t.name).returncode == 0                   # passes under any hash seed


def test_module_that_cannot_import_yields_no_cases(tmp_path: Path) -> None:
    put(tmp_path, "broken.py", "import definitely_not_a_module_xyz\n\n\ndef f(a: int):\n    return a\n")
    g = TG.generate(tmp_path, "broken.py")
    assert g.cases == 0 and g.tested == []


def test_empty_module_generates_only_smoke_tests(tmp_path: Path) -> None:
    put(tmp_path, "emptymod.py", "X = 1\n")
    g = TG.generate(tmp_path, "emptymod.py")
    assert g.cases == 0 and "test_module_imports" in g.source


# ------------------------------------------------------------------------------------------------ the worker

def test_worker_identity_and_registration_shape() -> None:
    w = TG.TestGenWorker()
    assert w.name == "self-testgen-v1" and w.steps == ("tested", "coverage")
    assert w.name.startswith("self-")                                    # creator.objective.self_share counts it


def test_worker_writes_a_passing_test_and_signs_it(tmp_path: Path) -> None:
    root = sample_repo(tmp_path)
    pkg = SimpleNamespace(outputs=("samplemod.py", "tests/test_x.py", "missing.py"))
    res = TG.TestGenWorker()(None, pkg, root)                           # type: ignore[arg-type]
    assert res.claimed_done and res.by == "self-testgen-v1" and "test_gen_samplemod.py" in res.notes
    assert (root / "tests" / "test_gen_samplemod.py").is_file()
    assert pytest_run(root, "tests/test_gen_samplemod.py").returncode == 0


def test_worker_does_nothing_for_modules_that_already_have_tests(tmp_path: Path) -> None:
    root = sample_repo(tmp_path)
    put(root, "tests/test_samplemod.py", "def test_x():\n    assert True\n")
    res = TG.TestGenWorker()(None, SimpleNamespace(outputs=("samplemod.py",)), root)      # type: ignore[arg-type]
    assert not res.claimed_done and res.by == ""
    assert not (root / "tests" / "test_gen_samplemod.py").exists()


def test_worker_reports_failure_without_targets(tmp_path: Path) -> None:
    res = TG.TestGenWorker()(None, SimpleNamespace(outputs=()), tmp_path)                 # type: ignore[arg-type]
    assert not res.claimed_done


def test_generated_tests_for_a_real_creator_module_pass(tmp_path: Path) -> None:
    """Integration (CR098): generate for creator/selfworkers.py (its pure rule functions run, its subprocess users are skipped)."""
    repo = Path(__file__).resolve().parents[1]
    g = TG.generate(repo, "creator/selfworkers.py")
    assert g.cases > 5 and "unused_imports" in g.tested
    assert "reset" in g.skipped                                         # runs git: side effect
    t = put(tmp_path, "test_gen_selfworkers.py", g.source)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(t)], cwd=tmp_path, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONPATH": str(repo)})
    assert r.returncode == 0, r.stdout[-1500:]
