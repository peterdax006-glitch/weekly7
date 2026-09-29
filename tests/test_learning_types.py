"""Type safety and structural integrity of the integration seam (S17a; contract C62 section 61 type safety, section 85 order).

Three things are checked, each with a planted defect it must catch:
  1. the wiring module is fully typed - every public function is annotated (independent of mypy) and mypy in strict mode is clean on it;
  2. the declared seams are really connected - every entry of wiring.HOOKS is called in each of its old files (a declared but
     uncalled hook is the silent integration gap), and every hook the module counts is declared;
  3. the records the seam produces are immutable and self-validating.
Also: the trader path guard (the curator cannot be reached from trader code) still passes with the new seam in the tree, and the
seam itself imports neither the curator nor trader_view.
Synthetic; IMPLEMENTED - NOT VALIDATED."""
import ast
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from engine.learning import trader_view as TV
from engine.learning import wiring as W
from engine.learning.core import FirewallBreach

ROOT = Path(__file__).resolve().parents[1]
WIRING_SRC = (ROOT / "engine" / "learning" / "wiring.py").read_text(encoding="utf-8")


def public_functions():
    return [(n, f) for n, f in vars(W).items() if inspect.isfunction(f) and f.__module__ == W.__name__ and not n.startswith("_")]


# ================================================================================================================ annotations
def test_every_public_wiring_function_is_fully_annotated():
    fns = public_functions()
    assert len(fns) >= 15
    gaps = []
    for name, fn in fns:
        target = inspect.unwrap(fn)                               # sink() wraps the real function; check the real one
        sig = inspect.signature(target)
        for p in sig.parameters.values():
            if p.annotation is inspect.Parameter.empty:
                gaps.append(f"{name}({p.name})")
        if sig.return_annotation is inspect.Signature.empty:
            gaps.append(f"{name} -> ?")
    assert gaps == []


def test_the_annotation_check_can_fail():
    def bare(a, b: int):
        return a

    sig = inspect.signature(bare)
    assert sig.parameters["a"].annotation is inspect.Parameter.empty and sig.return_annotation is inspect.Signature.empty


def test_wiring_is_clean_under_strict_mypy(tmp_path):
    api = pytest.importorskip("mypy.api")
    out, err, code = api.run(["--config-file", str(ROOT / "pyproject.toml"), "--cache-dir", str(tmp_path / "mypy"),
                              str(ROOT / "engine" / "learning" / "wiring.py")])
    mine = [ln for ln in out.splitlines() if "wiring.py" in ln and ": error" in ln]
    assert mine == [], "\n".join(mine)


def test_strict_mypy_would_catch_an_untyped_function(tmp_path):
    api = pytest.importorskip("mypy.api")
    f = tmp_path / "bad.py"
    f.write_text("def f(x):\n    return x\n\n\ndef g(x: int) -> str:\n    return x\n", encoding="utf-8")
    out, _, code = api.run(["--disallow-untyped-defs", "--cache-dir", str(tmp_path / "c"), str(f)])
    assert code != 0 and "no-untyped-def" in out and "return-value" in out                # planted: both defects are reported


# ================================================================================================================ seams are connected
def test_every_declared_hook_is_called_in_every_one_of_its_old_files():
    assert W.unwired_hooks() == []


def test_a_declared_hook_that_is_never_called_is_reported(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "lessons.py").write_text("from x import wiring\nwiring.on_lessons(new)\n", encoding="utf-8")
    hooks = {"lessons": W.Hook("on_lessons", ("engine/lessons.py",), "x"),
             "post_mortem": W.Hook("on_post_mortem", ("engine/lessons.py",), "x"),        # declared, never called
             "ghost": W.Hook("on_ghost", ("engine/nowhere.py",), "x")}                     # site file does not exist
    assert W.unwired_hooks(tmp_path, hooks) == ["post_mortem @ engine/lessons.py", "ghost @ engine/nowhere.py"]
    (tmp_path / "engine" / "lessons.py").write_text("_wiring().on_post_mortem(a)\nwiring.on_lessons(b)\n", encoding="utf-8")
    assert W.unwired_hooks(tmp_path, {k: v for k, v in hooks.items() if k != "ghost"}) == []


def test_every_counted_hook_is_declared_and_every_declared_hook_is_counted():
    counted = set(re.findall(r'(?:@sink|HUB\.calls)\(?\[?"(\w+)"', WIRING_SRC))
    assert counted == set(W.HOOKS) | set(W.PERIOD_HOOKS), (counted ^ (set(W.HOOKS) | set(W.PERIOD_HOOKS)))
    fn_names = {n for n, _ in public_functions()}
    assert {h.function for h in W.HOOKS.values()} <= fn_names


def test_hook_functions_named_in_the_table_exist_and_sinks_keep_their_identity():
    for name, h in W.HOOKS.items():
        fn = getattr(W, h.function)
        assert callable(fn) and fn.__name__ == h.function                                  # functools.wraps: the decorator hides nothing
        assert (fn.__doc__ or "").strip(), f"{h.function} has no docstring"


# ================================================================================================================ immutability
def test_the_records_the_seam_produces_are_frozen():
    v = W.PromotionVerdict("s", "2026-09-29", False, True, (W.Check("c", True, "ok"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.allowed = False                                                                  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.checks[0].ok = False                                                             # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        W.SinkError("h", "E", "m").hook = "x"                                              # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        W.HOOKS["lessons"].function = "x"                                                  # type: ignore[misc]
    assert v.digest() == W.PromotionVerdict("s", "2026-09-29", False, True, (W.Check("c", True, "ok"),)).digest()
    assert v.digest() != dataclasses.replace(v, allowed=False).digest()                    # the digest sees the verdict


def test_blockers_list_exactly_the_failed_checks():
    v = W.PromotionVerdict("s", "d", True, False, (W.Check("a", True, "ok"), W.Check("b", False, "why"), W.Check("c", False, "why2")))
    assert v.blockers == ("b: why", "c: why2")
    assert W.PromotionVerdict("s", "d", False, True, ()).blockers == ()


# ================================================================================================================ trader path
def test_the_trader_path_still_cannot_reach_the_curator_with_the_seam_in_the_tree():
    n = TV.assert_trader_path_clean()
    assert n > 10
    closure = TV.trader_closure()
    assert "engine.learning.curator" not in closure
    assert "engine.learning.wiring" in closure          # the old engine reaches the seam (that is its job); the seam reaches no curator


def test_a_module_that_imports_the_curator_would_be_caught(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "engine" / "trader_entry.py").write_text("from engine.learning import curator\n", encoding="utf-8")
    (tmp_path / "engine" / "learning").mkdir()
    (tmp_path / "engine" / "learning" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "engine" / "learning" / "curator.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(FirewallBreach):
        TV.assert_trader_path_clean(entries=["engine/trader_entry.py"], root=tmp_path)


def test_wiring_imports_nothing_from_the_curator_or_trader_view():
    tree = ast.parse(WIRING_SRC)
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
           [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in mods if m.endswith("curator") or m.endswith("trader_view")]
