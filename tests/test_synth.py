"""Tiny synthetic specs for creator.synth (no LLM, no network)."""
from pathlib import Path

from creator import synth as S


def _spec(name, params, doc, examples):
    return S.Spec(name, tuple(params), doc, tuple(examples))


def test_idiom_chosen_by_docstring_and_checked():
    sp = _spec("f", ["xs"], "Remove repeated items, keeping the first occurrence.", [(([1, 2, 1],), [1, 2])])
    label, body = S.synthesize(sp)
    assert label.startswith("idiom:dedupe") and S.passes(sp, body)


def test_idiom_rejected_when_example_contradicts():
    sp = _spec("f", ["xs"], "Remove repeated items.", [(([1, 2, 1],), [2, 1])])
    assert all(lbl != "idiom:dedupe" for lbl, b in S.candidates(sp) if S.passes(sp, b))


def test_enumeration_arithmetic():
    sp = _spec("f", ["a", "b"], "Mystery.", [((2, 3), 5), ((10, 1), 11), ((0, 0), 0)])
    got = S.synthesize(sp)
    assert got is not None and S.passes(sp, got[1])


def test_enumeration_list():
    sp = _spec("f", ["xs"], "Mystery.", [(([3, 1, 2],), 3), (([5, 9],), 9)])
    got = S.synthesize(sp)
    assert got is not None and S.passes(sp, got[1])


def test_doc_examples_extracted():
    ex = S.examples_from_doc("Run-length encode: 'aaab' -> 'a3b1'. f([1, 2]) == 3", "f", 1)
    assert (("aaab",), "a3b1") in ex and (([1, 2],), 3) in ex


def test_no_examples_no_enum():
    assert S.synthesize(_spec("f", ["x"], "Mystery.", [])) is None


def test_solver_fills_stub(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "solution.py").write_text('def total(a, b):\n    """Mystery."""\n    raise NotImplementedError()\n')
    (tmp_path / "tests" / "test_solution.py").write_text("def test_0():\n    assert total(*(2, 3)) == 5\n    assert total(*(4, 4)) == 8\n")
    res = S.SynthSolver()({"id": "x"}, tmp_path)
    assert res.claimed_done
    ns: dict = {}
    exec((tmp_path / "app" / "solution.py").read_text(), ns)
    assert ns["total"](1, 6) == 7


_DOC = "Remove repeated items, keeping the first occurrence."


def test_ambiguous_single_example_abstains():
    # [1, 2] -> [1, 2] is satisfied by dedupe AND by running_max; the probe [3, 1, 3] tells them apart
    sp = S.Spec("f", ("xs",), _DOC, ((([1, 2],), [1, 2]),), probes=(([3, 1, 3],),))
    assert S.synthesize(sp) is None


def test_two_examples_only_one_idiom_accepts():
    sp = S.Spec("f", ("xs",), _DOC, ((([1, 2, 1],), [1, 2]), (([3, 3],), [3])), probes=(([3, 1, 3],),))
    got = S.synthesize(sp)
    assert got is not None and got[0] == "idiom:dedupe"


def test_one_example_one_cue_abstains():
    sp = S.Spec("f", ("xs",), "Reverse it.", ((([1, 2],), [2, 1]),))
    assert S.synthesize(sp) is None


def test_one_example_two_cues_accepts_when_unambiguous():
    sp = S.Spec("f", ("xs",), _DOC, ((([1, 2, 1],), [1, 2]),))
    assert S.synthesize(sp) is not None


def test_idiom_disabled_by_name():
    sp = S.Spec("f", ("xs",), _DOC, ((([1, 2, 1],), [1, 2]), (([3, 3],), [3])))
    got = S.synthesize(sp, frozenset({"dedupe"}))
    assert got is None or got[0] != "idiom:dedupe"
