"""The learned chooser: deterministic training, a planted signal is learned, no-signal outcomes are ignored, flag off = old behaviour."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from creator import action_student as A
from creator import chooser as C
from creator.curriculum import Lesson

SRC = "import json\nimport csv\n\n\ndef f(x):\n    return json.dumps(x)\n\n\ndef g_helper():\n    return 1\n"


def _rows(planted_word: str = "zebra", n: int = 12) -> list[C.Row]:
    cands = A.enumerate_actions(SRC, "mod.py")
    kinds = [a.kind for a in cands]
    assert "remove_unused" in kinds and "lazy_import" in kinds
    rows = []
    for i in range(n):
        # the objective word picks the action kind; names are never mentioned
        want = "remove_unused" if i % 2 else "lazy_import"
        word = planted_word if want == "remove_unused" else "walrus"
        rows.append(C.Row(f"please do the {word} thing now", SRC, cands, [kinds.index(want)]))
    return rows


def test_training_is_deterministic() -> None:
    a, b = C.Chooser().fit(_rows(), epochs=40), C.Chooser().fit(_rows(), epochs=40)
    assert (a.w == b.w).all() and a.trained


def test_planted_signal_is_learned_and_roundtrips(tmp_path: Path) -> None:
    ch = C.Chooser().fit(_rows(), epochs=60)
    cands = A.enumerate_actions(SRC, "mod.py")
    srcs = {"mod.py": SRC}
    assert cands[ch.pick("do the zebra thing", cands, srcs) - 1].kind == "remove_unused"      # type: ignore[operator]
    assert cands[ch.pick("do the walrus thing", cands, srcs) - 1].kind == "lazy_import"        # type: ignore[operator]
    ch.save(tmp_path / "c.json")
    again = C.Chooser.load(tmp_path / "c.json")
    assert again is not None and again.pick("do the zebra thing", cands, srcs) == ch.pick("do the zebra thing", cands, srcs)
    assert C.Chooser.load(tmp_path / "missing.json") is None


def test_negative_outcome_pushes_the_action_down_and_no_signal_is_ignored() -> None:
    def les(adopted: Any, verdict: str, solver: str = "nupen-model-v2") -> Lesson:
        after = A.apply_action(SRC, A.enumerate_actions(SRC, "m.py")[0])
        return Lesson("l", "p", "c", "shrink", "make it lighter", files_before={"m.py": SRC}, files_after={"m.py": after or SRC},
                      solver=solver, adopted=adopted, verdict=verdict)
    assert C.lesson_rows([les(None, ""), les(False, "cancelled: pulled back before evaluation"), les(False, "error: boom")]) == []
    rows = C.lesson_rows([les(True, "ADOPTED"), les(False, "REJECTED: regression")])
    assert [r.sign for r in rows] == [1, -1]
    neg = C.Chooser().fit([rows[1]], epochs=60)
    cands = rows[1].cands
    sc = neg.scores(rows[1].objective, cands, {"m.py": SRC})
    assert sc[rows[1].chosen[0]] < max(sc)                       # the measured-bad action is not the favourite


def _student_run(tmp_path: Path, name: str, **kw: Any) -> tuple[bool, str]:
    class Llm:
        def chat(self, messages: Any, **k: Any) -> str:
            return "CHOICE: 1\nWHY: first."
    w = tmp_path / name
    w.mkdir()
    (w / "mod.py").write_text(SRC, encoding="utf-8")
    pkg = type("P", (), {"outputs": ["mod.py"], "objective": "banana", "component": "c"})()
    plan = type("Pl", (), {"component": "c", "step": "shrink", "requirement_key": "x.shrink"})()
    r = A.ActionStudent(tmp_path / "l.jsonl", llm=Llm(), **kw)(plan, pkg, w)
    return r.claimed_done, r.notes


def test_flag_off_is_the_old_behaviour_and_on_changes_the_pick(tmp_path: Path) -> None:
    cands = A.enumerate_actions(SRC, "mod.py")
    kinds = [a.kind for a in cands]
    ch = C.Chooser().fit([C.Row("banana", SRC, cands, [kinds.index("drop_unused_import")]) for _ in range(6)], epochs=60)
    old = _student_run(tmp_path, "old", use_chooser=False)
    off = _student_run(tmp_path, "off", use_chooser=False, chooser=ch)
    on = _student_run(tmp_path, "on", use_chooser=True, chooser=ch)
    assert off == old and "drop_unused_import" not in old[1]
    assert on[0] and "drop_unused_import" in on[1]
    assert A.CHOOSER_DEFAULT in (True, False)


TRICKY = ('import os\nimport sys as system\nfrom json import dumps\n\nVALUE = os.sep\n\n\ndef helper(a):\n    """doc"""\n    t = a + 1\n'
          '    u = t * t\n    return u + t\n\n\ndef outer(x):\n    def helper(y):\n        return y\n    tmp = dumps(x)\n    return helper(tmp)\n\n\n'
          'async def run(items):\n    for i in items:\n        return 1\n    return 0\n\n\nclass K:\n    def m(self):\n        v = system.argv\n        return v\n')


def test_indexed_structure_equals_the_reference_on_every_candidate() -> None:
    # h62: structure() reads one cached index per source instead of re-walking the AST per candidate; it must answer exactly as before
    for src in (SRC, TRICKY, "def broken(:\n", ""):
        cands = A.enumerate_actions(src, "m.py") + [A._act(k, "m.py", **kw) for k, kw in (
            ("inline_temp", {"function": "helper", "variable": "t"}), ("inline_temp", {"function": "nope"}),
            ("remove_unused", {"name": "helper"}), ("lazy_import", {"alias": "system"}), ("drop_unused_import", {"name": ""}),
            ("add_empty_guard", {"function": "run", "param": "items", "return_value": "0"}))]
        for a in cands:
            assert C.structure(src, a) == C._structure_ref(src, a), (src[:20], a)


def test_lesson_file_choices_are_cached_on_disk_and_identical(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", str(tmp_path / "cache"))
    after = SRC.replace("\n\ndef g_helper():\n    return 1\n", "")
    first = C._file_choice(SRC, after, "mod.py")
    calls: list[int] = []
    monkeypatch.setattr(A, "explain", lambda *a, **k: calls.append(1) or ([], False))
    assert C._file_choice(SRC, after, "mod.py") == first and calls == []                  # served from disk, same candidates and indices
    assert first[1] and first[0][first[1][0]].kind == "remove_unused"
    C._file_choice(SRC, after + "\n", "mod.py")
    assert calls == [1]                                                                # a different lesson text is computed afresh
