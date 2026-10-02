"""Model student v2: deterministic action enumeration, choice parsing, exact application, lesson mapping (FakeLLM, no server)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import action_student as A
from creator import curriculum as CUR

SRC = ('import csv\nimport sys\nimport os\n\nLIMIT = sys.maxsize\n\n\ndef rows(text):\n    return list(csv.reader(text.splitlines()))\n\n\n'
       'def unused_helper():\n    return 1\n\n\ndef total(xs):\n    t = sum(xs)\n    return t\n\n\n'
       'def first(items):\n    for i in items:\n        pass\n    return 0\n')


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply, self.seen = reply, []

    def chat(self, messages, **kw) -> str:                    # type: ignore[no-untyped-def]
        self.seen.append(messages)
        return self.reply


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    w = tmp_path / "w"
    (w / "app").mkdir(parents=True)
    (w / "app/u.py").write_text(SRC, encoding="utf-8")
    (w / ".creator_task.md").write_text("Move `import csv` into rows(); keep sys.", encoding="utf-8")
    return w


PLAN = SimpleNamespace(step="efficiency", requirement_key="x.shrink", component="util", package_id="P9")
PKG = SimpleNamespace(objective="move the csv import into rows", outputs=("app/u.py",))


def run(code: str, fn: str, *args):                           # type: ignore[no-untyped-def]
    ns: dict = {}
    exec(code, ns)
    return ns[fn](*args)


def test_candidates_are_valid_and_never_invent_names() -> None:
    acts = A.enumerate_actions(SRC, "u.py")
    short = {a.short() for a in acts}
    assert "lazy_import(csv)" in short and "drop_unused_import(os)" in short and "remove_unused(unused_helper)" in short
    assert "inline_temp(total,t)" in short and "add_empty_guard(first,items)" in short
    assert "lazy_import(sys)" not in short                    # sys is used at module level
    names = {"csv", "sys", "os", "rows", "unused_helper", "total", "first", "t", "items"}
    for a in acts:
        assert {v for k, v in a.params if k in ("alias", "name", "function", "variable", "param")} <= names
        assert A.apply_action(SRC, a) is not None
    assert A.enumerate_actions("def broken(:\n") == []


def test_choice_applies_exactly_that_action_and_preserves_behaviour(work: Path) -> None:
    st = A.ActionStudent(work / "none.jsonl", llm=FakeLLM(""))
    cands = st.candidates(PKG, work)
    i = [a.short() for a in cands].index("lazy_import(csv)") + 1
    st.llm = FakeLLM(f"WHY: csv is only used in rows.\nCHOICE: {i}")
    res = st(PLAN, PKG, work)
    assert res.claimed_done and res.by == "nupen-model-v2" and "csv" in res.reasoning
    new = (work / "app/u.py").read_text(encoding="utf-8")
    assert "import csv" in new.split("def rows")[1] and "import sys\n" in new.split("def rows")[0] and "\nimport csv" not in new.split("def rows")[0]
    assert run(new, "rows", "a,b\nc,d") == run(SRC, "rows", "a,b\nc,d")
    assert new.count("def unused_helper") == 1               # nothing else was touched


def test_garbage_and_out_of_range_replies_fail(work: Path) -> None:
    before = (work / "app/u.py").read_text(encoding="utf-8")
    for reply in ("I would rewrite the whole file", "CHOICE: 99", "CHOICE: none", "CHOICE: 0", "CHOICE: 1, banana", ""):
        res = A.ActionStudent(work / "n.jsonl", llm=FakeLLM(reply))(PLAN, PKG, work)
        assert not res.claimed_done
    assert (work / "app/u.py").read_text(encoding="utf-8") == before


def test_parse_choice() -> None:
    assert A.parse_choice("WHY: x\nCHOICE: 2, 1, 2", 3) == [2, 1]
    assert A.parse_choice("CHOICE: 4", 3) is None and A.parse_choice("no line 1", 3) is None


def test_missing_runtime_cannot_attempt(monkeypatch: pytest.MonkeyPatch, work: Path) -> None:
    monkeypatch.setattr(A.G, "SERVER_EXE", work / "nope.exe")
    st = A.ActionStudent(work / "n.jsonl")
    assert not st.can_attempt(None)
    assert not st(PLAN, PKG, work).claimed_done


def test_export_choices_maps_lessons_onto_actions(tmp_path: Path) -> None:
    after = SRC.replace("import csv\n", "").replace("def rows(text):\n", "def rows(text):\n    import csv\n")
    les = CUR.Lesson(lesson_id="L1", package_id="P1", component="u", task_kind="shrink", objective="move csv import into rows",
                     files_before={"u.py": SRC}, files_after={"u.py": after}, adopted=True, claimed_done=True)
    log = CUR.LessonLog(tmp_path / "l.jsonl")
    log.add(dataclasses_replace(les))
    log.outcome("L1", True, "ok")
    stats = A.export_choices(tmp_path / "l.jsonl", tmp_path / "out.jsonl")
    assert stats["mapped"] == 1 and stats["exact"] == 1
    row = __import__("json").loads((tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [row["candidates"][i - 1]["kind"] for i in row["chosen"]] == ["lazy_import"] and row["exact"]


def dataclasses_replace(les: CUR.Lesson) -> CUR.Lesson:
    import dataclasses
    return dataclasses.replace(les, adopted=None)


def test_lexical_pick_names_and_ties() -> None:
    acts = A.enumerate_actions(SRC, "u.py")
    pick = A.lexical_pick("Remove the dead helper unused_helper; nothing calls it.", acts)
    assert pick is not None and acts[pick - 1].short() == "remove_unused(unused_helper)"
    assert A.lexical_pick("tidy things up", acts) is None            # no evidence: never guess


def test_combine_choice_overrides_only_on_clear_lexical_margin() -> None:
    acts = A.enumerate_actions(SRC, "u.py")
    obj = "Delete the unused import os; the module never uses it."
    want = next(i for i, a in enumerate(acts, 1) if a.short() == "drop_unused_import(os)")
    wrong = next(i for i, a in enumerate(acts, 1) if a.short() == "inline_temp(total,t)")
    assert A.combine_choice(wrong, obj, acts) == want                # model wrong, lexical clear -> lexical
    assert A.combine_choice(want, obj, acts) == want
    assert A.combine_choice(None, obj, acts) == want                 # model silent -> lexical
    assert A.combine_choice(wrong, "tidy things up", acts) == wrong  # no lexical evidence -> model stands


def test_majority_vote_and_rich_label() -> None:
    assert A.majority_vote([2, None, 3, 3]) == 3 and A.majority_vote([None]) is None and A.majority_vote([4, 5]) == 4
    a = A.enumerate_actions(SRC, "u.py")[0]
    assert a.label(rich=True) != a.label() and a.short().split("(")[0] in a.label(rich=True)


def test_student_lexical_prior_flag_fixes_a_wrong_pick(work: Path) -> None:
    pkg = SimpleNamespace(objective="Delete the unused import os; the module never uses it.", outputs=("app/u.py",))
    (work / ".creator_task.md").write_text(pkg.objective, encoding="utf-8")
    acts = A.enumerate_actions(SRC, "app/u.py")
    wrong = next(i for i, a in enumerate(acts, 1) if a.short() == "inline_temp(total,t)")
    A.ActionStudent(work / "n.jsonl", llm=FakeLLM(f"CHOICE: {wrong}\nWHY: x"), lexical_prior=True, rich_labels=True)(PLAN, pkg, work)
    new = (work / "app/u.py").read_text(encoding="utf-8")
    assert "import os" not in new and "t = sum(xs)" in new          # the import was dropped, the model's inline_temp was overridden


def test_a_crlf_file_stays_crlf_after_an_action(work: Path) -> None:
    (work / "app/u.py").write_bytes(SRC.replace("\n", "\r\n").encode("utf-8"))
    st = A.ActionStudent(work / "none.jsonl", llm=FakeLLM(""))
    cands = st.candidates(PKG, work)
    drop = next(i for i, a in enumerate(cands, 1) if a.short() == "drop_unused_import(os)")
    st.llm = FakeLLM(f"CHOICE: {drop}\nWHY: os is unused")
    res = st(PLAN, PKG, work)
    assert res.claimed_done, res.detail if hasattr(res, "detail") else res
    raw = (work / "app/u.py").read_bytes()
    assert b"import os" not in raw and raw.count(b"\r\n") == raw.count(b"\n") > 5


def test_candidates_ignore_the_root_state_directory(work: Path) -> None:
    (work / "state" / "run1").mkdir(parents=True)
    (work / "state" / "run1" / "junk.py").write_text("import os\n\n\ndef _gone():\n    return 1\n", encoding="utf-8")
    st = A.ActionStudent(work / "none.jsonl", llm=FakeLLM(""))
    assert all(not a.path.startswith("state") for a in st.candidates(PKG, work))
    assert "state/run1/junk.py" not in A._py_texts(work) and "app/u.py" in A._py_texts(work)


def test_parse_choice_accepts_only_numbers_and_separators() -> None:
    assert A.parse_choice("CHOICE: 1, 3\nWHY: x", 5) == [1, 3]
    assert A.parse_choice("CHOICE: 2 and 4.", 5) == [2, 4]
    assert A.parse_choice("CHOICE: 1 2", 5) == [1, 2]
    for junk in ("CHOICE: 1 dnn 2", "CHOICE: a1", "CHOICE: and", "CHOICE: 1,", "CHOICE: 9"):
        assert A.parse_choice(junk, 5) is None, junk


def test_defaults_are_the_policy_that_won_the_heldout_hard_bench() -> None:
    """Held-out hard seeds 201-203 (n=90): rich labels + lexical prior 0.811 vs lexical 0.589 (CIs disjoint); flags stay switchable."""
    import inspect
    sig = inspect.signature(A.ActionStudent.__init__).parameters
    assert sig["rich_labels"].default is True and sig["lexical_prior"].default is True
    st = A.ActionStudent(Path("n.jsonl"), llm=FakeLLM(""), rich_labels=False, lexical_prior=False)
    assert (st.rich_labels, st.lexical_prior) == (False, False)
