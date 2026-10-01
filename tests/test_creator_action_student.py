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
