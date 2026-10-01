"""The model-backed student: retrieval of adopted lessons, prompt, edit application, safety, SFT export (FakeLLM, no server)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import curriculum as CUR
from creator import generator as G
from creator import model_student as MS

BEFORE = "import json\n\n\ndef dump(x):\n    return json.dumps(x)\n\n\ndef other():\n    return 1\n"
AFTER = "def dump(x):\n    import json\n    return json.dumps(x)\n\n\ndef other():\n    return 1\n"


def lesson(lid: str, kind: str, adopted: bool | None, objective: str = "move the json import into dump", reasoning: str = "json is used only in dump") -> CUR.Lesson:
    return CUR.Lesson(lesson_id=lid, package_id="P" + lid, component="util", task_kind=kind, objective=objective, reasoning=reasoning,
                      files_before={"app/u.py": BEFORE}, files_after={"app/u.py": AFTER}, adopted=adopted, claimed_done=True)


def log_with(tmp_path: Path, lessons: list[CUR.Lesson]) -> Path:
    path = tmp_path / "lessons.jsonl"
    log = CUR.LessonLog(path)
    for les in lessons:
        adopted = les.adopted
        les.adopted = None
        log.add(les)
        log.outcome(les.lesson_id, adopted, "x")
    return path


class FakeLLM:
    model = "fake"

    def __init__(self, reply: str) -> None:
        self.reply, self.seen = reply, []

    def chat(self, messages, **kw) -> str:                    # type: ignore[no-untyped-def]
        self.seen.append(messages)
        return self.reply


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    w = tmp_path / "w"
    (w / "app").mkdir(parents=True)
    (w / "app/u.py").write_text("import os\n\n\ndef where():\n    return os.getcwd()\n\n\ndef other():\n    return 1\n", encoding="utf-8")
    (w / ".creator_task.md").write_text("Task text: move the os import into where()", encoding="utf-8")
    return w


PLAN = SimpleNamespace(step="efficiency", requirement_key="x.shrink", component="util", package_id="P9")
PKG = SimpleNamespace(objective="move the os import into where", outputs=("app/u.py",))
GOOD = ("REASONING: os is only used in where, so import it there.\nFILE: app/u.py\n<<<<<<< SEARCH\nimport os\n\n\ndef where():\n"
        "    return os.getcwd()\n=======\ndef where():\n    import os\n    return os.getcwd()\n>>>>>>> REPLACE")


def test_retrieval_prefers_same_kind_and_ignores_unadopted(tmp_path: Path) -> None:
    ls = [lesson("a", "gap", True), lesson("b", "shrink", True, "unrelated words"), lesson("c", "shrink", False),
          lesson("d", "shrink", None), lesson("e", "shrink", True, "move the json import")]
    got = MS.retrieve(ls, "shrink", "move the os import", "util", k=3)
    assert [x.lesson_id for x in got] == ["e", "b", "a"]          # same kind first (best overlap first), then other kinds; no c, d


def test_prompt_carries_reasoning_task_and_target(tmp_path: Path, work: Path) -> None:
    st = MS.ModelStudent(log_with(tmp_path, [lesson("a", "shrink", True), lesson("r", "shrink", False, reasoning="REJECTED-REASON")]),
                         llm=FakeLLM(GOOD))
    p = st.build_prompt(PLAN, PKG, work)
    assert "json is used only in dump" in p and "REJECTED-REASON" not in p
    assert "Task text: move the os import" in p and "FILE: app/u.py" in p and "<<<<<<< SEARCH" in p


def test_edits_applied_and_result_carries_reasoning(tmp_path: Path, work: Path) -> None:
    st = MS.ModelStudent(log_with(tmp_path, [lesson("a", "shrink", True)]), llm=FakeLLM(GOOD))
    res = st(PLAN, PKG, work)
    assert res.claimed_done and res.by == "nupen-model-v1" and "os is only used in where" in res.reasoning
    assert "def where():\n    import os\n" in (work / "app/u.py").read_text(encoding="utf-8")


def test_edit_that_breaks_syntax_is_rolled_back(tmp_path: Path, work: Path) -> None:
    before = (work / "app/u.py").read_text(encoding="utf-8")
    bad = "REASONING: r\nFILE: app/u.py\n<<<<<<< SEARCH\ndef other():\n    return 1\n=======\ndef other(:\n    return 1\n>>>>>>> REPLACE"
    res = MS.ModelStudent(tmp_path / "none.jsonl", llm=FakeLLM(bad))(PLAN, PKG, work)
    assert not res.claimed_done and "parse" in res.notes and (work / "app/u.py").read_text(encoding="utf-8") == before


def test_garbage_reply_and_test_edits_are_refused(tmp_path: Path, work: Path) -> None:
    assert not MS.ModelStudent(tmp_path / "none.jsonl", llm=FakeLLM("I think you should refactor."))(PLAN, PKG, work).claimed_done
    (work / "tests").mkdir()
    (work / "tests/test_u.py").write_text("def test_a():\n    assert 1\n", encoding="utf-8")
    evil = "FILE: tests/test_u.py\n<<<<<<< SEARCH\nassert 1\n=======\npass\n>>>>>>> REPLACE"
    assert not MS.ModelStudent(tmp_path / "none.jsonl", llm=FakeLLM(evil))(PLAN, PKG, work).claimed_done
    assert "assert 1" in (work / "tests/test_u.py").read_text(encoding="utf-8")


def test_failing_model_call_never_crashes(tmp_path: Path, work: Path) -> None:
    class Boom:
        def chat(self, *a, **k):                              # type: ignore[no-untyped-def]
            raise OSError("server died")
    res = MS.ModelStudent(tmp_path / "none.jsonl", llm=Boom())(PLAN, PKG, work)
    assert not res.claimed_done and "server died" in res.notes


def test_missing_runtime_means_cannot_attempt(tmp_path: Path, work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(G, "SERVER_EXE", tmp_path / "nope.exe")
    st = MS.ModelStudent(tmp_path / "none.jsonl")
    assert not st.can_attempt(lesson("a", "shrink", None))
    assert not st(PLAN, PKG, work).claimed_done
    assert MS.ModelStudent(tmp_path / "none.jsonl", llm=FakeLLM("")).can_attempt(lesson("a", "shrink", None))


def test_export_sft_format(tmp_path: Path) -> None:
    path = log_with(tmp_path, [lesson("a", "shrink", True), lesson("b", "shrink", False), lesson("c", "gap", None)])
    out = tmp_path / "sft" / "train.jsonl"
    assert MS.export_sft(path, out) == 1
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    reply = rows[0]["messages"][2]["content"]
    assert reply.startswith("REASONING: json is used only in dump") and G.parse_edits(reply)
    # the exported edits, applied to the "before" text, reproduce the teacher's "after" text
    old, new = G.parse_edits(reply)[0][1:]
    assert G.apply_edit(BEFORE, old, new) is not None


def test_edits_between_reproduces_the_change() -> None:
    text = BEFORE
    for _path, old, new in G.parse_edits(MS.edits_between("app/u.py", BEFORE, AFTER)):
        text = G.apply_edit(text, old, new) or text
    assert text == AFTER


def test_swarm_wiring_has_a_flag() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("creator_swarm", Path(__file__).resolve().parents[1] / "scripts/creator_swarm.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [s.name for s in mod.make_students()]
    assert names[-1] == "nupen-model-v1" and "nupen-model-v1" not in [s.name for s in mod.make_students(model_student=False)]


def test_one_retry_is_told_why_the_first_reply_failed(tmp_path: Path, work: Path) -> None:
    class Script(FakeLLM):
        def __init__(self) -> None:
            super().__init__("")
            self.replies = ["REASONING: x\nFILE: app/u.py\n<<<<<<< SEARCH\nnot in file\n=======\ny\n>>>>>>> REPLACE", GOOD]

        def chat(self, messages, **kw) -> str:            # type: ignore[no-untyped-def]
            self.seen.append(list(messages))
            return self.replies.pop(0)
    llm = Script()
    res = MS.ModelStudent(tmp_path / "none.jsonl", llm=llm)(PLAN, PKG, work)
    assert res.claimed_done and res.calls == 2 and "did not work" in llm.seen[1][-1]["content"]


def test_a_noop_edit_is_not_a_claim(tmp_path: Path, work: Path) -> None:
    same = "FILE: app/u.py\n<<<<<<< SEARCH\ndef other():\n    return 1\n=======\ndef other():\n    return 1\n>>>>>>> REPLACE"
    res = MS.ModelStudent(tmp_path / "none.jsonl", llm=FakeLLM(same), attempts=1)(PLAN, PKG, work)
    assert not res.claimed_done and "changed nothing" in res.notes
