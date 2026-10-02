"""Student pre-screen: a non-improving or test-breaking candidate is rejected by the student itself, an improving one passes,
practice rows are a separate 'prescreen' source the shadow rule never reads, and nothing here adopts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import action_student as A
from creator import chooser as CH
from creator import prescreen as PS
from creator import shadow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import practice as PR  # noqa: E402

PLAN = SimpleNamespace(step="efficiency", requirement_key="x.shrink", component="util", package_id="P1")
PKG = SimpleNamespace(objective="shrink app/u.py", outputs=("app/u.py",))
LAZY = "import csv\n\n\ndef rows(text):\n    return list(csv.reader(text.splitlines()))\n"
DEAD = "def used():\n    return 1\n\n\ndef dead_helper():\n    return 2\n"
OSIMP = "import os\n\n\ndef f(x):\n    return x\n"


class FakeLLM:
    def __init__(self) -> None:
        self.n = 0

    def chat(self, messages, **kw) -> str:                    # type: ignore[no-untyped-def]
        self.n += 1
        return "CHOICE: 1\nWHY: first."


def tree(tmp: Path, src: str, test: str) -> Path:
    (tmp / "app").mkdir()
    (tmp / "tests").mkdir()
    (tmp / "app/u.py").write_text(src, encoding="utf-8", newline="\n")
    (tmp / "tests/test_u.py").write_text(test, encoding="utf-8", newline="\n")
    (tmp / ".creator_task.md").write_text("shrink app/u.py", encoding="utf-8")
    return tmp


def student(tmp: Path, llm: FakeLLM) -> A.ActionStudent:
    return A.ActionStudent(tmp / "none.jsonl", llm=llm, use_chooser=False, prescreen=True, state_dir=tmp / "st")


def test_non_improving_candidate_rejected_before_the_model_or_kernel(tmp_path: Path) -> None:
    w = tree(tmp_path, LAZY, "from app import u\n\n\ndef test_rows():\n    assert u.rows('a,b') == [['a', 'b']]\n")
    llm = FakeLLM()
    res = student(w, llm)(PLAN, PKG, w)
    assert not res.claimed_done and "prescreen" in res.notes and "cannot improve" in res.notes
    assert llm.n == 0 and (w / "app/u.py").read_text(encoding="utf-8") == LAZY       # no model call, no edit
    log = [json.loads(x) for x in (w / "st/prescreen_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and not log[0]["ok"] and log[0]["action"] == "lazy_import(csv)" and log[0]["size_delta"] >= 0


def test_test_breaking_candidate_rejected_with_reason(tmp_path: Path) -> None:
    w = tree(tmp_path, OSIMP, "from app import u\n\n\ndef test_f():\n    assert u.f(1) == 1\n    assert hasattr(u, 'os')\n")
    st = student(w, FakeLLM())
    res = st(PLAN, PKG, w)
    assert not res.claimed_done and "rejected" in res.notes
    assert (w / "app/u.py").read_text(encoding="utf-8") == OSIMP                       # restored byte for byte
    assert any(r.get("stage") == "tests" and "direct test failed" in r["reason"] for r in st.last_prescreen)


def test_improving_candidate_passes_and_is_applied(tmp_path: Path) -> None:
    w = tree(tmp_path, DEAD, "from app import u\n\n\ndef test_used():\n    assert u.used() == 1\n")
    res = student(w, FakeLLM())(PLAN, PKG, w)
    assert res.claimed_done and "remove_unused(dead_helper)" in res.notes
    assert "dead_helper" not in (w / "app/u.py").read_text(encoding="utf-8")


def test_prescreen_check_never_adopts_and_restores(tmp_path: Path) -> None:
    w = tree(tmp_path, DEAD, "from app import u\n\n\ndef test_used():\n    assert u.used() == 1\n")
    v = PS.prescreen(w, "app/u.py", "def used():\n    return 2\n", "size")
    assert not v.ok and v.stage == "tests" and (w / "app/u.py").read_text(encoding="utf-8") == DEAD
    v2 = PS.prescreen(w, "app/u.py", "def used(:\n", "size")
    assert not v2.ok and v2.stage == "metric"
    src = Path(PS.__file__).read_text(encoding="utf-8")
    assert "ledger" not in src.replace("never writes a ledger", "") and "import kernel" not in src and "adopt(" not in src


def test_practice_rows_are_a_separate_prescreen_source_the_shadow_rule_ignores(tmp_path: Path) -> None:
    w = tree(tmp_path, DEAD, "from app import u\n\n\ndef test_used():\n    assert u.used() == 1\n")
    out = tmp_path / "state/creator/practice_rows.jsonl"
    res = PR.practice(w, out, out.parent / "practice_src", minutes=2, run_tests=True, files=["app/u.py"])
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert res["rows"] == len(rows) >= 1 and all(r["source"] == "prescreen" for r in rows)
    assert any(r["good"] and r["action"]["kind"] == "remove_unused" and r["size_delta"] < 0 and r["tests_pass"] for r in rows)
    pr = CH.practice_rows(out, out.parent / "practice_src")
    assert pr and all(r.source == "prescreen" for r in pr) and all(r.weight < 1 for r in pr)
    # the shadow rule reads lessons only: practice rows give it no evidence and no flip
    s = shadow.update_policy(out.parent, lessons_path=out.parent / "lessons.jsonl")
    assert s["n"] == 0 and shadow.read_policy(out.parent) is None


def test_practice_holdout_runs_by_target_file() -> None:
    def mk(path: str, good: bool) -> list[CH.Row]:
        src = "import os\n\n\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n"
        c = [A._act("drop_unused_import", path, name="os"), A._act("remove_unused", path, name="g")]
        return [CH.Row(f"shrink {path} without losing capability", src, c, [1], 1, 0.3, "prescreen")]
    rows = [r for i in range(10) for r in mk(f"creator/m{i}.py", True)]
    res = CH.practice_holdout(rows, folds=5, epochs=60)
    assert res["decisions"] == 10 and res["chooser_top1"] is not None


def test_a_practice_run_without_tests_gives_the_chooser_no_positive_label(tmp_path: Path) -> None:
    """Validator 6: with --no-tests a candidate that merely imports was labelled good (tests_pass None) and trained the chooser as a
    measured-good outcome, although no test ever ran. Such rows are inconclusive and never become training rows."""
    w = tree(tmp_path, DEAD, "from app import u\n\n\ndef test_used():\n    assert u.used() == 1\n")
    out = tmp_path / "state/creator/practice_rows.jsonl"
    PR.practice(w, out, out.parent / "practice_src", minutes=2, run_tests=False, files=["app/u.py"])
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all(r["inconclusive"] for r in rows)
    assert CH.practice_rows(out, out.parent / "practice_src") == []
