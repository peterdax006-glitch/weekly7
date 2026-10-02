"""Resume (creator/pending.py): finished-but-unmeasured work re-applies to the current tree, exactly or function by function."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from creator import curriculum as C
from creator import pending as P

V0 = "def f():\n    return 1\ndef g():\n    return 2\ndef h():\n    return 3\n"
F_NEW = "def f():\n    return 10\ndef g():\n    return 2\ndef h():\n    return 3\n"


def _git(cwd: Path, *a: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def _repo(tmp: Path, text: str = V0) -> Path:
    r = tmp / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "m.py").write_text(text, encoding="utf-8", newline="\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base")
    return r


def _saved_patch(r: Path, pending: Path, new: str, name: str = "CP0001_x.patch") -> None:
    (r / "m.py").write_text(new, encoding="utf-8", newline="\n")
    pending.mkdir(parents=True, exist_ok=True)
    (pending / name).write_text(_git(r, "diff"), encoding="utf-8", newline="\n")
    _git(r, "checkout", "--", "m.py")


def _drift(r: Path, text: str) -> None:
    (r / "m.py").write_text(text, encoding="utf-8", newline="\n")
    _git(r, "commit", "-qam", "drift")


def _items(tmp: Path) -> list[P.Item]:
    return P.load_items(tmp / "pending", tmp / "lessons.jsonl")


def test_exact_patch_reapplies(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    _saved_patch(r, tmp_path / "pending", F_NEW)
    it = _items(tmp_path)[0]
    res = P.try_apply(it, r, write=False, repos=[r])
    assert res.ok and res.mode == "exact" and (r / "m.py").read_text() == V0          # report path writes nothing
    assert P.try_apply(it, r, repos=[r]).mode == "exact" and (r / "m.py").read_text() == F_NEW


def test_drifted_file_with_unchanged_target_function_reapplies_at_function_level(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    _saved_patch(r, tmp_path / "pending", F_NEW)
    _drift(r, "def f():\n    return 1\ndef g():\n    return 22\ndef h():\n    return 3\n")     # g moved next to f: the diff context breaks
    it = _items(tmp_path)[0]
    res = P.try_apply(it, r, repos=[r])
    assert res.ok and res.mode == "function" and any("::f" in a for a in res.applied)
    text = (r / "m.py").read_text()
    assert "return 10" in text and "return 22" in text                                    # f re-applied, g's drift kept
    compile(text, "m.py", "exec")


def test_a_changed_target_function_is_not_touched(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    _saved_patch(r, tmp_path / "pending", F_NEW)
    drifted = "def f():\n    return 5\ndef g():\n    return 22\ndef h():\n    return 3\n"
    _drift(r, drifted)
    res = P.try_apply(_items(tmp_path)[0], r, repos=[r])
    assert not res.ok and (r / "m.py").read_text() == drifted and any("::f" in s for s in res.skipped)


def test_lesson_function_level_added_function_and_credit_to_the_original_solver(tmp_path: Path) -> None:
    r = _repo(tmp_path, "def f():\n    return 1\ndef g():\n    return 22\n")
    log = C.LessonLog(tmp_path / "lessons.jsonl")
    for sol in ("claude", "nupen-model-v1"):
        log.add(C.Lesson(lesson_id=f"L-{sol}", package_id=f"CP9-{sol}", component="c", task_kind="gap", objective=f"obj {sol}",
                         files_before={"m.py": "def f():\n    return 1\ndef g():\n    return 2\n"},
                         files_after={"m.py": "def f():\n    return 10\ndef g():\n    return 2\ndef k():\n    return 7\n"},
                         reasoning="because", solver=sol, claimed_done=True))
    st = P.ResumeStudent(tmp_path / "lessons.jsonl", tmp_path / "pending")
    plan = SimpleNamespace(package_id="CP5")
    res = st(plan, SimpleNamespace(objective="obj claude", outputs=("m.py",), inputs=()), r)
    assert res.claimed_done and res.by == C.REPLAY and res.by in C.TEACHER and "L-claude" in res.reasoning   # teacher stays teacher
    text = (r / "m.py").read_text()
    assert "return 10" in text and "def k" in text and "return 22" in text
    r2 = _repo(tmp_path / "b", "def f():\n    return 1\ndef g():\n    return 22\n") if (tmp_path / "b").mkdir() is None else r
    res2 = st(plan, SimpleNamespace(objective="obj nupen-model-v1", outputs=("m.py",), inputs=()), r2)
    assert res2.claimed_done and res2.by == "nupen-model-v1" and res2.by not in C.TEACHER      # a student's own work counts to it
    assert st.can_attempt(SimpleNamespace(objective="obj claude")) and not st.can_attempt(SimpleNamespace(objective="zzz"))


def test_curriculum_records_the_lesson_under_the_original_solver(tmp_path: Path) -> None:
    r = _repo(tmp_path, "def f():\n    return 1\n")
    log = C.LessonLog(tmp_path / "lessons.jsonl")
    log.add(C.Lesson(lesson_id="L1", package_id="CP9", component="c", task_kind="gap", objective="o",
                     files_before={"m.py": "def f():\n    return 1\n"}, files_after={"m.py": "def f():\n    return 4\n"},
                     solver="nupen-model-v1", claimed_done=True))
    st = P.ResumeStudent(tmp_path / "lessons.jsonl", tmp_path / "pending")
    cur = C.Curriculum(tmp_path / "lessons.jsonl", [st])
    step = cur.student_steps()[0]
    out = step(SimpleNamespace(package_id="CP5", component="c", step="gap", requirement_key="k"),
               SimpleNamespace(objective="o", outputs=("m.py",), inputs=()), r)
    assert out.claimed_done
    new = [x for x in cur.log.lessons() if x.lesson_id != "L1"][0]
    assert new.solver == "nupen-model-v1" and new.package_id == "CP5"


def test_retire_rules_measured_and_three_strikes(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    _saved_patch(r, tmp_path / "pending", F_NEW)
    _drift(r, "def f():\n    return 5\ndef g():\n    return 22\ndef h():\n    return 3\n")       # f changed: never re-applies
    pend = tmp_path / "pending"
    st = P.ResumeStudent(tmp_path / "lessons.jsonl", pend, repos=[r])
    pkg = SimpleNamespace(objective="o", outputs=("m.py",), inputs=())
    for n in range(1, 4):
        assert not st(SimpleNamespace(package_id="CP1"), pkg, r).claimed_done
        assert (n < 3) == (pend / "CP0001_x.patch").is_file()
    assert (pend / "stale" / "CP0001_x.patch").is_file() and "3 failed" in (pend / "stale" / "CP0001_x.patch.json").read_text()
    assert _items(tmp_path) == []
    # measured: a resumed attempt that the kernel judged moves to measured/ with the verdict; no verdict leaves it
    r2 = _repo(tmp_path / "w")  if (tmp_path / "w").mkdir() is None else r
    _saved_patch(r2, pend, F_NEW, "CP0002_y.patch")
    st2 = P.ResumeStudent(tmp_path / "lessons.jsonl", pend)
    assert st2(SimpleNamespace(package_id="CP2"), SimpleNamespace(objective="o", outputs=("m.py",), inputs=()), r2).claimed_done
    st2.resolved("CP2", None, "CANCELLED: ram")
    assert (pend / "CP0002_y.patch").is_file()
    st2._open["CP2"] = _items(tmp_path)[0]
    st2.resolved("CP2", False, "REJECTED: regression")
    assert (pend / "measured" / "CP0002_y.patch").is_file() and "REJECTED" in (pend / "measured" / "CP0002_y.patch.json").read_text()
    assert _items(tmp_path) == []
