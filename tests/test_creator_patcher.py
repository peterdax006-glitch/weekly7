"""P0.4: patch applier (search/replace + unified diff, fuzzy, atomic, reversible, gated), static checks (content-hash cache), warm worker."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from creator import staticcheck as SC
from creator import testrun as T
from creator import warmworker as W
from creator.tools import patch as PT
from creator.tools import toolbox as TB

SRC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


def _box(tmp: Path) -> TB.ToolBox:
    sb = tmp / "sb"
    (sb / "creator").mkdir(parents=True)
    (sb / "tests").mkdir()
    (sb / "creator" / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    (sb / "creator" / "m.py").write_text(SRC, encoding="utf-8", newline="\n")
    (sb / "creator" / "n.py").write_text("X = 1\n", encoding="utf-8", newline="\n")
    (sb / "tests" / "test_m.py").write_text("from creator.m import add, sub\n\ndef test_m():\n    assert add(1, 2) == 3\n    assert sub(3, 1) == 2\n",
                                            encoding="utf-8", newline="\n")
    state = tmp / "state" / "creator"
    state.mkdir(parents=True)
    return TB.ToolBox(sb, state, "p4", "creator")


@pytest.fixture()
def box(tmp_path: Path):                                                  # type: ignore[no-untyped-def]
    b = _box(tmp_path)
    yield b
    b.close()


def sr(path: str, old: str, new: str) -> str:
    return f"{path}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n"


def read(box: TB.ToolBox, rel: str) -> str:
    return (box.policy.sandbox / rel).read_bytes().decode()


# ---------------------------------------------------------------------------------------------------------------- apply
def test_search_replace_basic_and_undo(box: TB.ToolBox) -> None:
    r = box.files.apply_patch(sr("creator/m.py", "    return a + b", "    return a + b + 0"))
    assert r["ok"] and r["fuzzy"] == 0 and "a + b + 0" in read(box, "creator/m.py")
    u = box.files.undo_patch(r["undo"])
    assert u["ok"] and read(box, "creator/m.py") == SRC


def test_fuzzy_whitespace_and_reindent(box: TB.ToolBox) -> None:
    # model dropped the indentation and added trailing blanks: matches, replacement re-indented to the file's indentation
    p = sr("creator/m.py", "return a + b   ", "if a:\n    return a + b\nreturn b")
    r = box.files.apply_patch(p)
    assert r["ok"] and r["fuzzy"] == 1
    assert "    if a:\n        return a + b\n    return b\n" in read(box, "creator/m.py")


def test_ambiguous_and_missing_fail_whole_patch(box: TB.ToolBox) -> None:
    (box.policy.sandbox / "creator" / "d.py").write_text("x = 1\nx = 1\n", encoding="utf-8", newline="\n")
    both = sr("creator/n.py", "X = 1", "X = 2") + sr("creator/d.py", "x = 1", "x = 9")
    r = box.files.apply_patch(both)
    assert not r["ok"] and "ambiguous" in r["error"]
    assert read(box, "creator/n.py") == "X = 1\n"                      # the first, valid block was NOT applied
    r = box.files.apply_patch(sr("creator/n.py", "X = 5", "X = 2"))
    assert not r["ok"] and "not found" in r["error"]


def test_unified_diff_with_wrong_counts_and_hint(box: TB.ToolBox) -> None:
    d = textwrap.dedent("""\
        --- a/creator/m.py
        +++ b/creator/m.py
        @@ -99,2 +99,2 @@
         def sub(a, b):
        -    return a - b
        +    return a - b - 1
        """)
    r = box.files.apply_patch(d)
    assert r["ok"] and "a - b - 1" in read(box, "creator/m.py") and "a + b\n" in read(box, "creator/m.py")


def test_unified_diff_creates_file_and_crlf_kept(box: TB.ToolBox) -> None:
    d = "--- /dev/null\n+++ b/creator/new.py\n@@ -0,0 +1,2 @@\n+A = 1\n+B = 2\n"
    assert box.files.apply_patch(d)["ok"] and read(box, "creator/new.py") == "A = 1\nB = 2\n"
    (box.policy.sandbox / "creator" / "w.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    assert box.files.apply_patch(sr("creator/w.py", "b = 2", "b = 3"))["ok"]
    assert read(box, "creator/w.py") == "a = 1\r\nb = 3\r\n"


def test_multi_file_patch_and_undo_refuses_changed_file(box: TB.ToolBox) -> None:
    r = box.files.apply_patch(sr("creator/n.py", "X = 1", "X = 2") + sr("creator/m.py", "    return a - b", "    return a - b + 0"))
    assert r["ok"] and len(r["files"]) == 2
    (box.policy.sandbox / "creator" / "n.py").write_text("X = 3\n", encoding="utf-8", newline="\n")
    u = box.files.undo_patch(r["undo"])
    assert not u["ok"] and "a - b + 0" in read(box, "creator/m.py")   # all-or-nothing: nothing was reverted
    assert box.files.undo_patch(r["undo"], force=True)["ok"] and read(box, "creator/m.py") == SRC


def test_policy_gate_denies_protected_and_outside(box: TB.ToolBox, tmp_path: Path) -> None:
    out = tmp_path / "outside.py"
    out.write_text("v = 1\n", encoding="utf-8", newline="\n")
    r = box.files.apply_patch(sr(str(out), "v = 1", "v = 2"))
    assert not r["ok"] and "denied" in r
    assert out.read_text(encoding="utf-8") == "v = 1\n"
    r = box.files.apply_patch(sr("creator/n.py", "X = 1", "X = 2") + sr(str(out), "v = 1", "v = 2"))
    assert "denied" in r and read(box, "creator/n.py") == "X = 1\n"


# ------------------------------------------------------------------------------------------------------ fault injection
def test_fault_between_replacements_leaves_everything_unchanged(box: TB.ToolBox, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = os.replace

    def flaky(a: Any, b: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:                                            # 1: journal, 2: first file, 3: second file -> boom
            raise OSError("injected")
        real(a, b)

    monkeypatch.setattr(PT, "_replace", flaky)
    sb = box.policy.sandbox
    before = {p: (sb / p).read_bytes() for p in ("creator/m.py", "creator/n.py")}
    with pytest.raises(OSError):
        box.files.apply_patch(sr("creator/m.py", "    return a - b", "    return a - b + 0") + sr("creator/n.py", "X = 1", "X = 2"))
    assert {p: (sb / p).read_bytes() for p in before} == before
    assert not list(sb.rglob("*.nupen*.tmp")) and not (box.policy.scratch / PT.JOURNAL).exists()


def test_hard_kill_mid_apply_is_rolled_back_by_next_apply(box: TB.ToolBox, tmp_path: Path) -> None:
    sb = box.policy.sandbox
    jd = Path(box.policy.scratch)
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from creator.tools import patch as PT
        real, n = os.replace, [0]
        def kill(a, b):
            n[0] += 1
            if n[0] == 3:
                os._exit(7)                                  # hard kill after the first file was replaced
            real(a, b)
        PT._replace = kill
        h = PT.parse({sr('creator/m.py', '    return a - b', '    return a - b + 0') + sr('creator/n.py', 'X = 1', 'X = 2')!r})
        PT.apply_to_files(h, lambda p: (None, os.path.join({str(sb)!r}, p)), __import__('pathlib').Path({str(jd)!r}))
    """)
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert cp.returncode == 7, cp.stderr
    assert "a - b + 0" in read(box, "creator/m.py") and read(box, "creator/n.py") == "X = 1\n"     # the crash left a half state ...
    assert (jd / PT.JOURNAL).is_file()
    assert box.files.apply_patch(sr("creator/n.py", "X = 1", "X = 5"))["ok"]                       # ... the next apply heals it first
    assert read(box, "creator/m.py") == SRC and read(box, "creator/n.py") == "X = 5\n"


# ------------------------------------------------------------------------------------------------------- static checks
def test_static_checks_compile_and_content_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", str(tmp_path / "cache"))
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    rep = SC.check_files(tmp_path, ["ok.py", "bad.py", "notes.md"])
    byp = {f.path: f for f in rep.files}
    assert byp["ok.py"].compile_ok and not byp["bad.py"].compile_ok and not rep.ok and "notes.md" not in byp
    assert any("bad.py" in m for m in rep.messages())
    again = SC.check_files(tmp_path, ["ok.py", "bad.py"])
    assert all(f.cached for f in again.files)
    (tmp_path / "ok.py").write_text("x = 2\n", encoding="utf-8", newline="\n")
    assert not {f.path: f for f in SC.check_files(tmp_path, ["ok.py"]).files}["ok.py"].cached


def test_static_checks_missing_tools_skip_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", "off")
    monkeypatch.setattr(SC, "_ruff_cmd", lambda: None)
    monkeypatch.setattr(SC, "_mypy_ok", lambda: False)
    (tmp_path / "a.py").write_text("import os\n", encoding="utf-8", newline="\n")
    rep = SC.check_files(tmp_path, ["a.py"], mypy=True)
    assert rep.ok and rep.tools == {"ruff": "skipped", "mypy": "skipped"} and rep.files[0].ruff == "skipped"


def test_static_checks_mypy_finds_type_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mypy")
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", "off")
    (tmp_path / "t.py").write_text("def f(x: int) -> str:\n    return x\n", encoding="utf-8", newline="\n")
    rep = SC.check_files(tmp_path, ["t.py"], mypy=True)
    assert not rep.ok and rep.files[0].mypy == "issues"


# ---------------------------------------------------------------------------------------------------------- warm worker
@pytest.fixture()
def worker():                                                              # type: ignore[no-untyped-def]
    w = W.WarmWorker(preload=("pytest",))
    yield w
    w.stop()


def test_warm_worker_runs_pytest_and_sees_edits(box: TB.ToolBox, worker: W.WarmWorker, tmp_path: Path) -> None:
    sb = box.policy.sandbox
    cfg = T.PytestConfig(warm=True, timeout=120)
    W._POOL[(cfg.python, "pytest,numpy,pandas")] = worker
    jp = tmp_path / "j.xml"
    r1 = T.run_pytest(sb, ["tests/test_m.py"], jp, label="candidate", config=cfg)
    assert r1.status is T.RunStatus.PASSED and r1.counts()["PASSED"] == 1
    assert box.files.apply_patch(sr("creator/m.py", "    return a + b", "    return a + b + 1"))["ok"]      # edit, SAME worker process
    r2 = T.run_pytest(sb, ["tests/test_m.py"], jp, label="candidate", config=cfg)
    assert r2.status is T.RunStatus.FAILED and "tests/test_m.py::test_m" in r2.cases and r2.cases["tests/test_m.py::test_m"].outcome.bad
    assert worker.runs == 2
    r3 = T.run_pytest(sb, [], jp, label="candidate", config=cfg)
    assert r3.status is T.RunStatus.NO_TESTS


def test_warm_worker_timeout_kills_and_recovers(box: TB.ToolBox, worker: W.WarmWorker, tmp_path: Path) -> None:
    sb = box.policy.sandbox
    (sb / "tests" / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8", newline="\n")
    res = worker.run(sb, ["tests/test_slow.py"], tmp_path / "s.xml", timeout=3)
    assert res["timed_out"] and not worker.alive()
    res = worker.run(sb, ["tests/test_m.py"], tmp_path / "m.xml", timeout=60)
    assert res["rc"] == 0 and not res["timed_out"]


def test_quick_cycle_applies_checks_and_runs(box: TB.ToolBox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", "off")
    w = W.WarmWorker(preload=("pytest",))
    W._POOL[(T.PytestConfig().python, "pytest,numpy,pandas")] = w
    try:
        sb = box.policy.sandbox
        r = T.quick_cycle(box.files, sb, sr("creator/m.py", "    return a - b", "    return a - b + 0"), tmp_path / "q.xml")
        assert r["ok"] and r["run"]["counts"]["PASSED"] == 1 and "tests/test_m.py" in r["selection"]["tests"]
        bad = T.quick_cycle(box.files, sb, sr("creator/m.py", "    return a - b + 0", "    return a - ("), tmp_path / "q2.xml")
        assert not bad["ok"] and bad["why"] == "static checks failed" and "run" not in bad
        assert box.files.undo_patch(bad["undo"])["ok"] and "a - b + 0" in read(box, "creator/m.py")
    finally:
        w.stop()


def test_warm_worker_purges_importers_of_an_edited_module(box: TB.ToolBox, worker: W.WarmWorker, tmp_path: Path) -> None:
    sb = box.policy.sandbox
    (sb / "creator" / "m.py").write_text("from creator.n import X\n\ndef val():\n    return X\n", encoding="utf-8", newline="\n")
    (sb / "tests" / "test_m.py").write_text("from creator.m import val\n\ndef test_v():\n    assert val() == 1\n", encoding="utf-8", newline="\n")
    assert worker.run(sb, ["tests/test_m.py"], tmp_path / "a.xml", timeout=60)["rc"] == 0
    assert box.files.apply_patch(sr("creator/n.py", "X = 1", "X = 2"))["ok"]       # only the LEAF changed; m.py kept its stale X unless purged
    assert worker.run(sb, ["tests/test_m.py"], tmp_path / "b.xml", timeout=60)["rc"] == 1
    assert box.files.apply_patch(sr("creator/n.py", "X = 2", "X = 1"))["ok"]
    assert worker.run(sb, ["tests/test_m.py"], tmp_path / "c.xml", timeout=60)["rc"] == 0
