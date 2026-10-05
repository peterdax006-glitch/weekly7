"""Locate index + context packer (creator.tools.index, MASTER_BLUEPRINT P0.3)."""
from __future__ import annotations

from pathlib import Path

from creator.tools.index import Index, _covered, _split


def _repo(tmp: Path) -> Path:
    (tmp / "pkg").mkdir()
    (tmp / "pkg" / "slowpath.py").write_text('"""Operation timer and slow path detector."""\n\n'
                                             "def record_timing(op, seconds):\n    return op, seconds\n", encoding="utf-8")
    (tmp / "pkg" / "runner.py").write_text("from pkg.slowpath import record_timing\n\n\n"
                                           "class TestSelector:\n    def select_tests(self, changed_files):\n"
                                           "        return [record_timing(f, 0) for f in changed_files]\n\n\n"
                                           "def parse_junit(path):\n    return path\n", encoding="utf-8")
    return tmp


def test_split_words() -> None:
    assert _split("select_tests") == ["select", "tests"]
    assert _split("ImportGraph") == ["import", "graph"]
    assert _split("pkg.slowpath") == ["pkg", "slowpath"]
    assert _covered("slow", ["slowpath"]) and not _covered("low", ["slowpath"])   # 4+ letters for substrings


def test_defs_callers_and_incremental_update(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix = Index(root, db=tmp_path / "ix.sqlite")
    first = ix.update(["pkg/slowpath.py", "pkg/runner.py"])
    assert first["parsed"] == 2 and first["failed"] == 0
    [d] = ix.defs("select_tests")
    assert (d.path, d.line, d.kind, d.qual) == ("pkg/runner.py", 5, "def", "TestSelector.select_tests")
    calls = ix.callers("record_timing")
    assert [(c.path, c.line, c.qual) for c in calls] == [("pkg/runner.py", 6, "TestSelector.select_tests")]
    again = ix.update(["pkg/slowpath.py", "pkg/runner.py"])
    assert again["parsed"] == 0                                   # unchanged bytes: nothing re-parsed
    (root / "pkg" / "runner.py").write_text("def parse_junit(path):\n    return path\n", encoding="utf-8")
    third = ix.update(["pkg/slowpath.py", "pkg/runner.py"])
    assert third["parsed"] == 1 and ix.defs("select_tests") == []  # the edited file only; its old symbols are gone


def test_locate_ranks_names_modules_and_compounds(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix = Index(root, db=tmp_path / "ix.sqlite")
    ix.update(["pkg/slowpath.py", "pkg/runner.py"])
    assert ix.locate("select tests for changed files")[0].qual == "TestSelector.select_tests"
    assert ix.locate("parse_junit")[0].qual == "parse_junit"                     # an exact identifier wins
    assert ix.locate("slow path detector")[0].qual == "pkg.slowpath"              # compound module name
    assert all(h.score > 0 for h in ix.locate("select tests"))


def test_pack_respects_the_token_budget(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix = Index(root, db=tmp_path / "ix.sqlite")
    ix.update(["pkg/slowpath.py", "pkg/runner.py"])
    hits = ix.locate("select tests for changed files")
    text = ix.pack(hits, max_tokens=1500)
    assert text.startswith("### pkg/runner.py:5-6 (TestSelector.select_tests)")
    assert "return [record_timing(f, 0)" in text
    tiny = ix.pack(hits, max_tokens=10)
    assert len(tiny) <= int(10 * 3.6)
