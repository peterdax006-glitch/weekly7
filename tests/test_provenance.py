"""Bible Phase 0 / T9: provenance. The code hash must cover exactly the code a run loaded (an unrelated new file is not
a change; an edit to a loaded file is), an edit after the process started must mark the run as mixed, legacy records
are stale, and the canon/Bible integrity check must fail closed on tampering."""
import importlib
import shutil
import sys
import time

import pytest

from engine import provenance as P


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    (tmp_path / "engine").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "engine" / "__init__.py").write_text("")
    (tmp_path / "engine" / "zz_mod.py").write_text("X = 1\n")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    return tmp_path


def test_hash_ignores_unrelated_new_files(fake_repo):
    files = ["engine/zz_mod.py"]
    h = P.code_hash(files)
    (fake_repo / "engine" / "unrelated.py").write_text("Y = 2\n")
    assert P.code_hash(files) == h


def test_hash_changes_when_a_loaded_file_changes(fake_repo):
    files = ["engine/zz_mod.py"]
    h = P.code_hash(files)
    (fake_repo / "engine" / "zz_mod.py").write_text("X = 2\n")
    assert P.code_hash(files) != h


def test_line_endings_do_not_change_the_hash(fake_repo):
    files = ["engine/zz_mod.py"]
    h = P.code_hash(files)
    (fake_repo / "engine" / "zz_mod.py").write_bytes(b"X = 1\r\n")
    assert P.code_hash(files) == h


def test_edit_after_process_start_is_mixed(fake_repo, monkeypatch):
    monkeypatch.setattr(P, "PROCESS_START", time.time() - 5)
    f = fake_repo / "engine" / "zz_mod.py"
    f.write_text("X = 3\n")
    assert P.code_mixed(["engine/zz_mod.py"]) == ["engine/zz_mod.py"]
    rec = {"code_hash": P.code_hash(["engine/zz_mod.py"]), "code_files": ["engine/zz_mod.py"], "code_mixed": ["engine/zz_mod.py"]}
    assert P.stale(rec)                                            # same hash, but the run straddled an edit


def test_clean_record_is_not_stale(fake_repo, monkeypatch):
    monkeypatch.setattr(P, "PROCESS_START", time.time() + 5)
    files = ["engine/zz_mod.py"]
    rec = {"code_hash": P.code_hash(files), "code_files": files, "code_mixed": P.code_mixed(files)}
    assert rec["code_mixed"] == [] and not P.stale(rec)


def test_record_goes_stale_after_edit(fake_repo, monkeypatch):
    monkeypatch.setattr(P, "PROCESS_START", time.time() + 5)
    files = ["engine/zz_mod.py"]
    rec = {"code_hash": P.code_hash(files), "code_files": files, "code_mixed": []}
    (fake_repo / "engine" / "zz_mod.py").write_text("X = 99\n")
    assert P.stale(rec)


def test_legacy_and_missing_records_are_stale():
    assert P.stale(None) and P.stale("4e9423971a4cc483") and P.stale({"code_hash": "x"})


def test_loaded_code_lists_real_engine_modules():
    import engine.candles  # noqa: F401
    files = P.loaded_code()
    assert "engine/candles.py" in files and "engine/provenance.py" in files
    assert all(f.startswith(("engine/", "scripts/")) for f in files)


def test_stamp_carries_code_and_config():
    s = P.stamp({"a": 1}, seed=5)
    for k in ("code_hash", "code_files", "code_mixed", "git_commit", "canon_sha", "config_hash", "data_snapshot", "seed"):
        assert k in s
    assert s["config_hash"] == P.config_hash({"a": 1}) != P.config_hash({"a": 2})


def test_integrity_fails_closed_on_tampered_bible(tmp_path, monkeypatch):
    real = P.ROOT
    for sub in ("canon",):
        shutil.copytree(real / sub, tmp_path / sub)
    shutil.copy(real / "BIBLE.md", tmp_path / "BIBLE.md")
    shutil.copy(real / "SELF_LEARNING_CONTRACT.md", tmp_path / "SELF_LEARNING_CONTRACT.md")
    shutil.copy(real / "RESEARCH_BRAIN_CONTRACT.md", tmp_path / "RESEARCH_BRAIN_CONTRACT.md")
    shutil.copy(real / "PREDICTION_ERROR_ADDITION.md", tmp_path / "PREDICTION_ERROR_ADDITION.md")
    shutil.copy(real / "TEN_HOUR_EXECUTION_CHECKLIST.md", tmp_path / "TEN_HOUR_EXECUTION_CHECKLIST.md")
    shutil.copy(real / "ULTIMATE_MASTER_PROMPT.md", tmp_path / "ULTIMATE_MASTER_PROMPT.md")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert P.verify_integrity() is True
    (tmp_path / "BIBLE.md").write_text((tmp_path / "BIBLE.md").read_text(encoding="utf-8") + "\nextra rule\n", encoding="utf-8")
    with pytest.raises(P.IntegrityError):
        P.verify_integrity()


@pytest.mark.parametrize("tamper", ["edit", "delete"])
def test_integrity_fails_closed_on_tampered_or_missing_contract(tmp_path, monkeypatch, tamper):
    real = P.ROOT
    shutil.copytree(real / "canon", tmp_path / "canon")
    shutil.copy(real / "BIBLE.md", tmp_path / "BIBLE.md")
    shutil.copy(real / "SELF_LEARNING_CONTRACT.md", tmp_path / "SELF_LEARNING_CONTRACT.md")
    shutil.copy(real / "RESEARCH_BRAIN_CONTRACT.md", tmp_path / "RESEARCH_BRAIN_CONTRACT.md")
    shutil.copy(real / "PREDICTION_ERROR_ADDITION.md", tmp_path / "PREDICTION_ERROR_ADDITION.md")
    shutil.copy(real / "TEN_HOUR_EXECUTION_CHECKLIST.md", tmp_path / "TEN_HOUR_EXECUTION_CHECKLIST.md")
    shutil.copy(real / "ULTIMATE_MASTER_PROMPT.md", tmp_path / "ULTIMATE_MASTER_PROMPT.md")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert P.verify_integrity() is True
    c = tmp_path / "SELF_LEARNING_CONTRACT.md"
    if tamper == "edit":
        c.write_text(c.read_text(encoding="utf-8").replace("BEGIN NOW.", "BEGIN LATER."), encoding="utf-8")
    else:
        c.unlink()
    with pytest.raises(P.IntegrityError):
        P.verify_integrity()


def test_engine_tree_hash_does_not_depend_on_what_is_imported():
    """C75 Phase 0: current_code_hash() changed as more modules were imported, so identical results disagreed."""
    import importlib
    from engine.learning.core import current_code_hash
    a = current_code_hash()
    importlib.import_module("engine.research.hypothesis_tree")          # load more engine modules
    importlib.import_module("engine.research.symmetry")
    assert current_code_hash() == a == P.engine_tree_hash() and len(a) == 16
