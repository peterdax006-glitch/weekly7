import os
import stat
import pytest
from engine import checkpoint as C


def mkb(tmp_path, rid="R1", **kw):
    a = tmp_path / "art.txt"
    a.write_text("weights")
    return C.write_checkpoint(tmp_path / "ck", rid, {"k": 4}, {"m": 0.07}, {"np": 1}, {"note": "x"}, "2026-01-01",
                              logs={"run.log": "line\n"}, artifacts={"model.bin": a}, provenance={"git": "abc"}, **kw)


def test_roundtrip_verifies_and_loads(tmp_path):
    d = mkb(tmp_path)
    assert C.verify(d) == {"ok": True, "problems": []}
    b = C.load(d)
    assert b["config"] == {"k": 4} and b["manifest"]["provenance"] == {"git": "abc"}
    assert "logs/run.log" in b["manifest"]["files"] and "artifacts/model.bin" in b["manifest"]["files"]


def test_never_overwrites(tmp_path):
    mkb(tmp_path)
    with pytest.raises(C.CheckpointError):
        mkb(tmp_path)
    assert C.list_checkpoints(tmp_path / "ck") == ["R1"]


def test_tamper_delete_and_add_are_caught(tmp_path):
    d = mkb(tmp_path)
    m = d / "metrics.json"
    os.chmod(m, stat.S_IWRITE)
    m.write_text('{"m": 0.5}')
    assert any("changed: metrics.json" in p for p in C.verify(d)["problems"])
    with pytest.raises(C.CheckpointError):
        C.load(d)
    os.chmod(d / "logs" / "run.log", stat.S_IWRITE)
    (d / "logs" / "run.log").unlink()
    (d / "extra.txt").write_text("added later")
    probs = C.verify(d)["problems"]
    assert "missing: logs/run.log" in probs and "unlisted: extra.txt" in probs


def test_manifest_edit_is_caught(tmp_path):
    d = mkb(tmp_path)
    mp = d / "MANIFEST.json"
    os.chmod(mp, stat.S_IWRITE)
    mp.write_text(mp.read_text().replace('"bytes": ', '"bytes": 1'), encoding="utf-8")
    assert not C.verify(d)["ok"]


def test_missing_manifest_fails_closed(tmp_path):
    (tmp_path / "half").mkdir()
    assert C.verify(tmp_path / "half")["ok"] is False
    assert C.verify_all(tmp_path)["ok"] is False
    assert C.verify(tmp_path / "nonexistent")["ok"] is False


def test_bad_inputs_leave_nothing_behind(tmp_path):
    root = tmp_path / "ck"
    root.mkdir()
    with pytest.raises(C.CheckpointError):
        C.write_checkpoint(root, "R2", {}, {}, {}, {}, "t")
    with pytest.raises(C.CheckpointError):
        C.write_checkpoint(root, "R3", {}, {}, {"s": 1}, {}, "t", artifacts={"a": tmp_path / "nope"})
    with pytest.raises(ValueError):
        C.write_checkpoint(root, "R4", {}, {"m": float("nan")}, {"s": 1}, {}, "t")
    with pytest.raises(C.CheckpointError):
        C.write_checkpoint(root, "../evil", {}, {}, {"s": 1}, {}, "t")
    assert C.list_checkpoints(root) == [] and not any(p.is_dir() for p in root.iterdir())


def test_same_config(tmp_path):
    a = mkb(tmp_path, "A")
    b = mkb(tmp_path, "B")
    assert C.same_config(a, b)
