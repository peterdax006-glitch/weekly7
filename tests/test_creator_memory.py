"""CR K12: the memory component."""
from __future__ import annotations

import pytest

from creator import memory as M


def _mem():
    m = M.Memory()
    m.remember("failure", "ImportError in sandbox build", "missing module creator.foo", ["import"])
    m.remember("repair", "added missing module", "created creator/foo.py", ["import"])
    m.remember("research", "sequential evidence", "SPRT thresholds")
    return m


def test_remember_and_by_kind():
    m = _mem()
    assert [e.seq for e in m.entries] == [1, 2, 3]
    assert [e.subject for e in m.by_kind("repair")] == ["added missing module"]
    assert m.by_kind("strategy") == []
    assert m.by_kind("bogus") == []


def test_remember_rejects_bad_input():
    m = M.Memory()
    with pytest.raises(ValueError):
        m.remember("nonsense", "x")
    with pytest.raises(ValueError):
        m.remember("failure", "   ")
    assert m.entries == []


def test_seen_before_ranks_and_filters():
    m = _mem()
    hits = m.seen_before("ImportError sandbox build missing module")
    assert hits and hits[0][1].kind == "failure"
    assert all(s >= 0.2 for s, _ in hits)
    only = m.seen_before("missing module foo import", kind="repair")
    assert [e.kind for _, e in only] == ["repair"]
    assert len(m.seen_before("missing module foo import", limit=1)) == 1


def test_seen_before_empty_cases():
    assert M.Memory().seen_before("anything") == []
    m = _mem()
    assert m.seen_before("") == []
    assert m.seen_before("zzz qqq unrelated") == []
    assert m.seen_before("import", limit=0) == []


def test_save_load_roundtrip(tmp_path):
    m = _mem()
    p = tmp_path / "sub" / "mem.jsonl"
    m.save(p)
    loaded = M.load(p)
    assert [e.to_dict() for e in loaded.entries] == [e.to_dict() for e in m.entries]


def test_load_missing_and_corrupt(tmp_path):
    assert M.load(tmp_path / "nope.jsonl").entries == []
    p = tmp_path / "bad.jsonl"
    p.write_text('not json\n\n{"kind": "zzz", "subject": "x"}\n'
                 '{"kind": "failure", "subject": "ok"}\n[1]\n', encoding="utf-8")
    loaded = M.load(p)
    assert [e.subject for e in loaded.entries] == ["ok"]
