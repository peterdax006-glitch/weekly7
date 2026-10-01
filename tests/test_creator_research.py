"""CR K07: the research component."""
from __future__ import annotations

from pathlib import Path

from creator import research as R


def test_generate_questions_empty_and_skips():
    assert R.generate_questions([]) == []
    assert R.generate_questions([{"id": "G1", "description": "  "}]) == []


def test_generate_questions_dedup_and_clamp():
    qs = R.generate_questions([
        {"kind": "KNOWLEDGE", "description": "x", "importance": 5},
        {"kind": "KNOWLEDGE", "description": "x"},
    ])
    assert len(qs) == 1
    assert qs[0].importance == 1.0 and qs[0].uncertainty == 1.0


def test_prioritise_order_and_budget():
    a = R.Question("a", importance=0.2)
    b = R.Question("b", importance=0.9)
    assert [q.text for q in R.prioritise([a, b])] == ["b", "a"]
    assert [q.text for q in R.prioritise([a, b], 1)] == ["b"]
    assert R.prioritise([a, b], 0) == []


def test_collect_repo_evidence(tmp_path: Path):
    (tmp_path / "m.py").write_text("x = 1\n# Sandbox here\n", encoding="utf-8")
    (tmp_path / "d.md").write_text("sandbox docs\n", encoding="utf-8")
    (tmp_path / "bin.py").write_bytes(b"\xff\xfe\x00sandbox")
    ev = R.collect_repo_evidence("q", tmp_path, ["sandbox"])
    assert {e.kind for e in ev} == {"repo", "docs"}
    assert any(e.source == "m.py:2" for e in ev)
    assert len(R.collect_repo_evidence("q", tmp_path, ["sandbox"], max_hits=1)) == 1


def test_collect_repo_evidence_empty_cases(tmp_path: Path):
    assert R.collect_repo_evidence("q", tmp_path / "missing", ["a"]) == []
    assert R.collect_repo_evidence("q", tmp_path, []) == []
    assert R.collect_repo_evidence("q", tmp_path, ["zzz"]) == []


def test_source_weight():
    assert R.source_weight("repo") > R.source_weight("web") > R.source_weight("unknown")


def test_contradictions():
    a = R.Evidence("q", "The cache is safe", "s1", supports=True)
    b = R.Evidence("q", "The cache is safe", "s2", supports=False)
    c = R.Evidence("q", "The cache is not safe", "s3", supports=True)
    d = R.Evidence("other", "The cache is safe", "s4", supports=False)
    pairs = R.detect_contradictions([a, b, c, d])
    assert (a, b) in pairs and (a, c) in pairs and (b, c) not in pairs
    assert all(d not in p for p in pairs)
    assert R.detect_contradictions([]) == []


def test_confidence():
    assert R.confidence([]) == 0.0
    one = [R.Evidence("q", "c", "s", "repo")]
    four = [R.Evidence("q", f"c{i}", "s", "repo") for i in range(4)]
    assert R.confidence(one) < R.confidence(four) == 1.0
    conflict = four + [R.Evidence("q", "c0", "s", "repo", supports=False)]
    assert R.confidence(conflict) < 1.0
    assert R.confidence([R.Evidence("q", "c", "s", supports=False)]) == 0.0


def test_memory_roundtrip_and_corrupt_lines(tmp_path: Path):
    p = tmp_path / "sub" / "mem.jsonl"
    m = R.ResearchMemory(p)
    assert m.recall("q") == []
    e = R.Evidence("q", "claim", "src", "docs")
    m.add(e)
    with p.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    m2 = R.ResearchMemory(p)
    assert m2.recall("q") == [e]
    assert m2.recall("nope") == []
    assert R.ResearchMemory().items == []


def test_design_bridge():
    empty = R.to_design_input("q", [])
    assert empty.open and empty.confidence == 0.0 and empty.findings == []
    ev = [R.Evidence("q", f"c{i}", f"s{i}", "repo") for i in range(4)]
    ok = R.to_design_input("q", ev)
    assert not ok.open and len(ok.findings) == 4
    bad = R.to_design_input("q", ev + [R.Evidence("q", "c0", "w", "web", supports=False)])
    assert bad.open and bad.contradictions
