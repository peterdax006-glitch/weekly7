"""Reuse index (creator.tools.reuse, R3): retrieval, adapt, direct, licences and the held-out leakage guard."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from creator.tools import reuse as RU

MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy of this software"
GPL = "GNU GENERAL PUBLIC LICENSE Version 3"
CLAMP = 'def clamp(x, lo, hi):\n    """Clamp a number into the closed interval between lo and hi."""\n    return max(lo, min(x, hi))\n'
LEAK = 'def secret_mean(values):\n    """Arithmetic mean of a list of numbers, 0.0 when empty."""\n    return sum(values) / len(values) if values else 0.0\n'


def _public(tmp: Path) -> Path:
    pub = tmp / "pub"
    for name, lic, code in (("mitlib", MIT, CLAMP), ("gpllib", GPL, 'def other(a, b):\n    """Add two numbers together always."""\n    x = a\n    return x + b\n'),
                            ("nolic", "", 'def third(a, b):\n    """Multiply two numbers always."""\n    x = a\n    return x * b\n')):
        (pub / name / "src").mkdir(parents=True)
        if lic:
            (pub / name / "LICENSE").write_text(lic, encoding="utf-8")
        (pub / name / "src" / "m.py").write_text(code, encoding="utf-8")
    (pub / "mitlib" / "src" / "leaked.py").write_text(LEAK, encoding="utf-8")
    (pub / "mitlib" / "src" / "vendored_copy.py").write_text(LEAK.replace("secret_mean", "avg").replace("values", "xs"), encoding="utf-8")
    return pub


def _task() -> dict:
    stub = 'def clip(v, a, b):\n    """Clamp a number into the closed interval between a and b."""\n    raise NotImplementedError\n'
    return {"id": "fn.clip", "family": "fn", "name": "clip", "stub": stub, "request": "clip " + stub, "query": "clip " + stub,
            "tests": [{"args": [5, 0, 3], "expect": 3}, {"args": [-1, 0, 3], "expect": 0}, {"args": [2, 0, 3], "expect": 2}]}


def _guard(tmp: Path, pub: Path) -> RU.LeakGuard:
    rl = tmp / "rl.jsonl"
    rl.write_text(json.dumps({"id": "pub:mitlib:src/leaked.py:secret_mean", "name": "secret_mean", "split": "eval"}) + "\n", encoding="utf-8")
    return RU.LeakGuard.from_sources([rl], public_root=pub, repo_root=tmp)


def test_licences(tmp_path: Path) -> None:
    pub = _public(tmp_path)
    assert RU.licence_of(pub / "mitlib") == "MIT"
    assert RU.licence_of(pub / "gpllib") is None and RU.licence_of(pub / "nolic") is None


def test_build_skips_unlicensed_and_records_source(tmp_path: Path) -> None:
    pub = _public(tmp_path)
    ix = RU.ReuseIndex(tmp_path / "r.sqlite", _guard(tmp_path, pub))
    res = ix.build(tmp_path / "selfrepo", pub)
    st = ix.stats()
    assert set(st["by_source"]) == {"mitlib"} and st["licences"]["mitlib"] == "MIT"
    assert sorted(st["skipped_repos"]) == ["gpllib", "nolic"]
    assert any(r.get("skipped") for r in res)


def test_leakage_guard_blocks_source_file_and_copies(tmp_path: Path) -> None:
    pub = _public(tmp_path)
    ix = RU.ReuseIndex(tmp_path / "r.sqlite", _guard(tmp_path, pub))
    ix.build(tmp_path / "selfrepo", pub)
    names = {r[0] for r in ix.db.execute("SELECT name FROM fn")}
    assert "clamp" in names
    assert "secret_mean" not in names                       # the source file of a held-out task
    assert "avg" not in names                               # a renamed copy elsewhere: same normalized body
    ix.assert_clean()
    # even a hit forced through retrieval can not surface it
    assert all(c.name not in ("secret_mean", "avg") for c in ix.query("arithmetic mean of a list of numbers", name="mean", args=["values"]))


def test_guard_sigdoc_blocks_task_signature(tmp_path: Path) -> None:
    pub = _public(tmp_path)
    g = _guard(tmp_path, pub)
    stub = 'def clamp(x, lo, hi):\n    """Clamp a number into the closed interval between lo and hi."""\n    raise NotImplementedError\n'
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "tasks.json").write_text(json.dumps({"tasks": [{"id": "fn.clamp", "family": "fn", "name": "clamp", "stub": stub}]}), encoding="utf-8")
    g2 = RU.LeakGuard.from_sources([], suite_dir=suite)
    ix = RU.ReuseIndex(tmp_path / "r2.sqlite", g2)
    ix.build(tmp_path / "selfrepo", pub)
    assert "clamp" not in {r[0] for r in ix.db.execute("SELECT name FROM fn")}
    assert g.task_names == {"secret_mean"}


def test_heldout_commit_files_are_excluded(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "gitrepo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "a@b.c")
    run("config", "user.name", "t")
    (repo / "old.py").write_text(CLAMP, encoding="utf-8")
    run("add", ".")
    run("commit", "-qm", "old")
    (repo / "new.py").write_text(LEAK, encoding="utf-8")
    run("add", ".")
    run("commit", "-qm", "heldout")
    sha = run("rev-parse", "HEAD").stdout.strip()
    ho = tmp_path / "ho.json"
    ho.write_text(json.dumps({"held_out_commits": [sha]}), encoding="utf-8")
    g = RU.LeakGuard.from_sources([], heldout_json=ho, repo_root=repo)
    ix = RU.ReuseIndex(tmp_path / "h.sqlite", g)
    ix.add_repo("self", repo)
    names = {r[0] for r in ix.db.execute("SELECT name FROM fn")}
    assert names == {"clamp"}
    ix.assert_clean()


def test_adapt_direct_and_adapt_modes(tmp_path: Path) -> None:
    pub = _public(tmp_path)
    ix = RU.ReuseIndex(tmp_path / "r.sqlite", _guard(tmp_path, pub))
    ix.build(tmp_path / "selfrepo", pub)
    out = RU.prepare(ix, _task(), tmp_path)
    assert out["mode"] == "direct" and out["cand"]["source"] == "mitlib" and out["cand"]["licence"] == "MIT"
    ast.parse(out["code"])
    assert "def clip(v, a, b)" in out["code"] and "max(a, min(v, b))" in out["code"]
    bad = _task()
    bad["tests"] = [{"args": [5, 0, 3], "expect": 99}]                 # the visible example disagrees: no direct, but still an adapt start
    out2 = RU.prepare(ix, bad, tmp_path)
    assert out2["mode"] == "adapt" and "max(a, min(v, b))" in out2["code"]


def test_non_function_task_is_none(tmp_path: Path) -> None:
    ix = RU.ReuseIndex(tmp_path / "r.sqlite")
    assert RU.prepare(ix, {"family": "app", "request": "add a thing"}, tmp_path)["mode"] == "none"


def test_free_names_not_adaptable(tmp_path: Path) -> None:
    c = RU.Cand("s", "MIT", "m.py", 1, 3, "f", "def f(a)", "doc", (), 0.9, 'def f(a):\n    """d"""\n    return helper(a)\n')
    r = RU.adapt(c, {"family": "fn", "name": "g", "stub": 'def g(x):\n    """d"""\n    raise NotImplementedError\n'})
    assert not r["ok"] and r["free"] == ["helper"]


def test_arity_mismatch_not_adapted() -> None:
    c = RU.Cand("s", "MIT", "m.py", 1, 3, "f", "def f(a, b)", "doc", (), 0.9, 'def f(a, b):\n    """d"""\n    return a\n')
    assert RU.adapt(c, {"family": "fn", "name": "g", "stub": 'def g(x):\n    """d"""\n    raise NotImplementedError\n'})["why"] == "arity differs"


REAL = Path.home() / "creator_runtime" / "gpuday" / "baseline_suite" / "tasks.json"


@pytest.mark.skipif(not REAL.is_file(), reason="the real suite is outside the repo")
def test_real_suite_never_in_index(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    pub = Path.home() / "creator_runtime" / "public_repos"
    g = RU.default_guard(root, pub)
    ix = RU.ReuseIndex(tmp_path / "real.sqlite", g)
    ix.add_repo("self", root, cap=4000)
    for r in ("requests", "sympy", "pydantic"):
        if (pub / r).is_dir():
            ix.add_repo(r, pub / r, cap=1500)
    ix.assert_clean()
    suite = json.loads(REAL.read_text(encoding="utf-8"))["tasks"]
    fns = [t for t in suite if t["family"] == "fn"]
    bad = {(s, p) for s, p in g.files}
    for s, p in bad:
        assert ix.db.execute("SELECT 1 FROM fn WHERE source=? AND path=?", (s, p)).fetchone() is None
    for t in fns:                                                       # no candidate for a suite task carries its own body or signature
        pt = RU.parse_task(t)
        for c in ix.query(t["query"], k=20, name=pt["name"], args=pt["args"], doc=pt["doc"]):
            assert (c.source, c.path) not in bad
    assert g.files and g.hashes and g.sigdocs and g.heldout_files


def test_near_duplicate_blocked_across_repos(tmp_path: Path) -> None:
    """A copy with renamed identifiers, other comments, type hints and one extra statement (httpx copying requests) is excluded."""
    pub = _public(tmp_path)
    body = ["    out = []", "    for item in values:", "        if item is None or item < 0:", "            continue",
            "        out.append(item * 2 + 1)", "    out.sort(reverse=True)", "    return out[:10]", ""]
    (pub / "mitlib" / "src" / "leaked.py").write_text(chr(10).join(['def secret_mean(values):', '    """Doc."""'] + body), encoding="utf-8")
    near = ["def tidy(xs: list) -> list:", '    """Other doc."""', "    # comment", "    out: list = []", "    for item in xs:",
            "        if item is None or item < 0:", "            continue", "        out.append(item * 2 + 1)", "    out.sort(reverse=True)",
            "    n = 10", "    return out[:n]", ""]
    (pub / "mitlib" / "src" / "near.py").write_text(chr(10).join(near), encoding="utf-8")
    ix = RU.ReuseIndex(tmp_path / "r.sqlite", _guard(tmp_path, pub))
    ix.build(tmp_path / "selfrepo", pub)
    names = {r[0] for r in ix.db.execute("SELECT name FROM fn")}
    assert "tidy" not in names and "clamp" in names


def test_real_parse_header_links_not_retrievable(tmp_path: Path) -> None:
    if not REAL.is_file():
        pytest.skip("real suite absent")
    root = Path(__file__).resolve().parents[1]
    pub = Path.home() / "creator_runtime" / "public_repos"
    ix = RU.ReuseIndex(tmp_path / "p.sqlite", RU.default_guard(root, pub))
    for r in ("requests", "httpx"):
        if (pub / r).is_dir():
            ix.add_repo(r, pub / r, cap=3000)
    assert not [c for c in ix.query("parse_header_links parse a link header into a list of dicts", k=20, name="parse_header_links", args=["value"])
                if c.name.lstrip("_") == "parse_header_links"]
