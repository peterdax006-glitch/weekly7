"""h38 speed (3 Oct 2026): drill jobs reuse what they already read - no git process when HEAD is unchanged, research files parsed once per
(size, mtime) - and every result stays identical to a fresh read."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from creator import drillsources as D


def _git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True,
                          text=True).stdout


def _repo(path: Path, n: int, start: int = 0) -> Path:
    path.mkdir(exist_ok=True)
    if not (path / ".git").exists():
        _git(path, "init", "-q")
    for i in range(start, start + n):
        (path / f"f{i}.py").write_text(f"v{i}\n")
        _git(path, "add", "-A")
        _git(path, "commit", "-qm", f"fix: {i}", "--date", f"2026-01-{i + 1:02d}T00:00:00")
    return path


def test_head_fast_matches_git_loose_and_packed(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", 2)
    assert D._head_fast(repo) == _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "pack-refs", "--all")                                   # the ref now lives only in packed-refs
    assert D._head_fast(repo) == _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "--detach")
    assert D._head_fast(repo) == _git(repo, "rev-parse", "HEAD").strip()
    assert D._head_fast(tmp_path / "nope") is None


def test_unchanged_head_runs_no_git_and_new_commits_still_arrive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "DEFAULT_STATE", None)
    repo, st = _repo(tmp_path / "r", 3), tmp_path / "st"
    assert D.refresh_git_cache(repo, st) == 3
    calls: list[list[str]] = []
    real = D._git
    monkeypatch.setattr(D, "_git", lambda r, a, t: (calls.append(a), real(r, a, t))[1])
    assert D.refresh_git_cache(repo, st) == 0 and calls == []          # old code: cat-file + log on every call
    row = D.compute_row("git_fixed", {"decay": 0.97, "k": 3.0}, st, repo, tmp_path / "J.md", tmp_path / "res")
    assert calls == [] and row["digest"] == _git(repo, "rev-parse", "HEAD").strip()[:12] and row["items"] == 3
    _repo(repo, 2, start=3)                                            # new commits: read incrementally, as before
    assert D.refresh_git_cache(repo, st) == 2 and any("..HEAD" in x for c in calls for x in c)
    assert len(D.git_commits(repo, st)) == 5


def test_moved_head_is_read_and_tip_follows_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "DEFAULT_STATE", None)
    repo, st = _repo(tmp_path / "r", 3), tmp_path / "st"
    D.refresh_git_cache(repo, st)
    _git(repo, "reset", "-q", "--hard", "HEAD~2")
    _repo(repo, 1, start=7)
    D.refresh_git_cache(repo, st)
    got = [c["s"] for c in D.git_commits(repo, st)]
    assert got[-1] == "fix: 7" and got[0] == "fix: 0"                     # same rows as before the tip existed (range last..HEAD)
    assert D.cache_path(st).with_suffix(".tip").read_text().strip() == _git(repo, "rev-parse", "HEAD").strip()


def _fresh(root: Path) -> list[tuple[object, ...]]:
    D._RESEARCH.clear()
    D._RESEARCH_DISK_LOADED.clear()
    return [(i.keys, i.created, i.resolved, i.y, i.subject) for i in D.research_scan(root)[0]]


def test_research_scan_reuses_parses_and_sees_every_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "DEFAULT_STATE", tmp_path / "st")
    root = tmp_path / "res"
    (root / "fam" / "sub").mkdir(parents=True)
    (root / "sealed_dir").mkdir()
    for i in range(5):
        (root / "fam" / "sub" / f"r{i}.json").write_text(json.dumps({"passed": i % 2 == 0, "n": i}))
    (root / "fam" / "bad.json").write_text("{not json")
    (root / "sealed_dir" / "x.json").write_text(json.dumps({"hidden": True}))       # counted by the digest, never read
    (root / "fam" / "notes.txt").write_text("x")
    reads: list[str] = []
    real = Path.read_text

    def spy(self: Path, *a: object, **k: object) -> str:
        if self.suffix == ".json" and "res" in self.parts:
            reads.append(self.name)
        return real(self, *a, **k)                                     # type: ignore[arg-type]
    monkeypatch.setattr(Path, "read_text", spy)
    first, dg = D.research_scan(root)
    assert dg == D.source_digest("research_bool", tmp_path, tmp_path, tmp_path / "J.md", root) == "8"
    assert len(reads) == 6 and "x.json" not in reads
    reads.clear()
    again, _ = D.research_scan(root)
    assert reads == [] and [(i.keys, i.y, i.subject) for i in again] == [(i.keys, i.y, i.subject) for i in first]
    p = root / "fam" / "sub" / "r1.json"                               # a changed file (new mtime) is re-read
    p.write_text(json.dumps({"passed": True, "n": 1}))
    os.utime(p, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    (root / "fam" / "sub" / "r0.json").unlink()                        # a deleted file disappears
    (root / "fam" / "new.json").write_text(json.dumps({"ok": False}))  # a new file appears
    now, dg2 = D.research_scan(root)
    assert sorted(reads) == ["new.json", "r1.json"] and dg2 == "8"
    monkeypatch.setattr(Path, "read_text", real)
    assert [(i.keys, i.created, i.resolved, i.y, i.subject) for i in now] == _fresh(root)
    assert {i.subject: i.y for i in now}["r1.json:passed"] == 1 and "r0.json:passed" not in {i.subject for i in now}


def test_research_disk_memo_starts_a_new_process_warm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "DEFAULT_STATE", tmp_path / "st")
    root = tmp_path / "res"
    root.mkdir()
    for i in range(4):
        (root / f"r{i}.json").write_text(json.dumps({"passed": bool(i % 2)}))
    want = _fresh(root)
    D._RESEARCH.clear()                                                # a new worker: empty memory, the disk memo is read
    D._RESEARCH_DISK_LOADED.clear()
    reads: list[str] = []
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self.name), real(self, *a, **k))[1])
    got = [(i.keys, i.created, i.resolved, i.y, i.subject) for i in D.research_scan(root)[0]]
    assert got == want and [r for r in reads if r[:1] == "r" and r[1:2].isdigit()] == []


def test_x_items_memo_is_invalidated_by_an_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    cache = tmp_path / "x.jsonl"
    monkeypatch.setattr(D, "extra_cache_path", lambda r: cache)
    rows = [{"h": f"h{i}", "t": float(i), "s": "fix -", "files": ["a/b"], "lines": 1, "add": 1, "del": 0} for i in range(30)]
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows[:25]))
    a = D.extra_git_items(5, "fixed", [repo])
    assert D.extra_git_items(5, "fixed", [repo]) == a
    with cache.open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows[25:]))
    b = D.extra_git_items(5, "fixed", [repo])
    assert len(b) == 30 and len(a) == 25
