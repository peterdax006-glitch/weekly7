"""h43 speed (3 Oct 2026): public cases kept per (size, mtime_ns) of each repository's text cache - identical to a fresh build, a grown
cache is re-read (only that repository), a repository that is not public stays empty - and drillsources._git_events, rewritten for speed,
gives exactly the events of the original code."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import publiccases as P


def _old_git_events(commits: list[dict[str, Any]], window: int, mode: str) -> list[D.BItem]:
    """The code before h43, verbatim (the reference)."""
    out: list[D.BItem] = []
    for i, c in enumerate(commits):
        top = sorted(c["files"])[0].split("/")[0] if c["files"] else "-"
        keys = (f"m:{D._msg_class(c['s'])}", f"n:{D._log2b(len(c['files']))}", f"l:{D._log2b(c['lines'])}", f"d:{top}")
        meta = {"t": c["t"], "files": c["files"], "s": f"{D._msg_class(c['s'])} {'fix' if D.FIX_WORDS.search(c['s']) else '-'}",
                "add": c.get("add", 0), "del": c.get("del", 0)}
        if i + window >= len(commits):
            out.append(D.BItem(keys, c["t"], None, 0, c["h"][:10], meta))
            continue
        nxt = commits[i + 1:i + 1 + window]
        if mode == "fixed":
            y = int(any(D.FIX_WORDS.search(n["s"]) and n["files"] & c["files"] for n in nxt))
        else:
            y = int(any(n["files"] & c["files"] for n in nxt))
        out.append(D.BItem(keys, c["t"], max(c["t"], nxt[-1]["t"]), y, c["h"][:10], meta))
    return out


def _commits(n: int, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    words = ["fix", "Fix:", "revert", "add", "docs(x):", "bug", "feat", "  refactor", "hotfix", "regression", "", "fixes", "prefix"]
    out = []
    for i in range(n):
        files = {f"{rnd.choice(['src', 'docs', 'Lib', 'a/b'])}/f{rnd.randrange(12)}.py" for _ in range(rnd.randrange(0, 4))}
        out.append({"h": f"{i:040x}", "t": 1000.0 + i * rnd.choice([0, 1, 60]), "s": f"{rnd.choice(words)} thing {i}", "files": files,
                    "lines": rnd.randrange(0, 900), "add": i % 3, "del": i % 5})
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_git_events_identical_to_the_original(seed: int) -> None:
    cs = _commits(400, seed)
    for window, mode in ((D.FIX_WINDOW, "fixed"), (D.CHURN_WINDOW, "churn"), (1, "fixed"), (3, "churn"), (500, "fixed")):
        assert D._git_events(cs, window, mode) == _old_git_events(cs, window, mode)


def _old_raw_cases(topic: str) -> list[tuple[str, D.BItem, str, str]]:
    mode, w = P.MODE[topic]
    out = []
    for repo in P.public_repos():
        cs = P.commits(repo)
        for c, it in zip(cs, _old_git_events(cs, w, mode)):
            keys = it.keys[:3] + (f"d:{repo.name}/{it.keys[3][2:]}", f"r:{repo.name}")
            top = sorted(c["files"])[0].split("/")[0] if c["files"] else "-"
            text = (f"A commit to the public open-source project '{repo.name}' at {P._iso(c['t'])} UTC with message '{c['s'][:P.SUBJECT_MAX]}'. "
                    f"It changed {len(c['files'])} files and {c['lines']} lines; first top folder '{top}'. {P.question(topic)}")
            out.append((repo.name, D.BItem(keys, it.created, it.resolved, it.y, f"{repo.name}:{it.subject}"), D._msg_class(c["s"]), text))
    out.sort(key=lambda x: x[1].created)
    return out


def _write(rt: Path, name: str, cs: list[dict[str, Any]], public: bool = True) -> Path:
    repo = (rt / "public_repos" / name) if public else (rt / "private" / name)
    (repo / ".git").mkdir(parents=True, exist_ok=True)
    p = P.text_cache_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in cs:
            f.write(json.dumps({"h": c["h"], "t": c["t"], "s": c["s"], "files": sorted(c["files"]), "lines": c["lines"]}) + "\n")
    return repo


def test_memo_is_identical_and_follows_each_cache_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rt = tmp_path / "rt"
    monkeypatch.setenv("NUPEN_RUNTIME", str(rt))
    monkeypatch.setattr(P, "_REPO_CASES", {})
    monkeypatch.setattr(P, "_ALL_CASES", {})
    _write(rt, "alpha", _commits(300, 5))
    beta = _write(rt, "beta", _commits(200, 6))
    for t in P.TOPICS:
        first = P.raw_cases(t)
        assert first == _old_raw_cases(t) and len(first) == 500
        again = P.raw_cases(t)
        assert again == first and again is not first                  # a fresh list each call
        assert P.items(t) == [x[1] for x in first]
    built: list[str] = []
    real = P._build_repo_cases
    monkeypatch.setattr(P, "_build_repo_cases", lambda repo: (built.append(repo.name), real(repo))[1])
    grown = _commits(260, 6)                                           # beta grows: only beta is rebuilt
    with P.text_cache_path(beta).open("a", encoding="utf-8") as f:
        for c in grown[200:]:
            f.write(json.dumps({"h": c["h"] + "x", "t": c["t"] + 99999.0, "s": c["s"], "files": sorted(c["files"]), "lines": c["lines"]}) + "\n")
    os.utime(P.text_cache_path(beta), ns=(1, 2))                       # a distinct mtime even on a coarse clock
    for t in P.TOPICS:
        assert P.raw_cases(t) == _old_raw_cases(t) and len(P.raw_cases(t)) == 560
    assert built == ["beta"]
    priv = _write(rt, "secret", _commits(50, 7), public=False)          # never public: its cases stay empty, memo or not
    assert P.raw_cases("pub_git_fixed", [priv]) == [] and P.raw_cases("pub_git_fixed", [priv]) == []


def test_disk_cache_identical_and_invalidated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """h48: a fresh process loads the built cases from disk (same values as a build); a changed text cache, another code stamp or a damaged
    file means a rebuild."""
    rt = tmp_path / "rt"
    monkeypatch.setenv("NUPEN_RUNTIME", str(rt))
    repo = _write(rt, "alpha", _commits(300, 5))
    want = {t: _old_raw_cases(t) for t in P.TOPICS}

    def fresh() -> None:
        monkeypatch.setattr(P, "_REPO_CASES", {})
        monkeypatch.setattr(P, "_ALL_CASES", {})

    built: list[str] = []
    real = P._build_repo_cases
    monkeypatch.setattr(P, "_build_repo_cases", lambda r: (built.append(r.name), real(r))[1])
    fresh()
    assert all(P.raw_cases(t) == want[t] for t in P.TOPICS) and built == ["alpha"]
    assert P._disk_path(repo).exists()
    fresh()
    assert all(P.raw_cases(t) == want[t] for t in P.TOPICS) and built == ["alpha"]          # from disk: no rebuild
    monkeypatch.setattr(P, "_CODE_STAMP", "other")
    fresh()
    assert all(P.raw_cases(t) == want[t] for t in P.TOPICS) and built == ["alpha", "alpha"]  # other code: rebuilt
    P._disk_path(repo).write_bytes(b"not a pickle")
    fresh()
    assert all(P.raw_cases(t) == want[t] for t in P.TOPICS) and built == ["alpha"] * 3       # damaged: rebuilt
    with P.text_cache_path(repo).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"h": "z" * 40, "t": 9e6, "s": "fix more", "files": ["src/f1.py"], "lines": 3}) + "\n")
    os.utime(P.text_cache_path(repo), ns=(1, 2))
    fresh()
    assert len(P.raw_cases("pub_git_fixed")) == 301 and built == ["alpha"] * 4              # changed cache: rebuilt
