"""h43 (3 Oct 2026): other projects' items keep ONE setup-facts dict per commit for both modes (fixed / churn), file names interned in a
frozenset - equal to what a fresh read gives, and refreshed when the cache grows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import drillsources as D


def test_modes_share_meta_equal_to_a_fresh_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "_PARSED", {})
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    cache = tmp_path / "x.jsonl"
    monkeypatch.setattr(D, "extra_cache_path", lambda r: cache)
    rows = [{"h": f"h{i:03d}", "t": float(i), "s": "fix bug" if i % 3 == 0 else "add x", "files": [f"a/f{i % 4}", "b/c"], "lines": i,
             "add": i, "del": i % 2} for i in range(30)]
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows[:25]))
    fx = D.extra_git_items(5, "fixed", [repo])
    ch = D.extra_git_items(8, "churn", [repo])
    assert all(a.meta is b.meta for a, b in zip(fx, ch))                       # one dict per commit, both modes
    assert all(isinstance(a.meta["files"], frozenset) for a in fx)
    commits = sorted(D._read_cache(cache), key=lambda c: c["t"])
    fresh = [{**it.meta, "repo": "proj"} for it in D._git_events(commits, 5, "fixed")]
    assert [a.meta for a in fx] == fresh                                       # equal to the uncached meta (frozenset == set)
    with cache.open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows[25:]))
    fx2 = D.extra_git_items(5, "fixed", [repo])
    assert len(fx2) == 30 and fx2[-1].meta["t"] == 29.0
