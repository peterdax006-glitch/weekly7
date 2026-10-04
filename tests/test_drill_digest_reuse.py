"""h43 speed (3 Oct 2026): a public fetch re-reads only the x sources' digests - the research digest (a walk over ~44,000 files) is reused -
and every digest is still re-read once it is older than its DIGEST_TTL_S, so new data is never missed for long."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import publicdata as PD


def test_fetch_reuses_research_digest_and_ttl_still_refreshes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    data = {"research_bool": "r1", "x_git_fixed": "x1"}

    def fake_digest(source: str, *a: Any) -> str:
        calls.append(source)
        return data[source]
    monkeypatch.setattr(D, "source_digest", fake_digest)
    monkeypatch.setattr(D, "extra_repos", lambda: [])
    due: list[Path] = []
    monkeypatch.setattr(PD, "due_fetches", lambda now=None: list(due))
    monkeypatch.setattr(PD, "refresh_public", lambda repo, now=None: 0)
    monkeypatch.setattr(PD, "acquisition_due", lambda state, now=None: False)
    monkeypatch.setattr(PD, "CHECK_EVERY_S", 0.0)
    clock = [1000.0]
    monkeypatch.setattr(D.time, "monotonic", lambda: clock[0])
    nxt = D.drill_filler(tmp_path / "st", tmp_path, tmp_path / "J.md", tmp_path / "res", sources=["research_bool", "x_git_fixed"], seed=1,
                         public=True)
    assert nxt() is not None                                       # a grid job: both digests read once
    for _ in range(len(D.VARIANTS) + 1):                           # reach the x source's grid jobs too
        nxt()
    assert calls.count("research_bool") == 1 and calls.count("x_git_fixed") == 1
    due.append(tmp_path / "proj")
    fj = nxt()
    assert fj is not None and "fjob" in fj.__qualname__
    due.clear()
    fj()                                                           # the fetch: only the x digest is dropped
    data["x_git_fixed"] = "x2"
    for _ in range(2 * len(D.VARIANTS) + 10):                       # through the grid into the open-ended search (asks every source)
        nxt()
    assert calls.count("research_bool") == 1, "the research walk is not redone after a public fetch"
    assert calls.count("x_git_fixed") == 2
    clock[0] += D.DIGEST_TTL_S["research"] + 1                     # stale: read again
    data["research_bool"] = "r2"
    for _ in range(5):
        nxt()
    assert calls.count("research_bool") == 2


def test_ttl_values_are_sane() -> None:
    assert D.DIGEST_TTL_DEFAULT_S <= 60.0 and D.DIGEST_TTL_S["research"] <= 900.0
