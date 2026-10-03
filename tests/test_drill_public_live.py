"""Prospective evidence from ACTIVE PUBLIC upstreams (3 Oct 2026): Nupen's own repository gets a few commits a day, so its git drills earn live
(before-the-fact) predictions slowly. Public clones in runtime_dir()/public_repos are fetched (no merge, no checkout), their new commits are
appended to the privacy-stripped cache, and live_pass_x records the best x variant's prediction for every commit whose outcome window is still
open - resolved when later upstream commits close it, counted as n_live by trust_section. Plus the autonomous acquisition of more public
repositories (owner 08:45: Nupen collects its own data when it runs out). Synthetic local repositories only: a bare 'upstream' that a
developer clone pushes to, and fake discovery - no network."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import publicdata as PD

ENV = {"GIT_AUTHOR_NAME": "Jane Secretname", "GIT_AUTHOR_EMAIL": "jane@example.com", "GIT_COMMITTER_NAME": "Jane Secretname",
       "GIT_COMMITTER_EMAIL": "jane@example.com"}
LEAKS = ("Jane", "Secretname", "jane@example.com", "acme", "clientname", "widgets_secret", "kickoff")
NOW = 1772323200.0                                                            # 1 Mar 2026: shallow-since dates make sense
FILES = ("widgets_secret/core.py", "widgets_secret/util.py", "docs/acme.md", "src/clientname.py")


def git(*args: str, cwd: Path | None = None, date: str | None = None) -> str:
    env = {**os.environ, **ENV}
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout


class Upstream:
    """A bare 'upstream' plus the developer clone that pushes to it."""

    def __init__(self, root: Path, name: str = "upstream") -> None:
        self.bare = root / f"{name}.git"
        self.dev = root / f"{name}_dev"
        git("init", "-q", "--bare", "-b", "main", str(self.bare))
        git("init", "-q", "-b", "main", str(self.dev))
        git("-C", str(self.dev), "remote", "add", "origin", str(self.bare))
        self.n = 0

    def commit(self, k: int, push: bool = True) -> None:
        for _ in range(k):
            i = self.n
            f = self.dev / FILES[i % len(FILES)]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"v{i}\n", encoding="utf-8")
            git("-C", str(self.dev), "add", "-A")
            msg = "fix: kickoff bug for acme" if i % 3 == 0 else "feat: widgets_secret for clientname"
            git("-C", str(self.dev), "commit", "-q", "-m", msg, date=f"2026-01-{1 + i // 600:02d}T{(i // 60) % 10:02d}:{i % 60:02d}:00")
            self.n += 1
        if push:
            git("-C", str(self.dev), "push", "-q", "origin", "main")

    @property
    def url(self) -> str:
        return self.bare.as_uri()


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from creator import device as DEV
    rt = tmp_path / "rt"
    monkeypatch.setattr(DEV, "runtime_dir", lambda *a, **k: rt)
    monkeypatch.setattr(D, "DEFAULT_STATE", D.DEFAULT_STATE)
    pub = rt / "public_repos"                                                    # this machine's ~/oldpc projects stay out of the tests
    monkeypatch.setattr(D, "extra_repos", lambda: sorted(d for d in pub.iterdir() if (d / ".git").exists()) if pub.is_dir() else [])
    getattr(D, "_XLIVE_MEMO", {}).clear()
    return rt


def public_clone(rt: Path, up: Upstream, name: str = "proj") -> Path:
    dest = rt / "public_repos" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "-q", "--no-tags", "--single-branch", up.url, str(dest))
    return dest


def lines(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def all_text(*dirs: Path) -> str:
    out = []
    for d in dirs:
        for f in d.rglob("*"):
            if f.is_file() and ".git" not in f.parts:
                out.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def test_public_commits_are_predicted_before_their_outcome_and_resolved_after_new_upstream_commits(tmp_path: Path, runtime: Path,
                                                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    up = Upstream(tmp_path)
    up.commit(60)
    repo = public_clone(runtime, up)
    state = tmp_path / "state"
    assert D.is_public(repo)
    assert D.build_extra_cache(repo) == 60
    for s in D.XLIVE_SOURCES:                                                    # the drills ran: a best variant per x source
        row = D.run_job(s, {"decay": 1.0, "k": 2.0}, state, tmp_path, tmp_path / "J.md", tmp_path)
        assert "error" not in row and row["items"] == 60
    r1 = D.live_pass_x(state, tmp_path / "J.md", now=1000.0)
    assert r1 == {"new": D.FIX_WINDOW + D.CHURN_WINDOW, "resolved": 0}         # exactly the commits whose window is still open
    rows = lines(D.live_x_path(state))
    assert all("resolved" not in r and "y" not in r and "outcome" not in r and r["made_at"] == 1000.0 for r in rows)
    assert D.live_pass_x(state, tmp_path / "J.md", now=1001.0)["new"] == 0       # nothing changed: skipped
    getattr(D, "_XLIVE_MEMO", {}).clear()
    assert D.live_pass_x(state, tmp_path / "J.md", now=1002.0) == {"new": 0, "resolved": 0}   # idempotent from the file too
    assert D.trust_section(state)["x_git_fixed"]["heldout"]["n_live"] == 0

    up.commit(25)                                                                # upstream moves on; nothing local knows yet
    assert D.live_pass_x(state, tmp_path / "J.md", now=1003.0)["resolved"] == 0
    assert PD.refresh_public(repo, now=2000.0) == 25                            # fetch + incremental stripped cache
    assert PD.refresh_public(repo, now=2001.0) == 0
    r2 = D.live_pass_x(state, tmp_path / "J.md", now=3000.0)
    assert r2 == {"new": D.FIX_WINDOW + D.CHURN_WINDOW, "resolved": D.FIX_WINDOW + D.CHURN_WINDOW}
    rows = lines(D.live_x_path(state))
    first = {r["id"]: i for i, r in enumerate(rows) if "p" in r}
    res = [(i, r) for i, r in enumerate(rows) if "resolved" in r]
    assert len(res) == 30 and all(first[r["id"]] < i for i, r in res)            # every outcome was written after its prediction
    lp = D.live_preds(state)
    assert len(lp["x_git_fixed"]) == D.FIX_WINDOW and len(lp["x_git_churn"]) == D.CHURN_WINDOW
    sec = D.trust_section(state)
    assert sec["x_git_fixed"]["heldout"]["n_live"] == D.FIX_WINDOW and sec["x_git_churn"]["heldout"]["n_live"] == D.CHURN_WINDOW
    assert sec["x_git_fixed"]["live"]["n"] == D.FIX_WINDOW
    assert "git_fixed" not in D.live_preds(state)                                # Nupen's own live file is untouched
    text = all_text(runtime / "thinking", state)
    for leak in LEAKS:
        assert leak not in text, leak


def test_archived_projects_are_never_fetched_nor_predicted_live(tmp_path: Path, runtime: Path) -> None:
    up = Upstream(tmp_path)
    up.commit(40)
    arch = tmp_path / "oldpc" / "proj"
    git("clone", "-q", up.url, str(arch))
    D.build_extra_cache(arch)
    state = tmp_path / "state"
    D.run_job("x_git_fixed", {"decay": 1.0, "k": 2.0}, state, tmp_path, tmp_path / "J.md", tmp_path)
    assert not D.is_public(arch)
    assert D.live_pass_x(state, tmp_path / "J.md", now=1.0, repos=[arch])["new"] == 0
    with pytest.raises(ValueError):
        PD.refresh_public(arch)


def test_a_mixed_set_predicts_only_the_public_open_commits(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = Upstream(tmp_path, "a"), Upstream(tmp_path, "b")
    a.commit(40)
    b.commit(40)
    arch = tmp_path / "oldpc" / "arch"
    git("clone", "-q", a.url, str(arch))
    pub = public_clone(runtime, b, "pubb")
    for r in (arch, pub):
        D.build_extra_cache(r)
    monkeypatch.setattr(D, "extra_repos", lambda: [arch, pub])
    state = tmp_path / "state"
    D.run_job("x_git_fixed", {"decay": 1.0, "k": 2.0}, state, tmp_path, tmp_path / "J.md", tmp_path)
    assert D.live_pass_x(state, tmp_path / "J.md", now=5.0)["new"] == D.FIX_WINDOW
    assert all(r["subject"].startswith("pubb:") for r in lines(D.live_x_path(state)))


def test_live_x_predictions_doing_worse_block_trust(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = {"n": 300, "n_live": 0, "gain_ci95": [0.01, 0.02], "ece": 0.02, "best_baseline": "base", "brier": 0.16}
    monkeypatch.setattr(D, "best_variant", lambda state, s: {"source": s, "variant": {"decay": 1.0, "k": 2.0}, "items": 0, "resolved": 0,
                                                              "heldout": good, "variants_tried": 1} if s == "x_git_fixed" else None)
    monkeypatch.setattr(D, "search_report", lambda state: {})
    path = D.live_x_path(tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(40):
            y = i % 2
            f.write(json.dumps({"id": f"x_git_fixed:p:c{i}", "source": "x_git_fixed", "subject": f"p:c{i}", "made_at": float(i),
                                "p": 0.95 if y == 0 else 0.05, "base": 0.5, "last": 0.5, "mode": "live"}) + "\n")
            f.write(json.dumps({"id": f"x_git_fixed:p:c{i}", "resolved": y, "at": float(i) + 1}) + "\n")
    sec = D.trust_section(tmp_path)["x_git_fixed"]
    assert sec["heldout"]["n_live"] == 40 and not sec["trusted"] and any("do WORSE" in w for w in sec["why_not"])


def test_a_shallow_clones_boundary_commit_never_enters_the_cache(tmp_path: Path, runtime: Path) -> None:
    up = Upstream(tmp_path)
    up.commit(30)
    dest = runtime / "public_repos" / "shal"
    dest.parent.mkdir(parents=True)
    git("clone", "-q", "--no-tags", "--single-branch", "--no-checkout", "--shallow-since=2026-01-01T00:20:00", up.url, str(dest))
    roots = D._shallow_roots(dest)
    assert roots
    n = D.build_extra_cache(dest)
    hs = {r["h"] for r in lines(D.extra_cache_path(dest))}
    assert n == len(hs) and not hs & roots and n < 30
    up.commit(5)
    assert PD.refresh_public(dest, now=10.0) == 5


def test_the_filler_hands_out_fetch_jobs_and_a_failing_fetch_never_raises(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    up = Upstream(tmp_path)
    up.commit(30)
    repo = public_clone(runtime, up)
    D.build_extra_cache(repo)
    state = tmp_path / "state"
    monkeypatch.setattr(PD, "acquisition_due", lambda state, now=None: False)
    up.commit(3)
    nxt = D.drill_filler(state, tmp_path, tmp_path / "J.md", tmp_path, sources=["x_git_fixed"], seed=1, public=True)
    job = nxt()
    assert job is not None
    job()                                                                        # the fetch job: 3 new upstream commits cached
    assert len(lines(D.extra_cache_path(repo))) == 33
    assert json.loads(PD.fetch_state_path().read_text(encoding="utf-8"))["proj"]["ok"]
    assert PD.due_fetches() == []                                                # not again before FETCH_EVERY_S
    git("-C", str(repo), "remote", "set-url", "origin", (tmp_path / "gone.git").as_uri())
    st = json.loads(PD.fetch_state_path().read_text(encoding="utf-8"))
    st["proj"]["at"] = 0.0
    PD.fetch_state_path().write_text(json.dumps(st), encoding="utf-8")
    nxt2 = D.drill_filler(state, tmp_path, tmp_path / "J.md", tmp_path, sources=["x_git_fixed"], seed=1, public=True)
    job2 = nxt2()
    assert job2 is not None
    job2()                                                                       # fails inside: logged, never raised
    assert json.loads(PD.fetch_state_path().read_text(encoding="utf-8"))["proj"]["ok"] is False
    assert any(r.get("source") == "public_fetch" and "error" in r for r in lines(D.runs_path(state)))
    nxt3 = D.drill_filler(state, tmp_path, tmp_path / "J.md", tmp_path, sources=["x_git_fixed"], seed=1)   # public off (tests, default)
    j = nxt3()
    assert j is None or "fjob" not in getattr(j, "__qualname__", "")


# ------------------------------------------------------------------------------------------------ acquisition
def test_acquire_clones_one_repo_records_it_and_the_machinery_picks_it_up(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ups = {"acme/one": Upstream(tmp_path, "one"), "acme/two": Upstream(tmp_path, "two")}
    for u in ups.values():
        u.commit(30)
    monkeypatch.setattr(PD, "disk_ok", lambda extra_mb=0.0: (True, "ok"))
    state = tmp_path / "state"
    urls = {k: u.url for k, u in ups.items()}
    urls["acme/broken"] = (tmp_path / "nothing.git").as_uri()
    disc = [{"repo": "acme/two", "lang": "python", "license": "mit", "stars": 9}]
    r1 = PD.acquire_one(state, now=NOW + 0, candidates=["acme/broken", "acme/one"], discover=lambda now: disc, url_of=urls.__getitem__)
    assert r1 is not None and not r1["ok"] and r1["repo"] == "acme/broken" and r1["why"] == "error"
    r2 = PD.acquire_one(state, now=NOW + 1, candidates=["acme/broken", "acme/one"], discover=lambda now: disc, url_of=urls.__getitem__)
    assert r2 is not None and r2["ok"] and r2["repo"] == "acme/one" and r2["commits"] >= 1 and r2["dir"] == "one"   # the failed one waits a day
    r3 = PD.acquire_one(state, now=NOW + 2, candidates=["acme/broken", "acme/one"], discover=lambda now: disc, url_of=urls.__getitem__)
    assert r3 is not None and r3["ok"] and r3["repo"] == "acme/two" and "github search" in r3["why"]
    assert PD.acquire_one(state, now=NOW + 3, candidates=["acme/broken", "acme/one"], discover=lambda now: disc, url_of=urls.__getitem__) is None
    repos = PD.public_repos()
    assert {r.name for r in repos} == {"one", "two"} and not list(PD.staging_dir().glob("*"))
    assert not (repos[0] / "widgets_secret").exists()                            # no checkout: history only
    assert len(lines(PD.acquisitions_path(state))) == 3
    nxt = D.drill_filler(state, tmp_path, tmp_path / "J.md", tmp_path, sources=["x_git_fixed"], seed=1)
    job = nxt()
    assert job is not None
    job()                                                                        # the existing cache job builds the new projects' caches
    assert all(D.extra_cache_path(r).exists() for r in repos)
    for leak in LEAKS:                                                           # the log names public repositories, never authors
        assert leak not in all_text(runtime / "thinking"), leak
    assert "Jane" not in all_text(state) and "example.com" not in all_text(state)


def test_acquisition_respects_disk_size_cap_and_cadence(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    up = Upstream(tmp_path, "big")
    up.commit(10)
    state = tmp_path / "state"
    monkeypatch.setattr(PD, "BUDGET_GB", 0.0)
    r = PD.acquire_one(state, now=NOW + 1, candidates=["acme/big"], discover=None, url_of=lambda f: up.url)
    assert r is not None and r["why"] == "disk" and not PD.public_repos()
    monkeypatch.setattr(PD, "disk_ok", lambda extra_mb=0.0: (True, "ok"))
    r = PD.acquire_one(state, now=NOW + 2, candidates=["acme/big"], discover=None, url_of=lambda f: up.url, cap_mb=0.0)
    assert r is not None and r["why"] == "too_big" and not PD.public_repos() and not list(PD.staging_dir().glob("*"))
    assert not PD.acquisition_due(state, now=NOW + 2 + PD.ACQ_DRY_GAP_S - 1)
    monkeypatch.setattr(PD, "drills_exhausted", lambda state: False)
    assert not PD.acquisition_due(state, now=NOW + 2 + PD.ACQ_DRY_GAP_S + 1)        # drills still learning: the hourly cadence
    assert PD.acquisition_due(state, now=NOW + 2 + PD.ACQ_EVERY_S)
    monkeypatch.setattr(PD, "drills_exhausted", lambda state: True)
    assert PD.acquisition_due(state, now=NOW + 2 + PD.ACQ_DRY_GAP_S + 1)            # ran dry: sooner


def test_drills_exhausted_reads_the_search_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    srcs = [s for s, k in D.SOURCES.items() if k in ("git", "gitx")]
    monkeypatch.setattr(D, "search_report", lambda state, sources=None: {s: {"stopped": True} for s in srcs})
    assert PD.drills_exhausted(tmp_path)
    monkeypatch.setattr(D, "search_report", lambda state, sources=None: {s: {"stopped": s != "x_git_churn"} for s in srcs})
    assert not PD.drills_exhausted(tmp_path)


def test_discovery_filters_licences_and_honours_rate_limits(tmp_path: Path, runtime: Path) -> None:
    calls: list[str] = []

    def limited(url: str) -> tuple[int, dict[str, Any], dict[str, str]]:
        calls.append(url)
        return 403, {}, {"X-RateLimit-Reset": "5000"}
    assert PD.github_discover(now=100.0, get=limited) == [] and len(calls) == 1
    assert PD.github_discover(now=200.0, get=limited) == [] and len(calls) == 1      # waits for the reset: no request
    items = [{"full_name": "a/mit", "license": {"spdx_id": "MIT"}, "size": 1000, "stargazers_count": 5},
             {"full_name": "a/gpl", "license": {"spdx_id": "GPL-3.0"}, "size": 1000},
             {"full_name": "a/huge", "license": {"spdx_id": "MIT"}, "size": PD.DISCOVER_MAX_KB + 1},
             {"full_name": "a/nolicence", "license": None, "size": 10}]

    def ok(url: str) -> tuple[int, dict[str, Any], dict[str, str]]:
        calls.append(url)
        assert "api.github.com/search/repositories" in url and "token" not in url.lower()
        return 200, {"items": items}, {}
    found = PD.github_discover(now=6000.0, get=ok)
    assert [f["repo"] for f in found] == ["a/mit"]
    assert PD.github_discover(now=6001.0, get=ok) == found and len(calls) == 2         # at most one request a minute
