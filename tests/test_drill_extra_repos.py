"""Other projects' and public git histories as drill material (owner 3 Oct 2026: other projects 'yes do it' - privacy-stripped; 'you can also
use public information it can learn off of'). Synthetic temporary repositories only: the stored cache carries no author, no message text and
no readable path, the drill features are identical to the unstripped ones, and windows never cross projects."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from creator import drillsources as D


def _repo(root: Path, name: str, msgs: list[tuple[str, str]]) -> Path:
    r = root / name
    r.mkdir(parents=True)
    env = {"GIT_AUTHOR_NAME": "Jane Secretname", "GIT_AUTHOR_EMAIL": "jane@example.com", "GIT_COMMITTER_NAME": "Jane Secretname",
           "GIT_COMMITTER_EMAIL": "jane@example.com"}
    import os
    full = {**os.environ, **env}
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    for i, (path, msg) in enumerate(msgs):
        f = r / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"v{i}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(r), "add", "-A"], check=True, env=full)
        subprocess.run(["git", "-C", str(r), "commit", "-q", "-m", msg, "--date", f"2026-01-01T00:{i:02d}:00"], check=True,
                       env={**full, "GIT_COMMITTER_DATE": f"2026-01-01T00:{i:02d}:00"})
    return r


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from creator import device as DEV
    rt = tmp_path / "rt"
    monkeypatch.setattr(DEV, "runtime_dir", lambda *a, **k: rt)
    return rt


def test_the_stored_history_is_privacy_stripped_and_keeps_the_features(tmp_path: Path, runtime: Path) -> None:
    msgs = [("clients/acme_contract.txt", "Add acme renewal for Bob Clientname"), ("clients/acme_contract.txt", "fix: typo in Bob's renewal"),
            ("src/app.py", "feat: secret project kickoff"), ("src/app.py", "Revert the kickoff")] * 6
    repo = _repo(tmp_path, "privateproj", msgs)
    assert D.build_extra_cache(repo) == len(msgs)
    text = D.extra_cache_path(repo).read_text(encoding="utf-8")
    for leak in ("Jane", "Secretname", "jane@example.com", "acme", "Bob", "Clientname", "secret", "kickoff", "renewal", "clients", "src/app.py"):
        assert leak not in text, leak
    raw = D._git_commits(repo)
    for window, mode in ((3, "fixed"), (3, "churn")):
        plain = D._git_events(raw, window, mode)
        strip = D._git_events(D._read_cache(D.extra_cache_path(repo)), window, mode)
        assert [(a.created, a.resolved, a.y) for a in plain] == [(b.created, b.resolved, b.y) for b in strip]     # same outcomes
        assert [a.keys[:3] for a in plain] == [b.keys[:3] for b in strip]                                          # same type/size keys
        assert len({a.keys[3] for a in plain}) == len({b.keys[3] for b in strip})                                  # top folders: same partition


def test_items_from_several_projects_never_share_a_window(tmp_path: Path, runtime: Path) -> None:
    a = _repo(tmp_path, "aproj", [("x.py", "fix: a")] * 6)
    b = _repo(tmp_path, "bproj", [("x.py", "feat: b")] * 6)
    for r in (a, b):
        D.build_extra_cache(r)
    items = D.extra_git_items(3, "churn", [a, b])
    assert len(items) == 12 and {i.keys[-1] for i in items} == {"r:aproj", "r:bproj"}
    assert sum(1 for i in items if i.resolved is None) == 6                   # each project's last 3 stay open: no window crosses projects
    assert [i.created for i in items] == sorted(i.created for i in items)


def test_a_missing_cache_is_built_by_the_filler_before_any_job(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, "cproj", [("x.py", "fix: c")] * 8)
    monkeypatch.setattr(D, "extra_repos", lambda: [repo])
    monkeypatch.setattr(D, "DEFAULT_STATE", D.DEFAULT_STATE)
    with pytest.raises(D.GitCacheMissing):
        D.extra_git_items(3, "fixed")
    assert D.source_digest("x_git_fixed", tmp_path, tmp_path, tmp_path / "J.md", tmp_path) == "-"
    nxt = D.drill_filler(tmp_path / "state", tmp_path, tmp_path / "J.md", tmp_path, sources=["x_git_fixed"], seed=1)
    job = nxt()
    assert job is not None
    job()                                                                     # the first job is the one full read
    assert D.extra_cache_path(repo).exists()
    assert D.source_digest("x_git_fixed", tmp_path, tmp_path, tmp_path / "J.md", tmp_path).startswith("x")
    row = D.run_job("x_git_fixed", {"decay": 1.0, "k": 2.0}, tmp_path / "state", tmp_path, tmp_path / "J.md", tmp_path)
    assert row["items"] == 8 and "error" not in row
    assert json.dumps(row).find("x.py") < 0                                   # results carry scores, never paths


def test_cached_projects_drill_while_others_are_still_being_read(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 3 Oct: cpython's first read and clones still in progress held back every other project's drills (digest '-' until ALL were cached)
    a = _repo(tmp_path / "pub", "aproj", [("x.py", "fix: a")] * 6)
    b = _repo(tmp_path / "pub", "bproj", [("x.py", "feat: b")] * 6)
    _repo(tmp_path / "pub", "cproj.tmp", [("x.py", "feat: c")] * 2)                    # a clone in progress
    monkeypatch.setattr(D.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    (runtime / "public_repos").mkdir(parents=True)
    for r in (a, b, tmp_path / "pub" / "cproj.tmp"):
        r.rename(runtime / "public_repos" / r.name)
    a, b = runtime / "public_repos" / "aproj", runtime / "public_repos" / "bproj"
    assert [r.name for r in D.extra_repos()] == ["aproj", "bproj"]                       # in-progress clones are not projects yet
    D.build_extra_cache(a)                                                               # b not read yet
    assert D.source_digest("x_git_churn", tmp_path, tmp_path, tmp_path / "J.md", tmp_path).startswith("x")
    items = D.extra_git_items(3, "churn")
    assert {i.keys[-1] for i in items} == {"r:aproj"}                                    # drills on what is cached now
    d1 = D.source_digest("x_git_churn", tmp_path, tmp_path, tmp_path / "J.md", tmp_path)
    D.build_extra_cache(b)
    assert D.source_digest("x_git_churn", tmp_path, tmp_path, tmp_path / "J.md", tmp_path) != d1     # a newly cached project = new data
