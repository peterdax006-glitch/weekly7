"""scripts/nupen_deploy.py on a temp repo with a bare 'origin': drain, merge, check, push, resume - or refuse and restore."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nupen_deploy", ROOT / "scripts" / "nupen_deploy.py")
D = importlib.util.module_from_spec(spec)                                    # type: ignore[arg-type]
spec.loader.exec_module(D)                                                   # type: ignore[union-attr]

PASS = [[sys.executable, "-c", "pass"]]
FAIL = [[sys.executable, "-c", "import sys; sys.exit(3)"]]


def sh(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    r = tmp_path / "repo"
    r.mkdir()
    sh(r, "init", "-q", "-b", "main")
    (r / ".gitignore").write_text("state/\n", encoding="utf-8")
    (r / "a.txt").write_text("one\n", encoding="utf-8")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    sh(r, "remote", "add", "origin", str(origin))
    sh(r, "push", "-q", "-u", "origin", "main")
    (r / "state" / "creator").mkdir(parents=True)
    return r


def branch(r: Path, name: str, fname: str, text: str) -> None:
    sh(r, "checkout", "-q", "-b", name)
    (r / fname).write_text(text, encoding="utf-8")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", name)
    sh(r, "checkout", "-q", "main")


def deploy(r: Path, b: str, **kw: object) -> tuple[int, list[str]]:
    out: list[str] = []
    kw.setdefault("checks", PASS)
    kw.setdefault("poll_s", 0.05)
    kw.setdefault("start_wait_s", 5.0)
    return D.deploy(b, r, say=out.append, **kw), out                          # type: ignore[arg-type]


def test_dry_run_changes_nothing(repo: Path) -> None:
    branch(repo, "feat", "b.txt", "x\n")
    head = sh(repo, "rev-parse", "HEAD")
    rc, out = deploy(repo, "feat", dry_run=True)
    assert rc == 0 and sh(repo, "rev-parse", "HEAD") == head
    assert not (repo / "state" / "creator" / "NUPEN_DRAIN").exists() and any("dry run" in x for x in out)


def test_a_clean_deploy_drains_waits_merges_pushes_resumes_and_prints_the_first_status(repo: Path) -> None:
    branch(repo, "feat", "b.txt", "x\n")
    st = repo / "state" / "creator"
    (st / "nupen_service.log").write_text("old swarm started pid=1\n", encoding="utf-8")
    (st / "kernel.lock").write_text(str(os.getpid()), encoding="utf-8")       # a live kernel: a cycle in flight
    seen: dict[str, object] = {}

    def service() -> None:
        while not (st / "NUPEN_DRAIN").exists():
            time.sleep(0.02)
        time.sleep(0.5)
        seen["drain_while_busy"] = (st / "NUPEN_DRAIN").exists()
        (st / "kernel.lock").unlink()                                        # the swarm finished its cycle and exited
        while (st / "NUPEN_DRAIN").exists():
            time.sleep(0.05)                                                 # the supervisor restarts only after the drain is gone
        with (st / "nupen_service.log").open("a", encoding="utf-8") as f:
            f.write("2026 swarm started pid=2\n")
        with (st / "swarm_service.log").open("a", encoding="utf-8") as f:
            f.write('STATUS {"running": 3}\n')
    t = threading.Thread(target=service)
    t.start()
    rc, out = deploy(repo, "feat")
    t.join()
    assert rc == 0, out
    assert seen["drain_while_busy"] is True and out[-1] == 'STATUS {"running": 3}'
    assert (repo / "b.txt").exists() and sh(repo, "rev-parse", "main") == sh(repo, "rev-parse", "origin/main")   # merged AND pushed
    assert not (st / "NUPEN_DRAIN").exists()


def test_a_merge_conflict_is_refused_and_everything_is_restored(repo: Path) -> None:
    branch(repo, "feat", "a.txt", "from branch\n")
    (repo / "a.txt").write_text("from main\n", encoding="utf-8")
    sh(repo, "commit", "-qam", "main edit")
    head = sh(repo, "rev-parse", "HEAD")
    rc, out = deploy(repo, "feat")
    assert rc == 2 and any("merge conflict" in x for x in out)
    assert sh(repo, "rev-parse", "HEAD") == head and sh(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert not (repo / "state" / "creator" / "NUPEN_DRAIN").exists()          # the run goes on


def test_a_failing_check_puts_main_back_and_never_pushes(repo: Path) -> None:
    branch(repo, "feat", "b.txt", "x\n")
    head = sh(repo, "rev-parse", "HEAD")
    rc, out = deploy(repo, "feat", checks=FAIL)
    assert rc == 2 and any("check failed" in x for x in out)
    assert sh(repo, "rev-parse", "HEAD") == head and not (repo / "b.txt").exists()
    assert sh(repo, "rev-parse", "origin/main") == head
    assert not (repo / "state" / "creator" / "NUPEN_DRAIN").exists()


def test_a_cycle_that_never_ends_is_refused_without_merging(repo: Path) -> None:
    branch(repo, "feat", "b.txt", "x\n")
    (repo / "state" / "creator" / "kernel.lock").write_text(str(os.getpid()), encoding="utf-8")
    head = sh(repo, "rev-parse", "HEAD")
    rc, out = deploy(repo, "feat", wait_s=0.3)
    assert rc == 2 and any("in flight" in x for x in out) and sh(repo, "rev-parse", "HEAD") == head
    assert not (repo / "state" / "creator" / "NUPEN_DRAIN").exists()


def test_a_drain_that_was_already_there_is_left_alone_on_refusal(repo: Path) -> None:
    branch(repo, "feat", "b.txt", "x\n")
    (repo / "state" / "creator" / "NUPEN_DRAIN").write_text("owner", encoding="utf-8")
    rc, _ = deploy(repo, "feat", checks=FAIL)
    assert rc == 2 and (repo / "state" / "creator" / "NUPEN_DRAIN").exists()


def test_an_unknown_branch_or_a_dirty_tree_is_refused(repo: Path) -> None:
    assert deploy(repo, "nope")[0] == 2
    branch(repo, "feat", "b.txt", "x\n")
    (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
    assert deploy(repo, "feat")[0] == 2
