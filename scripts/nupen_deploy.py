"""Deploy a branch to the running Nupen without losing in-flight work (2 Oct 2026: every deploy pause force-killed the swarm; a
stale ledger lock caused a crash loop and 35-45 min cycles were lost).

    python scripts/nupen_deploy.py h12/some-branch            # drain -> wait -> merge -> smoke + mypy -> push -> resume
    python scripts/nupen_deploy.py h12/some-branch --dry-run  # print what each step would do, change nothing
    python scripts/nupen_deploy.py h12/some-branch --hard     # NUPEN_STOP (immediate kill) instead of NUPEN_DRAIN

Steps: 1 create NUPEN_DRAIN (the swarm plans nothing new, finishes in-flight cycles, exits); 2 wait until no kernel holds
state/creator/kernel.lock (bounded by --wait-s); 3 `git merge --no-edit BRANCH` on main - a conflict is aborted; 4 smoke tests +
mypy - a failure puts main back where it was; 5 push; 6 remove the drain/stop file; 7 wait for 'swarm started' in the supervisor
log and print the first STATUS line. Any refusal restores the drain/stop state it found and exits non-zero. Never uses git stash."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

SMOKE_TESTS = ("tests/test_nupen_runsafe.py", "tests/test_nupen_monitor.py", "tests/test_nupen_service.py")


class DeployRefused(RuntimeError):
    """The deploy stopped before changing anything that is not restored."""


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise DeployRefused(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()[:400]}")
    return r


def pid_alive(pid: int) -> bool:
    try:
        import psutil
        return bool(psutil.pid_exists(pid)) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:                                                   # noqa: BLE001 - unknown means alive
        return True


def kernel_busy(state: Path) -> bool:
    """A live kernel (swarm) holds state/kernel.lock: cycles may still be in flight."""
    lock = state / "kernel.lock"
    try:
        holder = int(lock.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return lock.exists()
    return holder > 0 and pid_alive(holder)


def default_checks(repo: Path, python: str, tests: Sequence[str]) -> list[list[str]]:
    cmds = [[python, "-m", "pytest", t, "-q", "-x", "--deselect",
             "tests/test_creator_generator.py::test_the_real_local_model_answers_offline"] for t in tests if (repo / t).exists()]
    cmds.append([python, "-m", "mypy"])
    return cmds


def run_checks(repo: Path, cmds: Sequence[Sequence[str]], say: Callable[[str], None]) -> Optional[str]:
    """None when every command passes, else a description of the first failure."""
    for c in cmds:
        say("check: " + " ".join(c))
        r = subprocess.run(list(c), cwd=repo, capture_output=True, text=True)
        if r.returncode != 0:
            return f"{' '.join(c)} exited {r.returncode}: {(r.stdout + r.stderr).strip()[-600:]}"
    return None


def count_lines_with(path: Path, needle: str) -> int:
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if needle in ln)
    except OSError:
        return 0


def first_status_after(path: Path, offset: int) -> Optional[str]:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            for raw in f.read().decode("utf-8", "replace").splitlines():
                if raw.startswith("STATUS"):
                    return raw
    except OSError:
        pass
    return None


def deploy(branch: str, repo: Path = ROOT, main: str = "main", dry_run: bool = False, hard: bool = False,
           wait_s: float = 4200.0, start_wait_s: float = 300.0, push: bool = True, checks: Optional[Sequence[Sequence[str]]] = None,
           poll_s: float = 2.0, say: Callable[[str], None] = print) -> int:
    state = repo / "state" / "creator"
    flag = state / ("NUPEN_STOP" if hard else "NUPEN_DRAIN")
    svc_log, swarm_log = state / "nupen_service.log", state / "swarm_service.log"
    python = str(repo / ".venv" / "Scripts" / "python.exe") if (repo / ".venv" / "Scripts" / "python.exe").exists() else sys.executable
    cmds = list(checks) if checks is not None else default_checks(repo, python, SMOKE_TESTS)
    existed = flag.exists()
    pre = ""
    merged = False
    try:
        if git(repo, "rev-parse", "--verify", "--quiet", branch, check=False).returncode != 0:
            raise DeployRefused(f"branch {branch!r} does not exist")
        if git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != main:
            raise DeployRefused(f"{repo} is not on {main}")
        if git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
            raise DeployRefused("the working tree has uncommitted changes to tracked files")
        pre = git(repo, "rev-parse", "HEAD").stdout.strip()
        say(f"plan: {'STOP' if hard else 'drain'} -> wait for the kernel (max {wait_s:.0f}s) -> merge {branch} into {main} -> "
            f"{len(cmds)} checks -> {'push -> ' if push else ''}resume -> wait for 'swarm started'")
        if dry_run:
            say(f"dry run: main is at {pre[:10]}; checks: " + "; ".join(" ".join(c) for c in cmds))
            return 0
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("deploy", encoding="utf-8")
        say(f"{flag.name} created")
        t0 = time.monotonic()
        while kernel_busy(state):
            if time.monotonic() - t0 > wait_s:
                raise DeployRefused(f"a cycle was still in flight after {wait_s:.0f}s (kernel.lock is held by a live process)")
            time.sleep(poll_s)
        say(f"no cycle in flight after {time.monotonic() - t0:.0f}s")
        m = git(repo, "merge", "--no-edit", branch, check=False)
        if m.returncode != 0:
            git(repo, "merge", "--abort", check=False)
            raise DeployRefused(f"merge conflict: {(m.stdout + m.stderr).strip()[-400:]}")
        merged = git(repo, "rev-parse", "HEAD").stdout.strip() != pre
        bad = run_checks(repo, cmds, say)
        if bad:
            raise DeployRefused("check failed: " + bad)
        if push:
            p = git(repo, "push", check=False)
            if p.returncode != 0:
                raise DeployRefused(f"push failed: {(p.stderr or p.stdout).strip()[-400:]}")
            say("pushed")
        merged = False                                                  # accepted: never rolled back from here on
        offset = swarm_log.stat().st_size if swarm_log.exists() else 0
        starts = count_lines_with(svc_log, "swarm started")
        if not existed:
            flag.unlink(missing_ok=True)
        say(f"{flag.name} removed; waiting for the swarm to start")
        t1 = time.monotonic()
        while time.monotonic() - t1 < start_wait_s:
            if count_lines_with(svc_log, "swarm started") > starts:
                break
            time.sleep(poll_s)
        else:
            say("WARNING: no 'swarm started' within the wait (is the supervisor running?)")
            return 3
        while time.monotonic() - t1 < start_wait_s + 120:
            line = first_status_after(swarm_log, offset)
            if line:
                say(line)
                return 0
            time.sleep(poll_s)
        say("swarm started; no STATUS line yet (it prints one a minute)")
        return 0
    except DeployRefused as e:
        say(f"REFUSED: {e}")
        if merged and pre:                                              # our own merge only: put main back where it was
            git(repo, "reset", "--hard", pre, check=False)
            say(f"main restored to {pre[:10]}")
        if not existed and not dry_run:
            flag.unlink(missing_ok=True)                                # the run goes on as it was
            say(f"{flag.name} removed (state restored)")
        return 2


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branch")
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--main", default="main")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hard", action="store_true", help="use NUPEN_STOP (immediate) instead of the graceful NUPEN_DRAIN")
    ap.add_argument("--wait-s", type=float, default=4200.0, help="max wait for in-flight cycles (cycles take 35-45 min)")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args(list(argv))
    os.chdir(a.repo)
    return deploy(a.branch, a.repo, a.main, a.dry_run, a.hard, a.wait_s, push=not a.no_push)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
