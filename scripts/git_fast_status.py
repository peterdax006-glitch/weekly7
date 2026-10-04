"""Turn git's file-system monitor and untracked cache on or off for ONE worktree, and measure `git status` before/after.

h43 speed (3 Oct 2026): `git status` in a Nupen worktree took 0.8-1.2 s (48,000 tracked files, a large state/ tree to scan for untracked
files). Measured on a temp clone of the repository (BELOW_NORMAL, 5 runs, median): neither 1.10-1.31 s; untrackedCache alone 0.83 s;
fsmonitor alone 5.7 s (worse: never enable it without the untracked cache); BOTH 0.18 s. Both are set with `git config --worktree` (the
repository already has extensions.worktreeConfig=true): the shared .git/config is not touched.

CAVEAT (measured): `git worktree add` (git >= 2.36) COPIES the current worktree's config.worktree into the new worktree, so every sandbox
the kernel creates from a worktree with these keys inherits them and would start its own fsmonitor daemon. Before turning this on for the
live repository, creator/sandbox.py must run `git -C <new worktree> config --worktree --unset core.fsmonitor` right after each
`worktree add` (or `sandbox_guard(<new worktree>)` below); this script does not change the kernel.

Reversible: `off` unsets both keys and stops that worktree's daemon. Usage:
    python scripts/git_fast_status.py status  <worktree>
    python scripts/git_fast_status.py measure <worktree> [runs]
    python scripts/git_fast_status.py on      <worktree>
    python scripts/git_fast_status.py off     <worktree>
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

KEYS = {"core.fsmonitor": "true", "core.untrackedCache": "true"}


def git(wt: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(wt), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", check=check)


def worktree_config_enabled(wt: Path) -> bool:
    return git(wt, "config", "--bool", "--get", "extensions.worktreeConfig", check=False).stdout.strip() == "true"


def status(wt: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"worktree": str(wt), "worktreeConfig": worktree_config_enabled(wt)}
    for k in KEYS:
        out[k] = git(wt, "config", "--worktree", "--get", k, check=False).stdout.strip() or None if out["worktreeConfig"] else None
    out["daemon"] = git(wt, "fsmonitor--daemon", "status", check=False).stdout.strip()[:120]
    return out


def measure(wt: Path, runs: int = 7) -> dict[str, float]:
    git(wt, "status", "--porcelain")                                   # warm-up (first run builds the caches)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        git(wt, "status", "--porcelain")
        ts.append(time.perf_counter() - t)
    return {"median_s": round(statistics.median(ts), 3), "min_s": round(min(ts), 3), "max_s": round(max(ts), 3)}


def on(wt: Path) -> None:
    if not worktree_config_enabled(wt):
        raise SystemExit("extensions.worktreeConfig is not enabled: refusing to write these keys into the shared config")
    for k, v in KEYS.items():
        git(wt, "config", "--worktree", k, v)
    git(wt, "update-index", "--untracked-cache", check=False)


def off(wt: Path) -> None:
    git(wt, "fsmonitor--daemon", "stop", check=False)
    for k in KEYS:
        git(wt, "config", "--worktree", "--unset", k, check=False)
    git(wt, "update-index", "--no-untracked-cache", check=False)


def sandbox_guard(new_worktree: Path) -> None:
    """For a worktree just created by `git worktree add`: drop the inherited fsmonitor key (no daemon per sandbox)."""
    git(new_worktree, "config", "--worktree", "--unset", "core.fsmonitor", check=False)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("status", "measure", "on", "off"):
        print(__doc__)
        return 2
    wt = Path(argv[1]).resolve()
    if argv[0] == "status":
        print(status(wt))
    elif argv[0] == "measure":
        print(measure(wt, int(argv[2]) if len(argv) > 2 else 7))
    elif argv[0] == "on":
        on(wt)
        print(status(wt))
    else:
        off(wt)
        print(status(wt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
