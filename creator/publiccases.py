"""PUBLIC COMMIT CASES for the local-model judgment drills (owner, 3 Oct 2026: "you can also use public information it can learn off of";
teacher: give the judge far more cases so its strategy search stops choosing on noise).

Nupen's own history gives ~850 git cases; the public open-source repositories cloned under runtime_dir()/public_repos/ (flask, requests, pytest,
... - more arrive during the day) give tens of thousands. This module turns them into judgment cases the local model can READ:

  PUBLIC ONLY. Commit subjects are shown to the model, so text is read only from repositories that sit directly in runtime_dir()/public_repos
  (a symlink or a path elsewhere - e.g. ~/oldpc, the owner's private projects - is refused; an unfinished clone '<name>.tmp' is skipped). The text
  cache keeps hash, commit time, subject (e-mail addresses and @handles removed), paths and line counts: NO author, NO body. It lives OUTSIDE the
  Nupen repository (runtime_dir()/thinking/public_text/<repo>.jsonl), is built once per repository and extended incrementally (`<last>..<ref>`,
  where ref is origin/HEAD when present - another agent only `git fetch`es - else HEAD).

  NO OUTCOME IN THE SETUP. The case text is the repository name, the commit time, its subject, files changed, lines changed and its top folder.
  The outcome (a later fix touching one of its files within FIX_WINDOW commits / any of its files changed again within CHURN_WINDOW commits) is
  computed exactly as creator.drillsources does and is never part of the text.

  SELF-RENEWING. New repositories and new commits are picked up by `refresh_all` (a low-priority job of the judgment filler) and `load_cases` reads
  whatever caches exist; nothing here runs git while cases are loaded."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

from creator import drillsources as D

TOPICS = ("pub_git_fixed", "pub_git_churn")
MODE = {"pub_git_fixed": ("fixed", D.FIX_WINDOW), "pub_git_churn": ("churn", D.CHURN_WINDOW)}
SUBJECT_MAX = 110
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
HANDLE = re.compile(r"(?<![\w/])@[A-Za-z0-9][A-Za-z0-9-]{0,38}")
FORMAT = "--format=\x01%H\x02%ct\x02%s\x03"          # hash, commit time, subject: never %an/%ae/%b


def public_root() -> Path:
    from creator import device as DEV
    return DEV.runtime_dir() / "public_repos"


def is_public(repo: Path) -> bool:
    """A repository counts as public only when it sits DIRECTLY in public_root() (resolved: a symlink to a private tree is not public)."""
    try:
        r = Path(repo).resolve()
        root = public_root().resolve()
    except OSError:
        return False
    return r.parent == root and not r.name.endswith(".tmp") and (r / ".git").exists()


def public_repos() -> list[Path]:
    root = public_root()
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir()) if d.is_dir() and is_public(d)]


def text_cache_path(repo: Path) -> Path:
    from creator import device as DEV
    return DEV.runtime_dir() / "thinking" / "public_text" / f"{Path(repo).name}.jsonl"


def scrub(subject: str) -> str:
    s = HANDLE.sub("@someone", EMAIL.sub("<email>", " ".join(str(subject).split())))
    return s[:SUBJECT_MAX]


def _ref(repo: Path) -> str:
    try:
        D._git(repo, ["rev-parse", "--verify", "-q", "origin/HEAD^{commit}"], 60.0)
        return "origin/HEAD"
    except (RuntimeError, subprocess.SubprocessError, OSError):
        return "HEAD"


def _read(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("h") and "t" in r:
                    out.append(r)
    except OSError:
        pass
    return out


def refresh(repo: Path, full_timeout: float = D.GIT_FULL_TIMEOUT_S, inc_timeout: float = D.GIT_INC_TIMEOUT_S) -> int:
    """Bring one public repository's text cache up to its ref (full read the first time, `<last>..<ref>` later). Returns new commits."""
    repo = Path(repo)
    if not is_public(repo):
        raise PermissionError(f"{repo} is not a public repository under {public_root()}: its commit text is never read")
    path = text_cache_path(repo)
    have = _read(path)
    ref = _ref(repo)
    rng, timeout = ref, full_timeout
    if have:
        try:
            D._git(repo, ["cat-file", "-e", f"{have[-1]['h']}^{{commit}}"], 60.0)
            rng, timeout = f"{have[-1]['h']}..{ref}", inc_timeout
        except RuntimeError:                                       # history rewritten: rebuild
            have = []
    text = D._git(repo, ["log", "--reverse", "--no-merges", "--no-renames", "--numstat", FORMAT, rng], timeout)
    seen = {r["h"] for r in have}
    new = [c for c in D._parse_log(text) if c["h"] not in seen]
    rows = [{"h": c["h"], "t": c["t"], "s": scrub(c["s"]), "files": sorted(c["files"]), "lines": c["lines"]} for c in new]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not have:                                                   # a first build is written whole or not at all
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        tmp.replace(path)
    elif rows:
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(rows)


def refresh_all(repos: Optional[Sequence[Path]] = None) -> dict[str, Any]:
    """Every public repository (new ones get their first full read); a repository that fails is reported and retried next time."""
    out: dict[str, Any] = {}
    for r in (public_repos() if repos is None else repos):
        try:
            out[r.name] = refresh(r)
        except Exception as e:                                     # noqa: BLE001 - one broken clone never stops the others
            out[r.name] = f"{type(e).__name__}: {e}"[:200]
    return out


def signature() -> str:
    """Changes when a public cache grows or a repository appears: the judgment filler reloads its cases then."""
    parts = []
    for r in public_repos():
        p = text_cache_path(r)
        try:
            st = p.stat()
            parts.append(f"{r.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{r.name}:-")
    return "|".join(parts)


def commits(repo: Path) -> list[dict[str, Any]]:
    """The cached public commits of one repository, time ordered ([] for anything not public or not cached yet)."""
    if not is_public(repo):
        return []
    rows = _read(text_cache_path(repo))
    out = [{"h": r["h"], "t": float(r["t"]), "s": str(r.get("s", "")), "files": set(r.get("files") or []), "lines": int(r.get("lines") or 0)}
           for r in rows]
    out.sort(key=lambda c: c["t"])
    return out


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def question(topic: str) -> str:
    mode, w = MODE[topic]
    if mode == "fixed":
        return f"Will a later fix or revert commit of this project touch one of its files within the next {w} commits?"
    return f"Will any of its files be changed again within the next {w} commits of this project?"


def setup_text(repo: str, c: dict[str, Any], topic: str) -> str:
    """What was known when the commit landed: never the outcome, never anything later."""
    top = sorted(c["files"])[0].split("/")[0] if c["files"] else "-"
    return (f"A commit to the public open-source project '{repo}' at {_iso(c['t'])} UTC with message '{c['s'][:SUBJECT_MAX]}'. "
            f"It changed {len(c['files'])} files and {c['lines']} lines; first top folder '{top}'. {question(topic)}")


def raw_cases(topic: str, repos: Optional[Sequence[Path]] = None) -> list[tuple[str, D.BItem, str, str]]:
    """(repo name, item, message class, setup text) for every commit of every public repository; windows never cross repositories.
    Unresolved items (the last `window` commits) are included with resolved None."""
    return _cases(topic, repos, True)


def _cases(topic: str, repos: Optional[Sequence[Path]], text: bool) -> list[tuple[str, D.BItem, str, str]]:
    mode, w = MODE[topic]
    out: list[tuple[str, D.BItem, str, str]] = []
    for repo in (public_repos() if repos is None else repos):
        cs = commits(repo)
        for c, it in zip(cs, D._git_events(cs, w, mode)):
            keys = it.keys[:3] + (f"d:{repo.name}/{it.keys[3][2:]}", f"r:{repo.name}")
            out.append((repo.name, D.BItem(keys, it.created, it.resolved, it.y, f"{repo.name}:{it.subject}"),
                        D._msg_class(c["s"]) if text else "", setup_text(repo.name, c, topic) if text else ""))
    out.sort(key=lambda x: x[1].created)
    return out


def items(topic: str, repos: Optional[Sequence[Path]] = None) -> list[D.BItem]:
    """The same cases as statistical items (for the walk-forward baseline the judge is compared with). h59: built without the setup texts
    and message classes the items never carry (measured 3 Oct on 157,024 public commits: ~10 s of setup_text + ~4 s of _msg_class per call
    of raw_cases' 56 s, and ~50 MB of strings built only to be dropped); the same items in the same order."""
    return [x[1] for x in _cases(topic, repos, False)]
