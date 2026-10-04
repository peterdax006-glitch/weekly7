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
import gc
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
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
    return _setup_head(repo, c) + question(topic)


def _setup_head(repo: str, c: dict[str, Any], top: Optional[str] = None) -> str:
    """setup_text without the question (the same for every topic); `top` when the caller already has it (drillsources' 'd:' key)."""
    if top is None:
        top = min(c["files"]).split("/")[0] if c["files"] else "-"
    return (f"A commit to the public open-source project '{repo}' at {_iso(c['t'])} UTC with message '{c['s'][:SUBJECT_MAX]}'. "
            f"It changed {len(c['files'])} files and {c['lines']} lines; first top folder '{top}'. ")


# Speed h43: the cases of each repository kept per exact (size, mtime_ns) of its text cache, every topic built from ONE parse (3 Oct: the
# first thinking.run spent 98 of 120 s re-reading ~240,000 JSON rows and rebuilding 120,000 cases per topic; a fetch that grows one repository
# now recomputes only that repository). Identical cases; callers get fresh lists (the tuples and items are shared and never modified).
_REPO_CASES: dict[str, tuple[tuple[int, int], dict[str, list[tuple[str, D.BItem, str, str]]]]] = {}
_ALL_CASES: dict[str, tuple[tuple[Any, ...], list[tuple[str, D.BItem, str, str]]]] = {}


def _file_key(path: Path) -> Optional[tuple[int, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _build_repo_cases(repo: Path) -> dict[str, list[tuple[str, D.BItem, str, str]]]:
    cs = commits(repo)
    name = repo.name
    events = {t: D._git_events(cs, MODE[t][1], MODE[t][0]) for t in TOPICS}
    first = events[TOPICS[0]]
    heads = [_setup_head(name, c, it.keys[3][2:]) for c, it in zip(cs, first)]         # top folder = the 'd:' key (no second min())
    mcls = [it.keys[0][2:] for it in first]                                            # message class = the 'm:' key
    tag = f"r:{name}"
    out: dict[str, list[tuple[str, D.BItem, str, str]]] = {}
    for topic in TOPICS:
        q = question(topic)
        rows = []
        for i, it in enumerate(events[topic]):
            keys = tuple(sys.intern(k) for k in it.keys[:3]) + (sys.intern(f"d:{name}/{it.keys[3][2:]}"), tag)   # kept: shared strings
            rows.append((name, D.BItem(keys, it.created, it.resolved, it.y, f"{name}:{it.subject}"), mcls[i], heads[i] + q))
        out[topic] = rows
    return out


def _repo_key(repo: Path) -> Optional[tuple[int, int]]:
    """The memo key of a repository's cases: its text cache's (size, mtime_ns), None (never memoised) while it has none. Public status is
    part of it: commits() of a repository that is not public is empty, cached or not."""
    k = _file_key(text_cache_path(repo))
    return None if k is None else (k[0], k[1] * 2 + int(is_public(repo)))


# Speed h48: a fresh process (every thinking.run, every drill worker) used to re-parse ~370 MB of JSON rows - 38 s - for cases that only change
# when a text cache changes. The built cases of each repository are also kept on disk (pickle, runtime_dir()/thinking/public_cases/<repo>.pkl),
# valid only for the exact (size, mtime_ns, public) of the text cache AND the exact source of the two modules that build them; any mismatch, a
# damaged or foreign file is ignored and the cases are rebuilt (the result is the same objects' values either way).
_CODE_STAMP: Optional[str] = None


def _code_stamp() -> str:
    global _CODE_STAMP
    if _CODE_STAMP is None:
        h = hashlib.sha256()
        for m in (__file__, D.__file__):
            try:
                h.update(Path(m).read_bytes())
            except OSError:
                h.update(b"?")
        _CODE_STAMP = h.hexdigest()
    return _CODE_STAMP


def _disk_path(repo: Path) -> Path:
    from creator import device as DEV
    return DEV.runtime_dir() / "thinking" / "public_cases" / f"{Path(repo).name}.pkl"


def _disk_cases(repo: Path, key: tuple[int, int]) -> Optional[dict[str, list[tuple[str, D.BItem, str, str]]]]:
    was = gc.isenabled()
    gc.disable()                                                    # ~1M small objects built at once: the collector only re-scans them
    try:
        with _disk_path(repo).open("rb") as f:
            stamp, k, built = pickle.load(f)
    except Exception:                                               # missing, truncated, other version: rebuild
        return None
    finally:
        if was:
            gc.enable()
    if stamp != _code_stamp() or k != key or not isinstance(built, dict) or set(built) != set(TOPICS):
        return None
    return built


def _disk_cases_save(repo: Path, key: tuple[int, int], built: dict[str, list[tuple[str, D.BItem, str, str]]]) -> None:
    path = _disk_path(repo)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as f:
            pickle.dump((_code_stamp(), key, built), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass


def _repo_cases(repo: Path, topic: str) -> list[tuple[str, D.BItem, str, str]]:
    path = text_cache_path(repo)
    key = _repo_key(repo)
    mk = str(path)
    hit = _REPO_CASES.get(mk)
    if key is not None and hit is not None and hit[0] == key:
        return hit[1][topic]
    built = _disk_cases(repo, key) if key is not None else None
    if built is None:
        built = _build_repo_cases(repo)
        if key is not None and _repo_key(repo) == key:
            _disk_cases_save(repo, key, built)
    if key is not None and _repo_key(repo) == key:                 # unchanged while read: safe to keep
        _REPO_CASES[mk] = (key, built)
    else:
        _REPO_CASES.pop(mk, None)
    return built[topic]


def raw_cases(topic: str, repos: Optional[Sequence[Path]] = None) -> list[tuple[str, D.BItem, str, str]]:
    """(repo name, item, message class, setup text) for every commit of every public repository; windows never cross repositories.
    Unresolved items (the last `window` commits) are included with resolved None."""
    MODE[topic]                                                    # an unknown topic raises KeyError, as before
    rs = list(public_repos() if repos is None else repos)
    sig = tuple((str(text_cache_path(r)), _repo_key(r)) for r in rs)
    hit = _ALL_CASES.get(topic)
    if hit is not None and hit[0] == sig and None not in (k for _p, k in sig):
        return list(hit[1])
    out: list[tuple[str, D.BItem, str, str]] = []
    for repo in rs:
        out.extend(_repo_cases(repo, topic))
    out.sort(key=lambda x: x[1].created)
    if None not in (k for _p, k in sig) and sig == tuple((str(text_cache_path(r)), _repo_key(r)) for r in rs):
        _ALL_CASES[topic] = (sig, out)
    return list(out)


def items(topic: str, repos: Optional[Sequence[Path]] = None) -> list[D.BItem]:
    """The same cases as statistical items (for the walk-forward baseline the judge is compared with). h59: built without the setup texts
    and message classes the items never carry (measured 3 Oct on 157,024 public commits: ~10 s of setup_text + ~4 s of _msg_class per call
    of raw_cases' 56 s, and ~50 MB of strings built only to be dropped); the same items in the same order."""
    return [x[1] for x in raw_cases(topic, repos)]
