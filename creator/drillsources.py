"""MORE THINKING DATA and PARALLEL DRILL JOBS (owner, 2 Oct 2026: "Nupen should use all the data it has access to just to have as many agents as
the computer can handle just to make Nupen a better thinker").

Every local source with ground truth becomes a binary, strictly time-ordered drill (an event is predicted at its creation time from the items
RESOLVED before it; the resolution time of a windowed event is the end of its window, never earlier). Sources and events:
  git_fixed      a commit of the Weekly7 repo: will a later fix/revert commit touch one of its files within FIX_WINDOW commits?
  git_churn      will any file of this commit be changed again within CHURN_WINDOW commits?
  journal_persist  owner's JOURNAL.md: will the next entry stay on the same topic as the last one (what the owner prioritises next)?
  plan_choice    plan_explanations.jsonl: will the scheduler choose this node (predicted from step / component / value)?
  research_bool  state/research/**.json top-level yes/no fields: predicted from the experiment family and the file's name pattern (setup)
HARD EXCLUSIONS: state/livesim never read; any path that looks secret (.env, credential, secret, token, key, password, .pem) or like an answer
key (key, answers, sealed) is skipped by name before it is opened; nothing leaves the machine (local subprocess `git log` only; the one
network use is creator.publicdata fetching/cloning PUBLIC upstreams, unauthenticated, and live_pass_x predicts their open commits).

PARALLELISM: one drill batch = one JOB (source x model variant). `drill_filler(...)` returns a next_job() callable exactly like
creator.swarm.self_bench_filler: the swarm / Governor admits as many jobs as CPU and RAM allow, and each job loads its WHOLE dataset in memory once
(RAM-heavy by design) and is independent of every other. Results append to state/creator/thinking/drill_runs.jsonl; a (source, variant, data
digest) triple already done is not repeated, so new data (new commits, new journal entries) re-queues the jobs. Model selection is honest:
`best_variant` picks on the first SELECT_SPLIT of the time-ordered predictions and reports the score of the LAST part only."""
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from creator import thinking as T

FIX_WINDOW = 20
CHURN_WINDOW = 10
SELECT_SPLIT = 0.6
DECAYS = (0.9, 0.97, 1.0)
SHRINKS = (2.0, 5.0, 10.0)
VARIANTS = [{"decay": d, "k": k} for d, k in itertools.product(DECAYS, SHRINKS)]
SECRET = re.compile(r"(\.env|credential|secret|token|password|passwd|\.pem|\.key|api[_-]?key|answers?|sealed|livesim)", re.I)
FIX_WORDS = re.compile(r"\b(fix|fixes|fixed|revert|bug|regress\w*|hotfix|repair)\b", re.I)


def excluded(path: str) -> bool:
    """Never read: secret-looking names, sealed/answer material, and state/livesim."""
    return bool(SECRET.search(path.replace("\\", "/")))


@dataclass
class BItem:
    keys: tuple[str, ...]           # setup features known at creation (never the outcome)
    created: float
    resolved: Optional[float]       # when the outcome became known (None = not yet)
    y: int = 0
    subject: str = ""
    meta: Any = None                # setup-only facts a feature FAMILY needs (git: the commit's files, sizes, message type; never the outcome)


# ------------------------------------------------------------------------------------------------ the generic time-ordered binary model
def walk_forward(items: Sequence[BItem], topic: str, decay: float = 0.97, k: float = 3.0, agg: str = "mean", cap: float = 0.02) -> list[T.Pred]:
    """The compiled loop (creator.fastwalk, numba; same predictions as walk_forward_reference, tests/test_fastwalk.py) when available,
    else the reference."""
    try:
        from creator import fastwalk as FW
        if FW.enabled():
            return list(FW.walk_forward(items, topic, decay, k, agg, cap))
    except Exception:                                                  # noqa: BLE001 - never lose a drill to the accelerator
        pass
    return walk_forward_reference(items, topic, decay, k, agg, cap)


def walk_forward_reference(items: Sequence[BItem], topic: str, decay: float = 0.97, k: float = 3.0, agg: str = "mean", cap: float = 0.02) -> list[T.Pred]:
    """Predict each resolved item at its creation time using only items resolved strictly before; learn from every resolution as it happens.
    Per feature key a decayed (n, positives); p = mean over the item's keys of (pos_k + k*global)/(n_k + k). Baselines: running base rate, last value."""
    ev: list[tuple[float, int, int]] = []                    # (time, 0=resolve first / 1=create, index)
    for i, it in enumerate(items):
        if it.resolved is not None:
            ev.append((it.created, 1, i))
            ev.append((it.resolved, 0, i))
    ev.sort()
    scale = 1.0
    cnt: dict[str, list[float]] = {}                          # key -> [n, pos] in units of 1/scale
    gn = gp = 0.0
    hist: list[int] = []
    out: list[T.Pred] = []
    for t, kind, i in ev:
        it = items[i]
        if kind == 0:                                         # resolution: the model learns
            scale *= decay
            w = 1.0 / scale
            for key in it.keys:
                c = cnt.setdefault(key, [0.0, 0.0])
                c[0] += w
                c[1] += w * it.y
            gn += w
            gp += w * it.y
            hist.append(it.y)
            if scale < 1e-100:                            # renormalise (decay^n underflows on long histories)
                for c in cnt.values():
                    c[0] *= scale
                    c[1] *= scale
                gn *= scale
                gp *= scale
                scale = 1.0
        else:                                                 # creation: predict from what is resolved so far
            glob = (gp * scale + 1.0) / (gn * scale + 2.0)
            ps = []
            for key in it.keys:
                cc = cnt.get(key)
                if cc:
                    c = cc
                    ps.append((c[1] * scale + k * glob) / (c[0] * scale + k))
            if ps and agg == "logit":
                p = 1.0 / (1.0 + math.exp(-sum(math.log(max(q, 1e-6) / max(1.0 - q, 1e-6)) for q in ps) / len(ps)))
            else:
                p = sum(ps) / len(ps) if ps else glob
            p = min(1.0 - cap, max(cap, p))
            out.append(T.Pred(topic, it.subject, t, p, T._laplace(hist), T._last(hist), it.y))
    return out


def split_score(preds: Sequence[T.Pred]) -> dict[str, Any]:
    cut = int(len(preds) * SELECT_SPLIT)
    return {"select": T.score(preds[:cut]), "heldout": T.score(preds[cut:]), "all": T.score(preds)}


# ------------------------------------------------------------------------------------------------ sources
def _log2b(n: int) -> str:
    return str(int(math.log2(n + 1)))


def _msg_class(subj: str) -> str:
    m = re.match(r"\s*([A-Za-z][A-Za-z0-9_-]{0,15})[:(\s]", subj + " ")
    return m.group(1).lower() if m else "other"


_GIT_CACHE: dict[str, list[dict[str, Any]]] = {}
GIT_FULL_TIMEOUT_S = 6 * 3600.0     # the first full read of a big repo on a loaded machine is slow: generous, and done once, in its own job
GIT_INC_TIMEOUT_S = 300.0           # an incremental `<last>..HEAD` read
GIT_RETRY_S = 900.0                 # a source whose data could not be read is retried after this long, never dropped for good
IDLE_PRIORITY = 0x00004000          # BELOW_NORMAL_PRIORITY_CLASS: true IDLE starves for hours on an always-busy machine (a full git log never finished)
DEFAULT_STATE: Optional[Path] = None   # set by load(): lets callers without a state argument (judgment) reach the same persistent cache


class GitCacheMissing(RuntimeError):
    """No parsed history yet and this caller may not do the slow full read (the dedicated git_cache job does it)."""


def _git(repo: Path, args: list[str], timeout: float) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=IDLE_PRIORITY if os.name == "nt" else 0)
    if p.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {p.stderr.strip()[:200]}")
    return p.stdout


def _git_head(repo: Path, timeout: float = 60.0) -> str:
    return _git(repo, ["rev-parse", "HEAD"], timeout).strip() or "-"


def cache_path(state: Path) -> Path:
    return Path(state) / "thinking" / "git_history.jsonl"


def _read_cache(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in T._jsonl(path):
        if r.get("h") and "t" in r:
            out.append({"h": r["h"], "t": float(r["t"]), "s": r.get("s", ""), "files": set(r.get("files") or []), "lines": int(r.get("lines") or 0),
                        "add": int(r.get("add") or 0), "del": int(r.get("del") or 0), "au": r.get("au", ""), "body": r.get("body", "")})
    return out


def _append_cache(path: Path, commits: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for c in commits:
            f.write(json.dumps({**c, "files": sorted(c["files"])}, sort_keys=True) + "\n")


def refresh_git_cache(repo: Path, state: Path, full_timeout: float = GIT_FULL_TIMEOUT_S, inc_timeout: float = GIT_INC_TIMEOUT_S) -> int:
    """Bring state/creator/thinking/git_history.jsonl up to HEAD: the first call parses the whole history once, later calls read only
    `<last cached>..HEAD`. Appends in log order (a half-finished read writes nothing). A timeout raises and leaves the cache as it was."""
    path = cache_path(state)
    have = _read_cache(path)
    rng = ""
    timeout = full_timeout
    if have:
        last = have[-1]["h"]
        try:
            _git(repo, ["cat-file", "-e", f"{last}^{{commit}}"], 60.0)
            rng, timeout = f"{last}..HEAD", inc_timeout
        except RuntimeError:                                       # history was rewritten: rebuild from scratch
            have = []
            path.unlink(missing_ok=True)
    args = ["log", "--reverse", "--no-merges", "--no-renames", "--numstat", "--format=\x01%H\x02%ct\x02%s\x02%an\x02%b\x03"]
    new = _parse_log(_git(repo, args + ([rng] if rng else []), timeout))
    seen = {c["h"] for c in have}
    new = [c for c in new if c["h"] not in seen]
    if new:
        _append_cache(path, new)
    return len(new)


def _parse_log(text: str) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    for block in text.split("\x01")[1:]:
        head, _, rest = block.partition("\x03")
        parts = head.split("\x02")
        if len(parts) < 3:
            continue
        files, add, dele = [], 0, 0
        for ln in rest.splitlines():
            f = ln.split("	")
            if len(f) == 3 and not excluded(f[2]):
                files.append(f[2])
                add += int(f[0]) if f[0].isdigit() else 0
                dele += int(f[1]) if f[1].isdigit() else 0
        commits.append({"h": parts[0], "t": float(parts[1]), "s": parts[2], "au": parts[3] if len(parts) > 3 else "",
                        "body": (parts[4] if len(parts) > 4 else "")[-300:], "files": set(files), "lines": add + dele, "add": add, "del": dele})
    return commits


def _git_commits(repo: Path) -> list[dict[str, Any]]:
    commits = _parse_log(_git(repo, ["log", "--reverse", "--no-merges", "--no-renames", "--numstat", "--format=\x01%H\x02%ct\x02%s\x02%an\x02%b\x03"], GIT_FULL_TIMEOUT_S))
    commits.sort(key=lambda c: c["t"])
    return commits


def git_commits(repo: Path, state: Optional[Path] = None, allow_full: bool = False) -> list[dict[str, Any]]:
    """Parsed history, time ordered. With a state dir it comes from the persistent cache (refreshed incrementally; if that refresh times out the
    cached history is used as is). The full first read happens only when allow_full (the dedicated git_cache job); otherwise GitCacheMissing."""
    state = state if state is not None else DEFAULT_STATE
    if state is None:
        head = _git_head(repo)
        ck = f"{repo}|{head}"
        if ck not in _GIT_CACHE:
            _GIT_CACHE.clear()
            _GIT_CACHE[ck] = _git_commits(repo)
        return _GIT_CACHE[ck]
    path = cache_path(state)
    if not path.exists() and not allow_full:
        raise GitCacheMissing(f"{path} not built yet; the git_cache job builds it")
    try:
        refresh_git_cache(Path(repo), state, GIT_FULL_TIMEOUT_S if allow_full else GIT_INC_TIMEOUT_S)
    except (subprocess.SubprocessError, RuntimeError, OSError):
        if not path.exists():
            raise
    commits = _read_cache(path)
    commits.sort(key=lambda c: c["t"])
    _GIT_CACHE.clear()
    _GIT_CACHE[f"{repo}|{commits[-1]['h'] if commits else '-'}"] = commits
    return commits


def git_items(repo: Path, window: int, mode: str, state: Optional[Path] = None) -> list[BItem]:
    """mode 'fixed' | 'churn'. Read-only `git log`; parsed once into the persistent cache and kept in memory."""
    return _git_events(git_commits(repo, state), window, mode)


# ------------------------------------------------------------------------------------------------ other projects' histories (replay only)
def extra_repos() -> list[Path]:
    """Other projects on this machine (owner 2 Oct 2026: 'use all the data it has access to'; 3 Oct: 'souly only focus on Nupens thinking
    capacity'): archived git histories that train the same git judgments on ~2,850 more commits than Nupen's own. They never change, so they are
    replayed only - never live predictions, never trust on their own."""
    base = Path.home() / "oldpc"
    own = [base / "Highlighting-Utah", base / "spanish-app", base / "Desktop" / "tico_project"]
    from creator import device as DEV
    pub = DEV.runtime_dir() / "public_repos"                          # owner 3 Oct: "you can also use public information it can learn off of"
    public = sorted(d for d in pub.iterdir() if d.is_dir()) if pub.is_dir() else []
    return [r for r in own + public if (r / ".git").exists() and not r.name.endswith(".tmp") and not r.name.startswith(".")]   # .tmp = clone in progress


def cached_extra_repos() -> list[Path]:
    """The projects whose history is already cached: drills use these now; a missing one is built in the background without blocking the rest
    (3 Oct: one huge first read - cpython - and clones still in progress held back every other project's drills)."""
    return [r for r in extra_repos() if extra_cache_path(r).exists()]


def public_dir() -> Path:
    from creator import device as DEV
    return DEV.runtime_dir() / "public_repos"


def is_public(repo: Path) -> bool:
    """A clone of an ACTIVE public upstream (fetched, predicted live). Archived projects (~/oldpc) are never fetched."""
    try:
        return Path(repo).resolve().parent == public_dir().resolve()
    except OSError:
        return False


def upstream_ref(repo: Path) -> str:
    """What a public clone's history is read up to: the fetched upstream branch (origin/HEAD), never the stale checkout."""
    if is_public(repo):
        try:
            return _git(repo, ["rev-parse", "--verify", "-q", "refs/remotes/origin/HEAD"], 60.0).strip() or "HEAD"
        except (RuntimeError, OSError, subprocess.SubprocessError):
            pass
    return "HEAD"


def _shallow_roots(repo: Path) -> set[str]:
    """A shallow clone's boundary commits look parentless: their numstat is the WHOLE tree as 'added' - never a real commit, dropped."""
    p = Path(repo) / ".git" / "shallow"
    try:
        return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except OSError:
        return set()


def extra_cache_path(repo: Path) -> Path:
    """OUTSIDE the (public) repository: other projects' commit messages never sit where a commit could publish them."""
    from creator import device as DEV
    return DEV.runtime_dir() / "thinking" / f"git_history_x_{repo.name}.jsonl"


def _h8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def stripped(c: dict[str, Any]) -> dict[str, Any]:
    """PRIVACY-STRIPPED commit (owner, 3 Oct 2026: other projects' histories only without personal data): no author, no message text, no
    body, no readable path. Kept: time, line counts, the message's first word (its type) plus 'fix' when a fix word occurred, and each path as
    '<hash of top folder>/<hash of path>' - the features the git drills use (type, fix, top folder, file overlap) come out identical."""
    s = str(c.get("s", ""))
    kind = _msg_class(s)
    return {"h": c["h"], "t": c["t"], "s": f"{kind} fix" if FIX_WORDS.search(s) else f"{kind} -",
            "files": sorted(f"{_h8(f.split('/')[0])}/{_h8(f)}" for f in c["files"]), "lines": c["lines"], "add": c["add"], "del": c["del"]}


_LOG_ARGS = ["log", "--reverse", "--no-merges", "--no-renames", "--numstat", "--format=\x01%H\x02%ct\x02%s\x02%an\x02%b\x03"]


def _tip_path(repo: Path) -> Path:
    return extra_cache_path(repo).with_suffix(".tip")


_XCACHE_LOCK = threading.RLock()            # a fetch job and the cache-building job never write one cache at once


def build_extra_cache(repo: Path) -> int:
    """The one full read of an archived history (BELOW_NORMAL priority, like every git read here), stored privacy-stripped; written whole or not at all.
    A public clone is read up to its fetched upstream tip (recorded beside the cache, so the next read is incremental)."""
    with _XCACHE_LOCK:
        return _build_extra_cache(repo)


def _build_extra_cache(repo: Path) -> int:
    ref = upstream_ref(repo)
    tip = _git(repo, ["rev-parse", ref], 60.0).strip()
    roots = _shallow_roots(repo)
    commits = [c for c in _parse_log(_git(repo, _LOG_ARGS + [tip], GIT_FULL_TIMEOUT_S)) if c["h"] not in roots]
    commits.sort(key=lambda c: c["t"])
    path = extra_cache_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in commits:
            f.write(json.dumps(stripped(c)) + "\n")
    tmp.replace(path)
    _tip_path(repo).write_text(tip, encoding="utf-8")
    return len(commits)


def refresh_extra_cache(repo: Path, timeout: float = GIT_INC_TIMEOUT_S) -> int:
    """Append a public clone's commits since the last read (`<tip>..origin/HEAD`, privacy-stripped) to its cache - incremental, like
    refresh_git_cache for Nupen's own repo. No cache yet, or the old tip vanished (upstream rewrote history): one full rebuild."""
    with _XCACHE_LOCK:
        return _refresh_extra_cache(repo, timeout)


def _refresh_extra_cache(repo: Path, timeout: float) -> int:
    path = extra_cache_path(repo)
    try:
        old = _tip_path(repo).read_text(encoding="utf-8").strip()
    except OSError:
        old = ""
    if not path.exists() or not old:
        return build_extra_cache(repo)
    tip = _git(repo, ["rev-parse", upstream_ref(repo)], 60.0).strip()
    if tip == old:
        return 0
    try:
        _git(repo, ["cat-file", "-e", f"{old}^{{commit}}"], 60.0)
    except RuntimeError:
        return build_extra_cache(repo)
    seen = {c["h"] for c in _read_cache(path)} | _shallow_roots(repo)
    new = [c for c in _parse_log(_git(repo, _LOG_ARGS + [f"{old}..{tip}"], timeout)) if c["h"] not in seen]
    if new:
        with path.open("a", encoding="utf-8") as f:
            for c in new:
                f.write(json.dumps(stripped(c)) + "\n")
    _tip_path(repo).write_text(tip, encoding="utf-8")
    return len(new)


def extra_git_items(window: int, mode: str, repos: Optional[Sequence[Path]] = None) -> list[BItem]:
    """Every archived project's commits as git items (windows never cross projects), tagged 'r:<project>', in creation order."""
    out: list[BItem] = []
    if repos is None:
        repos = cached_extra_repos()
        if not repos:
            raise GitCacheMissing("no other project's history is cached yet; the extra-history job builds them")
    for repo in repos:
        path = extra_cache_path(repo)
        if not path.exists():
            raise GitCacheMissing(f"{path} not built yet; the extra-history job builds it")
        commits = _read_cache(path)
        commits.sort(key=lambda c: c["t"])
        for it in _git_events(commits, window, mode):
            out.append(BItem(it.keys + (f"r:{repo.name}",), it.created, it.resolved, it.y, f"{repo.name[:8]}:{it.subject}",
                             {**it.meta, "repo": repo.name} if it.meta else None))
    out.sort(key=lambda it: it.created)
    return out


def _git_events(commits: list[dict[str, Any]], window: int, mode: str) -> list[BItem]:
    out: list[BItem] = []
    for i, c in enumerate(commits):
        top = sorted(c["files"])[0].split("/")[0] if c["files"] else "-"
        keys = (f"m:{_msg_class(c['s'])}", f"n:{_log2b(len(c['files']))}", f"l:{_log2b(c['lines'])}", f"d:{top}")
        meta = {"t": c["t"], "files": c["files"], "s": f"{_msg_class(c['s'])} {'fix' if FIX_WORDS.search(c['s']) else '-'}",
                "add": c.get("add", 0), "del": c.get("del", 0)}
        if i + window >= len(commits):
            out.append(BItem(keys, c["t"], None, 0, c["h"][:10], meta))
            continue
        nxt = commits[i + 1:i + 1 + window]
        if mode == "fixed":
            y = int(any(FIX_WORDS.search(n["s"]) and n["files"] & c["files"] for n in nxt))
        else:
            y = int(any(n["files"] & c["files"] for n in nxt))
        out.append(BItem(keys, c["t"], max(c["t"], nxt[-1]["t"]), y, c["h"][:10], meta))
    return out


_TOPICS = (("creator", r"nupen|creator|kernel|swarm|ledger|goal|constraint"), ("spanish", r"sendero|spanish|tico|godot|app store"),
           ("weekly7", r"weekly7|stock|backtest|trader|pattern|blind|memory bank|alpaca|portfolio|vol"))


def topic_of(text: str) -> str:
    low = text.lower()
    best = max(_TOPICS, key=lambda tp: len(re.findall(tp[1], low)))
    return best[0] if re.search(best[1], low) else "other"


def journal_items(journal: Path) -> list[BItem]:
    """Top-level journal entries '- YYYY-MM-DD...' in order; event: the NEXT entry has the same topic as this one. Resolved when the next entry exists."""
    if not journal.is_file() or excluded(str(journal.name)):
        return []
    entries: list[tuple[float, str]] = []
    for ln in journal.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^- (\d{4})-(\d\d)-(\d\d)\b(.*)", ln)
        if m:
            entries.append((T._ts(f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00") + len(entries), m.group(4)))
        elif entries and ln.startswith("  "):
            entries[-1] = (entries[-1][0], entries[-1][1] + " " + ln)
    tops = [topic_of(e[1]) for e in entries]
    out: list[BItem] = []
    for i, (t, _txt) in enumerate(entries):
        prev = tops[i - 1] if i else "none"
        keys = (f"cur:{tops[i]}", f"prev:{prev}", f"pair:{prev}>{tops[i]}")
        nxt = entries[i + 1][0] if i + 1 < len(entries) else None
        out.append(BItem(keys, t, nxt, int(i + 1 < len(tops) and tops[i + 1] == tops[i]), f"j{i}"))
    return out


def plan_items(state: Path) -> list[BItem]:
    out: list[BItem] = []
    for i, r in enumerate(T._jsonl(state / "plan_explanations.jsonl")):
        if "chosen" not in r or "at" not in r:
            continue
        t = T._local_ts(r["at"].replace("Z", "")) if not r["at"].endswith("Z") else T._ts(r["at"])
        v = r.get("value")
        keys = (f"step:{r.get('step')}", f"comp:{r.get('component')}", f"val:{round(float(v), 1) if isinstance(v, (int, float)) else '?'}")
        out.append(BItem(keys, t + i * 1e-3, t + i * 1e-3, int(bool(r["chosen"])), str(r.get("node", i))))
    return out


def research_items(root: Path, max_files: int = 60000) -> list[BItem]:
    out: list[BItem] = []
    n = 0
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if not excluded(d)]
        for f in fs:
            if not f.endswith(".json") or excluded(f) or n >= max_files:
                continue
            p = os.path.join(dp, f)
            n += 1
            try:
                if os.path.getsize(p) > 200_000:
                    continue
                d = json.loads(Path(p).read_text(encoding="utf-8"))
                mt = os.path.getmtime(p)
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict):
                continue
            rel = os.path.relpath(p, root).replace("\\", "/").split("/")
            fam = "/".join(rel[:2]) if len(rel) > 2 else rel[0]
            stem = re.sub(r"\d+", "#", Path(f).stem)
            for k, v in d.items():
                if isinstance(v, bool):
                    out.append(BItem((f"{k}|fam:{fam}", f"{k}|stem:{stem}", f"{k}"), mt, mt, int(v), f"{rel[-1]}:{k}"))
    out.sort(key=lambda b: b.created)
    return out


SOURCES: dict[str, str] = {"git_fixed": "git", "git_churn": "git", "journal_persist": "journal", "plan_choice": "plan", "research_bool": "research",
                           "x_git_fixed": "gitx", "x_git_churn": "gitx"}   # gitx: other projects' histories (archived + active public clones)
XDIGEST_BYTES = 1_000_000


def load(source: str, state: Path, repo: Path, journal: Path, research: Path) -> list[BItem]:
    global DEFAULT_STATE
    DEFAULT_STATE = Path(state)
    if source == "git_fixed":
        return git_items(repo, FIX_WINDOW, "fixed", state)
    if source == "git_churn":
        return git_items(repo, CHURN_WINDOW, "churn", state)
    if source == "x_git_fixed":
        return extra_git_items(FIX_WINDOW, "fixed")
    if source == "x_git_churn":
        return extra_git_items(CHURN_WINDOW, "churn")
    if source == "journal_persist":
        return journal_items(journal)
    if source == "plan_choice":
        return plan_items(state)
    if source == "research_bool":
        return research_items(research)
    raise KeyError(source)


def source_digest(source: str, state: Path, repo: Path, journal: Path, research: Path) -> str:
    """Cheap fingerprint of a source's data (no full load): new commits / journal lines / plan rows / research files change it and re-queue jobs."""
    try:
        if SOURCES[source] == "git":
            return _git_head(repo)[:12]
        if SOURCES[source] == "gitx":                       # the cached projects' sizes (a newly cached project = new data); "-" while none is
            xs = cached_extra_repos()
            if not xs:
                return "-"
            # public clones grow with every fetch: a digest per byte would re-queue the whole x grid every few minutes and leave the live
            # predictions' best variant chosen from one row. Steps of XDIGEST_BYTES (~5,000 stripped commits) or a new project re-queue it.
            return f"x{len(xs)}.{sum(extra_cache_path(r).stat().st_size for r in xs) // XDIGEST_BYTES}"
        if SOURCES[source] == "journal":                    # the ITEMS (owner's audit, 3 Oct: the byte size restarted the search with
            return digest(journal_items(journal))           # unchanged items whenever the journal was edited)
        if SOURCES[source] == "plan":
            return str(sum(1 for _ in (state / "plan_explanations.jsonl").open("rb")))
        return str(sum(len(fs) for _d, _ds, fs in os.walk(research)))
    except (OSError, subprocess.SubprocessError, RuntimeError):
        if SOURCES[source] == "git":                         # head unreadable right now (machine loaded): fall back to what the cache holds
            c = _read_cache(cache_path(state))
            return c[-1]["h"][:12] if c else "-"
        return "-"


def digest(items: Sequence[BItem]) -> str:
    return hashlib.sha256(f"{len(items)}|{sum(1 for i in items if i.resolved is not None)}|{items[-1].created if items else 0}".encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ parallel jobs
def runs_path(state: Path) -> Path:
    return state / "thinking" / "drill_runs.jsonl"


def compute_row(source: str, variant: dict[str, Any], state: Path, repo: Path, journal: Path, research: Path) -> dict[str, Any]:
    """One drill batch's result (no write): load the whole dataset, walk forward with this variant, score. Module-level so a worker process runs it."""
    items = apply_variant(load(source, state, repo, journal, research), variant, journal)
    preds = walk_forward(items, source, float(variant["decay"]), float(variant["k"]), str(variant.get("agg", "mean")), float(variant.get("cap", 0.02)))
    row = {"source": source, "variant": variant, "search": bool(variant.get("search")), "digest": source_digest(source, state, repo, journal, research), "items": len(items), "resolved": len(preds), **split_score(preds)}
    row["at"] = T.dt.datetime.now(T.dt.timezone.utc).isoformat(timespec="seconds")
    return row


# 3 Oct 2026: drill jobs ran as THREADS of the swarm process - pure-Python walk-forwards share one interpreter lock, so however many fillers
# the Governor admitted they used about one core (CPU 57%, 19 GB free). With processes=True the arithmetic runs in a pool of worker processes
# (one per core, below-normal priority: kernel tests and the owner come first); the filler thread only waits, and the parent still does every write.
_POOL: Any = None
_POOL_LOCK = threading.Lock()


def _low_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32                                      # type: ignore[attr-defined]
            k.SetPriorityClass(k.GetCurrentProcess(), IDLE_PRIORITY)
        else:
            getattr(os, "nice")(10)                                       # POSIX only
    except Exception:                                                       # noqa: BLE001 - a worker at normal priority still works
        pass


def _pool() -> Any:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            from concurrent.futures import ProcessPoolExecutor
            _POOL = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 2), initializer=_low_priority)
        return _POOL


def run_job(source: str, variant: dict[str, float], state: Path, repo: Path, journal: Path, research: Path,
            lock: Optional[threading.Lock] = None, processes: bool = False, why: str = "") -> dict[str, Any]:
    """One drill batch: compute (in a worker process when `processes`), then append the row. Independent of every other job."""
    global _POOL
    if processes:
        from concurrent.futures.process import BrokenProcessPool
        try:
            row = _pool().submit(compute_row, source, dict(variant), Path(state), Path(repo), Path(journal), Path(research)).result()
        except BrokenProcessPool:                                           # a worker died: a fresh pool next time, this job in-thread
            with _POOL_LOCK:
                _POOL = None
            row = compute_row(source, dict(variant), state, repo, journal, research)
    else:
        row = compute_row(source, dict(variant), state, repo, journal, research)
    if why:
        row["why"] = why
    p = runs_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with (lock or threading.Lock()), p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _error_row(state: Path, s: str, v: dict[str, Any], e: Exception, lock: threading.Lock) -> None:
    with lock:
        p = runs_path(state)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"source": s, "variant": v, "digest": "-", "error": f"{type(e).__name__}: {e}"}) + "\n")


def _fast_mod() -> Any:
    try:
        from creator import registry as REG
        return REG.optional("fastpred")
    except Exception:                                                  # noqa: BLE001
        return None


def _pack_bytes(repo: Path) -> int:
    """Rough size of a clone's history (its pack files): which missing cache is cheapest to build."""
    try:
        return sum(f.stat().st_size for f in (Path(repo) / ".git" / "objects" / "pack").glob("*.pack"))
    except OSError:
        return 0


def drill_filler(state: Path, repo: Path, journal: Path, research: Path, sources: Optional[Iterable[str]] = None, seed: Optional[int] = None,
                 processes: bool = False, public: bool = False, react: bool = True) -> Callable[[], Optional[Callable[[], None]]]:
    """Like creator.swarm.self_bench_filler: returns next_job(); each call hands out one independent drill job. First the fixed VARIANTS grid per
    source, then (OPEN-ENDED, owner 2 Oct: the filler must never run dry while there is thinking left to improve) NEW variants proposed by
    `propose` - round-robin over the sources, at most SEARCH_OUTSTANDING in flight per source - until SEARCH_STOP_K consecutive variants fail to
    beat that source's best on the select part; a changed data digest (new commits, journal lines, plan rows, research files) reopens it.
    Returns None only when every source is stopped or done for the present data. The swarm's Governor sets how many run at once.
    With `public` (the swarm; never tests) and an x source: also the network jobs of creator.publicdata - a `git fetch` of one active public
    clone at a time (each at most every FETCH_EVERY_S) and, when the drills run dry or once an hour, the acquisition of one more public repo."""
    state = Path(state)
    srcs = list(sources) if sources else list(SOURCES)
    done = {(r.get("source"), json.dumps(r.get("variant"), sort_keys=True), r.get("digest")) for r in T._jsonl(runs_path(state))}
    lock = threading.Lock()
    queue: list[tuple[str, dict[str, Any]]] = [(s, v) for s in srcs for v in VARIANTS]
    digests: dict[str, str] = {}
    flight: dict[str, int] = dict.fromkeys(srcs, 0)
    rnd = random.Random(seed if seed is not None else int.from_bytes(os.urandom(4), "big"))
    rr = itertools.count()

    def digest_of(s: str) -> str:
        if s not in digests:
            digests[s] = source_digest(s, state, Path(repo), Path(journal), Path(research))
        return digests[s]

    def wrap(s: str, v: dict[str, Any], search: bool, why: str = "") -> Callable[[], None]:
        def job() -> None:
            fp, pid, prior = _fast_mod(), None, None
            try:
                if fp is not None and fp.enabled():                  # FAST-PREDICTION HOOK: P(this variant beats its source's best), before it runs
                    prior = fp.drill_prior_best(state, s, digest_of(s))
                    pid = fp.begin_safe("drill_beats_best", f"{s}:{json.dumps(v, sort_keys=True)}:{digest_of(s)}",
                                        f"{s}|{'search' if search else 'grid'}", state=state) if prior is not None else None
            except Exception:                                       # noqa: BLE001
                pid = None
            try:
                row = run_job(s, v, state, Path(repo), Path(journal), Path(research), lock, processes, why)
                if pid and fp is not None and row.get("select", {}).get("n"):
                    fp.end_safe(pid, int(row["select"]["brier"] < prior), state)
                _react(row)
            except Exception as e:                                 # noqa: BLE001
                _error_row(state, s, v, e, lock)
            finally:
                if search:
                    with lock:
                        flight[s] -= 1
        return job

    deferred: list[tuple[str, dict[str, Any]]] = []             # jobs whose data could not be read now: retried later, never dropped
    state_t = {"retry_at": 0.0, "cache_flight": 0.0, "xretry_at": 0.0, "xcache_flight": 0.0}
    xgits = [s for s in srcs if SOURCES[s] == "gitx"]

    xfail: dict[str, float] = {}

    def xcache_job() -> Callable[[], None]:
        def job() -> None:
            now = time.time()
            missing = sorted((r for r in extra_repos() if not extra_cache_path(r).exists() and now - xfail.get(r.name, 0.0) >= GIT_RETRY_S),
                             key=_pack_bytes)                # a project that just failed waits; the others go on
            if not missing:
                state_t["xcache_flight"] = 0.0
                return
            r = missing[0]                                         # ONE project per job, the smallest first: cpython's first read never
            try:                                                   # delays the small ones
                build_extra_cache(r)
                digests.clear()
                state_t["xretry_at"] = 0.0                         # the next missing one may start at once
            except Exception as e:                                 # noqa: BLE001 - retried after GIT_RETRY_S
                xfail[r.name] = time.time()
                state_t["xretry_at"] = 0.0                         # the others need not wait for it
                _error_row(state, "git_cache_x", {"repo": r.name}, e, lock)
            finally:
                state_t["xcache_flight"] = 0.0
        return job
    gits = [s for s in srcs if SOURCES[s] == "git"]

    def cache_job() -> Callable[[], None]:
        def job() -> None:
            try:
                refresh_git_cache(Path(repo), state, GIT_FULL_TIMEOUT_S if not cache_path(state).exists() else GIT_INC_TIMEOUT_S)
                digests.clear()
            except Exception as e:                                 # noqa: BLE001 - a timeout leaves the cache as it was; the next retry continues
                _error_row(state, "git_cache", {}, e, lock)
            finally:
                state_t["cache_flight"] = 0.0
        return job

    def _react(row: dict[str, Any]) -> None:
        """FAST REACTION (creator.trialerror.react): a finished result can re-diagnose its source at once and queue targeted variants."""
        if not react:
            return
        try:
            from creator import registry as REG
            te = REG.optional("trialerror")
            if te is not None:
                te.react(state, row, Path(repo), Path(journal), Path(research))
        except Exception as e:                                     # noqa: BLE001 - a reaction never loses the drill result
            _error_row(state, "react", {"source": row.get("source")}, e, lock)

    def learn_queue() -> list[tuple[str, dict[str, Any]]]:
        try:
            from creator import registry as REG
            ll = REG.optional("learnloop")
            return list(ll.queued_variants(state)) if ll is not None else []
        except Exception:                                          # noqa: BLE001 - the queue is optional; the drills run without it
            return []

    pub_t = {"fetch_flight": 0.0, "acq_flight": 0.0, "check_at": 0.0, "acq_check_at": 0.0}

    def public_job() -> Optional[Callable[[], None]]:
        """One public-data network job, or None. Every check is throttled; a failure is logged and retried later, never raised."""
        now = time.time()
        try:
            from creator import publicdata as PD
            if not pub_t["fetch_flight"] and now >= pub_t["check_at"]:
                pub_t["check_at"] = now + PD.CHECK_EVERY_S
                due = PD.due_fetches(now)
                if due:
                    pub_t["fetch_flight"] = now

                    def fjob() -> None:
                        try:
                            PD.refresh_public(due[0], time.time())
                            digests.clear()
                        except Exception as e:                     # noqa: BLE001 - recorded in public_fetch.json too; retried after FETCH_EVERY_S
                            _error_row(state, "public_fetch", {"repo": due[0].name}, e, lock)
                        finally:
                            pub_t["fetch_flight"] = 0.0
                            pub_t["check_at"] = 0.0
                    return fjob
            if not pub_t["acq_flight"] and now >= pub_t["acq_check_at"]:
                pub_t["acq_check_at"] = now + PD.ACQ_CHECK_EVERY_S
                if PD.acquisition_due(state, now):
                    pub_t["acq_flight"] = now

                    def ajob() -> None:
                        try:
                            PD.acquire_one(state)
                            digests.clear()
                        except Exception as e:                     # noqa: BLE001
                            _error_row(state, "public_acquire", {}, e, lock)
                        finally:
                            pub_t["acq_flight"] = 0.0
                    return ajob
        except Exception as e:                                     # noqa: BLE001 - a broken public-data module never stops the drills
            _error_row(state, "public_data", {}, e, lock)
            pub_t["check_at"] = pub_t["acq_check_at"] = now + GIT_RETRY_S
        return None

    def next_job() -> Optional[Callable[[], None]]:
        now = time.time()
        if public and xgits:
            pj = public_job()
            if pj is not None:
                return pj
        if deferred and now >= state_t["retry_at"]:
            queue[:0] = deferred
            deferred.clear()
            digests.clear()
        if (xgits and not state_t["xcache_flight"] and now >= state_t["xretry_at"]
                and any(not extra_cache_path(r).exists() for r in extra_repos())):
            state_t["xcache_flight"] = now                           # the archived projects' one full read, in its own job
            state_t["xretry_at"] = now + GIT_RETRY_S
            return xcache_job()
        if gits and not state_t["cache_flight"] and now >= state_t["retry_at"] and not cache_path(state).exists():
            state_t["cache_flight"] = now                            # the one slow full read, in its own job
            state_t["retry_at"] = now + GIT_RETRY_S
            return cache_job()
        while queue:
            s, v = queue.pop(0)
            if digest_of(s) == "-" or (SOURCES[s] == "git" and not cache_path(state).exists()):
                deferred.append((s, v))                              # unreadable right now: retry after GIT_RETRY_S
                if state_t["retry_at"] <= now:
                    state_t["retry_at"] = now + GIT_RETRY_S
                continue
            if (s, json.dumps(v, sort_keys=True), digest_of(s)) in done:
                continue
            return wrap(s, v, False)
        for s, v in learn_queue():                                 # remedies queued by the learning loop (creator.learnloop) for a diagnosed
            d = digest_of(s)                                       # weakness: ordinary search variants, run even when the search had stopped;
            if s not in srcs or d == "-" or (s, json.dumps(v, sort_keys=True), d) in done:   # selection is unchanged (select part only)
                continue
            done.add((s, json.dumps(v, sort_keys=True), d))
            return wrap(s, v, False, "diagnosis")
        for _ in range(len(srcs)):                                 # fixed grid exhausted: open-ended search
            s = srcs[next(rr) % len(srcs)]
            d = digest_of(s)
            with lock:
                if d == "-" or flight[s] >= SEARCH_OUTSTANDING:
                    continue
                rows = [r for r in T._jsonl(runs_path(state)) if r.get("source") == s]
                if search_state(rows, d)["stopped"]:
                    continue
                for _try in range(20):
                    v = propose(s, [r for r in rows if r.get("digest") == d], rnd)
                    why = _proposal_why(v)
                    if (s, json.dumps(v, sort_keys=True), d) not in done:
                        break
                else:
                    continue
                done.add((s, json.dumps(v, sort_keys=True), d))
                flight[s] += 1
            return wrap(s, v, True, why)
        return None
    return next_job


def search_report(state: Path, sources: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Per source: how far the open-ended search has got (variants tried, best select Brier, failures in a row, stopped?)."""
    out: dict[str, Any] = {}
    allrows = T._jsonl(runs_path(state))
    for s in (list(sources) if sources else list(SOURCES)):
        rows = [r for r in allrows if r.get("source") == s and r.get("select", {}).get("n")]
        if rows:
            out[s] = search_state(rows, rows[-1]["digest"])                 # the LATEST data (file order), not the largest hash string
    return out


def trust_section(state: Path) -> dict[str, Any]:
    """For trust.json: per drill source the variant chosen on the SELECT part with its HELD-OUT score, the gate's verdict on that score and the
    search status. Selection used only the earlier 60% of the time order; the held-out tail never chose anything."""
    out: dict[str, Any] = {}
    rep = search_report(state)
    live = live_preds(state)
    for s in SOURCES:
        b = best_variant(state, s)
        if b is None:
            continue
        sc = dict(b["heldout"])
        lp = live.get(s, [])
        sc["n_live"] = len(lp)                                          # the prospective ones are the stored live predictions, resolved since
        ok, why = T.trust_of(sc) if sc.get("n") else (False, ["no held-out predictions"])
        lsc = T.score(lp) if lp else {}
        hi = (lsc.get("gain_ci95") or [None, None])[1]
        if len(lp) >= T.MIN_LIVE and hi is not None and hi < 0:        # the future contradicts the replay: never trusted on the replay alone
            ok = False
            why.append(f"its {len(lp)} prospective predictions do WORSE than the best baseline (gain CI upper bound {hi})")
        out[s] = {"trusted": ok, "why_not": why, "variant": b["variant"], "heldout": sc, "live": lsc, "variants_tried": b["variants_tried"],
                  "search": rep.get(s)}
    return out


# ------------------------------------------------------------------------------------------------ prospective (live) drill predictions
LIVE_SOURCES = ("git_fixed", "git_churn")   # subjects are commit hashes (stable; main is never rewritten). Journal subjects are line positions: not live.


def live_path(state: Path) -> Path:
    return state / "thinking" / "drill_live.jsonl"


def _live_records(state: Path, path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in T._jsonl(live_path(state) if path is None else path):
        if "resolved" in r and r.get("id") in out:
            out[r["id"]].update(outcome=r["resolved"], resolved_at=r.get("at"))
        elif "p" in r:
            out[r["id"]] = dict(r)
    return out


def predict_open(items: Sequence[BItem], source: str, variant: dict[str, Any]) -> list[T.Pred]:
    """The best variant's P for every item whose outcome does not exist yet, from the history resolved strictly before the item was created -
    exactly what the walk-forward replay would have said. The open items get a resolution at +infinity (after every creation), so their
    placeholder outcome can never reach any prediction."""
    shadow = [BItem(it.keys, it.created, it.resolved if it.resolved is not None else math.inf, it.y if it.resolved is not None else 0, it.subject)
              for it in items]
    open_subjects = {it.subject for it in items if it.resolved is None}
    preds = walk_forward(shadow, source, float(variant["decay"]), float(variant["k"]), str(variant.get("agg", "mean")), float(variant.get("cap", 0.02)))
    return [p for p in preds if p.subject in open_subjects]


def live_pass(state: Path, repo: Path, journal: Optional[Path] = None, now: Optional[float] = None, max_new: int = 200) -> dict[str, int]:
    """Record BEFORE-the-fact predictions for the commits whose outcome window is still open (the best variant at this moment), and resolve
    the stored ones whose window has since closed. Idempotent: one prediction per (source, commit), never after its outcome exists."""
    now = time.time() if now is None else now
    state, repo = Path(state), Path(repo)
    journal = journal if journal is not None else Path.home() / "Masterstock" / "JOURNAL.md"
    known = _live_records(state)
    new = resolved = 0
    rows: list[dict[str, Any]] = []
    for s in LIVE_SOURCES:
        b = best_variant(state, s)
        if b is None:
            continue
        try:
            items = apply_variant(load(s, state, repo, journal, state), b["variant"], journal)
        except (GitCacheMissing, OSError, RuntimeError, subprocess.SubprocessError):
            continue
        truth = {it.subject: it for it in items if it.resolved is not None}
        for pid, rec in known.items():
            if rec.get("source") == s and rec.get("outcome") is None and rec["subject"] in truth:
                rows.append({"id": pid, "resolved": truth[rec["subject"]].y, "at": truth[rec["subject"]].resolved})
                resolved += 1
        for p in predict_open(items, s, b["variant"]):
            pid = f"{s}:{p.subject}"
            if pid in known or new >= max_new:
                continue
            rows.append({"id": pid, "source": s, "subject": p.subject, "made_at": now, "created": p.made_at, "p": p.p, "base": p.base,
                         "last": p.last, "mode": "live", "variant": b["variant"]})
            new += 1
    if rows:
        path = live_path(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return {"new": new, "resolved": resolved}


def live_preds(state: Path) -> dict[str, list[T.Pred]]:
    """Per source: the stored live predictions whose outcome has arrived (mode 'live')."""
    out: dict[str, list[T.Pred]] = {}
    recs = list(_live_records(state).values()) + list(_live_records(state, live_x_path(state)).values())   # Nupen's repo + public upstreams
    for rec in recs:
        if rec.get("outcome") is not None:
            out.setdefault(rec["source"], []).append(T.Pred(rec["source"], rec["subject"], rec["made_at"], rec["p"], rec["base"], rec["last"],
                                                           int(rec["outcome"]), "live"))
    for v in out.values():
        v.sort(key=lambda p: p.made_at)
    return out


# ------------------------------------------------------------------------------------------------ prospective predictions on PUBLIC upstreams
XLIVE_SOURCES = ("x_git_fixed", "x_git_churn")   # 3 Oct 2026: Nupen's own repo gets a few commits a day; active public upstreams get hundreds
XLIVE_MAX_NEW = 5000
_XLIVE_MEMO: dict[str, str] = {}


def live_x_path(state: Path) -> Path:
    return state / "thinking" / "drill_live_x.jsonl"


def live_pass_x(state: Path, journal: Optional[Path] = None, now: Optional[float] = None, max_new: int = XLIVE_MAX_NEW,
                repos: Optional[Sequence[Path]] = None) -> dict[str, int]:
    """live_pass for the x sources: the best x variant's P for every commit of an ACTIVE public clone whose outcome window is still open in
    its fetched history (never an archived project: those windows never close), resolved when later fetched upstream commits close it.
    The model sees exactly the replay's items (all projects); only the public open commits are recorded. Stored apart from Nupen's own
    (drill_live_x.jsonl, ids 'x_git_*:<project>:<hash>'); live_preds - and so trust_section's n_live and its 'live does worse' veto -
    reads both. Cheap: nothing is loaded while no cache grew and no best variant changed since the last pass of this process."""
    now = time.time() if now is None else now
    state = Path(state)
    journal = journal if journal is not None else Path.home() / "Masterstock" / "JOURNAL.md"
    xs = list(cached_extra_repos() if repos is None else repos)          # the projects cached now (the replay's items)
    pub = {r.name for r in xs if is_public(r)}
    idle = {"new": 0, "resolved": 0, "skipped": 1}
    if not pub:
        return idle
    bests = {s: best_variant(state, s) for s in XLIVE_SOURCES}
    try:
        sig = json.dumps([[r.name, extra_cache_path(r).stat().st_size] for r in xs] + [[s, (b or {}).get("variant")] for s, b in bests.items()],
                         sort_keys=True)
    except OSError:
        return idle                                                  # a cache is still being built: the next pass
    memo = str(state.resolve())
    if _XLIVE_MEMO.get(memo) == sig:
        return idle
    known = _live_records(state, live_x_path(state))
    new = resolved = 0
    rows: list[dict[str, Any]] = []
    for s in XLIVE_SOURCES:
        b = bests[s]
        if b is None:
            continue
        window, mode = (FIX_WINDOW, "fixed") if s == "x_git_fixed" else (CHURN_WINDOW, "churn")
        raw = extra_git_items(window, mode, xs)
        public_subjects = {it.subject for it in raw if it.keys[-1][2:] in pub}
        items = apply_variant(raw, b["variant"], journal)
        truth = {it.subject: it for it in items if it.resolved is not None}
        for pid, rec in known.items():
            if rec.get("source") == s and rec.get("outcome") is None and rec["subject"] in truth:
                rows.append({"id": pid, "resolved": truth[rec["subject"]].y, "at": truth[rec["subject"]].resolved})
                resolved += 1
        for p in predict_open(items, s, b["variant"]):
            pid = f"{s}:{p.subject}"
            if p.subject not in public_subjects or pid in known or new >= max_new:
                continue
            rows.append({"id": pid, "source": s, "subject": p.subject, "made_at": now, "created": p.made_at, "p": p.p, "base": p.base,
                         "last": p.last, "mode": "live", "variant": b["variant"]})
            new += 1
    if rows:
        path = live_x_path(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    if new < max_new:
        _XLIVE_MEMO[memo] = sig
    return {"new": new, "resolved": resolved}


# ------------------------------------------------------------------------------------------------ open-ended variant search
SEARCH_STOP_K = 12                 # a source stops proposing after this many consecutive variants that fail to beat its best (select part)
SEARCH_OUTSTANDING = 3             # proposals in flight per source (results of the others are not in yet)
MASKS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)   # bit i set = drop key position i (0 = all kept)
EXTRAS = ("none", "jtopic", "hour", "both")
CAPS = (0.02, 0.05, 0.1)
AGGS = ("mean", "logit")
P_FAMILY = 0.3                     # share of proposals that try a feature FAMILY (creator.trialerror) instead of a knob


def _proposal_why(v: dict[str, Any]) -> str:
    return f"idea: feature family {v['fam']}" if v.get("fam") else "search"


def apply_variant(items: list[BItem], variant: dict[str, Any], journal: Path) -> list[BItem]:
    """Feature set of a search variant: drop key positions (`mask`) and add CROSS-SOURCE context keys known at creation time only:
    'jt' = topic of the last journal entry from a day BEFORE the item's day (an entry is dated midnight but written later that day), 'hr' = UTC
    hour band and weekday. Never the outcome, never a later record."""
    mask, extra, fam = int(variant.get("mask", 0)), variant.get("extra", "none"), str(variant.get("fam") or "")
    if not mask and extra == "none" and not fam:
        return items
    jt: list[tuple[float, str]] = []
    if extra in ("jtopic", "both") and journal.is_file() and not excluded(journal.name):
        for ln in journal.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^- (\d{4})-(\d\d)-(\d\d)\b(.*)", ln)
            if m:
                jt.append((T._ts(f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00"), topic_of(m.group(4))))
        jt.sort()
    stamps = [x[0] for x in jt]
    out = []
    for it in items:
        keys = tuple(k for i, k in enumerate(it.keys) if not mask >> i & 1) or it.keys[:1]
        add: list[str] = []
        if extra in ("jtopic", "both"):
            j = bisect.bisect_right(stamps, it.created - 86400.0) - 1
            add.append(f"jt:{jt[j][1] if j >= 0 else 'none'}")
        if extra in ("hour", "both"):
            d = T.dt.datetime.fromtimestamp(it.created, T.dt.timezone.utc)
            add += [f"hr:{d.hour // 6}", f"wd:{d.weekday()}"]
        out.append(BItem(keys + tuple(add), it.created, it.resolved, it.y, it.subject, it.meta))
    if fam:                                             # a feature FAMILY (creator.trialerror): walk-forward history keys, setup facts only
        from creator import registry as REG
        out = REG.get("trialerror").with_family(out, fam)
    return out


def propose(source: str, rows: Sequence[dict[str, Any]], rnd: random.Random) -> dict[str, Any]:
    """Next variant to try: half the time a fresh random point, otherwise a one-parameter mutation of the best-on-select so far (hill climb)."""
    def fresh() -> dict[str, Any]:
        return {"decay": rnd.choice((0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0)), "k": rnd.choice((0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)),
                "mask": rnd.choice(MASKS), "extra": rnd.choice(EXTRAS), "agg": rnd.choice(AGGS), "cap": rnd.choice(CAPS), "search": 1}
    ok = [r for r in rows if r.get("select", {}).get("n")]
    if rnd.random() < P_FAMILY:                                     # IDEAS, not only knobs: a feature family on top of the best so far
        from creator import registry as REG
        fv = REG.get("trialerror").propose_family(source, dict(min(ok, key=lambda r: r["select"]["brier"])["variant"]) if ok else None, rnd)
        if fv is not None:
            return dict(fv)
    if not ok or rnd.random() < 0.5:
        return fresh()
    b = dict(min(ok, key=lambda r: r["select"]["brier"])["variant"])
    base = fresh()
    base.update({kk: vv for kk, vv in b.items() if kk != "search"})
    knob = rnd.choice(("decay", "k", "mask", "extra", "agg", "cap"))
    base[knob] = fresh()[knob]
    base["search"] = 1
    return base


def search_state(rows: Sequence[dict[str, Any]], digest_now: str) -> dict[str, Any]:
    """From a source's rows at the current data digest (file order = completion order): the best select Brier, the run of consecutive search
    variants that failed to beat the best so far, and whether proposing is over. A new best counts as an improvement only when it beats the
    old by more than the paired-SE threshold (owner's audit, 3 Oct: any 1e-9 gain reset the stop rule, so the search chased noise), and a
    source whose select part does not predict its held-out part (rank correlation across variants below NOISE_RHO) stops: searching it only
    follows noise until new data arrives."""
    from creator import registry as REG
    te = REG.get("trialerror")
    best, fails, n = float("inf"), 0, 0
    inc: Optional[dict[str, Any]] = None
    cur = []
    for r in rows:
        if r.get("digest") != digest_now or not r.get("select", {}).get("n"):
            continue
        cur.append(r)
        b = r["select"]["brier"]
        if inc is None or b < best - te.improve_threshold(inc):
            best, fails, inc = b, 0, r
        elif r.get("search"):
            fails += 1
        n += 1
    rho = te.select_predicts_heldout(cur)
    noise = rho is not None and rho < te.NOISE_RHO
    return {"best_select_brier": None if best == float("inf") else best, "consecutive_fails": fails, "tried": n,
            "select_heldout_rho": None if rho is None else round(rho, 3), "noise_stopped": noise, "stopped": fails >= SEARCH_STOP_K or noise}


def best_variant(state: Path, source: str) -> Optional[dict[str, Any]]:
    """Choose the variant with the best Brier on the SELECT part; report its HELD-OUT score (the number to believe) with the baselines."""
    rows = [r for r in T._jsonl(runs_path(state)) if r.get("source") == source and r.get("select", {}).get("n")]
    if not rows:
        return None
    last = rows[-1]["digest"]                                       # the latest data: file order = completion order (max() of hashes was arbitrary)
    rows = [r for r in rows if r["digest"] == last] or rows
    b = min(rows, key=lambda r: r["select"]["brier"])
    return {"source": source, "variant": b["variant"], "items": b["items"], "resolved": b["resolved"], "heldout": b["heldout"],
            "variants_tried": len(rows)}
