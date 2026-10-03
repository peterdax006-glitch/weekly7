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
key (key, answers, sealed) is skipped by name before it is opened; nothing leaves the machine (no network, local subprocess `git log` only).

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


# ------------------------------------------------------------------------------------------------ the generic time-ordered binary model
def walk_forward(items: Sequence[BItem], topic: str, decay: float = 0.97, k: float = 3.0, agg: str = "mean", cap: float = 0.02) -> list[T.Pred]:
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


def _git_events(commits: list[dict[str, Any]], window: int, mode: str) -> list[BItem]:
    out: list[BItem] = []
    for i, c in enumerate(commits):
        top = sorted(c["files"])[0].split("/")[0] if c["files"] else "-"
        keys = (f"m:{_msg_class(c['s'])}", f"n:{_log2b(len(c['files']))}", f"l:{_log2b(c['lines'])}", f"d:{top}")
        if i + window >= len(commits):
            out.append(BItem(keys, c["t"], None, 0, c["h"][:10]))
            continue
        nxt = commits[i + 1:i + 1 + window]
        if mode == "fixed":
            y = int(any(FIX_WORDS.search(n["s"]) and n["files"] & c["files"] for n in nxt))
        else:
            y = int(any(n["files"] & c["files"] for n in nxt))
        out.append(BItem(keys, c["t"], max(c["t"], nxt[-1]["t"]), y, c["h"][:10]))
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


SOURCES: dict[str, str] = {"git_fixed": "git", "git_churn": "git", "journal_persist": "journal", "plan_choice": "plan", "research_bool": "research"}


def load(source: str, state: Path, repo: Path, journal: Path, research: Path) -> list[BItem]:
    global DEFAULT_STATE
    DEFAULT_STATE = Path(state)
    if source == "git_fixed":
        return git_items(repo, FIX_WINDOW, "fixed", state)
    if source == "git_churn":
        return git_items(repo, CHURN_WINDOW, "churn", state)
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
        if SOURCES[source] == "journal":
            st = journal.stat()
            return f"{st.st_size}"
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


def run_job(source: str, variant: dict[str, float], state: Path, repo: Path, journal: Path, research: Path,
            lock: Optional[threading.Lock] = None) -> dict[str, Any]:
    """One drill batch: load the whole dataset, walk forward with this variant, score, append the row. Independent of every other job."""
    items = apply_variant(load(source, state, repo, journal, research), variant, journal)
    preds = walk_forward(items, source, float(variant["decay"]), float(variant["k"]), str(variant.get("agg", "mean")), float(variant.get("cap", 0.02)))
    row = {"source": source, "variant": variant, "search": bool(variant.get("search")), "digest": source_digest(source, state, repo, journal, research), "items": len(items), "resolved": len(preds), **split_score(preds)}
    row["at"] = T.dt.datetime.now(T.dt.timezone.utc).isoformat(timespec="seconds")
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


def drill_filler(state: Path, repo: Path, journal: Path, research: Path, sources: Optional[Iterable[str]] = None, seed: Optional[int] = None,
                 ) -> Callable[[], Optional[Callable[[], None]]]:
    """Like creator.swarm.self_bench_filler: returns next_job(); each call hands out one independent drill job. First the fixed VARIANTS grid per
    source, then (OPEN-ENDED, owner 2 Oct: the filler must never run dry while there is thinking left to improve) NEW variants proposed by
    `propose` - round-robin over the sources, at most SEARCH_OUTSTANDING in flight per source - until SEARCH_STOP_K consecutive variants fail to
    beat that source's best on the select part; a changed data digest (new commits, journal lines, plan rows, research files) reopens it.
    Returns None only when every source is stopped or done for the present data. The swarm's Governor sets how many run at once."""
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

    def wrap(s: str, v: dict[str, Any], search: bool) -> Callable[[], None]:
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
                row = run_job(s, v, state, Path(repo), Path(journal), Path(research), lock)
                if pid and fp is not None and row.get("select", {}).get("n"):
                    fp.end_safe(pid, int(row["select"]["brier"] < prior), state)
            except Exception as e:                                 # noqa: BLE001
                _error_row(state, s, v, e, lock)
            finally:
                if search:
                    with lock:
                        flight[s] -= 1
        return job

    deferred: list[tuple[str, dict[str, Any]]] = []             # jobs whose data could not be read now: retried later, never dropped
    state_t = {"retry_at": 0.0, "cache_flight": 0.0}
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

    def next_job() -> Optional[Callable[[], None]]:
        now = time.time()
        if deferred and now >= state_t["retry_at"]:
            queue[:0] = deferred
            deferred.clear()
            digests.clear()
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
                    if (s, json.dumps(v, sort_keys=True), d) not in done:
                        break
                else:
                    continue
                done.add((s, json.dumps(v, sort_keys=True), d))
                flight[s] += 1
            return wrap(s, v, True)
        return None
    return next_job


def search_report(state: Path, sources: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Per source: how far the open-ended search has got (variants tried, best select Brier, failures in a row, stopped?)."""
    out: dict[str, Any] = {}
    allrows = T._jsonl(runs_path(state))
    for s in (list(sources) if sources else list(SOURCES)):
        rows = [r for r in allrows if r.get("source") == s and r.get("select", {}).get("n")]
        if rows:
            out[s] = search_state(rows, max(r["digest"] for r in rows))
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


def _live_records(state: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in T._jsonl(live_path(state)):
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
    for rec in _live_records(state).values():
        if rec.get("outcome") is not None:
            out.setdefault(rec["source"], []).append(T.Pred(rec["source"], rec["subject"], rec["made_at"], rec["p"], rec["base"], rec["last"],
                                                           int(rec["outcome"]), "live"))
    for v in out.values():
        v.sort(key=lambda p: p.made_at)
    return out


# ------------------------------------------------------------------------------------------------ open-ended variant search
SEARCH_STOP_K = 12                 # a source stops proposing after this many consecutive variants that fail to beat its best (select part)
SEARCH_OUTSTANDING = 3             # proposals in flight per source (results of the others are not in yet)
MASKS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)   # bit i set = drop key position i (0 = all kept)
EXTRAS = ("none", "jtopic", "hour", "both")
CAPS = (0.02, 0.05, 0.1)
AGGS = ("mean", "logit")


def apply_variant(items: list[BItem], variant: dict[str, Any], journal: Path) -> list[BItem]:
    """Feature set of a search variant: drop key positions (`mask`) and add CROSS-SOURCE context keys known at creation time only:
    'jt' = topic of the last journal entry from a day BEFORE the item's day (an entry is dated midnight but written later that day), 'hr' = UTC
    hour band and weekday. Never the outcome, never a later record."""
    mask, extra = int(variant.get("mask", 0)), variant.get("extra", "none")
    if not mask and extra == "none":
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
        out.append(BItem(keys + tuple(add), it.created, it.resolved, it.y, it.subject))
    return out


def propose(source: str, rows: Sequence[dict[str, Any]], rnd: random.Random) -> dict[str, Any]:
    """Next variant to try: half the time a fresh random point, otherwise a one-parameter mutation of the best-on-select so far (hill climb)."""
    def fresh() -> dict[str, Any]:
        return {"decay": rnd.choice((0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0)), "k": rnd.choice((0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)),
                "mask": rnd.choice(MASKS), "extra": rnd.choice(EXTRAS), "agg": rnd.choice(AGGS), "cap": rnd.choice(CAPS), "search": 1}
    ok = [r for r in rows if r.get("select", {}).get("n")]
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
    variants that failed to beat the best so far, and whether proposing is over."""
    best, fails, n = float("inf"), 0, 0
    for r in rows:
        if r.get("digest") != digest_now or not r.get("select", {}).get("n"):
            continue
        b = r["select"]["brier"]
        if b < best - 1e-9:
            best, fails = b, 0
        elif r.get("search"):
            fails += 1
        n += 1
    return {"best_select_brier": None if best == float("inf") else best, "consecutive_fails": fails, "tried": n, "stopped": fails >= SEARCH_STOP_K}


def best_variant(state: Path, source: str) -> Optional[dict[str, Any]]:
    """Choose the variant with the best Brier on the SELECT part; report its HELD-OUT score (the number to believe) with the baselines."""
    rows = [r for r in T._jsonl(runs_path(state)) if r.get("source") == source and r.get("select", {}).get("n")]
    if not rows:
        return None
    last = max(r["digest"] for r in rows)
    rows = [r for r in rows if r["digest"] == last] or rows
    b = min(rows, key=lambda r: r["select"]["brier"])
    return {"source": source, "variant": b["variant"], "items": b["items"], "resolved": b["resolved"], "heldout": b["heldout"],
            "variants_tried": len(rows)}
