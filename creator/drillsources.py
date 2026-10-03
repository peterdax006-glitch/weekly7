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

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import threading
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
def walk_forward(items: Sequence[BItem], topic: str, decay: float = 0.97, k: float = 3.0) -> list[T.Pred]:
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
            p = min(0.98, max(0.02, sum(ps) / len(ps) if ps else glob))
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


def _git_head(repo: Path) -> str:
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    return p.stdout.strip() or "-"


def git_items(repo: Path, window: int, mode: str) -> list[BItem]:
    """mode 'fixed' | 'churn'. Read-only `git log`; the whole history is parsed once per process and kept in memory (per HEAD)."""
    head = _git_head(repo)
    ck = f"{repo}|{head}"
    if ck not in _GIT_CACHE:
        _GIT_CACHE.clear()
        _GIT_CACHE[ck] = _git_commits(repo)
    commits = _GIT_CACHE[ck]
    return _git_events(commits, window, mode)


def _git_commits(repo: Path) -> list[dict[str, Any]]:
    p = subprocess.run(["git", "-C", str(repo), "log", "--reverse", "--no-merges", "--numstat", "--format=\x01%H\x02%ct\x02%s"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    commits: list[dict[str, Any]] = []
    for block in p.stdout.split("\x01")[1:]:
        head, _, rest = block.partition("\n")
        parts = head.split("\x02")
        if len(parts) < 3:
            continue
        files, lines = [], 0
        for ln in rest.splitlines():
            f = ln.split("\t")
            if len(f) == 3 and not excluded(f[2]):
                files.append(f[2])
                lines += (int(f[0]) if f[0].isdigit() else 0) + (int(f[1]) if f[1].isdigit() else 0)
        commits.append({"h": parts[0], "t": float(parts[1]), "s": parts[2], "files": set(files), "lines": lines})
    commits.sort(key=lambda c: c["t"])
    return commits


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
    if source == "git_fixed":
        return git_items(repo, FIX_WINDOW, "fixed")
    if source == "git_churn":
        return git_items(repo, CHURN_WINDOW, "churn")
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
    except OSError:
        return "-"


def digest(items: Sequence[BItem]) -> str:
    return hashlib.sha256(f"{len(items)}|{sum(1 for i in items if i.resolved is not None)}|{items[-1].created if items else 0}".encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ parallel jobs
def runs_path(state: Path) -> Path:
    return state / "thinking" / "drill_runs.jsonl"


def run_job(source: str, variant: dict[str, float], state: Path, repo: Path, journal: Path, research: Path,
            lock: Optional[threading.Lock] = None) -> dict[str, Any]:
    """One drill batch: load the whole dataset, walk forward with this variant, score, append the row. Independent of every other job."""
    items = load(source, state, repo, journal, research)
    preds = walk_forward(items, source, variant["decay"], variant["k"])
    row = {"source": source, "variant": variant, "digest": source_digest(source, state, repo, journal, research), "items": len(items), "resolved": len(preds), **split_score(preds)}
    row["at"] = T.dt.datetime.now(T.dt.timezone.utc).isoformat(timespec="seconds")
    p = runs_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with (lock or threading.Lock()), p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def drill_filler(state: Path, repo: Path, journal: Path, research: Path, sources: Optional[Iterable[str]] = None,
                 ) -> Callable[[], Optional[Callable[[], None]]]:
    """Like creator.swarm.self_bench_filler: returns next_job(); each call hands out one independent drill job (or None when all current ones are
    done for the present data digest). The swarm keeps calling it while the Governor admits work, so the number of parallel drill workers is
    set by free CPU and RAM, not by this module."""
    state = Path(state)
    done = {(r.get("source"), json.dumps(r.get("variant"), sort_keys=True), r.get("digest")) for r in T._jsonl(runs_path(state))}
    lock = threading.Lock()
    queue: list[tuple[str, dict[str, float]]] = [(s, v) for s in (list(sources) if sources else list(SOURCES)) for v in VARIANTS]
    digests: dict[str, str] = {}

    def next_job() -> Optional[Callable[[], None]]:
        while queue:
            s, v = queue.pop(0)
            if s not in digests:
                digests[s] = source_digest(s, state, Path(repo), Path(journal), Path(research))
            if (s, json.dumps(v, sort_keys=True), digests[s]) in done or digests[s] == "-":
                continue

            def job(s: str = s, v: dict[str, float] = v) -> None:
                try:
                    run_job(s, v, state, Path(repo), Path(journal), Path(research), lock)
                except Exception as e:                             # noqa: BLE001
                    with lock:
                        p = runs_path(state)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with p.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"source": s, "variant": v, "digest": "-", "error": f"{type(e).__name__}: {e}"}) + "\n")
            return job
        return None
    return next_job


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
