"""REASONING DRILLS: a self-renewing stream of verifiable multiple-choice questions for Nupen's reasoning (owner, 3 Oct 2026: massively increase
Nupen's thinking/reasoning capacity; 08:45: 'learn from itself, improve its question-answering, revisit its data and repeat, collect new data when
it runs out').

QUESTIONS are mined from git histories: every PUBLIC repository under runtime_dir()/public_repos/ (whatever the acquisition job has cloned or
fetched - new repositories and new commits become new questions with no teacher step) and Nupen's OWN repository. Never ~/oldpc, never an author
name or e-mail (only commit subjects without '@', file paths and dates are read). Each question has 4 options, exactly one correct by construction,
options shuffled by a seed of the question's own id, and a difficulty label:

  files_changed  'which of these 4 files did the commit "<subject>" change?'  distractors were live files at that time (hard: same directory)
  which_first    'which of these 4 commits came first?'                     hard: all touched one file; easy: spread over the history
  revert_of      'a revert at <date> changed files F; which commit did it revert?'  (the revert's own subject, which names it, is never shown)
  co_change      'up to <date>, which source file changed most often together with test file T?'  (unique top by a margin; public repos only)

NO LEAKAGE: a question shows only what was known at its time `t` (distractor files existed then; co-change counts stop at `t`); a question whose
subject names its answer (or a distractor) is dropped. THE FROZEN BENCHMARK (creator.thinkbench, part c) IS NEVER TRAINING DATA: its kinds are not
generated here, and `collides` drops any question sharing an id, subject, answer-subject or text with a frozen item (tested).

STRATEGY SEARCH (creator.judgment's successive halving, made endless): answers are grouped in EPOCHS of EPOCH_Q fresh questions. Inside an epoch
the strategies answer the same questions and are halved by accuracy (HALVING); the 'plain' prompt always stays as the control. The next epoch
starts with the previous winners (re-tested OUT OF SAMPLE on questions they never saw) plus CHALLENGERS: losers rotated back in, so a strategy cut
early gets another chance. Strategy 'retrieve' shows solved questions from Nupen's OWN earlier answers (other repositories, or the same repository
strictly earlier in time; never the same commit; never the frozen benchmark) with the correct answer and Nupen's own earlier pick - it learns
from itself. Accuracy per epoch is the improvement curve; when fresh questions run low the report says so (`need_acquisition`).

Records: state/creator/thinking/reasoning.jsonl (question id, strategy, pick, correct, model, seconds; never the prompt text). Loaded on demand
through creator.registry (name `reasondrills`)."""
from __future__ import annotations

import datetime as dt
import bisect
import dataclasses
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

LETTERS = "ABCD"
ANSWER = re.compile(r"ANSWER\s*[:=]\s*\(?\**\s*([A-D])\b", re.I)
KINDS = ("files_changed", "which_first", "revert_of", "co_change")
EPOCH_Q = 40                                    # fresh questions per epoch
HALVING = ((10, 4), (20, 2))                    # (common questions answered, non-control strategies kept)
CARRY, CHALLENGERS = 2, 2                       # next epoch: the winners plus rotated-back losers
# ONLINE ALLOCATION (3 Oct 2026, creator.trialerror): epochs started from now on give every question the control and the carried winners (the
# floor: paired comparisons and the out-of-sample re-test) plus ONLINE_PICKS strategies picked by top-two Thompson sampling on Beta posteriors of
# the epoch's answers so far; a strategy whose paired gain over the control is clearly negative (95% CI) is raced out of the epoch. Epochs
# whose rows carry no 'alloc' keep their halving replay.
REASON_ALLOCATION = "online"
ONLINE_PICKS = 3                                # simulation (creator.trialerror): 3 picks per question beat halving at equal calls; 2 did not
ONLINE_MIN_N = 8                                # answers before a strategy can be carried as a winner
BATCH = 4                                       # model calls per job (one server lease)
REVISIT = "#revisit"                            # qid suffix of a second pass over a question the control got wrong
LOW_FRESH = 300                                 # fewer unanswered questions than this: the report asks for acquisition
REFRESH_S = 1800.0                              # how often the filler re-reads the repositories (new commits, new repositories)
MAX_FILES = 6                                   # commits touching more files are not used for files_changed (no single-subject answer)
BAD_SUBJECT = re.compile(r"@|<[^>]*>|^merge\b|signed-off|co-authored|https?://", re.I)
SYSTEM = ("You answer multiple-choice questions about the development history of software repositories. Exactly one option is correct. "
          "Use only what the question shows and general software knowledge.")


@dataclass
class Strategy:
    name: str
    style: str                  # plain | cot | eliminate | think
    samples: int = 1
    retrieve: int = 0
    max_tokens: int = 220
    late: bool = False          # expensive: not in the first epoch; enters as a challenger once the cheap strategies set the bar


STRATEGIES = [Strategy("plain", "plain", max_tokens=16), Strategy("cot", "cot"), Strategy("eliminate", "eliminate", max_tokens=480),
              Strategy("retrieve4", "cot", retrieve=4), Strategy("retrieve4_elim", "eliminate", retrieve=4, max_tokens=480),
              Strategy("vote3", "cot", samples=3), Strategy("think", "think", max_tokens=1400, late=True)]
BY_NAME = {s.name: s for s in STRATEGIES}
CONTROL = "plain"


@dataclass
class Question:
    qid: str
    kind: str
    source: str                 # repository name ('nupen' = Nupen's own)
    subject: str                # what the question is about (commit hash, file) - never shown as an answer hint
    t: float                    # when everything the question shows was known
    stem: str
    options: list[str]
    answer: int
    difficulty: str
    keys: tuple[str, ...] = field(default_factory=tuple)      # commits involved (retrieval never shows a question about the same commit)

    def prompt(self) -> str:
        return self.stem + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(self.options))


# ------------------------------------------------------------------------------------------------ git histories
@dataclass
class Commit:
    h: str
    t: float
    s: str
    changes: list[tuple[str, str]]          # (status letter, path); a rename appears as D old + A new


def repos(repo: Optional[Path] = None, public_dir: Optional[Path] = None) -> list[tuple[str, Path]]:
    """(name, path): every git clone under public_repos (half-finished '*.tmp' clones skipped) and Nupen's own repository as 'nupen'."""
    if public_dir is None:
        from creator import device as DEV
        public_dir = DEV.runtime_dir() / "public_repos"
    out: list[tuple[str, Path]] = []
    if public_dir.is_dir():
        for d in sorted(public_dir.iterdir()):
            if d.is_dir() and not d.name.endswith(".tmp") and (d / ".git").exists():
                out.append((d.name, d))
    if repo is not None and (Path(repo) / ".git").exists():
        out.append(("nupen", Path(repo)))
    return out


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    return r.stdout if r.returncode == 0 else ""


def head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


_SHA = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")


def head_fast(repo: Path) -> str:
    """`git rev-parse HEAD` read straight from the files (a git process costs ~2 s on the busy PC): HEAD -> loose ref -> packed-refs, worktrees
    (a '.git' file pointing at its gitdir + commondir) included. Anything unusual falls back to the git process."""
    try:
        g = Path(repo) / ".git"
        if g.is_file():
            gd = g.read_text(encoding="utf-8").strip()
            if not gd.startswith("gitdir:"):
                return head(repo)
            g = (Path(repo) / gd[len("gitdir:"):].strip()).resolve()
        common = g
        if (g / "commondir").is_file():
            common = (g / (g / "commondir").read_text(encoding="utf-8").strip()).resolve()
        h = (g / "HEAD").read_text(encoding="utf-8").strip()
        if _SHA.fullmatch(h):
            return h
        if not h.startswith("ref: refs/"):
            return head(repo)
        ref = h[len("ref: "):]
        for base in ((g, common) if g != common else (g,)):
            p = base / ref
            if p.is_file():
                v = p.read_text(encoding="utf-8").strip()
                return v if _SHA.fullmatch(v) else head(repo)
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[1].strip() == ref and _SHA.fullmatch(parts[0]):
                    return parts[0]
    except (OSError, ValueError):
        pass
    return head(repo)


def parse_log(text: str) -> list[Commit]:
    """`git log --no-merges --name-status --format=@@%H%x09%at%x09%s` (newest first) -> commits oldest first. Authors are never requested."""
    out: list[Commit] = []
    cur: Optional[Commit] = None
    for line in text.splitlines():
        if line.startswith("@@"):
            parts = line[2:].split("\t", 2)
            if len(parts) < 3:
                cur = None
                continue
            try:
                cur = Commit(parts[0], float(parts[1]), parts[2].strip(), [])
            except ValueError:
                cur = None
                continue
            out.append(cur)
        elif line.strip() and cur is not None:
            f = line.split("\t")
            st = f[0][:1]
            if st == "R" and len(f) >= 3:
                cur.changes += [("D", f[1]), ("A", f[2])]
            elif st == "C" and len(f) >= 3:
                cur.changes.append(("A", f[2]))
            elif len(f) >= 2:
                cur.changes.append((st, f[1]))
    out.reverse()
    return out


_LOGS: dict[str, tuple[str, list[Commit]]] = {}
_LOGS_LOCK = threading.Lock()


def history(repo: Path) -> list[Commit]:
    """All non-merge commits, oldest first; re-read only when HEAD moved (a fetch brings new commits -> new questions)."""
    h = head(repo)
    key = str(repo)
    with _LOGS_LOCK:
        hit = _LOGS.get(key)
        if hit and hit[0] == h:
            return hit[1]
    cs = parse_log(_git(repo, "log", "--no-merges", "--no-renames", "--name-status", "--format=@@%H%x09%at%x09%s"))
    with _LOGS_LOCK:
        _LOGS[key] = (h, cs)
    return cs


# ------------------------------------------------------------------------------------------------ question generators
def _rng(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16], 16))


def _date(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")


def _stem_of(p: str) -> str:
    b = p.rsplit("/", 1)[-1]
    return b.rsplit(".", 1)[0].lower() if "." in b else b.lower()


def _names(subject: str, path: str) -> bool:
    """Does the commit subject name this file (its basename, or a stem of 4+ characters as a word)?"""
    s = subject.lower()
    b = path.rsplit("/", 1)[-1].lower()
    st = _stem_of(path)
    return b in s or (len(st) >= 4 and _word_in(st, s))


_BEFORE, _AFTER = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_"), frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def _word_in(st: str, s: str) -> bool:
    """re.search(r"(?<![a-z0-9_])" + re.escape(st) + r"(?![a-z0-9])", s) without compiling a pattern per file name (the regex cache thrashed:
    one compile per distinct stem, millions of calls over a large history)."""
    i, n = s.find(st), len(st)
    while i >= 0:
        if (i == 0 or s[i - 1] not in _BEFORE) and (i + n >= len(s) or s[i + n] not in _AFTER):
            return True
        i = s.find(st, i + 1)
    return False


def _ok_subject(s: str) -> bool:
    return 8 <= len(s) and not BAD_SUBJECT.search(s)


def _mk(kind: str, source: str, subject: str, t: float, stem: str, truth: str, distractors: Sequence[str], difficulty: str,
        keys: Sequence[str]) -> Optional[Question]:
    ds = [d for d in dict.fromkeys(distractors) if d != truth][:3]
    if len(ds) < 3:
        return None
    qid = f"r:{kind}:{source}:{subject}"
    opts = ds + [truth]
    _rng(qid, "options").shuffle(opts)
    return Question(qid, kind, source, subject, t, stem, opts, opts.index(truth), difficulty, tuple(keys))


def _short(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _dir(p: str) -> str:
    return p.rsplit("/", 1)[0] if "/" in p else ""


def gen_files_changed(source: str, cs: Sequence[Commit]) -> list[Question]:
    out: list[Question] = []
    live: dict[str, int] = {}                                          # path -> index of its last commit; insertion order = recency
    by_dir: dict[tuple[str, str], set[str]] = {}                       # (directory, extension) -> live files (the hard distractor pool)
    for i, c in enumerate(cs):
        changed = [p for st, p in c.changes if st != "D"]
        if 1 <= len(c.changes) <= MAX_FILES and changed and _ok_subject(c.s) and len(live) >= 20:
            r = _rng(source, c.h, "fc")
            truth = r.choice(sorted(changed))
            touched = {p for _st, p in c.changes}                      # distractors: files that existed BEFORE this commit, untouched by it
            hard = r.random() < 0.5
            cand: list[str] = []
            if hard:
                ext = truth.rsplit(".", 1)[-1]
                cand = sorted(p for p in by_dir.get((_dir(truth), ext), ()) if p not in touched)
            if len(cand) < 3:
                hard = False
                cand = sorted(itertools.islice((p for p in reversed(live) if p not in touched), 300))
            r.shuffle(cand)
            ds = list(itertools.islice((p for p in cand if not _names(c.s, p)), 3))
            if not any(_names(c.s, p) for p in changed) and len(ds) == 3:
                q = _mk("files_changed", source, c.h[:12], c.t,
                        f"Repository '{source}'. A commit made on {_date(c.t)} has the message: \"{_short(c.s)}\". "
                        f"Which ONE of these files did this commit change (the other three existed but were not changed by it)?",
                        truth, ds, "hard" if hard else "easy", [c.h[:12]])
                if q:
                    out.append(q)
        for st, p in c.changes:
            live.pop(p, None)
            if st == "D":
                by_dir.get((_dir(p), p.rsplit(".", 1)[-1]), set()).discard(p)
            else:
                live[p] = i
                by_dir.setdefault((_dir(p), p.rsplit(".", 1)[-1]), set()).add(p)
    return out


def gen_which_first(source: str, cs: Sequence[Commit], every: int = 3) -> list[Question]:
    out: list[Question] = []
    ok = [i for i, c in enumerate(cs) if _ok_subject(c.s) and not c.s.lower().startswith("revert")]
    by_file: dict[str, list[int]] = {}
    for i in ok:
        for _st, p in cs[i].changes:
            by_file.setdefault(p, []).append(i)
    for n, i in enumerate(ok[30::every]):
        c = cs[i]
        r = _rng(source, c.h, "wf")
        hard = r.random() < 0.5
        group: list[int] = []
        if hard and c.changes:
            f = r.choice(sorted(p for _st, p in c.changes))
            prev = [j for j in by_file.get(f, []) if j < i]
            if len(prev) >= 3:
                group = sorted(r.sample(prev[-12:], 3)) + [i]
        if not group:
            hard = False
            pos = bisect.bisect_left(ok, i)
            prev = ok[max(0, pos - 400):pos]
            if len(prev) < 3:
                continue
            group = sorted(r.sample(prev, 3)) + [i]
        ts = [cs[j].t for j in group]
        subs = [_short(cs[j].s, 120) for j in group]
        if len(set(ts)) < 4 or len(set(subs)) < 4 or min(ts) != cs[group[0]].t or ts.count(min(ts)) > 1:
            continue
        first = min(group, key=lambda j: cs[j].t)
        stem = (f"Repository '{source}'. Four commits are listed below in random order" +
                (", all of which changed the same file" if hard else "") + ". Which one was committed FIRST?")
        q = _mk("which_first", source, cs[first].h[:12] + "+" + c.h[:12], c.t, stem, _short(cs[first].s, 120),
                [_short(cs[j].s, 120) for j in group if j != first], "hard" if hard else "easy", [cs[j].h[:12] for j in group])
        if q:
            out.append(q)
    return out


_REVERT = re.compile(r'^Revert "(.+)"$')


def gen_revert_of(source: str, cs: Sequence[Commit]) -> list[Question]:
    out: list[Question] = []
    for i, c in enumerate(cs):
        m = _REVERT.match(c.s)
        if not m or not c.changes:
            continue
        target = next((j for j in range(i - 1, -1, -1) if cs[j].s == m.group(1)), None)
        if target is None or not _ok_subject(cs[target].s):
            continue
        files = sorted({p for _st, p in c.changes})
        r = _rng(source, c.h, "rv")
        near = [j for j in range(max(0, i - 300), i) if j != target and _ok_subject(cs[j].s) and not cs[j].s.lower().startswith("revert")
                and cs[j].s != cs[target].s]
        overlap = [j for j in near if {p for _st, p in cs[j].changes} & set(files)]
        pool = overlap if len(overlap) >= 3 else near
        r.shuffle(pool)
        q = _mk("revert_of", source, c.h[:12], c.t,
                f"Repository '{source}'. A commit made on {_date(c.t)} REVERTED an earlier commit; the revert changed: {', '.join(files[:6])}"
                f"{' ...' if len(files) > 6 else ''}. Which earlier commit did it revert?",
                _short(cs[target].s, 120), [_short(cs[j].s, 120) for j in pool], "hard" if len(overlap) >= 3 else "easy",
                [c.h[:12], cs[target].h[:12]])
        if q:
            out.append(q)
    return out


def _is_test(p: str) -> bool:
    b = p.rsplit("/", 1)[-1]
    return p.endswith(".py") and (b.startswith("test_") or b.endswith("_test.py"))


def gen_co_change(source: str, cs: Sequence[Commit], checkpoints: Sequence[int] = (10, 20, 40, 80, 160)) -> list[Question]:
    if source == "nupen":                                   # Nupen's own test <-> module map is the frozen benchmark's ground: never trained on
        return []
    out: list[Question] = []
    co: dict[str, dict[str, int]] = {}
    seen: dict[str, int] = {}
    live: set[str] = set()
    live_src: set[str] = set()                              # live non-test .py files, kept as the walk goes (was a rescan of `live` per question)
    for c in cs:
        for st, p in c.changes:
            if st == "D":
                live.discard(p)
                live_src.discard(p)
            else:
                live.add(p)
                if p.endswith(".py") and not _is_test(p):
                    live_src.add(p)
        if len(c.changes) > 12:
            continue
        tests = [p for _st, p in c.changes if _is_test(p)]
        srcs = [p for _st, p in c.changes if p.endswith(".py") and not _is_test(p) and "/conftest" not in "/" + p]
        for tf in tests:
            d = co.setdefault(tf, {})
            for s in srcs:
                d[s] = d.get(s, 0) + 1
            if srcs:
                seen[tf] = seen.get(tf, 0) + 1
                if seen[tf] in checkpoints and tf in live:
                    top = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
                    if top[0][1] >= 4 and (len(top) == 1 or top[0][1] >= 1.5 * top[1][1]) and top[0][0] in live:
                        truth = top[0][0]
                        zero = sorted(p for p in live_src if p not in d)
                        r = _rng(source, tf, str(seen[tf]), "cc")
                        r.shuffle(zero)
                        easy = _stem_of(truth) in _stem_of(tf)
                        q = _mk("co_change", source, f"{tf}@{seen[tf]}", c.t,
                                f"Repository '{source}'. Counting commits up to {_date(c.t)}, which source file was changed most often in the SAME "
                                f"commit as the test file {tf}?", truth, zero, "easy" if easy else "hard", [c.h[:12]])
                        if q:
                            out.append(q)
    return out


GENERATORS: dict[str, Callable[[str, Sequence[Commit]], list[Question]]] = {
    "files_changed": gen_files_changed, "which_first": gen_which_first, "revert_of": gen_revert_of, "co_change": gen_co_change}


# ------------------------------------------------------------------------------------------------ the frozen benchmark is never training data
def frozen_c(state: Path) -> list[dict[str, Any]]:
    try:
        body = json.loads((Path(state) / "thinkbench" / "items.json").read_text(encoding="utf-8"))
        return list(body.get("c") or [])
    except (OSError, ValueError):
        return []


class Frozen:
    """What the frozen benchmark's part c is about: ids, subjects (module paths, package ids, snapshot times), answers and texts."""

    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.ids = {str(q.get("id")) for q in items}
        self.subjects = {str(q.get("id", "")).split(":", 2)[-1] for q in items}
        self.texts = {" ".join(str(q.get("prompt", "")).split()) for q in items}
        self.stems = {" ".join(str(q.get("prompt", "")).split("\n", 1)[0].split()) for q in items}
        self.answers: set[str] = set()
        for q in items:
            lines = str(q.get("prompt", "")).split("\n")[1:]
            a = q.get("answer")
            if isinstance(a, int) and 0 <= a < len(lines):
                self.answers.add(lines[a][3:].strip())

    def collides(self, q: Question) -> bool:
        if q.qid in self.ids or q.subject in self.subjects or any(k in self.subjects for k in q.keys):
            return True
        if " ".join(q.prompt().split()) in self.texts or " ".join(q.stem.split()) in self.stems:
            return True
        return q.source == "nupen" and self.answers.__contains__(q.options[q.answer])


_SRC_HASH: dict[str, str] = {}


def _src_hash() -> str:
    """The generators' code: a cached question list is only reused by the code that made it."""
    if "h" not in _SRC_HASH:
        _SRC_HASH["h"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:20]
    return _SRC_HASH["h"]


def qcache_dir() -> Path:
    from creator import device as DEV
    return DEV.runtime_dir() / "thinking" / "reasondrills_q"


def _repo_questions(name: str, path: Path, cache: Optional[Path], cached_only: bool = False) -> Optional[dict[str, list[Question]]]:
    """Every kind's questions of one repository (before the frozen filter). Generating them reads the whole git history and walks it (about a
    minute per large repository on a busy PC), so the result is kept on disk outside the repository, keyed on the repository's HEAD and this
    module's source: unchanged HEAD + unchanged code = the identical list without a git log. Never fatal: any cache problem regenerates."""
    h = head_fast(path) if cache is not None and (Path(path) / ".git").exists() else ""
    key = f"{name}|{h}|{_src_hash()}"
    f = (cache / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.json") if cache is not None and h else None
    if f is not None:
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
            if body.get("key") == key:
                return {k: [Question(**dict(d, keys=tuple(d["keys"]))) for d in v] for k, v in body["kinds"].items()}
        except (OSError, ValueError, TypeError, KeyError):
            pass
    if cached_only:
        return None
    cs = history(path)
    got = {k: GENERATORS[k](name, cs) for k in KINDS}
    if f is not None and head_fast(path) == h:                     # HEAD moved while reading: do not file new questions under the old key
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f"{f.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"key": key, "kinds": {k: [dataclasses.asdict(q) for q in v] for k, v in got.items()}}), encoding="utf-8")
            tmp.replace(f)
        except OSError:
            pass
    return got


def generate(sources: Sequence[tuple[str, Path]], state: Path, kinds: Sequence[str] = KINDS, cache: Optional[Path] = None,
             use_cache: bool = True) -> list[Question]:
    """Every question available now (deterministic ids and options), frozen-benchmark collisions removed. Per repository the generated list is
    cached on disk (qcache_dir(), outside the repository) by HEAD and source hash: identical output, no git log while nothing changed."""
    fz = Frozen(frozen_c(state))
    cdir = (cache or qcache_dir()) if use_cache else None
    out: list[Question] = []
    for name, path in sources:
        try:
            got = _repo_questions(name, path, cdir)
        except Exception:                                             # noqa: BLE001 - a broken clone is skipped, never fatal
            continue
        for k in kinds:
            out += [q for q in got[k] if not fz.collides(q)] if got else []
    return out


def generate_stream(sources: Sequence[tuple[str, Path]], state: Path, cache: Optional[Path] = None) -> Iterator[list[Question]]:
    """generate() in pieces, so a consumer (gpupulse.traces) can start at once: first every repository the disk cache already holds (one list),
    then each repository that has to be generated, one list each. Together exactly generate()'s questions."""
    fz = Frozen(frozen_c(state))
    cdir = cache or qcache_dir()
    warm: list[Question] = []
    cold: list[tuple[str, Path]] = []
    for name, path in sources:
        try:
            got = _repo_questions(name, path, cdir, cached_only=True)
        except Exception:                                             # noqa: BLE001
            got = None
        if got is None:
            cold.append((name, path))
        else:
            warm += [q for k in KINDS for q in got[k] if not fz.collides(q)]
    yield warm
    for name, path in cold:
        try:
            got = _repo_questions(name, path, cdir)
        except Exception:                                             # noqa: BLE001 - a broken clone is skipped, never fatal
            continue
        yield [q for k in KINDS for q in (got or {}).get(k, []) if not fz.collides(q)]


def _round_robin(groups: dict[str, list[Question]]) -> list[Question]:
    out: list[Question] = []
    lists = [list(groups[k]) for k in sorted(groups)]
    while any(lists):
        for g in lists:
            if g:
                out.append(g.pop(0))
    return out


def order(qs: Iterable[Question]) -> list[Question]:
    """Fresh-question order: kinds take turns (so the first halving cut is not judged on one kind), and inside a kind the repositories take
    turns; inside one repository a stable hash order (difficulty and time mixed)."""
    by: dict[str, dict[str, list[Question]]] = {}
    for q in qs:
        by.setdefault(q.kind, {}).setdefault(q.source, []).append(q)
    for srcs in by.values():
        for g in srcs.values():
            g.sort(key=lambda q: hashlib.sha256(q.qid.encode("utf-8")).hexdigest())
    return _round_robin({k: _round_robin(srcs) for k, srcs in by.items()})


# ------------------------------------------------------------------------------------------------ prompts, answers, the model-free control
def parse_choice(reply: Optional[str]) -> Optional[int]:
    m = ANSWER.findall(reply or "")
    return LETTERS.index(m[-1].upper()) if m else None


def solved_examples(q: Question, solved: Sequence[tuple[Question, Optional[int], float]], k: int, now: float) -> list[tuple[Question, Optional[int]]]:
    """Nupen's OWN earlier answered questions (answered before `now`) shown as worked examples: same kind; another repository, or the same
    repository strictly earlier in time; never one sharing a commit with `q`; most similar first (word overlap)."""
    from creator import reasonmethods as RM
    if not k:
        return []
    want = set(RM.tokens(q.prompt()))
    keys = set(q.keys)
    cand = []
    for s, pick, at in solved:
        if at >= now or s.qid == q.qid or s.kind != q.kind or keys & set(s.keys):
            continue
        if s.source == q.source and s.t >= q.t:
            continue
        have = set(RM.tokens(s.prompt()))
        cand.append((len(want & have) / (len(want | have) or 1), s.qid, s, pick))
    cand.sort(key=lambda x: (-x[0], x[1]))
    return [(s, pick) for _sc, _id, s, pick in cand[:k]]


def build_messages(st: Strategy, q: Question, examples: Sequence[tuple[Question, Optional[int]]] = (),
                   traces: Optional[Mapping[str, str]] = None) -> list[dict[str, str]]:
    sysm = SYSTEM
    if st.style == "plain":
        sysm += " Reply with exactly one line: 'ANSWER: <letter>'."
    elif st.style == "cot":
        sysm += " Think briefly (at most three short lines), then finish with exactly 'ANSWER: <letter>'."
    elif st.style == "eliminate":
        sysm += (" Go through the options one by one: for each write one short line 'X: keep' or 'X: drop' with the reason. Then compare the kept "
                 "ones and finish with exactly 'ANSWER: <letter>'.")
    else:
        sysm += " Reason it through, then finish with exactly 'ANSWER: <letter>'."
    user = ""
    if examples:
        parts = []
        tr = traces or {}
        for s, pick in examples:
            line = s.prompt() + (f"\nWorked reasoning: {tr[s.qid]}" if s.qid in tr else "") + f"\nCorrect answer: {LETTERS[s.answer]}"
            if pick is not None and pick != s.answer:
                line += f" (you answered {LETTERS[pick]} before - that was wrong)"
            parts.append(line)
        head = ("Solved examples (worked reasoning checked against the known answer):" if any(s.qid in tr for s, _p in examples)
                else "Solved examples from your own earlier practice:")
        user = head + "\n\n" + "\n\n".join(parts) + "\n\nNow the question:\n"
    return [{"role": "system", "content": sysm}, {"role": "user", "content": user + q.prompt()}]


def lexical_pick(q: Question) -> int:
    """The model-free control: the option sharing the most words with the question stem (ties: the first). If a strategy cannot beat this,
    it is not reasoning."""
    from creator import reasonmethods as RM
    want = set(RM.tokens(q.stem))
    sc = [len(want & set(RM.tokens(o.replace("/", " ").replace(".", " ").replace("_", " ")))) for o in q.options]
    return sc.index(max(sc))


def vote(picks: Sequence[Optional[int]]) -> Optional[int]:
    """Self-consistency: the most frequent parsed pick (ties: the earliest sample's)."""
    ok = [p for p in picks if p is not None]
    if not ok:
        return None
    best = max(ok.count(p) for p in ok)
    return next(p for p in ok if ok.count(p) == best)


def ask(llm: Any, st: Strategy, msgs: list[dict[str, str]]) -> tuple[Optional[int], list[Optional[int]], str, int]:
    """(pick, sample picks, first reply, approximate tokens). 'think' lets a reasoning model think (no /no_think); the others go through
    judgment.chat_text (the /no_think switch, <think> stripped)."""
    from creator import generator as G
    from creator import judgment as J
    picks: list[Optional[int]] = []
    first, toks = "", 0
    for i in range(max(1, st.samples)):
        temp = 0.7 if st.samples > 1 else 0.2
        if st.style == "think":
            raw = str(llm.chat(msgs, max_tokens=st.max_tokens, temperature=0.6, seed=i, timeout=600.0))
            reply = G.THINK_BLOCK.sub("", raw).strip()
            if "<think>" in reply:                                     # ran out of tokens while thinking: only an answer after it counts
                reply = ""
        else:
            raw = reply = J.chat_text(llm, msgs, max_tokens=st.max_tokens, temperature=temp, seed=i, timeout=300.0)
        first = first or reply
        toks += (sum(len(m["content"]) for m in msgs) + len(raw)) // 4
        picks.append(parse_choice(reply))
    return vote(picks), picks, first, toks


# ------------------------------------------------------------------------------------------------ records, epochs, halving
def path(state: Path) -> Path:
    return Path(state) / "thinking" / "reasoning.jsonl"


def bank_path(state: Path) -> Path:
    return Path(state) / "thinking" / "trace_bank.jsonl"


TRACE_TOKENS = 260
_BANK: dict[str, Any] = {"key": None, "rows": {}}


def trace_bank(state: Path) -> dict[str, dict[str, Any]]:
    """qid -> banked worked example: a correct, outcome-blind reasoning trace written by a stronger model on a GPU pulse (creator.gpupulse.traces).
    The retrieval strategies show these beside Nupen's own solved questions, under the same rules (other repository or strictly earlier, never
    a shared commit, answered before now). Cached by file size and mtime."""
    p = bank_path(state)
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (str(p), st.st_size, st.st_mtime_ns)
    if _BANK["key"] != key:
        _BANK["rows"] = {str(r["qid"]): r for r in _jsonl(p) if r.get("qid") and r.get("trace")}
        _BANK["key"] = key
    return dict(_BANK["rows"])


def trace_ok(reply: str) -> bool:
    """A usable worked example: some reasoning before the final answer line (not just 'ANSWER: B') and no talk of the outcome."""
    body = ANSWER.split(reply or "")[0].strip()
    return len(body) >= 20 and parse_choice(reply) is not None and "correct answer" not in body.lower()


def bank_add(state: Path, q: Question, trace: str, model: str, pulse: str) -> None:
    p = bank_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"qid": q.qid, "kind": q.kind, "source": q.source, "t": q.t, "model": model, "gpu_pulse": pulse, "trace": trace.strip()[-1200:],
           "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "ts": round(time.time(), 1)}
    with _BANK_LOCK, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


_BANK_LOCK = threading.Lock()


def status_path(state: Path) -> Path:
    return Path(state) / "thinking" / "reasoning_status.json"


def _jsonl(p: Path) -> list[dict[str, Any]]:
    from creator import reasonmethods as RM
    return RM._jsonl(p)


def _append(state: Path, row: dict[str, Any]) -> None:
    p = path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _model_rows(rows: Sequence[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("model") == tag and r.get("qid") and r.get("strategy")]


def epoch_pool(e: int, prev_alive: Sequence[str], prev_acc: dict[str, float]) -> list[str]:
    """Epoch 0: every strategy. Later: the control, the previous epoch's winners, and CHALLENGERS rotated back in from the losers."""
    names = [s.name for s in STRATEGIES if s.name != CONTROL]
    if e == 0:
        return [CONTROL] + [n for n in names if not BY_NAME[n].late]
    win = [n for n in sorted(prev_alive, key=lambda n: -prev_acc.get(n, 0.0)) if n != CONTROL][:CARRY]
    losers = sorted((n for n in names if n not in win), key=lambda n: not BY_NAME[n].late)      # the never-tried expensive ones first
    ch = [losers[((e - 1) * CHALLENGERS + i) % len(losers)] for i in range(min(CHALLENGERS, len(losers)))] if losers else []
    return [CONTROL] + win + [n for n in dict.fromkeys(ch) if n not in win]


def carried(prev_alive: Sequence[str], prev_acc: dict[str, float]) -> list[str]:
    """The previous epoch's winners that the next epoch re-tests (epoch_pool's 'win')."""
    return [n for n in sorted(prev_alive, key=lambda n: -prev_acc.get(n, 0.0)) if n != CONTROL][:CARRY]


def online_epoch(pool: Sequence[str], rows: Sequence[dict[str, Any]]) -> tuple[list[str], dict[str, float], list[str], dict[str, list[float]]]:
    """An online epoch's standing: (alive = control + not raced out, best posterior first; posterior-mean accuracy of the strategies with
    ONLINE_MIN_N answers; raced out; per strategy [mean paired gain over the control, se, n])."""
    from creator import registry as REG
    te = REG.get("trialerror")
    by: dict[str, dict[str, int]] = {}
    for r in rows:
        by.setdefault(r["strategy"], {})[r["qid"]] = int(r.get("correct") or 0)
    ctl = by.get(CONTROL, {})
    gains = {n: [float(v - ctl[q]) for q, v in by.get(n, {}).items() if q in ctl] for n in pool if n != CONTROL}
    raced = [n for n, d in gains.items() if te.raced_out(d)]
    acc = {n: (sum(m.values()) + 1.0) / (len(m) + 2.0) for n, m in by.items() if n in pool and len(m) >= ONLINE_MIN_N}
    rest = sorted((n for n in pool if n != CONTROL and n not in raced), key=lambda n: -acc.get(n, -1.0))
    return [CONTROL] + rest, acc, raced, {n: [round(x, 4) for x in te.mean_se(d)] + [len(d)] for n, d in gains.items() if d}


def online_picks(pool: Sequence[str], floor: Sequence[str], rows: Sequence[dict[str, Any]], raced: Sequence[str], e: int) -> list[str]:
    """The strategies one new question of an online epoch gets: control + floor + ONLINE_PICKS top-two Thompson picks (Beta posteriors),
    seeded by the epoch and its number of answers (the same records give the same picks)."""
    from creator import registry as REG
    te = REG.get("trialerror")
    k: dict[str, int] = {}
    n: dict[str, int] = {}
    for r in rows:
        n[r["strategy"]] = n.get(r["strategy"], 0) + 1
        k[r["strategy"]] = k.get(r["strategy"], 0) + int(r.get("correct") or 0)
    fixed = list(dict.fromkeys([CONTROL, *floor]))
    cand = [x for x in pool if x not in fixed and x not in raced]

    def beta(name: str) -> Any:
        return lambda rnd: te.beta_sample(k.get(name, 0), n.get(name, 0), rnd)
    return fixed + te.thompson_order({x: beta(x) for x in cand}, te.seeded("reason", e, len(rows)), ONLINE_PICKS)


def halve(pool: Sequence[str], rows: Sequence[dict[str, Any]]) -> tuple[list[str], dict[str, float]]:
    """Successive halving inside one epoch: compared on the questions every surviving strategy answered; accuracy, ties to the cheaper.
    The control is never cut. Returns (alive, accuracy on the last common set)."""
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["strategy"], {})[r["qid"]] = r
    live = list(pool)
    acc: dict[str, float] = {}

    def score(n: str, common: set[str]) -> tuple[float, float]:
        rs = by.get(n, {})
        return (sum(int(rs[k].get("correct") or 0) for k in common) / max(1, len(common)),
                -sum(float(rs[k].get("seconds") or 0) for k in common if not rs[k].get("gpu_pulse")) / max(1, len(common)))
    for cutoff, keep in HALVING:
        rest = [n for n in live if n != CONTROL]
        if len(rest) <= keep:
            continue
        common = set.intersection(*[set(by.get(n, {})) for n in live]) if live else set()
        if len(common) < cutoff:
            break
        acc = {n: score(n, common)[0] for n in live}
        rest.sort(key=lambda n: score(n, common), reverse=True)
        live = [n for n in live if n == CONTROL or n in rest[:keep]]
    common = set.intersection(*[set(by.get(n, {})) for n in live]) if live else set()
    if common:
        acc = {n: score(n, common)[0] for n in live}
    return live, acc


def epochs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay the epochs from the records: pool, alive, questions, accuracy - the current epoch last."""
    out: list[dict[str, Any]] = []
    top = max((int(r.get("epoch", 0)) for r in rows), default=0)
    prev_alive: list[str] = []
    prev_acc: dict[str, float] = {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(int(r.get("epoch", 0)), []).append(r)
    for e in range(top + 1):
        pool = epoch_pool(e, prev_alive, prev_acc)
        floor = carried(prev_alive, prev_acc) if e else []
        er = [r for r in grouped.get(e, []) if r["strategy"] in pool]
        qids = list(dict.fromkeys(r["qid"] for r in er))
        if any(r.get("alloc") == "online" for r in er):
            live, acc, raced, gains = online_epoch(pool, er)
            ctl = {r["qid"] for r in er if r["strategy"] == CONTROL}
            given: dict[str, set[str]] = {}
            got: dict[str, set[str]] = {}
            for r in er:
                given.setdefault(r["qid"], set()).update(r.get("arms") or [])
                got.setdefault(r["qid"], set()).add(r["strategy"])
            out.append({"epoch": e, "pool": pool, "alive": live, "questions": qids, "accuracy": acc, "allocation": "online", "floor": floor,
                        "raced": raced, "gains_vs_control": gains,
                        "complete": len(ctl) >= EPOCH_Q and all(given[q] <= got[q] for q in given)})
        else:
            live, acc = halve(pool, er)
            have = {(r["qid"], r["strategy"]) for r in er}
            out.append({"epoch": e, "pool": pool, "alive": live, "questions": qids, "accuracy": acc, "floor": floor,
                        "complete": len(qids) >= EPOCH_Q and all((q, n) in have for q in qids for n in live)})
        prev_alive, prev_acc = live, acc
    return out


# ------------------------------------------------------------------------------------------------ the filler
class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


_NOLOCK = _NoLock()


class _Skip(Exception):
    """The thinking model has no RAM right now: the batch is dropped and offered again later."""


def active_tag() -> str:
    from creator import judgment as J
    return J.active_tag()


def write_status(state: Path, qs: Sequence[Question], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    done = {r["qid"] for r in rows}
    fresh = [q for q in qs if q.qid not in done]
    st = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "available": len(qs), "answered": len(done & {q.qid for q in qs}),
          "fresh": len(fresh), "low": len(fresh) < LOW_FRESH, "need_acquisition": len(fresh) < LOW_FRESH,
          "by_kind": {k: sum(1 for q in qs if q.kind == k) for k in KINDS},
          "by_source": {s: sum(1 for q in qs if q.source == s) for s in sorted({q.source for q in qs})},
          "by_difficulty": {d: sum(1 for q in qs if q.difficulty == d) for d in ("easy", "hard")}}
    p = status_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8")
    return st


def run_batch(state: Path, batch: Sequence[tuple[Question, str, int]], make_llm: Callable[[], Any], solved: Sequence[tuple[Question, Optional[int], float]],
              traces: Optional[Mapping[str, str]] = None, extra: Optional[dict[str, Any]] = None) -> int:
    """Ask each (question, strategy, epoch) under one server lease; the correct answer is attached to the record only after the answer exists."""
    n = 0
    with make_llm() as llm:
        tag = Path(str(getattr(llm, "model", ""))).name or active_tag()
        for q, name, e in batch:
            st = BY_NAME[name]
            now = time.time()
            ex = solved_examples(q, solved, st.retrieve, now)
            t0 = time.monotonic()
            pick, picks, first, toks = ask(llm, st, build_messages(st, q, ex, traces))
            row = {"qid": q.qid, "kind": q.kind, "source": q.source, "difficulty": q.difficulty, "strategy": name, "epoch": e, "model": tag,
                   "pick": pick, "correct": int(pick == q.answer), "lexical_correct": int(lexical_pick(q) == q.answer),
                   "seconds": round(time.monotonic() - t0, 2), "tokens": toks, "examples": [s.qid for s, _p in ex],
                   "reply": first[-160:], "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "ts": round(now, 1)}
            if len(picks) > 1:
                row["picks"] = picks
            row.update({k: v for k, v in (extra or {}).items() if k != "arms_by_qid"})
            arms = ((extra or {}).get("arms_by_qid") or {}).get(q.qid)
            if arms:                                            # online epochs: the strategies this question was given
                row["arms"] = arms
            if traces and any(s.qid in traces for s, _p in ex):
                row["bank_examples"] = sum(1 for s, _p in ex if s.qid in traces)
            if isinstance(getattr(llm, "pulse", None), str) and llm.pulse:     # answered on a rented GPU (creator.gpupulse): timings kept apart
                row["gpu_pulse"] = llm.pulse
            _append(state, row)
            n += 1
    return n


def reasoning_filler(state: Path, repo: Path, max_servers: int = 1, llm_factory: Optional[Callable[[], Any]] = None,
                     public_dir: Optional[Path] = None, questions: Optional[Callable[[], list[Question]]] = None, tag: Optional[str] = None,
                     ) -> Callable[[], Optional[Callable[[], None]]]:
    """next_job() hands out one batch of BATCH (question, strategy) calls of the current epoch; None when `max_servers` batches are in flight.
    The question pool is re-read every REFRESH_S (new repositories and fetched commits become new questions); never idles silently: when fresh
    questions run low the status file says need_acquisition. `tag` = the model whose records count (default: the active thinking model)."""
    state = Path(state)
    lock = threading.Lock()
    flight = {"n": 0}
    taken: set[tuple[str, str]] = set()                         # (qid, strategy) handed out
    assigned: dict[str, int] = {}                               # qid -> epoch it was handed out in (until its records exist)
    cache: dict[str, Any] = {"at": -1e18, "qs": [], "by": {}}

    def make_llm() -> Any:
        if llm_factory is not None:
            return llm_factory()
        from creator import generator as G
        lm = G.thinker()
        if Path(str(lm.model)).name != active_tag():
            raise _Skip()
        return lm

    def refresh() -> None:
        try:
            qs = questions() if questions is not None else generate(repos(Path(repo), public_dir), state)
            oq = order(qs)
            with lock if questions is None else _NOLOCK:
                cache.update(at=time.time(), qs=oq, by={q.qid: q for q in qs})
            write_status(state, qs, _jsonl(path(state)))
        except Exception:                                          # noqa: BLE001 - a failed re-read keeps the old pool
            cache["at"] = time.time()
        finally:
            cache["busy"] = False

    def pool() -> list[Question]:
        """The question pool; re-read every REFRESH_S. Reading ~15 git histories takes seconds, so in the swarm it happens on a thread (the
        swarm never waits; until the first read is done there is simply no reasoning job)."""
        if time.time() - cache["at"] >= REFRESH_S and not cache.get("busy"):
            cache["busy"] = True
            if questions is not None:
                refresh()
            else:
                threading.Thread(target=refresh, daemon=True, name="reasondrills-refresh").start()
        return list(cache["qs"])

    arms_of: dict[str, list[str]] = {}                          # online epochs: qid -> the strategies it was given (fixed once handed out)
    mode = {"online": False}

    def pending_online(rows: Sequence[dict[str, Any]], eps: list[dict[str, Any]]) -> list[tuple[Question, str, int]]:
        cur = eps[-1]
        if cur["complete"] or not any(r.get("alloc") == "online" for r in rows if int(r.get("epoch", 0)) == int(cur["epoch"])):
            if cur["complete"]:
                e, pl, floor = int(cur["epoch"]) + 1, epoch_pool(int(cur["epoch"]) + 1, cur["alive"], cur["accuracy"]), carried(cur["alive"], cur["accuracy"])
            else:                                               # a fresh (empty) epoch starts online
                e, pl, floor = int(cur["epoch"]), list(cur["pool"]), list(cur.get("floor") or [])
            raced: list[str] = []
        else:
            e, pl, floor, raced = int(cur["epoch"]), list(cur["pool"]), list(cur.get("floor") or []), list(cur.get("raced") or [])
        er = [r for r in rows if int(r.get("epoch", 0)) == e and r["strategy"] in pl]
        done = {(r["qid"], r["strategy"]) for r in rows}
        out: list[tuple[Question, str, int]] = []
        qids = list(dict.fromkeys([r["qid"] for r in er] + [q for q, ep in assigned.items() if ep == e]))
        stored: dict[str, list[str]] = {}
        for r in er:
            stored.setdefault(r["qid"], []).extend(a for a in r.get("arms") or [] if a not in stored.get(r["qid"], []))
        for qid in qids:                                        # handed-out questions: only the strategies they were given
            q = cache["by"].get(qid)
            if q is not None:
                out += [(q, n, e) for n in (arms_of.get(qid) or stored.get(qid, [])) if (qid, n) not in done and (qid, n) not in taken]
            if len(out) >= BATCH:
                return out
        used = {r["qid"] for r in rows} | set(assigned)
        nctl = len({r["qid"] for r in er if r["strategy"] == CONTROL} | {q for q, ep in assigned.items() if ep == e})
        for q in pool():
            if nctl >= EPOCH_Q or len(out) >= BATCH:
                break
            if q.qid not in used:
                assigned[q.qid] = e
                arms_of[q.qid] = online_picks(pl, floor, er, raced, e)
                nctl += 1
                out += [(q, n, e) for n in arms_of[q.qid]]
        if not out and nctl < EPOCH_Q:                          # every question used: REVISIT the ones the control got wrong (as in halving)
            for r in rows:
                q = cache["by"].get(r["qid"])
                rq = f"{r['qid']}{REVISIT}"
                if q is None or r["strategy"] != CONTROL or r.get("correct") or REVISIT in r["qid"] or rq in used:
                    continue
                q2 = dataclasses.replace(q, qid=rq)
                cache["by"][rq] = q2
                assigned[rq] = e
                arms_of[rq] = online_picks(pl, floor, er, raced, e)
                used.add(rq)
                nctl += 1
                out += [(q2, n, e) for n in arms_of[rq]]
                if nctl >= EPOCH_Q or len(out) >= BATCH:
                    break
        return out

    def pending(rows: Sequence[dict[str, Any]]) -> list[tuple[Question, str, int]]:
        qs = pool()
        eps = epochs(rows)
        cur = eps[-1]
        cur_rows = [r for r in rows if int(r.get("epoch", 0)) == int(cur["epoch"])]
        legacy_open = not cur["complete"] and cur_rows and not any(r.get("alloc") == "online" for r in cur_rows)
        mode["online"] = REASON_ALLOCATION == "online" and not legacy_open      # an unfinished halving epoch finishes as it began
        if mode["online"]:
            return pending_online(rows, eps)
        e, live, qids = int(cur["epoch"]), list(cur["alive"]), list(cur["questions"])
        if cur["complete"]:                                     # the next epoch: winners re-tested on fresh questions, challengers back in
            e, live, qids = e + 1, epoch_pool(e + 1, cur["alive"], cur["accuracy"]), []
        qids += [q for q, ep in assigned.items() if ep == e and q not in qids]
        done = {(r["qid"], r["strategy"]) for r in rows}
        out: list[tuple[Question, str, int]] = []
        for qid in qids:
            q = cache["by"].get(qid)
            if q is not None:
                out += [(q, n, e) for n in live if (qid, n) not in done and (qid, n) not in taken]
            if len(out) >= BATCH:
                return out
        used = {r["qid"] for r in rows} | set(assigned)
        for q in qs:
            if len(qids) >= EPOCH_Q or len(out) >= BATCH:
                break
            if q.qid not in used:
                assigned[q.qid] = e
                qids.append(q.qid)
                out += [(q, n, e) for n in live]
        if not out and len(qids) < EPOCH_Q:                     # every question used: REVISIT the ones the control got wrong (a second pass;
            for r in rows:                                      # the report keeps it apart - it is not fresh) until acquisition brings more
                q = cache["by"].get(r["qid"])
                rq = f"{r['qid']}{REVISIT}"
                if q is None or r["strategy"] != CONTROL or r.get("correct") or REVISIT in r["qid"] or rq in used:
                    continue
                q2 = dataclasses.replace(q, qid=rq)
                cache["by"][rq] = q2
                assigned[rq] = e
                qids.append(rq)
                out += [(q2, n, e) for n in live]
                if len(qids) >= EPOCH_Q or len(out) >= BATCH:
                    break
        return out

    def next_job() -> Optional[Callable[[], None]]:
        with lock:
            if flight["n"] >= max_servers:
                return None
            try:
                rows = _model_rows(_jsonl(path(state)), tag or active_tag())
                todo = pending(rows)
            except Exception:                                      # noqa: BLE001 - no data / broken source: nothing to do now
                return None
            if not todo:
                return None
            batch = todo[:BATCH]
            for q, n, _e in batch:
                taken.add((q.qid, n))
            flight["n"] += 1
            by = cache["by"]
            mine = {r["qid"]: (by[r["qid"]], r.get("pick"), float(r.get("ts") or 0)) for r in rows if r["qid"] in by and r["strategy"] == CONTROL}
            bank = {k: v for k, v in trace_bank(state).items() if k in by}       # correct worked examples a GPU pulse banked (creator.gpupulse)
            mine.update({k: (by[k], by[k].answer, float(v.get("ts") or 0)) for k, v in bank.items()})
            solved = list(mine.values())
            traces = {k: str(v.get("trace") or "") for k, v in bank.items()}

        extra: Optional[dict[str, Any]] = None
        if mode["online"]:
            extra = {"alloc": "online", "arms_by_qid": {q.qid: arms_of[q.qid] for q, _n, _e in batch if q.qid in arms_of}}

        def job() -> None:
            try:
                run_batch(state, batch, make_llm, solved, traces=traces, extra=extra)
            except Exception as e:                                 # noqa: BLE001 - a filler never stops the swarm; _Skip = no RAM now
                with lock:
                    for q, n, _e in batch:
                        taken.discard((q.qid, n))
                if not isinstance(e, _Skip):
                    _append(state, {"error": f"{type(e).__name__}: {e}"[:300], "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
            finally:
                with lock:
                    flight["n"] -= 1
        return job
    return next_job


def online_report(state: Path, tag: Optional[str] = None) -> dict[str, Any]:
    """For the experiments section: the current epoch's allocation (pool, floor, raced out, paired gains over the control [mean, se, n])."""
    rows = [r for r in _model_rows(_jsonl(path(state)), tag or active_tag()) if REVISIT not in r["qid"]]
    if not rows:
        return {}
    e = epochs(rows)[-1]
    return {k: e.get(k) for k in ("epoch", "allocation", "pool", "floor", "alive", "raced", "gains_vs_control", "complete")} | {
        "questions": len(e["questions"])}


# ------------------------------------------------------------------------------------------------ the report
def _cpu_per_item(rows: Sequence[dict[str, Any]]) -> Optional[float]:
    cpu = [r for r in rows if not r.get("gpu_pulse")]
    return round(sum(float(r.get("seconds") or 0) for r in cpu) / len(cpu), 2) if cpu else None


def wilson(k: int, n: int) -> list[Optional[float]]:
    if n == 0:
        return [None, None, None]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def paired(a: Sequence[int], b: Sequence[int]) -> list[float]:
    """Mean of b - a on the same questions and its 95% CI (normal; positive = b is better)."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return [round(sum(d) / n, 4) if n else 0.0, 0.0, 0.0, n]
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return [round(m, 4), round(m - 1.96 * se, 4), round(m + 1.96 * se, 4), n]


def _by_kind(m: dict[str, int], kind: dict[str, str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for kd in KINDS:
        qs = [q for q in m if kind[q] == kd]
        if qs:
            out[kd] = [wilson(sum(m[q] for q in qs), len(qs))[0], len(qs)]
    return out


def report_section(state: Path, tag: Optional[str] = None) -> dict[str, Any]:
    """The 'reasoning' section for creator.thinking's report: per strategy accuracy, n, Wilson CI vs chance 0.25, paired gain vs the plain
    control, the lexical control, the epochs (pool, survivors, out-of-sample re-test of the carried winners: the improvement curve) and the
    fresh-question supply. Pure measurement: it enables nothing."""
    rows_all = _jsonl(path(state))
    tag = tag or active_tag()
    rows = _model_rows(rows_all, tag)
    rev = [r for r in rows if REVISIT in r["qid"]]
    rows = [r for r in rows if REVISIT not in r["qid"]]                # accuracy is measured on FRESH questions only
    out: dict[str, Any] = {"model": tag, "answers": len(rows), "errors": sum(1 for r in rows_all if r.get("error")), "chance": 0.25}
    if rev:
        out["revisits"] = {n: wilson(sum(int(r.get("correct") or 0) for r in rev if r["strategy"] == n), sum(1 for r in rev if r["strategy"] == n))
                           for n in sorted({r["strategy"] for r in rev})}
    try:
        out["supply"] = json.loads(status_path(state).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        out["supply"] = {"available": None, "need_acquisition": None}
    if out["supply"].get("available") is not None:                     # the status file is written at each re-read; answers since then count
        fresh = max(0, int(out["supply"]["available"]) - len({r["qid"] for r in rows}))
        out["supply"].update(fresh=fresh, low=fresh < LOW_FRESH, need_acquisition=fresh < LOW_FRESH)
    if not rows:
        return out
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        latest[(r["qid"], r["strategy"])] = r
    per: dict[str, dict[str, int]] = {}
    for (qid, n), r in latest.items():
        per.setdefault(n, {})[qid] = int(r.get("correct") or 0)
    lex = {qid: int(r.get("lexical_correct") or 0) for (qid, _n), r in latest.items()}
    strat: dict[str, Any] = {}
    for n, m in sorted(per.items()):
        k = sum(m.values())
        a = wilson(k, len(m))
        both = sorted(set(m) & set(per.get(CONTROL, {})))
        rs = [latest[(q, n)] for q in m]
        strat[n] = {"n": len(m), "accuracy": a, "beats_chance": bool(a[1] is not None and a[1] > 0.25),
                    "gain_vs_plain": paired([per[CONTROL][q] for q in both], [m[q] for q in both]) if n != CONTROL and both else None,
                    "gain_vs_lexical": paired([lex[q] for q in m], [m[q] for q in m]),
                    "seconds_per_item": _cpu_per_item(rs),                     # this PC only (GPU pulse rows: creator.gpupulse)
                    "by_kind": _by_kind(m, {q: str(latest[(q, n)].get("kind")) for q in m})}
    out["strategies"] = strat
    out["lexical_control"] = wilson(sum(lex.values()), len(lex))
    eps = epochs(rows)
    curve = []
    for i, e in enumerate(eps):
        er = [r for r in rows if int(r.get("epoch", 0)) == e["epoch"]]
        accs = {n: wilson(sum(int(r.get("correct") or 0) for r in er if r["strategy"] == n), sum(1 for r in er if r["strategy"] == n))[0]
                for n in e["pool"]}
        item = {"epoch": e["epoch"], "pool": e["pool"], "alive": e["alive"], "questions": len(e["questions"]), "complete": e["complete"],
                "accuracy": accs, "allocation": e.get("allocation", "halving")}
        if e.get("allocation") == "online":
            item.update(raced=e.get("raced"), gains_vs_control=e.get("gains_vs_control"), floor=e.get("floor"))
        if i:                                                              # carried winners re-tested on questions they never saw
            carried = [n for n in eps[i - 1]["alive"] if n != CONTROL and n in e["pool"]]
            item["out_of_sample"] = {n: {"before": eps[i - 1]["accuracy"].get(n), "now": accs.get(n)} for n in carried}
        curve.append(item)
    out["epochs"] = curve
    out["curve"] = [{"epoch": e["epoch"], "plain": e["accuracy"].get(CONTROL),
                     "best": max(((n, a) for n, a in e["accuracy"].items() if n != CONTROL), key=lambda x: x[1], default=None)}
                    for e in eps if e["accuracy"]]                     # accuracy on each epoch's common questions: the improvement curve
    out["current_alive"] = eps[-1]["alive"]
    return out
