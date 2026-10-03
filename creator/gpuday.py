"""GPU DAY PREP (owner, 3 Oct 2026: a 24-hour RTX 4090 rental where no paid minute goes to setup). Loaded ON DEMAND only
(scripts/gpuday/*.py); never part of the swarm's eager start load.

This module turns Nupen's own records into training/evaluation data that may leave the PC, and plans the day:

  export       (a) coder SFT pairs: the task text the worker got (.creator_task.md) -> the change the kernel ADOPTED, as search/replace
               edits in the creator.generator.EDIT_RE format (so a fine-tuned coder plugs straight into the model student / harness);
               (b) preference pairs: same requirement -> adopted vs rejected/failed answer; (c) worked-example SFT for the thinker from the
               worked bank the GPU runner writes (WORKED_BANK_FIELDS); plus public-commit SFT (Nupen's own public git history: commit
               message -> its diff) as a separate, clearly lower tier.
  splits       time-ordered held-out splits (every training example is strictly EARLIER than every eval example; a package/requirement
               never straddles the cut) for handoffs, reasoning questions and judgment cases.
  plan         the day's job list with time/cost per block and what comes home.

PRIVACY (the repo and the exports are public material): every exported text passes `private_reason` (owner journal / Masterstock,
state/livesim data, the ~/oldpc projects, secrets/keys/tokens) - a row that trips it is DROPPED, never redacted into training data - and
`scrub` (author names, e-mail addresses, home-directory paths). FROZEN BENCHMARK: the thinkbench items (state/creator/thinkbench/items.json,
hash FROZEN_HASH) never become training data: their verdict subjects (packages whose outcome IS the label), their commit subjects, their
journal subjects, their question prompts and their planning contexts are excluded (`Frozen`); with strict=True the planning candidates'
packages are excluded too. Chat rows are {"messages": [...]} (accepted by Unsloth/TRL SFTTrainer and by Axolotl type chat_template);
preference rows are TRL's conversational {"prompt", "chosen", "rejected"} (Unsloth DPOTrainer/ORPOTrainer); ids live in sidecar files."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASH = "1f520b202ac4"                    # thinkbench items.json 'hash' (prefix) frozen 3 Oct 2026 06:04 UTC

# ------------------------------------------------------------------------------------------------ privacy
PRIVATE_MARKERS = ("Masterstock", "MASTERSTOCK", "Latest owner directives", "owner journal", "JOURNAL.md", "livesim", "oldpc",
                   "BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "PRIVATE KEY-----", "OPEN_BUTTON_TOKEN", "pulse.json",
                   "Google Voice")
MARKER_KIND = {"Masterstock": "owner notes", "MASTERSTOCK": "owner notes", "Latest owner directives": "owner notes",
               "owner journal": "owner notes", "JOURNAL.md": "owner notes", "livesim": "live trading state", "oldpc": "old-PC projects",
               "Google Voice": "owner contact", "pulse.json": "credentials", "OPEN_BUTTON_TOKEN": "credentials"}
SECRET_RES = (re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"), re.compile(r"\bhf_[A-Za-z0-9]{20,}"), re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
              re.compile(r"\bAKIA[0-9A-Z]{16}\b"), re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[=:]\s*['\"][^'\"\s]{12,}['\"]"))
# harmless mentions of private NAMES inside Nupen's own prompts (the protected-path list) are removed before the marker check
SCRUB_FIRST = ((re.compile(r",?\s*state/livesim/\*"), ""),)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NAME_RES = ((re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s'\"]+"), "~"), (re.compile(r"/c/Users/[^/\s'\"]+"), "~"),
            (re.compile(r"\bpeterdax006(?:-glitch)?\b"), "owner"), (re.compile(r"\bPeter\b"), "the owner"), (re.compile(r"\bDax\b"), "the owner"))


def private_reason(text: str) -> str:
    """'' when `text` may leave the PC; otherwise what makes it private (the caller drops the row)."""
    t = text
    for rx, rep in SCRUB_FIRST:
        t = rx.sub(rep, t)
    for k in PRIVATE_MARKERS:
        if k in t:
            return f"marker {k!r}"
    for rx in SECRET_RES:
        if rx.search(t):
            return "secret-like token"
    return ""


def scrub(text: str) -> str:
    """Remove author names, e-mail addresses and home-directory paths (and the protected-path mention of the sealed live state)."""
    t = text
    for rx, rep in SCRUB_FIRST:
        t = rx.sub(rep, t)
    t = EMAIL_RE.sub("<email>", t)
    for rx, rep in NAME_RES:
        t = rx.sub(rep, t)
    return t


def scrub_obj(obj: Any) -> Any:
    """`scrub` applied to every string inside a JSON-like value."""
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, list):
        return [scrub_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------------------------------------------ the frozen benchmark
@dataclasses.dataclass
class Frozen:
    """Everything of the frozen thinkbench that must never be trained on."""
    packages: set[str]                 # verdict subjects (+ planning candidates when strict)
    commits: set[str]                  # git_fixed / git_churn subjects (10-char prefixes)
    subjects: set[str]                 # every subject / id (journal ids, qids ...)
    texts: list[str]                   # question prompts and planning contexts (exact text never appears in an export)
    hash: str = ""

    @classmethod
    def load(cls, path: Path, strict: bool = False, expect: str = FROZEN_HASH) -> "Frozen":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        h = str(d.get("hash", ""))
        if expect and not h.startswith(expect):
            raise ValueError(f"frozen thinkbench hash {h[:12]} != {expect}: refusing to export against an unknown benchmark")
        pk: set[str] = set()
        cm: set[str] = set()
        sj: set[str] = set()
        tx: list[str] = []
        for part in ("a", "b", "c", "d"):
            for it in d.get(part, []) or []:
                src = str(it.get("source") or it.get("topic") or it.get("kind") or "")
                s = str(it.get("subject") or "")
                sj.update(x for x in (s, str(it.get("id", ""))) if x)
                if src == "verdict" and s:
                    pk.add(s)
                if src.startswith("git") and s:
                    cm.add(s[:10])
                for k in ("prompt", "context"):
                    if isinstance(it.get(k), str) and len(it[k]) >= 40:
                        tx.append(it[k])
                for c in it.get("candidates", []) or []:
                    if strict and c.get("pkg"):
                        pk.add(str(c["pkg"]))
        return cls(pk, cm, sj, tx, h)

    @classmethod
    def empty(cls) -> "Frozen":
        return cls(set(), set(), set(), [])

    def leak(self, *, package: str = "", commit: str = "", subject: str = "", text: str = "") -> str:
        """'' when allowed; otherwise why the row touches the frozen benchmark."""
        if package and package in self.packages:
            return f"frozen package {package}"
        if commit and commit[:10] in self.commits:
            return f"frozen commit {commit[:10]}"
        if subject and subject in self.subjects:
            return f"frozen subject {subject}"
        if text:
            for t in self.texts:
                if t in text or (len(t) > 200 and t[:200] in text):
                    return "frozen item text"
        return ""


# ------------------------------------------------------------------------------------------------ diffs -> search/replace edits
@dataclasses.dataclass
class FileDiff:
    path: str
    status: str                                   # A added, M modified, D deleted, R renamed/binary (not expressible)
    hunks: list[tuple[list[str], list[str]]]      # (old lines incl. context, new lines incl. context)


def parse_diff(patch: str) -> list[FileDiff]:
    out: list[FileDiff] = []
    cur: Optional[FileDiff] = None
    old: list[str] = []
    new: list[str] = []
    in_hunk = False

    def close_hunk() -> None:
        nonlocal old, new
        if cur is not None and (old or new):
            cur.hunks.append((old, new))
        old, new = [], []
    for ln in patch.split("\n"):
        if ln.startswith("diff --git "):
            close_hunk()
            m = re.match(r"diff --git a/(.+?) b/(.+)$", ln)
            cur = FileDiff(m.group(2) if m else ln[11:], "M", [])
            out.append(cur)
            in_hunk = False
            continue
        if cur is None:
            continue
        if not in_hunk:
            if ln.startswith("new file mode"):
                cur.status = "A"
            elif ln.startswith("deleted file mode"):
                cur.status = "D"
            elif ln.startswith(("rename from", "Binary files", "GIT binary patch")):
                cur.status = "R"
        if ln.startswith("@@"):
            close_hunk()
            in_hunk = True
            continue
        if not in_hunk or ln.startswith("\\"):
            continue
        if ln.startswith("+"):
            new.append(ln[1:])
        elif ln.startswith("-"):
            old.append(ln[1:])
        elif ln.startswith(" ") or ln == "":
            old.append(ln[1:])
            new.append(ln[1:])
    close_hunk()
    for fd in out:                                # a trailing '' from the final newline is not a context line
        fd.hunks = [(o[:-1], n[:-1]) if o and n and o[-1] == "" and n[-1] == "" else (o, n) for o, n in fd.hunks]
    return out


def diff_to_edits(patch: str, py_only: bool = True) -> tuple[str, list[str]]:
    """(edits text in the EDIT_RE format, paths it covers). Deleted/renamed/binary files and (py_only) non-.py files are left out."""
    parts: list[str] = []
    paths: list[str] = []
    for fd in parse_diff(patch):
        if fd.status in ("D", "R") or (py_only and not fd.path.endswith(".py")):
            continue
        if fd.status == "A":
            body = "\n".join(n for _, nn in fd.hunks for n in nn)
            parts.append(f"FILE: {fd.path}\n<<<<<<< SEARCH\n\n=======\n{body.rstrip()}\n>>>>>>> REPLACE\n")
        else:
            for o, n in fd.hunks:
                parts.append(f"FILE: {fd.path}\n<<<<<<< SEARCH\n" + "\n".join(o) + "\n=======\n" + "\n".join(n) + "\n>>>>>>> REPLACE\n")
        paths.append(fd.path)
    return "".join(parts), paths


# ------------------------------------------------------------------------------------------------ the coder prompt (shared with the harness)
CODER_SYSTEM = ("You are Nupen's coder. You get one work package of a Python project (the Creator) and the current text of the files it "
                "names. Reply with one short line starting 'REASONING:' saying what you change and why, then ONLY search/replace edits, "
                "each as:\nFILE: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n<replacement lines>\n>>>>>>> REPLACE\n"
                "An empty SEARCH creates the file (or replaces it whole). SEARCH text must match the file exactly and be unique. "
                "Write the tests the package asks for; never weaken, skip or delete existing tests. Change as little as possible.")
MAX_FILE_CHARS = 9000


def coder_prompt(task_text: str, files: Mapping[str, str], max_file_chars: int = MAX_FILE_CHARS) -> str:
    parts = [f"Your task:\n{task_text.strip()}"]
    for rel, body in files.items():
        if rel.endswith(".py"):
            parts.append(f"FILE: {rel}\n```python\n{body[:max_file_chars]}\n```" if body else f"FILE: {rel}\n(does not exist yet)")
    return "\n\n".join(parts)


def coder_messages(task_text: str, files: Mapping[str, str], reply: Optional[str] = None) -> list[dict[str, str]]:
    m = [{"role": "system", "content": CODER_SYSTEM}, {"role": "user", "content": coder_prompt(task_text, files)}]
    if reply is not None:
        m.append({"role": "assistant", "content": reply})
    return m


def render_task(wp: Mapping[str, Any]) -> str:
    """A work package (ledger record data) as the worker saw it (creator.kernel.render_package's verbatim fields; the registry/tool
    lines that render_package adds at run time are not reconstructed here - lessons carry the exact text when they exist)."""
    lines = [f"You are working in a git worktree of a Python project (the Creator). Package {wp.get('package_id', '')}.",
             f"Objective: {wp.get('objective', '')}", f"Why: {wp.get('why_it_exists', '')}", "", "Do:"]
    lines += [f"- {s}" for s in wp.get("implementation_requirements", []) or []]
    lines += ["", "Tests:"] + [f"- {s}" for s in wp.get("test_requirements", []) or []]
    lines += ["", "Known interfaces:"] + [f"- {s}" for s in (wp.get("interfaces", []) or [])[:20]]
    lines += ["", "Ways this commonly goes wrong (avoid them):"] + [f"- {s}" for s in wp.get("expected_failure_modes", []) or []]
    lines += ["", "Done means (computed by the system, not by you):"] + [f"- {s}" for s in wp.get("completion_criteria", []) or []]
    lines += ["", "Rules:", "- Python 3.11.", "- Never weaken, skip or delete tests to make them pass.", "- Keep changes small and focused on this package."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ sources (read-only)
def _ts(s: Any) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if d.tzinfo is None:                          # lessons' 'at' is local wall time without zone: treat as UTC (ordering only)
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def work_packages(ledger: Path) -> dict[str, dict[str, Any]]:
    """package_id -> {'data': WorkPackage data, 'ts': creation time, 'base': git commit at creation}."""
    out: dict[str, dict[str, Any]] = {}
    try:
        fh = Path(ledger).open(encoding="utf-8")
    except OSError:
        return out
    with fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            d = r.get("data") or {}
            pid = d.get("package_id")
            if pid and "objective" in d and pid not in out:
                prov = r.get("provenance") or {}
                out[pid] = {"data": d, "ts": _ts(prov.get("timestamp")), "base": prov.get("git_commit") or ""}
    return out


def lessons(path: Path) -> list[dict[str, Any]]:
    """Teacher/student lesson rows (context = the exact .creator_task.md text) joined with their kernel outcome."""
    rows: list[dict[str, Any]] = []
    outs: dict[str, dict[str, Any]] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    for ln in text.splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("event") == "outcome":
            outs[str(r.get("lesson_id"))] = r
        elif "objective" in r:
            rows.append(r)
    for r in rows:
        o = outs.get(str(r.get("lesson_id")))
        r["outcome"] = "ADOPTED" if o and o.get("adopted") is True else (str(o.get("verdict", "")).split(":")[0] if o else "")
    return rows


def cycles(state: Path) -> list[dict[str, Any]]:
    """state/creator/cycles/CP*/ with a diff: outcome, verdict, requirement, base commit, patch."""
    out = []
    for d in sorted((Path(state) / "cycles").glob("CP*")):
        patch_p, cyc_p, ev_p = d / "diff.patch", d / "cycle.json", d / "evaluation.json"
        if not patch_p.is_file():
            continue
        try:
            c = json.loads(cyc_p.read_text(encoding="utf-8")) if cyc_p.is_file() else {}
            ev = json.loads(ev_p.read_text(encoding="utf-8")) if ev_p.is_file() else {}
        except ValueError:
            continue
        out.append({"package": d.name, "outcome": c.get("outcome") or "", "verdict": c.get("verdict") or "",
                    "requirement": c.get("requirement") or "", "base": ev.get("base") or "",
                    "patch": patch_p.read_text(encoding="utf-8", errors="replace"), "by": ((c.get("details") or {}).get("worker") or {}).get("by", "")})
    return out


def _git(repo: Path, *args: str, timeout: float = 60) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=timeout)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""


def file_at(repo: Path, rev: str, rel: str) -> str:
    return _git(repo, "show", f"{rev}:{rel}") if rev else ""


def commits(repo: Path, rev: str = "HEAD", paths: Sequence[str] = ("creator", "tests"), limit: int = 0) -> list[dict[str, Any]]:
    """Non-merge commits touching `paths`: sha, parent, time, message (author identity is never read)."""
    fmt = "%H%x1f%P%x1f%ct%x1f%B%x1e"
    args = ["log", rev, "--no-merges", f"--format={fmt}"] + (["-n", str(limit)] if limit else []) + ["--", *paths]
    out = []
    for rec in _git(Path(repo), *args, timeout=300).split("\x1e"):
        f = rec.strip("\n").split("\x1f")
        if len(f) < 4 or not f[0]:
            continue
        out.append({"sha": f[0], "parent": f[1].split(" ")[0] if f[1] else "", "ts": float(f[2]), "message": f[3].strip()})
    return out


# ------------------------------------------------------------------------------------------------ examples
@dataclasses.dataclass
class Example:
    id: str
    kind: str                 # sft_handoff | sft_commit | sft_unjudged | pref | worked
    ts: float
    group: str                # package / requirement / question id: never split across train and eval
    row: dict[str, Any]       # the JSONL row itself (messages, or prompt/chosen/rejected)
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def text(self) -> str:
        return json.dumps(self.row, ensure_ascii=False)


@dataclasses.dataclass
class Drops:
    counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def add(self, why: str) -> None:
        k = why.split(" ")[0] if why.startswith(("frozen", "marker")) else why
        k = {"frozen": "frozen benchmark"}.get(k, k)
        if k == "marker":                          # a category, never the marker itself (the manifest travels with the data)
            k = "private: " + MARKER_KIND.get(why.split("'")[1] if "'" in why else "", "other")
        self.counts[k] = self.counts.get(k, 0) + 1


def _admit(ex: Example, frozen: Frozen, drops: Drops, *, package: str = "", commit: str = "", subject: str = "",
           max_chars: int = 0) -> Optional[Example]:
    """Privacy + frozen checks on the FINAL row text; scrubbed copy returned, or None (the reason is counted)."""
    raw = ex.text()
    why = frozen.leak(package=package, commit=commit, subject=subject, text=raw) or private_reason(raw)
    if not why and max_chars and len(raw) > max_chars:
        why = "too long for the context window"
    if why:
        drops.add(why)
        return None
    ex.row = scrub_obj(ex.row)
    return ex


def _files_before(repo: Path, base: str, paths: Sequence[str], added: Sequence[str]) -> dict[str, str]:
    return {p: ("" if p in added else file_at(repo, base, p)) for p in paths}


def handoff_examples(state: Path, repo: Path, frozen: Frozen, drops: Drops, max_chars: int = 48000) -> dict[str, list[Example]]:
    """{'adopted': gold SFT, 'pref': preference pairs, 'unjudged': teacher answers the kernel never judged (separate file)}."""
    wps = work_packages(Path(state) / "ledger.jsonl")
    les = lessons(Path(state) / "lessons.jsonl")
    ctx_by_pkg: dict[str, str] = {}
    for r in les:
        if r.get("context"):
            ctx_by_pkg.setdefault(str(r.get("package_id")), str(r["context"]))
    answers: list[dict[str, Any]] = []                 # every judged-or-not answer: requirement, pkg, ts, task, files_before, reply, outcome
    for c in cycles(state):
        wp = wps.get(c["package"], {})
        task = ctx_by_pkg.get(c["package"]) or (render_task(wp["data"]) if wp else "")
        reply_edits, paths = diff_to_edits(c["patch"])
        if not task or not paths:
            drops.add("no task text or no python change")
            continue
        added = [fd.path for fd in parse_diff(c["patch"]) if fd.status == "A"]
        base = c["base"] or wp.get("base", "")
        answers.append({"pkg": c["package"], "req": c["requirement"] or (wp.get("data", {}).get("objective", "")), "ts": wp.get("ts", 0.0),
                        "task": task, "files": _files_before(repo, base, paths, added), "edits": reply_edits,
                        "reasoning": "", "outcome": c["outcome"], "src": "cycle", "by": c["by"], "base": base})
    seen_cycle = {(a["pkg"], a["outcome"]) for a in answers}
    for r in les:
        fa, fb = r.get("files_after") or {}, r.get("files_before") or {}
        py = {p: v for p, v in fa.items() if p.endswith(".py") and v != fb.get(p, "")}
        if not py or not r.get("context"):
            continue
        pkg, outcome = str(r.get("package_id")), str(r.get("outcome") or "")
        if outcome in ("ADOPTED", "REJECTED") and (pkg, outcome) in seen_cycle:
            continue                                   # the cycle diff already is this answer
        from creator.model_student import edits_between
        edits = "".join(edits_between(p, fb.get(p, ""), v, max_chars=1_000_000) for p, v in py.items())
        wp = wps.get(pkg, {})
        answers.append({"pkg": pkg, "req": str(wp.get("data", {}).get("why_it_exists", "")).split(":")[0] or pkg, "ts": _ts(r.get("at")),
                        "task": str(r["context"]), "files": {p: fb.get(p, "") for p in py}, "edits": edits,
                        "reasoning": str(r.get("reasoning") or ""), "outcome": outcome, "src": f"lesson:{r.get('solver')}",
                        "by": str(r.get("solver") or ""), "base": wp.get("base", "")})
    for a in answers:                                  # requirement key for grouping: 'K07.exists'-style when the gap names it
        m = re.search(r"\(([A-Z]+\d*\.[a-z_]+)\)", a["task"])
        a["req"] = m.group(1) if m else (a["req"] or a["pkg"])

    def reply(a: Mapping[str, Any]) -> str:
        return f"REASONING: {(a['reasoning'] or 'implement the package as specified').strip()[:1500]}\n{a['edits']}"
    out: dict[str, list[Example]] = {"adopted": [], "pref": [], "unjudged": []}
    seen: set[str] = set()
    for a in sorted(answers, key=lambda x: x["ts"]):
        h = hashlib.sha256((a["task"] + a["edits"]).encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        ex = Example(f"{a['pkg']}:{a['src']}:{h[:8]}", "", a["ts"], a["req"], {"messages": coder_messages(a["task"], a["files"], reply(a))},
                     {"package": a["pkg"], "outcome": a["outcome"], "source": a["src"], "base": a["base"]})
        if a["outcome"] == "ADOPTED":
            ex.kind = "sft_handoff"
            got = _admit(ex, frozen, drops, package=a["pkg"], max_chars=max_chars)
            if got:
                out["adopted"].append(got)
        elif a["by"].startswith("claude") and a["outcome"] not in ("REJECTED", "ROLLED_BACK"):
            ex.kind = "sft_unjudged"
            got = _admit(ex, frozen, drops, package=a["pkg"], max_chars=max_chars)
            if got:
                out["unjudged"].append(got)
    good = [a for a in answers if a["outcome"] == "ADOPTED"]
    bad = [a for a in answers if a["outcome"] in ("REJECTED", "ROLLED_BACK", "not claimed done", "error") and a["edits"]]
    for b in bad:
        for g in good:
            if g["req"] != b["req"]:
                continue
            msgs = coder_messages(b["task"], b["files"])
            ex = Example(f"pref:{g['pkg']}>{b['pkg']}:{b['src']}", "pref", max(g["ts"], b["ts"]), g["req"],
                         {"prompt": msgs, "chosen": [{"role": "assistant", "content": reply(g)}],
                          "rejected": [{"role": "assistant", "content": reply(b)}]},
                         {"chosen": g["pkg"], "rejected": b["pkg"], "rejected_outcome": b["outcome"]})
            got = _admit(ex, frozen, drops, package=b["pkg"], max_chars=max_chars * 2)
            if got and not frozen.leak(package=g["pkg"]):
                out["pref"].append(got)
    return out


def commit_examples(repo: Path, frozen: Frozen, drops: Drops, rev: str = "HEAD", max_chars: int = 48000,
                    limit: int = 0) -> list[Example]:
    """Nupen's own PUBLIC history: commit message (the change's specification, written by the teacher) -> the change as edits."""
    out = []
    for c in commits(repo, rev, limit=limit):
        patch = _git(Path(repo), "show", "--format=", "--no-color", "--no-ext-diff", c["sha"], "--", "creator", "tests", "scripts")
        edits, paths = diff_to_edits(patch)
        if not paths:
            drops.add("no task text or no python change")
            continue
        if len(edits) > max_chars:
            drops.add("too long for the context window")
            continue
        added = [fd.path for fd in parse_diff(patch) if fd.status == "A"]
        files = _files_before(Path(repo), c["parent"], paths, added)
        subj = c["message"].split("\n")[0]
        task = f"Implement this change to the project (Nupen's own commit message):\n{c['message']}"
        ex = Example(f"commit:{c['sha'][:12]}", "sft_commit", c["ts"], f"commit:{c['sha'][:12]}",
                     {"messages": coder_messages(task, files, f"REASONING: {subj[:400]}\n{edits}")},
                     {"sha": c["sha"][:12], "package": f"commit-{c['sha'][:10]}", "base": c["parent"]})
        got = _admit(ex, frozen, drops, commit=c["sha"], max_chars=max_chars)
        if got:
            out.append(got)
    return out


# ------------------------------------------------------------------------------------------------ worked examples (the runner's bank)
WORKED_BANK_FIELDS = ("qid", "kind", "source", "prompt", "reasoning", "answer", "correct", "model", "created")
THINKER_SYSTEM = "You are Nupen's thinker. Reason briefly and concretely, then give the final line exactly as the question asks."


def worked_examples(bank: Path, frozen: Frozen, drops: Drops, max_chars: int = 12000) -> list[Example]:
    """Rows of the worked bank (JSONL, WORKED_BANK_FIELDS; written by the GPU runner): only correct ones become thinker SFT
    (question -> short worked reasoning + the answer line)."""
    out: list[Example] = []
    try:
        lines = Path(bank).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    seen: set[str] = set()
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if not r.get("correct") or not r.get("prompt") or not r.get("answer"):
            drops.add("worked example not correct or incomplete")
            continue
        key = hashlib.sha256((str(r["prompt"]) + str(r["reasoning"])).encode()).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        ans = str(r["answer"]).strip()
        reply = f"{str(r.get('reasoning') or '').strip()}\n{ans if ans.upper().startswith(('ANSWER', 'PROBABILITY')) else 'ANSWER: ' + ans}".strip()
        ex = Example(f"worked:{r['qid']}:{key[:6]}", "worked", _ts(r.get("created")), str(r["qid"]),
                     {"messages": [{"role": "system", "content": THINKER_SYSTEM}, {"role": "user", "content": str(r["prompt"])},
                                   {"role": "assistant", "content": reply}]}, {"kind": r.get("kind"), "model": r.get("model")})
        got = _admit(ex, frozen, drops, subject=str(r["qid"]), max_chars=max_chars)
        if got:
            out.append(got)
    return out


# ------------------------------------------------------------------------------------------------ time-ordered splits
def time_split(items: Sequence[Any], eval_frac: float = 0.2, ts: Any = None, group: Any = None,
               min_eval: int = 1) -> tuple[list[Any], list[Any], float]:
    """(train, eval, cut): every train item is strictly earlier than every eval item, and no group has members on both sides
    (a group belongs to the side of its EARLIEST member; members after the cut of a group that started before it are dropped,
    so the eval side never shares a task with training). Items without a time go nowhere."""
    tsf = ts or (lambda x: x.ts)
    gf = group or (lambda x: x.group)
    xs = sorted([x for x in items if tsf(x) > 0], key=tsf)
    if not xs:
        return [], [], 0.0
    n_eval = max(min_eval, int(round(len(xs) * eval_frac))) if eval_frac > 0 else 0
    if n_eval >= len(xs):
        n_eval = len(xs) // 2
    if n_eval <= 0:
        return xs, [], float("inf")
    cut = tsf(xs[len(xs) - n_eval])
    first: dict[str, float] = {}
    for x in xs:
        first.setdefault(gf(x), tsf(x))
    train = [x for x in xs if tsf(x) < cut and first[gf(x)] < cut]
    ev = [x for x in xs if tsf(x) >= cut and first[gf(x)] >= cut]
    return train, ev, cut


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for ln in text.splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def question_splits(state: Path, frozen: Frozen, eval_frac: float = 0.2) -> dict[str, dict[str, Any]]:
    """Held-out id lists (time-ordered) for reasoning questions and judgment cases; frozen subjects never appear on either side."""
    th = Path(state) / "thinking"
    out: dict[str, dict[str, Any]] = {}
    for name, fn, idf, tf in (("reasoning", "reasoning.jsonl", "qid", "at"), ("judgment", "judgment.jsonl", "subject", "created")):
        first: dict[str, float] = {}
        for r in jsonl_rows(th / fn):
            q = str(r.get(idf) or "")
            if not q or frozen.leak(subject=q, package=q, commit=q if len(q) >= 10 and re.fullmatch(r"[0-9a-f]+", q) else ""):
                continue
            t = _ts(r.get(tf))
            if t > 0:
                first[q] = min(first.get(q, t), t)
        items = sorted(first.items(), key=lambda kv: kv[1])
        tr, ev, cut = time_split(items, eval_frac, ts=lambda kv: kv[1], group=lambda kv: kv[0])
        out[name] = {"train": [k for k, _ in tr], "eval": [k for k, _ in ev], "cut": cut}
    return out


# ------------------------------------------------------------------------------------------------ export
def outside_repo(p: Path, root: Path = ROOT) -> Path:
    rp, rr = Path(p).resolve(), Path(root).resolve()
    if rp == rr or rr in rp.parents:
        raise ValueError(f"refusing to write GPU-day data inside the repository: {rp}")
    return Path(p)


def default_out() -> Path:
    return Path.home() / "creator_runtime" / "gpuday" / "export"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(state: Path, repo: Path, out: Path, *, strict: bool = False, bank: Optional[Path] = None, eval_frac: float = 0.2,
           with_commits: bool = True, commit_limit: int = 0, frozen_path: Optional[Path] = None,
           expect_hash: str = FROZEN_HASH) -> dict[str, Any]:
    """Write every dataset + splits + MANIFEST.json into `out` (outside the repo). Returns the manifest."""
    outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    fp = frozen_path or Path(state) / "thinkbench" / "items.json"
    frozen = Frozen.load(fp, strict=strict, expect=expect_hash) if Path(fp).is_file() else Frozen.empty()
    drops = Drops()
    h = handoff_examples(Path(state), Path(repo), frozen, drops)
    com = commit_examples(Path(repo), frozen, drops, limit=commit_limit) if with_commits else []
    wk = worked_examples(bank or Path(state) / "thinking" / "worked_bank.jsonl", frozen, drops)
    files: dict[str, dict[str, Any]] = {}

    def put(name: str, exs: Sequence[Example]) -> None:
        p = out / f"{name}.jsonl"
        n = _write_jsonl(p, (e.row for e in exs))
        _write_jsonl(out / f"{name}.ids.jsonl", ({"id": e.id, "kind": e.kind, "ts": e.ts, "group": e.group, **e.meta} for e in exs))
        files[name] = {"rows": n, "bytes": p.stat().st_size, "sha256": _sha(p)}
    for name, exs in (("handoff", h["adopted"]), ("pref", h["pref"]), ("commit", com), ("worked", wk)):
        tr, ev, cut = time_split(exs, eval_frac)
        put(f"{name}_train", tr)
        put(f"{name}_eval", ev)
        files[f"{name}_train"]["cut_utc"] = files[f"{name}_eval"]["cut_utc"] = (
            dt.datetime.fromtimestamp(cut, dt.timezone.utc).isoformat() if 0 < cut < float("inf") else None)
    put("handoff_unjudged", h["unjudged"])
    # coder fine-tune mix: gold handoffs + public commits (train sides only); held-out handoffs stay for the harness
    mix = [e for n in ("handoff_train", "commit_train") for e in jsonl_rows(out / f"{n}.jsonl")]
    files["coder_sft_mix"] = {"rows": _write_jsonl(out / "coder_sft_mix.jsonl", mix)}
    docs = embed_corpus(Path(state), Path(repo), frozen, drops, with_commits=with_commits)
    files["embed_docs"] = {"rows": _write_jsonl(out / "embed_docs.jsonl", docs)}
    qs = question_splits(Path(state), frozen, eval_frac)
    (out / "splits.json").write_text(json.dumps(qs, indent=1), encoding="utf-8")
    man = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "strict": strict, "frozen_hash": frozen.hash[:12],
           "frozen_excluded": {"packages": sorted(frozen.packages), "commits": len(frozen.commits), "texts": len(frozen.texts)},
           "files": files, "dropped": drops.counts,
           "splits": {k: {"train": len(v["train"]), "eval": len(v["eval"])} for k, v in qs.items()},
           "format": {"sft": "{'messages': [system, user, assistant]} - Unsloth/TRL SFTTrainer; Axolotl type chat_template",
                      "pref": "{'prompt': [system, user], 'chosen': [assistant], 'rejected': [assistant]} - TRL/Unsloth DPO, ORPO"},
           "privacy": "rows with private markers or secrets dropped (never: the live trading state, the old-PC projects, the owner's notes); "
                      "names, e-mails, home paths scrubbed"}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _strings(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _strings(x)


def audit_export(out: Path, frozen: Frozen) -> list[str]:
    """Problems found by re-reading every exported file (used by tests and before any upload): private markers, secrets, e-mails,
    frozen texts/packages."""
    bad = []
    for p in sorted(Path(out).glob("*.jsonl")):
        text = p.read_text(encoding="utf-8")
        why = private_reason(text)
        if why:
            bad.append(f"{p.name}: {why}")
        if any(EMAIL_RE.search(v) for r in jsonl_rows(p) for v in _strings(r)):     # on the decoded text ('\n@pytest' is no e-mail)
            bad.append(f"{p.name}: e-mail address")
        for t in frozen.texts:
            if t in text:
                bad.append(f"{p.name}: frozen item text")
                break
        if p.name.endswith(".ids.jsonl"):
            for r in jsonl_rows(p):
                for k in ("package", "chosen", "rejected"):
                    if r.get(k) in frozen.packages:
                        bad.append(f"{p.name}: frozen package {r.get(k)}")
    return bad


# ------------------------------------------------------------------------------------------------ coder harness (evaluation-only)
@dataclasses.dataclass
class Case:
    """One held-out handoff: the prompt messages (system + user) and where to replay it."""
    id: str
    package: str
    base: str
    task: str
    messages: list[dict[str, str]]
    outputs: list[str]


def harness_cases(eval_jsonl: Path) -> list[Case]:
    """Held-out handoffs from an export (<name>.jsonl + <name>.ids.jsonl): the same prompt the training rows carry."""
    rows, ids = jsonl_rows(eval_jsonl), jsonl_rows(Path(str(eval_jsonl)[: -len(".jsonl")] + ".ids.jsonl"))
    out = []
    for r, m in zip(rows, ids):
        msgs = [x for x in r.get("messages", []) if x.get("role") != "assistant"]
        user = next((x["content"] for x in msgs if x["role"] == "user"), "")
        task = user.split("\n\nFILE: ")[0].removeprefix("Your task:\n")
        outs = [ln[len("FILE: "):].strip() for ln in user.split("\n") if ln.startswith("FILE: ")]
        out.append(Case(str(m.get("id")), str(m.get("package", "")), str(m.get("base", "")), task, msgs, outs))
    return out


def chat_http(url: str, timeout: float = 900.0, think: bool = False) -> Any:
    """An OpenAI-compatible chat call (llama-server /v1/chat/completions, local or through the pulse tunnel)."""
    import urllib.request

    def call(messages: Sequence[Mapping[str, str]], temperature: float, max_tokens: int, seed: int) -> str:
        body = {"messages": list(messages), "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": think}}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        return str(d["choices"][0]["message"].get("content") or "")
    return call


def default_judge(sb: Any, base: str, outputs: Sequence[str]) -> dict[str, Any]:
    """The kernel's sandbox gates that need no ledger: build + base-vs-candidate affected tests CLEAN, no weakened tests, every
    declared .py output present. (Not run here: the ledger's claim verdict, the post-merge re-assessment and the start-load guard -
    'would pass' is therefore necessary, not sufficient, for an adoption.)"""
    from creator import build as B
    from creator import kernel as K
    from creator.audit import checks as AC
    ev = sb.evaluate(build_config=B.BuildConfig(run_typecheck=False, require_typecheck=False))
    before, after = K._test_sources(sb.path, base, list(ev.change.files))
    weak = AC.check_test_weakening(before, after)
    missing = [o for o in outputs if o.endswith(".py") and not (sb.path / o).is_file()]
    rep = ev.report.verdict.value if ev.report else "NO_REPORT"
    return {"builds": ev.builds, "report": rep, "weakened": [f.detail for f in weak][:3], "missing": missing,
            "would_pass": bool(ev.clean and not weak and not missing)}


def replay_case(repo: Path, case: Case, ask: Any, n: int = 1, temperature: float = 0.6, max_tokens: int = 6000,
                judge: Any = None, scratch: Optional[Path] = None) -> dict[str, Any]:
    """Best-of-N for one held-out handoff: N answers, each applied in its OWN fresh sandbox at the recorded base and judged;
    the tests pick (the first candidate that would pass). Every sandbox is discarded - nothing is ever adopted or merged."""
    from creator import generator as G
    from creator import sandbox as S
    judge = judge or default_judge
    base = case.base if case.base and S.commit_exists(repo, case.base) else "HEAD"
    cands = []
    for i in range(max(1, n)):
        t0 = time.monotonic()
        rec: dict[str, Any] = {"i": i}
        try:
            reply = ask(case.messages, temperature if n > 1 else 0.2, max_tokens, 1000 + i)
        except Exception as e:                                      # noqa: BLE001 - a dead server is a recorded failure
            rec.update(error=f"model: {type(e).__name__}: {str(e)[:200]}", would_pass=False)
            cands.append(rec)
            continue
        rec["model_s"] = round(time.monotonic() - t0, 1)
        edits = G.parse_edits(reply)
        sb = S.Sandbox.open(repo, base, scratch=scratch, label=f"gpuday-{case.package}")
        try:
            (sb.path / S.HANDOFF_FILES[0]).write_text(case.task, encoding="utf-8")
            applied, refused = G.apply_edits(sb.path, edits) if edits else ([], ["no usable search/replace edit"])
            (sb.path / S.HANDOFF_FILES[0]).unlink(missing_ok=True)
            rec.update(edits=len(edits), applied=len(applied), refused=refused[:3])
            if not applied or not sb.changes().paths:
                rec["would_pass"] = False
            else:
                rec.update(judge(sb, base, case.outputs))
        except Exception as e:                                      # noqa: BLE001
            rec.update(error=f"evaluation: {type(e).__name__}: {str(e)[:300]}", would_pass=False)
        finally:
            S.discard(sb)
        rec["seconds"] = round(time.monotonic() - t0, 1)
        cands.append(rec)
    pick = next((c["i"] for c in cands if c.get("would_pass")), None)
    return {"case": case.id, "package": case.package, "base": base, "n": n, "candidates": cands, "picked": pick,
            "pass_at_1": bool(cands and cands[0].get("would_pass")), "pass_best_of_n": pick is not None}


def harness_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    k = len(results)
    p1 = sum(1 for r in results if r.get("pass_at_1"))
    pn = sum(1 for r in results if r.get("pass_best_of_n"))
    return {"cases": k, "pass_at_1": p1, "pass_best_of_n": pn, "rate_at_1": round(p1 / k, 3) if k else None,
            "rate_best_of_n": round(pn / k, 3) if k else None,
            "note": "'would pass' = the kernel's sandbox gates (build, affected tests CLEAN vs base, no weakened tests, outputs present); "
                    "evaluation only - nothing was merged"}



# ------------------------------------------------------------------------------------------------ embedding index (GPU builds, CPU queries)
# ONE model on both sides, run by llama.cpp both times (llama-server --embedding): on the pod at GPU speed for the corpus, on the PC on the
# CPU for a query - so the vectors are comparable. The index is small: float16, L2-normalised rows + a docs.jsonl with ids and snippets.
EMBED_MODELS: dict[str, dict[str, Any]] = {
    "Qwen3-Embedding-0.6B-Q8_0.gguf": {"repo": "Qwen/Qwen3-Embedding-0.6B-GGUF", "bytes": 639150592, "pooling": "last",
                                       "sha256": "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"},
    "bge-small-en-v1.5-q8_0.gguf": {"repo": "CompendiumLabs/bge-small-en-v1.5-gguf", "bytes": 36806944, "pooling": "cls",
                                    "sha256": "ec38e8da142596baa913124ae50550de284b6916bf59577ef2f0cb9660c2f514"},
}


def embed_corpus(state: Path, repo: Path, frozen: Frozen, drops: Drops, max_chars: int = 2000,
                 with_commits: bool = True) -> list[dict[str, Any]]:
    """Public, privacy-filtered texts worth finding again: commit messages, lesson objectives + reasoning, plan explanations."""
    docs: list[dict[str, Any]] = []

    def add(did: str, kind: str, text: str, ts: float, package: str = "", commit: str = "") -> None:
        text = text.strip()
        if not text:
            return
        why = frozen.leak(text=text, package=package, commit=commit) or private_reason(text)
        if why:
            drops.add(why)
            return
        docs.append({"id": did, "kind": kind, "ts": ts, "text": scrub(text)[:max_chars]})
    if with_commits:
        for c in commits(repo, paths=(".",)):
            add(f"commit:{c['sha'][:12]}", "commit", c["message"], c["ts"], commit=c["sha"])
    seen: set[str] = set()
    for r in lessons(Path(state) / "lessons.jsonl"):
        t = f"{r.get('objective', '')}\n{r.get('reasoning') or ''}"
        if t in seen:
            continue
        seen.add(t)
        add(f"lesson:{r.get('lesson_id')}", "lesson", t, _ts(r.get("at")), package=str(r.get("package_id", "")))
    for i, r in enumerate(jsonl_rows(Path(state) / "plan_explanations.jsonl")):
        t = f"{r.get('component', '')} {r.get('step', '')}: {r.get('why', '')}"
        if t in seen:
            continue
        seen.add(t)
        add(f"plan:{i}", "plan", t, _ts(r.get("at")))
    return docs


def embed_http(url: str, timeout: float = 300.0) -> Any:
    """texts -> vectors through llama-server's OpenAI-compatible /v1/embeddings."""
    import urllib.request

    def call(texts: Sequence[str]) -> list[list[float]]:
        req = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=json.dumps({"input": list(texts)}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]
    return call


def build_index(docs: Sequence[Mapping[str, Any]], embed: Any, out: Path, model: str, batch: int = 32,
                model_sha256: str = "", allow_repo: bool = False) -> dict[str, Any]:
    import numpy as np
    if not allow_repo:
        outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    vecs: list[list[float]] = []
    for i in range(0, len(docs), batch):
        vecs += embed([str(d["text"]) for d in docs[i:i + batch]])
    m = np.asarray(vecs, dtype=np.float32).reshape(len(vecs), -1)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    np.save(out / "vectors.npy", m.astype(np.float16))
    _write_jsonl(out / "docs.jsonl", ({"id": d["id"], "kind": d["kind"], "ts": d["ts"], "text": str(d["text"])[:600]} for d in docs))
    meta = {"model": model, "model_sha256": model_sha256, "dim": int(m.shape[1]) if m.size else 0, "rows": int(m.shape[0]),
            "dtype": "float16", "normalised": True, "seconds": round(time.monotonic() - t0, 1),
            "query_note": "embed the query with the SAME gguf through llama-server --embedding, then dot product"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def query_index(index: Path, embed: Any, text: str, k: int = 5, kinds: Sequence[str] = ()) -> list[dict[str, Any]]:
    import numpy as np
    m = np.load(Path(index) / "vectors.npy").astype(np.float32)
    docs = jsonl_rows(Path(index) / "docs.jsonl")
    q = np.asarray(embed([text])[0], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    sims = m @ q
    order = [int(i) for i in np.argsort(-sims) if not kinds or docs[int(i)]["kind"] in kinds][:k]
    return [dict(docs[i], score=round(float(sims[i]), 4)) for i in order]


# ------------------------------------------------------------------------------------------------ the day's plan (jobs, time, cost, what comes home)
USD_PER_HR = 0.343


@dataclasses.dataclass(frozen=True)
class Block:
    start_h: float
    end_h: float
    name: str
    jobs: tuple[str, ...]           # 'thinkbench:/judgment:/drills:' = gpupulse jobs; 'pod:' = scripts/gpuday/pod.sh on the pod; 'home:' = this PC
    comes_home: str
    ready: str                      # READY / READY-UNPROVEN-ON-GPU / THIN-DATA / NOT-READY - honest, from the CPU tests
    kind: str = "gen"               # what bounds its speed: 'train' (tensor throughput) or 'gen' (memory bandwidth)

    @property
    def hours(self) -> float:
        return self.end_h - self.start_h

    @property
    def usd(self) -> float:
        return round(self.hours * USD_PER_HR, 2)


DAY: tuple[Block, ...] = (
    Block(0.0, 0.5, "setup + throughput probe", ("pod:setup", "home:gpu_pulse setup", "home:gpu_pulse bench"), "probe.json (<1 KB)", "READY"),
    Block(0.5, 2.5, "bake-off + coder trial", ("thinkbench:Qwen3-1.7B-Q4_K_M.gguf", "thinkbench:Qwen3-4B-Q4_K_M.gguf",
                                                "thinkbench:Qwen3-8B-Q4_K_M.gguf", "thinkbench:Qwen3-14B-Q4_K_M.gguf",
                                                "home:coder_harness --n 1 (commit_eval + handoff_eval, served model through the tunnel)"),
          "thinkbench results (~50 KB each); harness_*.json (~100 KB)", "READY"),
    Block(2.5, 6.5, "worked-example bank + embedding index", ("drills:<best>:0:200 (rows go to the worked bank)", "pod:embed",
                                                               "home:embed_index build --url <tunnelled embedding server>"),
          "worked_bank.jsonl (~5-30 MB); index: vectors.npy float16 (~2-10 MB) + docs.jsonl", "READY (bank writer is the runner's)"),
    Block(6.5, 10.0, "judgment/reasoning decision rounds", ("judgment:<best>:0:120", "drills:<best>:0:80"), "state records", "READY (exists)"),
    Block(10.0, 13.0, "FINE-TUNE 1: distil correct worked examples into Qwen3-1.7B / 4B (LoRA bf16)",
          ("home:export_data --bank worked_bank.jsonl", "pod:upload", "pod:ft1 Qwen/Qwen3-1.7B", "pod:ft1 Qwen/Qwen3-4B",
           "thinkbench:<tuned gguf> (frozen benchmark; keep only if better, paired CI)"),
          "LoRA adapters (~35-70 MB) + adapter GGUF (~35-70 MB); Q4_K_M GGUF of a winner only (1.1 / 2.5 GB)", "READY-UNPROVEN-ON-GPU", "train"),
    Block(13.0, 18.0, "FINE-TUNE 2: coder on Nupen's handoff history (SFT + DPO)",
          ("pod:ft2 Qwen/Qwen3-14B (QLoRA, coder_sft_mix)", "pod:dpo (pref_train; skipped below 20 pairs)",
           "home:coder_harness --url <tuned server> (held-out commit_eval + handoff_eval)"),
          "LoRA adapter (~130-260 MB); the 9 GB 14B GGUF ONLY if it beats the untuned 14B on held-out", "THIN-DATA", "train"),
    Block(18.0, 21.0, "best-of-N (N=8) on the held-out handoffs/commits; home sandbox judges", ("call creator.gpuday:harness_bon_job",),
          "candidate patches (KB); judged by the home sandbox, never merged",
          "READY-UNPROVEN-ON-GPU on held-out cases; on LIVE open goals NOT READY (needs the kernel to export its open packages as cases)"),
    Block(21.0, 23.0, "RL proof of concept (GRPO, unit-test reward)", ("home:rl_grpo tasks", "pod:rl Qwen/Qwen3-1.7B"),
          "rl result.json (+ adapter ~35 MB if the reward rose on held-out tasks)", "READY-UNPROVEN-ON-GPU (tiny CPU run only)", "train"),
    Block(23.0, 24.0, "results home + teardown", ("pod:pack", "home:gpu_pulse teardown --destroy"), "home.tar (result.json files, adapters, index)",
          "READY"),
)


def day_plan(gpu: Any = None, usd_per_hr: float = USD_PER_HR) -> dict[str, Any]:
    """The blocks on the rented card: each block keeps its slot, and 'work_hours' is the 4090-sized work scaled to this card; work beyond
    the slot ('overflow_hours') is cut by the job's max_minutes, so a slower card does less in the same slot rather than overrunning."""
    prof = gpu_profile(parse_gpu(gpu) if isinstance(gpu, str) else gpu)
    blocks = []
    for b in DAY:
        work = scaled_minutes(b.hours * 60, b.kind, prof) / 60
        blocks.append(dict(dataclasses.asdict(b), hours=b.hours, usd=round(b.hours * usd_per_hr, 2), work_hours=round(work, 2),
                           overflow_hours=round(max(0.0, work - b.hours), 2)))
    return {"gpu": prof, "usd_per_hr": usd_per_hr, "total_hours": sum(b.hours for b in DAY),
            "total_usd": round(sum(b.hours for b in DAY) * usd_per_hr, 2), "blocks": blocks}



# ------------------------------------------------------------------------------------------------ runner hook: the day's job list (gpupulse ext jobs)
POD_DIR = "gpuday"                                  # under the runner's remote_dir on the pod
SCRIPTS = ("scripts/gpuday/finetune.py", "scripts/gpuday/embed_pod.py", "scripts/gpuday/rl_grpo.py", "scripts/gpuday/pod_setup.sh")
UPLOAD_DATA = ("handoff_train.jsonl", "handoff_eval.jsonl", "pref_train.jsonl", "pref_eval.jsonl", "commit_train.jsonl", "commit_eval.jsonl",
               "worked_train.jsonl", "worked_eval.jsonl", "coder_sft_mix.jsonl", "embed_docs.jsonl", "rl_tasks.jsonl", "MANIFEST.json")
MODELS_HF = {"1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B", "8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
             "14b": "unsloth/Qwen3-14B-unsloth-bnb-4bit", "coder30b": "unsloth/Qwen3-Coder-30B-A3B-Instruct"}
MERGE_BASE = {"8b": "Qwen/Qwen3-8B", "14b": "Qwen/Qwen3-14B", "coder30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct"}
# Peak pod disk of one fine-tune, GB: the base download (HF cache) + the 16-bit merge base (QLoRA) + merged weights + f16 GGUF + the quantised
# GGUF (merged and f16 are deleted by finetune.py only after the quantised file exists). bf16 sizes: 1.7B 3.4, 4B 8.0, 8B 16.4, 14B 29.5,
# 30B-A3B 61 GB; the bnb-4bit repos ~1/3 of that. ft_script skips itself (with the reason) instead of filling the pod's disk mid-run.
FT_DISK_GB = {"1.7b": 12, "4b": 27, "8b": 60, "14b": 110, "coder30b": 210}
FT_DL_GB = {"1.7b": 5, "4b": 10, "8b": 8, "14b": 12, "coder30b": 64}      # adapters only (gpuday_ft_gguf false): the base download + headroom


def upload_bundle(export_dir: Path, frozen: Frozen, repo: Path = ROOT) -> tuple[bytes, dict[str, Any]]:
    """The ONE upload of the day (gzip tar, sent inside a remote job's script): the pod scripts, this module as gpuday_lib.py and the
    training files of an export. Refused whole when the export audit finds anything private or frozen."""
    import io
    import tarfile
    bad = audit_export(export_dir, frozen)
    if bad:
        raise ValueError(f"export audit failed, nothing is uploaded: {bad[:5]}")
    members: list[tuple[str, bytes]] = [(Path(s).name, (Path(repo) / s).read_bytes()) for s in SCRIPTS if (Path(repo) / s).is_file()]
    members.append(("gpuday_lib.py", (Path(repo) / "creator" / "gpuday.py").read_bytes()))
    for n in UPLOAD_DATA:
        p = Path(export_dir) / n
        if p.is_file():
            members.append((f"data/{n}", p.read_bytes()))
    for name, data in members:
        why = private_reason(data.decode("utf-8", "replace")) if name.startswith("data/") else ""
        if why:
            raise ValueError(f"{name}: {why} - nothing is uploaded")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            tf.addfile(ti, io.BytesIO(data))
    blob = buf.getvalue()
    return blob, {"files": [m[0] for m in members], "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def upload_script(blob: bytes, info: Mapping[str, Any]) -> str:
    import base64
    b64 = base64.encodebytes(blob).decode("ascii")
    return (f"set -e\nmkdir -p {POD_DIR}\nbase64 -d > {POD_DIR}/upload.tgz <<'GPUDAY_B64'\n{b64}GPUDAY_B64\n"
            f"test \"$(sha256sum {POD_DIR}/upload.tgz | cut -d' ' -f1)\" = \"{info['sha256']}\"\n"
            f"tar -xzf {POD_DIR}/upload.tgz -C {POD_DIR} && rm -f {POD_DIR}/upload.tgz\n"
            f"echo '@@result={json.dumps({'uploaded': len(info['files']), 'bytes': info['bytes']})}'\n")


def ft_script(name: str, base: str, data: str, *, qlora: bool = False, pref: str = "", method: str = "dpo", max_seq: int = 8192,
              epochs: float = 2.0, batch: int = 2, accum: int = 8, lr: float = 2e-4, quant: str = "Q4_K_M", gguf: bool = True,
              min_rows: int = 1, serve_as: str = "", merge_base: str = "", disk_gb: int = 0) -> str:
    """Remote script of one fine-tune: SFT (+ optional DPO/ORPO) -> merge -> GGUF -> sha256; with `serve_as` the GGUF is also placed in the
    runner's models/ with its .ok hash so the runner can serve it without a download (see docs/GPU_DAY_PREP.md, hook H3; the runner serves
    it once register_tuned_job has recorded its size + sha256). `disk_gb` > 0: skipped with the reason when the pod's working directory or
    $HOME (the Hugging Face cache) has less free space than that."""
    d = f"{POD_DIR}/runs/{name}"
    opts = (f"--base {base} --data {POD_DIR}/data/{data} --out {d} --max-seq {max_seq} --epochs {epochs} --batch {batch} --accum {accum} "
            f"--lr {lr}" + (" --qlora" if qlora else "") + (f" --merge-base {merge_base}" if merge_base else ""))
    if gguf:
        opts += f" --llama-cpp {POD_DIR}/llama.cpp --quantize \"$(cat {POD_DIR}/quantize_path)\" --quant {quant}"
    if pref:
        opts += f" --pref {POD_DIR}/data/{pref} --method {method}"
    count = f"n=$(wc -l < {POD_DIR}/data/{data} 2>/dev/null || echo 0)"
    skip = f"if [ \"$n\" -lt {min_rows} ]; then echo '@@result={{\"skipped\": \"fewer than {min_rows} rows in {data}\"}}'; exit 0; fi"
    if disk_gb > 0:
        skip += "\n" + disk_check(int(disk_gb))
    place = ""
    if gguf and serve_as:                     # a hard link (same disk, no second copy of a 1-9 GB file); a copy only across filesystems
        place = (f"\ng={d}/model-{quant}.gguf; if [ -f \"$g\" ]; then ln -f \"$g\" models/{serve_as} 2>/dev/null || cp -f \"$g\" models/{serve_as}; "
                 f"sha256sum models/{serve_as} | cut -d' ' -f1 > models/{serve_as}.ok; echo \"@@served_as={serve_as}\"; fi")
    return (f"set -e\n{POD_PY}\n{count}\n{skip}\nmkdir -p {d}\nexport PIP_NO_CACHE_DIR=1\n\"$PY\" {POD_DIR}/finetune.py pipeline {opts} > {d}/train.log 2>&1 "
            f"|| {{ tail -40 {d}/train.log; exit 5; }}{place}\ncat {d}/result.json | tr -d '\\n' | sed 's/^/@@result=/'\necho\n")


# The pod's Python for every GPU-day job: the interpreter pod_setup.sh chose and wrote to gpuday/python (a venv: the image's system Python
# is PEP 668 externally-managed on Ubuntu 24), else python3 on PATH.
POD_PY = f"PY=\"$(cat {POD_DIR}/python 2>/dev/null || true)\"; [ -n \"$PY\" ] && [ -x \"$PY\" ] || PY=$(command -v python3 || command -v python)"


def disk_check(need_gb: int) -> str:
    """Shell lines: skip the job (exit 0, '@@result={"skipped": ...}') when the working directory or $HOME has under `need_gb` GB free."""
    return ("free=$(df -Pk . \"$HOME\" 2>/dev/null | awk 'NR>1 {g=int($4/1048576); if (m==\"\" || g<m) m=g} END {print m+0}')\n"
            f"if [ \"$free\" -lt {int(need_gb)} ]; then printf '@@result={{\"skipped\": \"disk: %s GB free, needs ~{int(need_gb)} GB\"}}\\n' \"$free\"; "
            "exit 0; fi")


def day_jobs(cfg: Mapping[str, Any], export_dir: Optional[Path] = None) -> list[Any]:
    """`gpu_pulse.py run --jobs-from creator.gpuday:day_jobs`: the whole day after setup, as gpupulse job specs. The upload job carries
    the audited bundle; the training jobs free the GPU; every job's small outputs come home through the runner's 'outputs'."""
    ex = Path(export_dir or cfg.get("gpuday_export") or default_out())
    frozen_p = Path(str(cfg.get("gpuday_frozen") or ROOT / "state" / "creator" / "thinkbench" / "items.json"))
    frozen = Frozen.load(frozen_p) if frozen_p.is_file() else Frozen.empty()
    blob, info = upload_bundle(ex, frozen)
    m17, m4, m8, m14 = "Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf"
    best = str(cfg.get("best_model") or m8)
    gpu = cfg.get("gpuday_gpu")
    prof = gpu_profile(parse_gpu(gpu) if isinstance(gpu, str) else gpu)      # e.g. "NVIDIA GeForce RTX 3090, 24576 MiB, 8.6" from the probe
    coder = str(cfg.get("gpuday_coder") or prof["coder"])
    serve_coder = f"Qwen3-{coder}-gpuday-coder.gguf"
    tm = lambda m: scaled_minutes(m, "train", prof)                         # noqa: E731
    gm = lambda m: scaled_minutes(m, "gen", prof)                           # noqa: E731
    out = lambda name, extra=(): [f"{POD_DIR}/runs/{name}/result.json", f"{POD_DIR}/runs/{name}/adapter", *extra]  # noqa: E731
    thinkers = [m17, m4, m8] + ([m14] if prof["vram_gb"] >= 14 else [])
    gg = bool(cfg.get("gpuday_ft_gguf", True))       # False: adapters only (no merge / GGUF / serving) - for a small pod disk
    disk = (lambda k: 0) if cfg.get("gpuday_skip_disk_check") else (lambda k: FT_DISK_GB[k] if gg else FT_DL_GB[k])  # noqa: E731

    def register(src: str, name: str) -> list[Any]:
        """After a fine-tune: its GGUF's bytes + sha256 into extra_models, then the jobs that serve it (they skip if it never came)."""
        if not gg:
            return []
        return [{"name": f"register_{src}", "call": "creator.gpuday:register_tuned_job", "args": {"source": src, "serve_as": name},
                 "minutes": 1, "low_util_abort_minutes": 0}]
    jobs: list[Any] = [
        "probe:" + m4,
        {"name": "gpuday_upload", "remote": upload_script(blob, info), "minutes": 1},
        *[f"thinkbench:{m}" for m in thinkers],
        {"name": "coder_trial_base", "call": "creator.gpuday:harness_job", "model": prof["best_of_n_model"], "minutes": gm(40),
         "max_minutes": gm(60), "low_util_abort_minutes": 0},
        f"traces:{best}:0:200",
        {"name": "embed_index", "remote": f"{POD_PY}; \"$PY\" {POD_DIR}/embed_pod.py build "
         f"--docs {POD_DIR}/data/embed_docs.jsonl --out {POD_DIR}/index --model Qwen3-Embedding-0.6B-Q8_0.gguf --models-dir models",
         "free_gpu": True, "minutes": gm(10), "outputs": [f"{POD_DIR}/index"]},
        f"judgment:{best}:0:120", f"drills:{best}:0:80",
        {"name": "gpuday_reexport", "call": "creator.gpuday:reexport_job", "minutes": 5},
        {"name": "ft1_17b", "remote": ft_script("ft1_17b", MODELS_HF["1.7b"], "worked_train.jsonl", max_seq=4096, batch=prof["ft1_batch_17b"],
                                                accum=max(1, 16 // prof["ft1_batch_17b"]), min_rows=200, serve_as="Qwen3-1.7B-gpuday-ft1.gguf",
                                                gguf=gg, disk_gb=disk("1.7b")),
         "free_gpu": True, "minutes": tm(50), "max_minutes": tm(80), "outputs": out("ft1_17b")},
        *register("ft1_17b", "Qwen3-1.7B-gpuday-ft1.gguf"),
        {"name": "ft1_4b", "remote": ft_script("ft1_4b", MODELS_HF["4b"], "worked_train.jsonl", max_seq=4096, batch=prof["ft1_batch_4b"],
                                               accum=max(1, 16 // prof["ft1_batch_4b"]), min_rows=200, serve_as="Qwen3-4B-gpuday-ft1.gguf",
                                               gguf=gg, disk_gb=disk("4b")),
         "free_gpu": True, "minutes": tm(80), "max_minutes": tm(110), "outputs": out("ft1_4b")},
        *register("ft1_4b", "Qwen3-4B-gpuday-ft1.gguf"),
        *(["thinkbench:Qwen3-1.7B-gpuday-ft1.gguf", "thinkbench:Qwen3-4B-gpuday-ft1.gguf"] if gg else []),
        {"name": "ft2_coder", "remote": ft_script("ft2_coder", MODELS_HF[coder], "coder_sft_mix.jsonl", qlora=bool(prof["coder_qlora"]),
                                                  pref="pref_train.jsonl" if int(cfg.get("gpuday_pref_rows", 0)) >= 20 else "",
                                                  max_seq=int(prof["coder_max_seq"]), batch=int(prof["coder_batch"]),
                                                  accum=max(1, 16 // int(prof["coder_batch"])), epochs=2, lr=1e-4, min_rows=50,
                                                  merge_base=MERGE_BASE[coder], serve_as=serve_coder, gguf=gg, disk_gb=disk(coder)),
         "free_gpu": True, "minutes": tm(150), "max_minutes": tm(240), "outputs": out("ft2_coder")},
        *register("ft2_coder", serve_coder),
        *([{"name": "coder_trial_tuned", "call": "creator.gpuday:harness_job", "model": serve_coder, "minutes": gm(40),
            "max_minutes": gm(60), "low_util_abort_minutes": 0}] if gg else []),
        {"name": "best_of_n_goals", "call": "creator.gpuday:harness_bon_job", "model": str(cfg.get("gpuday_bon_model") or prof["best_of_n_model"]),
         "minutes": gm(150), "max_minutes": gm(180), "low_util_abort_minutes": 0},
        {"name": "rl_poc", "remote": f"{POD_PY}; \"$PY\" {POD_DIR}/rl_grpo.py --base {MODELS_HF['1.7b']} "
         f"--tasks {POD_DIR}/data/rl_tasks.jsonl --out {POD_DIR}/runs/rl_poc --steps 150",
         "free_gpu": True, "minutes": tm(100), "max_minutes": tm(120), "outputs": [f"{POD_DIR}/runs/rl_poc/result.json"]},
    ]
    start = str(cfg.get("gpuday_start_at") or "")
    if start:                  # a second run of the day (e.g. after gpuday_reexport): a fresh upload carries the grown export, then `start` on
        label = [j.get("name") if isinstance(j, Mapping) else str(j) for j in jobs]
        if start not in label:
            raise ValueError(f"gpuday_start_at {start!r} is not a job of the day: {label}")
        jobs = [jobs[1]] + [j for j in jobs[label.index(start):] if j is not jobs[1]]
    return jobs


def harness_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job: replay held-out cases against the served model (best-of-N from cfg). Runs on the PC: the process drops itself
    to IDLE priority first, and evaluations run ONE at a time while N generations per case are in flight on the GPU."""
    _idle_priority()
    ep = ctx.get("endpoints") or {}
    urls = ((ep.get("models") or {}).get(str(ctx.get("model")), {}) or {}).get("urls") or []
    if not urls:
        return {"error": f"model {ctx.get('model')} is not served"}
    ex = default_out()
    n = int(ctx.get("n") or os.environ.get("GPUDAY_N", "1"))
    deadline = float(ctx.get("deadline") or 0)
    repo = harness_clone(ex.parent / "harness_repo", Path(str(ctx["repo"])))     # never the main repository: sandboxes make branches
    res = run_harness(repo, [ex / "commit_eval.jsonl", ex / "handoff_eval.jsonl"], urls[0].rsplit("/v1", 1)[0], n=n,
                      deadline=deadline, scratch=ex.parent / "harness_sandboxes", out=ex.parent / f"harness_{ctx.get('model')}.json")
    return res


def harness_bon_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Best-of-N (N = env GPUDAY_N or 8): N candidates per case, the home sandbox tests pick."""
    return harness_job(dict(ctx, n=int(os.environ.get("GPUDAY_N", "8"))))


def harness_clone(path: Path, source: Path) -> Path:
    """A separate clone for the harness's sandboxes (their branches never touch the main repository's refs); refreshed on each use."""
    if not (Path(path) / ".git").exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", str(source), str(path)], check=True, timeout=1800)
    else:
        subprocess.run(["git", "-C", str(path), "fetch", "-q", "origin"], check=False, timeout=600)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", "--detach", "origin/HEAD"], check=False, timeout=600)
    return Path(path)


def reexport_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Re-export after the bank grew (the worked examples for FINE-TUNE 1), at IDLE priority; the next upload carries it."""
    _idle_priority()
    man = export(Path(str(ctx["state"])), Path(str(ctx["repo"])), default_out())
    return {k: v.get("rows") for k, v in man["files"].items()}


def tuned_gguf(row: Mapping[str, Any], quant: str = "Q4_K_M") -> Optional[dict[str, Any]]:
    """{"bytes", "sha256"} of the quantised GGUF a fine-tune job's runner row reports (its result.json, sent back as '@@result='), or None
    when the job failed, skipped itself, or made no GGUF."""
    if row.get("rc") not in (0, None) or row.get("stopped"):
        return None
    res = row.get("result")
    files = ((res.get("gguf") or {}).get("files") or {}) if isinstance(res, Mapping) else {}
    e = files.get(f"model-{quant}.gguf")
    if not isinstance(e, Mapping) or not e.get("sha256") or not int(e.get("bytes") or 0):
        return None
    return {"bytes": int(e["bytes"]), "sha256": str(e["sha256"]).lower()}


def register_tuned(name: str, entry: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Add/replace one pod-local model in the extra_models file (atomic write; other entries kept) and validate the whole file the way the
    runner reads it (gpupulse.extra_models): a malformed entry raises and the previous file stays."""
    from creator import gpupulse as GP
    p = Path(path)
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    if not isinstance(cur, dict):
        cur = {}
    cur[name] = {"bytes": int(entry["bytes"]), "sha256": str(entry["sha256"]).lower()}
    GP.extra_models({"extra_models": cur, "extra_models_file": str(p.with_name(p.name + ".absent"))})   # validate before writing
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return cur


def register_tuned_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job (hook H3), right after a fine-tune job: args {"source": <ft job name>, "serve_as": <models/ file name>[, "quant",
    "extra_models_file"]}. Reads the fine-tune's row of THIS pulse from <state>/thinking/gpu_pulse_runs.jsonl and records the GGUF's bytes +
    sha256 in the extra_models file (ctx/args 'extra_models_file', else env NUPEN_GPU_EXTRA_MODELS, else <runtime>/gpu/extra_models.json),
    which the runner lays over the config's extra_models on every read. The runner then re-hashes the file ON THE POD before serving it."""
    from creator import gpupulse as GP
    a = dict(ctx.get("args") or {})
    src, name, quant = str(a.get("source") or ""), str(a.get("serve_as") or ""), str(a.get("quant") or "Q4_K_M")
    if not src or not name:
        return {"registered": False, "reason": "args need 'source' (fine-tune job name) and 'serve_as' (model file name)"}
    pulse = str(ctx.get("pulse") or "")
    row: Optional[dict[str, Any]] = None
    try:
        for ln in (Path(str(ctx["state"])) / "thinking" / "gpu_pulse_runs.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("job") == f"ext:{src}" and (not pulse or str(r.get("gpu_pulse") or "") == pulse):
                row = r                                                # the last one of this pulse wins
    except OSError:
        pass
    if row is None:
        return {"registered": False, "reason": f"no run of {src} in this pulse"}
    e = tuned_gguf(row, quant)
    if e is None:
        why = (row.get("result") or {}).get("skipped") if isinstance(row.get("result"), Mapping) else None
        return {"registered": False, "reason": f"{src} made no model-{quant}.gguf" + (f" (skipped: {why})" if why else f" (rc {row.get('rc')})")}
    path = Path(str(a.get("extra_models_file") or ctx.get("extra_models_file") or GP.extra_models_file()))
    register_tuned(name, e, path)
    return {"registered": True, "model": name, "bytes": e["bytes"], "sha256": e["sha256"], "file": str(path)}


def run_harness(repo: Path, eval_files: Sequence[Path], url: str, n: int = 1, deadline: float = 0.0, scratch: Optional[Path] = None,
                out: Optional[Path] = None, gen_threads: int = 4, think: bool = False) -> dict[str, Any]:
    """Every case of the eval files: generations run `gen_threads` cases ahead (the GPU stays busy) while sandbox evaluations run one at a
    time on this PC (they are the heavy home-side part)."""
    import concurrent.futures as cf
    cases = [c for f in eval_files if Path(f).is_file() for c in harness_cases(Path(f))]
    ask = chat_http(url, think=think)
    replies: dict[str, list[str]] = {}

    def gen(c: Case) -> tuple[str, list[str]]:
        rs = []
        for i in range(max(1, n)):
            try:
                rs.append(ask(c.messages, 0.6 if n > 1 else 0.2, 6000, 1000 + i))
            except Exception as e:                                     # noqa: BLE001
                rs.append(f"(model error: {type(e).__name__}: {str(e)[:200]})")
        return c.id, rs
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, gen_threads)) as pool:
        futs = {pool.submit(gen, c): c for c in cases}
        for fut in cf.as_completed(futs):
            c = futs[fut]
            if deadline and time.monotonic() > deadline:
                break
            cid, rs = fut.result()
            replies[cid] = rs
            it = iter(rs)
            results.append(replay_case(repo, c, lambda *a, **k: next(it), n=len(rs), scratch=scratch))
    rep = dict(harness_report(results), url=url, n=n, cases_total=len(cases))
    if out:
        outside_repo(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"report": rep, "results": results}, indent=1), encoding="utf-8")
    return rep


def _idle_priority() -> None:
    """Teacher rule (3 Oct): home-side GPU-day work yields the CPU to Nupen (IDLE class on Windows, nice 19 elsewhere)."""
    import sys as _sys
    try:
        if _sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32                                 # type: ignore[attr-defined]
            k.GetCurrentProcess.restype = ctypes.c_void_p
            k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00000040)
        else:
            getattr(os, "nice")(19)
    except (OSError, AttributeError):
        pass



# ------------------------------------------------------------------------------------------------ provider- and GPU-agnostic choices
# Relative speed vs an RTX 4090 (= 1.0). 'train' ~ dense bf16 tensor throughput as LoRA training sees it; 'gen' ~ memory bandwidth (token
# generation is bandwidth-bound). Sources: vendor specs (bandwidth 3090 936 GB/s, 4090 1008, 5090 1792, A6000 768, L40S 864, A100-80 2039,
# H100 SXM 3350 GB/s); the teacher's note of 3 Oct: a 3090 trains ~1.5-2x slower than a 4090 with similar generation speed.
GPU_SPEED: dict[str, dict[str, float]] = {
    "3090": {"train": 0.55, "gen": 0.93}, "4090": {"train": 1.0, "gen": 1.0}, "5090": {"train": 1.35, "gen": 1.75},
    "a6000": {"train": 0.6, "gen": 0.76}, "l40s": {"train": 0.95, "gen": 0.86}, "a100": {"train": 1.1, "gen": 1.9},
    "h100": {"train": 2.2, "gen": 3.0}, "4080": {"train": 0.7, "gen": 0.71}, "3080": {"train": 0.45, "gen": 0.75},
}
DEFAULT_GPU: dict[str, Any] = {"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "cc": 8.9}


def parse_gpu(text: str) -> dict[str, Any]:
    """'NVIDIA GeForce RTX 3090, 24576 MiB, ..., 8.6' (nvidia-smi csv: name, memory.total[, ...][, compute_cap]) or 'RTX 3090,24,8.6'."""
    parts = [p.strip() for p in str(text).split(",")]
    name = parts[0] if parts and parts[0] else DEFAULT_GPU["name"]
    vram, cc = 0.0, 0.0
    for p in parts[1:]:
        m = re.match(r"^([\d.]+)\s*(MiB|GiB|GB)?$", p)
        if not m:
            continue
        v = float(m.group(1))
        if m.group(2) == "MiB" or v > 1000:
            vram = vram or round(v / 1024, 1)
        elif m.group(2) in ("GiB", "GB") or (v >= 10 and not vram):
            vram = vram or v
        elif v < 13:
            cc = v
    return {"name": name, "vram_gb": vram or float(DEFAULT_GPU["vram_gb"]), "cc": cc or _cc_from_name(name)}


def _cc_from_name(name: str) -> float:
    n = name.lower()
    for k, cc in (("5090", 12.0), ("5080", 12.0), ("4090", 8.9), ("4080", 8.9), ("l40", 8.9), ("3090", 8.6), ("3080", 8.6), ("a6000", 8.6),
                  ("a100", 8.0), ("h100", 9.0), ("h200", 9.0), ("v100", 7.0), ("t4", 7.5)):
        if k in n:
            return cc
    return 8.0


def gpu_profile(gpu: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Every size/precision choice of the day from the detected card: VRAM picks the coder base and QLoRA vs LoRA and the batch; the
    compute capability picks bf16 (>= 8.0) vs fp16 and never an FP8 path (we use none; Ampere has none); Blackwell (cc >= 12) needs CUDA
    >= 12.8 wheels (pod_setup.sh enforces it). Speeds scale the plan's minutes."""
    g = dict(DEFAULT_GPU, **(gpu or {}))
    vram, cc, name = float(g["vram_gb"]), float(g["cc"]), str(g["name"])
    key = next((k for k in GPU_SPEED if k in name.lower()), "4090")
    sp = GPU_SPEED[key]
    if vram >= 40:        # 48 GB modded 4090, A6000, L40S, A100: the MoE coder fits in QLoRA with room; 4B trains in bigger batches
        coder, coder_q, coder_batch, coder_seq, ft1_batch, judge_model = "coder30b", True, 2, 16384, 16, "Qwen3-14B-Q4_K_M.gguf"
    elif vram >= 30:      # 32 GB (5090)
        coder, coder_q, coder_batch, coder_seq, ft1_batch, judge_model = "14b", True, 2, 16384, 8, "Qwen3-14B-Q4_K_M.gguf"
    elif vram >= 22:      # 24 GB (3090, 4090)
        coder, coder_q, coder_batch, coder_seq, ft1_batch, judge_model = "14b", True, 1, 12288, 4, "Qwen3-14B-Q4_K_M.gguf"
    else:                 # 16 GB and below: no 14B; the 8B coder in QLoRA
        coder, coder_q, coder_batch, coder_seq, ft1_batch, judge_model = "8b", True, 1, 8192, 2, "Qwen3-8B-Q4_K_M.gguf"
    return {"gpu": name, "vram_gb": vram, "cc": cc, "speed_key": key, "train_speed": sp["train"], "gen_speed": sp["gen"],
            "dtype": "bf16" if cc >= 8.0 else "fp16", "fp8": False, "min_cuda": "12.8" if cc >= 12.0 else "12.1",
            "coder": coder, "coder_qlora": coder_q, "coder_batch": coder_batch, "coder_max_seq": coder_seq,
            "ft1_batch_4b": ft1_batch, "ft1_batch_17b": ft1_batch * 2, "best_of_n_model": judge_model}


def scaled_minutes(minutes: float, kind: str, prof: Mapping[str, Any]) -> float:
    """A 4090-based estimate scaled to the detected card ('train' jobs by training speed, everything else by generation speed)."""
    s = float(prof["train_speed"] if kind == "train" else prof["gen_speed"])
    return round(minutes / max(s, 0.1), 1)
