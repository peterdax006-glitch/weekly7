"""Backfill curriculum lessons from work packages solved before lesson capture existed.

Sources (read-only): <state>/pending/CP*.patch and <state>/cycles/CP*/diff.patch (unified diffs against the base commit of the
sandbox), the development ledger (WorkPackage / ChangeProposal / Experiment / Gap / Requirement records), kernel_log.jsonl (cycle
outcomes and the teacher's notes) and the git history of --repo. Nothing in the repo is ever modified: before-texts come from
`git show <base>:<path>` and the patch is applied in a temporary directory.

Output: one creator.curriculum.Lesson per package (JSONL), solver "claude". A reasoning that is not the teacher's own notes is
prefixed "[reconstructed] ". adopted/verdict come from the kernel's cycle outcome (None when no cycle outcome is known).
With --infer-adopted-from-git an otherwise-unknown lesson whose post-image is a blob that exists in the repo history becomes
adopted=True with a verdict starting "[inferred]"; off by default."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creator import curriculum as C  # noqa: E402
from creator import kernel as K  # noqa: E402

PREFIX = "[reconstructed] "
DEC = {"ADOPTED": True, "REJECTED": False, "ROLLED_BACK": False}
MACHINE_NOTE = re.compile(r"^(OK|CONTAMINATED|ERROR|TIMEOUT)\b.*\brun=")      # agent-run markers, not the teacher's reasoning
STAMP = re.compile(r"^(CP\d+)_(\d{8}T\d{6})")


def git(repo: Path, *args: str, cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess[bytes]:
    p = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd or repo, capture_output=True, timeout=120)
    if check and p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:300])
    return p


def jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return out


# ------------------------------------------------------------------------------------------------ patches

def patch_files(patch: str) -> list[str]:
    """Paths (b-side) of every file the diff touches, in order, without duplicates."""
    seen: list[str] = []
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+)$", patch, re.M):
        if m.group(2) not in seen:
            seen.append(m.group(2))
    return seen


def apply_at(repo: Path, base: str, patch: bytes, paths: Sequence[str], check_only: bool = False) -> Optional[dict[str, str]]:
    """Apply `patch` to the `paths` as they are at `base`, in a temp dir. Returns {path: after-text} or None if it does not apply."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for p in paths:
            r = git(repo, "show", f"{base}:{p}")
            if r.returncode == 0:
                (tmp / p).parent.mkdir(parents=True, exist_ok=True)
                (tmp / p).write_bytes(r.stdout)
        (tmp / "_p.patch").write_bytes(patch)
        a = git(repo, "apply", "--check" if check_only else "--whitespace=nowarn", "_p.patch", cwd=tmp)
        if a.returncode:
            return None
        if check_only:
            return {}
        return {p: ((tmp / p).read_bytes().decode("utf-8", "replace") if (tmp / p).is_file() else "") for p in paths}


def before_texts(repo: Path, base: str, paths: Sequence[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        r = git(repo, "show", f"{base}:{p}")
        out[p] = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""
    return out


def find_base(repo: Path, patch: bytes, paths: Sequence[str], hints: Sequence[str], max_scan: int = 400) -> Optional[str]:
    """First commit the patch applies cleanly to: ledger hints first, then every commit newest-first."""
    tried: set[str] = set()
    scan = [h for h in hints if h]
    revs = git(repo, "rev-list", "--all", f"--max-count={max_scan}").stdout.decode().split()
    for c in scan + revs:
        if c in tried:
            continue
        tried.add(c)
        full = git(repo, "rev-parse", "--verify", "--quiet", f"{c}^{{commit}}").stdout.decode().strip()
        if full and apply_at(repo, full, patch, paths, check_only=True) is not None:
            return full
    return None


# ------------------------------------------------------------------------------------------------ ledger

class Facts:
    """Everything the ledger and kernel log know, indexed by package id."""

    def __init__(self, state: Path) -> None:
        rows = jsonl(state / "ledger.jsonl")
        self.by_id = {r["id"]: r for r in rows if "id" in r}
        self.wps: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            if r.get("rtype") == "WorkPackage":
                self.wps.setdefault(r["data"]["package_id"], []).append(r)
        self.chg_of_wp: dict[str, dict[str, Any]] = {}
        for r in rows:
            if r.get("rtype") == "ChangeProposal":
                for p in r["data"].get("parents", []):
                    self.chg_of_wp[p] = r
        self.exp_of_chg: dict[str, dict[str, Any]] = {}
        for r in rows:
            if r.get("rtype") == "Experiment":
                for p in r["data"].get("parents", []):
                    self.exp_of_chg[p] = r
        self.log: dict[str, list[dict[str, Any]]] = {}
        for d in jsonl(state / "kernel_log.jsonl"):
            self.log.setdefault(str(d.get("package")), []).append(d)
        self.state = state

    def wp(self, pkg: str, stamp: str = "") -> Optional[dict[str, Any]]:
        cands = self.wps.get(pkg) or []
        if stamp:
            iso = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}"
            early = [r for r in cands if r["provenance"]["timestamp"][:19] <= iso]
            cands = early or cands
        return cands[-1] if cands else None

    def baseline(self, wp: Optional[dict[str, Any]]) -> list[str]:
        if not wp:
            return []
        chg = self.chg_of_wp.get(wp["id"])
        exp = self.exp_of_chg.get(chg["id"]) if chg else None
        return [str(exp["data"].get("baseline_ref", "")).split("+")[0]] if exp else []

    def requirement_key(self, wp: Optional[dict[str, Any]]) -> str:
        if not wp:
            return ""
        for ref in wp["data"].get("parents", []):
            node = self.by_id.get(ref)
            while node is not None:
                if node.get("rtype") == "Requirement":
                    return str(node["data"].get("key", ""))
                ps = node["data"].get("parents", [])
                node = self.by_id.get(ps[0]) if ps else None
        for s in wp["data"].get("inputs", []):
            if str(s).startswith("requirement "):
                return str(s).split(" ", 1)[1]
        return ""

    def outcome(self, pkg: str) -> Optional[dict[str, Any]]:
        """The last kernel cycle record of the package that adjudicated it, else the last record of any kind."""
        recs = self.log.get(pkg, [])
        decided = [d for d in recs if d.get("outcome") in DEC]
        cj = self.state / "cycles" / pkg / "cycle.json"
        if not decided and cj.is_file():
            try:
                d = json.loads(cj.read_text(encoding="utf-8"))
                if d.get("outcome") in DEC:
                    return d  # type: ignore[no-any-return]
            except (OSError, json.JSONDecodeError):
                pass
        return (decided or recs or [None])[-1]  # type: ignore[list-item]


def kind_of(key: str) -> str:
    step = key.split(".", 1)[1] if "." in key else ""
    if key.startswith("EFF"):
        return "activation" if step == "activation" else "shrink"
    return "bugfix" if step == "tested" else "gap"


# ------------------------------------------------------------------------------------------------ reasoning

def diff_summary(patch: str) -> str:
    parts = []
    for sec in re.split(r"(?m)^diff --git ", patch)[1:]:
        m = re.match(r"a/(.+?) b/(.+)", sec)
        if not m:
            continue
        add = sum(1 for ln in sec.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        rem = sum(1 for ln in sec.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        new = "new file mode" in sec
        names = re.findall(r"(?m)^([+-])[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)", sec)
        imps = re.findall(r"(?m)^([+-])[ \t]*(?:from[ \t]+\S+[ \t]+)?import[ \t]+([^\n]+)", sec)
        s = f"{m.group(2)}: {'new file, ' if new else ''}+{add}/-{rem} lines"
        if names:
            s += "; defs " + ", ".join(f"{'added' if a == '+' else 'removed'} {n}" for a, n in names[:8])
        if imps:
            s += "; imports " + ", ".join(f"{'added' if a == '+' else 'removed'} {i.strip()[:40]}" for a, i in imps[:6])
        parts.append(s)
    return " | ".join(parts)


def reasoning_for(notes: str, objective: str, kind: str, patch: str) -> str:
    if notes and not MACHINE_NOTE.match(notes):
        return notes
    return (PREFIX + f"Task kind {kind}: {objective}. The accepted-shape of the solution, read from the diff: " + diff_summary(patch)
            + (f". Solver note: {notes}" if notes else ""))


# ------------------------------------------------------------------------------------------------ build

def in_history(repo: Path, after: dict[str, str]) -> bool:
    """True when every non-empty post-image file is, byte for byte, a blob that some commit in the history holds at that path."""
    if not any(after.values()):
        return False
    for p, text in after.items():
        if not text:
            continue
        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(text.encode("utf-8"))
            tmpname = fh.name
        want = git(repo, "hash-object", tmpname).stdout.decode().strip()
        Path(tmpname).unlink(missing_ok=True)
        revs = git(repo, "log", "--all", "--format=%H", "--max-count=150", "--", p).stdout.decode().split()
        if not any(git(repo, "rev-parse", f"{c}:{p}").stdout.decode().strip() == want for c in revs):
            return False
    return True


def sources(state: Path) -> list[tuple[str, Path, str]]:
    """(package id, patch file, filename stamp), cycle diffs preferred over pending ones for the same package."""
    found: dict[str, tuple[str, Path, str]] = {}
    for p in sorted((state / "pending").glob("CP*.patch")):
        m = STAMP.match(p.name)
        if m:
            found[m.group(1)] = (m.group(1), p, m.group(2))
    for p in sorted((state / "cycles").glob("CP*/diff.patch")):
        found[p.parent.name] = (p.parent.name, p, "")
    return [found[k] for k in sorted(found)]


def build(repo: Path, state: Path, infer_adopted: bool = False) -> tuple[list[C.Lesson], list[tuple[str, str]]]:
    facts = Facts(state)
    lessons: list[C.Lesson] = []
    skipped: list[tuple[str, str]] = []
    for pkg, pfile, stamp in sources(state):
        raw = pfile.read_bytes()
        if raw.split(b"\n", 1)[0].endswith(b"\r"):               # written in Windows text mode: every line ended CRLF
            raw = raw.replace(b"\r\n", b"\n")
        text = raw.decode("utf-8", "replace")
        paths = patch_files(text)
        if not paths:
            skipped.append((pkg, "empty patch (the worker changed nothing)"))
            continue
        if "GIT binary patch" in text or "Binary files" in text:
            skipped.append((pkg, "binary patch"))
            continue
        wp = facts.wp(pkg, stamp)
        if wp is None:
            skipped.append((pkg, "no WorkPackage record in the ledger"))
            continue
        out = facts.outcome(pkg)
        hints = facts.baseline(wp) + ([str(out.get("merge_commit")) + "^1"] if out and out.get("merge_commit") else [])
        base = find_base(repo, raw, paths, hints)
        if base is None:
            skipped.append((pkg, "patch applies to no commit in the history (base not found)"))
            continue
        after = apply_at(repo, base, raw, paths)
        if after is None:
            skipped.append((pkg, "patch did not apply"))
            continue
        d = wp["data"]
        key = facts.requirement_key(wp)
        kind = kind_of(key) if key else "gap"
        ns = SimpleNamespace(**{k: d.get(k) or [] for k in ("implementation_requirements", "test_requirements", "interfaces",
                                                            "expected_failure_modes", "completion_criteria")},
                             package_id=pkg, objective=d.get("objective", ""), why_it_exists=d.get("why_it_exists", ""))
        context = K.render_package(None, ns)        # type: ignore[arg-type]  # render_package reads only package fields
        worker = (out or {}).get("details", {}).get("worker", {}) or {}
        notes = str(worker.get("notes") or "")
        adopted: Optional[bool] = DEC.get(str((out or {}).get("outcome")))
        verdict = ""
        if out:
            verdict = f"{out.get('outcome')}: {out.get('reason', '')}" + (f" ({out['verdict']})" if out.get("verdict") else "")
        if adopted is None and infer_adopted and in_history(repo, after):
            adopted, verdict = True, "[inferred] post-image blobs of every changed file exist in the git history; " + verdict
        at = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}" if stamp else str(wp["provenance"]["timestamp"])
        lessons.append(C.Lesson(
            lesson_id=f"bf-{pkg}", package_id=pkg, component=key.split(".", 1)[0], task_kind=kind, objective=str(d.get("objective", "")),
            context=context, files_before=before_texts(repo, base, paths), files_after=after,
            reasoning=reasoning_for(notes, str(d.get("objective", "")), kind, text), solver=C.CLAUDE, adopted=adopted,
            verdict=verdict, claimed_done=bool(worker.get("claimed_done", True)) if out else True, at=at))
    return lessons, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("output", nargs="?", default=str(root / "state" / "creator" / "lessons_backfill.jsonl"))
    ap.add_argument("--repo", default=r"C:\Users\Peter\weekly7")
    ap.add_argument("--state", default=None, help="state/creator directory (default <repo>/state/creator)")
    ap.add_argument("--infer-adopted-from-git", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    state = Path(a.state) if a.state else repo / "state" / "creator"
    lessons, skipped = build(repo, state, a.infer_adopted_from_git)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(les.to_dict(), sort_keys=True) + "\n" for les in lessons), encoding="utf-8")
    n = lambda v: sum(1 for les in lessons if les.adopted is v)  # noqa: E731
    print(f"lessons {len(lessons)}: adopted {n(True)}, rejected {n(False)}, unknown {n(None)}; skipped {len(skipped)}")
    for pkg, why in skipped:
        print(f"  skipped {pkg}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
