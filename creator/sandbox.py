"""Creator K06 - the controlled environment for every autonomous code change (C77 secs 22-25, 28, 29, 44-46, 59-61; package CR03)
- IMPLEMENTED, NOT VALIDATED.

A Sandbox is a `git worktree` on its own branch, created from a RECORDED base commit, in a scratch directory outside the main
working tree. A change is applied there (never in the main tree), built (creator.build) and tested against the base commit on the
same test selection (creator.testrun), producing a regression report. Nothing becomes authoritative by executing: adoption needs an
explicit decision object and merges the sandbox branch with `--no-ff` only when the merge is clean; a conflicting merge is aborted
and refused, never forced. Rollback discards the sandbox, or reverts an adopted merge commit. A protected-path firewall refuses any
change to canon, contracts, lock files, the sealed development benchmark, the auditor, CI and the evaluator thresholds.

Git rules (CONTEXT.md, memory): the main tree is never stashed, reset or rebased; its index is touched only by the adopt/rollback
merge or revert, under a lock, and only when it is clean of conflicts with the change."""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from creator import build as B
from creator import device
from creator import testrun as T

PROTECTED: tuple[str, ...] = (
    "canon/*", "canon/**", "*.lock.json", "**/*.lock.json",
    "*_PROMPT.md", "BIBLE.md", "SELF_LEARNING_CONTRACT.md", "RESEARCH_BRAIN_CONTRACT.md", "PREDICTION_ERROR_ADDITION.md",
    "TEN_HOUR_EXECUTION_CHECKLIST.md",
    ".github/*", ".github/**",
    "creator/devbench/sealed/*", "creator/devbench/sealed/**",
    "creator/audit/*", "creator/audit/**",
    "creator/evaluate_thresholds.json",
    # the measuring sticks (1 Oct): the Creator may PROPOSE changes to these, but only the owner adopts them - otherwise it could
    # make itself look better by lowering its floors, softening the scorer, the verdict/status rules or the ledger's integrity
    # capabilities_approved.json is written only by creator.goals.approve (owner/teacher): a worker editing it would turn a goal into
    # work, or lower an approved capability's floor, without any approval (validator 6)
    "creator/capabilities.json", "creator/capabilities_approved.json", "creator/devbench.py", "creator/model.py", "creator/ledger.py", "creator/sandbox.py",
    "state/creator/ledger.jsonl",
    # thinking focus + trust (3 Oct, F7): owner/teacher-owned focus switch and the drill records/report a worker could otherwise forge
    "state/creator/focus.json", "state/creator/trust.json", "state/creator/thinking/*", "state/creator/thinking/**",
    # the self-teaching measuring code (3 Oct, h58 self-teach gate check g): the trust gate (thinking.trust_of), the frozen benchmark and its
    # items, the walk-forward / select-held-out split, the learn loop's verdicts, judgment / fast-topic scoring, decision arms, the search's
    # stop rules and the self-teach gate itself. Nupen improves through variants (data) and new goal_think_* modules, never by editing these.
    "creator/thinking.py", "creator/thinkbench.py", "creator/learnloop.py", "creator/drillsources.py", "creator/fastwalk.py",
    "creator/judgment.py", "creator/fastpred.py", "creator/decide.py", "creator/trialerror.py", "creator/selfteach.py", "creator/gpuselfteach.py",
    "state/creator/thinkbench/*", "state/creator/thinkbench/**",
    # the coding trust gate (3 Oct, h56): its yardstick, held-out set and records decide what Nupen may change unsupervised
    "creator/codetrust.py", "creator/codetrust_heldout.json", "state/creator/codetrust/*", "state/creator/codetrust/**",
    "state/livesim/*", "state/livesim/**",
)
SANDBOX_PREFIX = "creator/sbx-"
GIT_TIMEOUT = 120.0


class SandboxError(RuntimeError):
    """A refused or failed sandbox operation; never caught-and-continued."""


class ProtectedPathError(SandboxError):
    """The change touches a path the Creator may never modify (C77 secs 33, 34, 44)."""


def is_protected(rel: str, patterns: Sequence[str] = PROTECTED) -> bool:
    rel = rel.replace("\\", "/")
    while rel.startswith("./"):                     # strip a literal "./" prefix only: lstrip("./") ate the dot of ".github" (30 Sep)
        rel = rel[2:]
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def _safe_rel(rel: str) -> str:
    p = Path(rel.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise SandboxError(f"path must be relative inside the repository: {rel!r}")
    return p.as_posix()


def git(repo: str | Path, *args: str, check: bool = True, timeout: float = GIT_TIMEOUT) -> subprocess.CompletedProcess:
    env = B.clean_env(None, {"GIT_TERMINAL_PROMPT": "0"})
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)
    if check and proc.returncode != 0:
        raise SandboxError(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    return proc


def head(repo: str | Path) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def commit_exists(repo: str | Path, rev: str) -> bool:
    return git(repo, "cat-file", "-e", f"{rev}^{{commit}}", check=False).returncode == 0


# ------------------------------------------------------------------------------------------------ results

@dataclasses.dataclass(frozen=True)
class ChangeSet:
    """What a sandbox changed relative to its base: path -> 'A'dded / 'M'odified / 'D'eleted."""
    base: str
    paths: Mapping[str, str]

    @property
    def files(self) -> list[str]:
        return sorted(self.paths)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(sorted(self.paths.items())).encode()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """One sandbox's build + test comparison against its base (inputs to the evaluator; no verdict about IMPROVEMENT here)."""
    change: ChangeSet
    build: B.BuildResult
    selection: T.TestSelection
    base_run: Optional[T.TestRun]
    candidate_run: Optional[T.TestRun]
    report: Optional[T.RegressionReport]
    evidence_dir: str

    @property
    def builds(self) -> bool:
        return bool(self.build.ok)

    @property
    def clean(self) -> bool:
        return self.builds and self.report is not None and self.report.verdict is T.Verdict.CLEAN

    def to_record(self) -> dict[str, Any]:
        return {"base": self.change.base, "changed": dict(self.change.paths), "change_digest": self.change.digest(),
                "builds": self.builds, "build": self.build.to_record() if hasattr(self.build, "to_record") else str(self.build.ok),
                "selection": self.selection.to_dict(), "report": self.report.to_record() if self.report else None,
                "evidence_dir": self.evidence_dir}


@dataclasses.dataclass(frozen=True)
class AdoptResult:
    sandbox_id: str
    branch: str
    sandbox_commit: str
    merge_commit: str
    main_before: str


# ------------------------------------------------------------------------------------------------ the sandbox

class Sandbox:
    """    sb = Sandbox.open(repo, base="HEAD", scratch=...)
           sb.write("creator/x.py", text)          # refused for protected paths
           ev = sb.evaluate()                      # build + base-vs-candidate tests
           sb.close()                              # always; discards the worktree (the branch stays until adopt/discard)
    """

    def __init__(self, repo: Path, sid: str, base: str, path: Path, branch: str, scratch: Path):
        self.repo, self.id, self.base, self.path, self.branch, self.scratch = repo, sid, base, path, branch, scratch
        self.closed = False
        self.omit: tuple[str, ...] = ()

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def open(cls, repo: str | Path, base: str = "HEAD", scratch: str | Path | None = None, label: str = "",
             hide: Sequence[str] = (), omit: Sequence[str] = ()) -> "Sandbox":
        repo = Path(repo).resolve()
        if not commit_exists(repo, base):
            raise SandboxError(f"base {base!r} is not a known commit - a sandbox never starts from an unknown state")
        base_sha = git(repo, "rev-parse", base).stdout.strip()
        scratch_dir = Path(scratch).resolve() if scratch else device.sandbox_root(repo)
        if scratch_dir == repo or repo in scratch_dir.parents:
            raise SandboxError("the scratch directory must be OUTSIDE the main working tree")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        sid = f"{stamp}-{hashlib.sha256(f'{base_sha}{label}{time.time_ns()}'.encode()).hexdigest()[:8]}"
        path = scratch_dir / sid
        branch = f"{SANDBOX_PREFIX}{sid}"
        if hide or omit:
            # omit (2 Oct): paths never needed to develop or test the Creator - state/research holds 44k of the repo's 47k
            # tracked files, and checking them out timed out (120 s) under load; unlike `hide` they stay out after reveal().
            # a worker developing the Creator must not see the sealed answer keys (1 Oct): the hidden paths are left out of the
            # worktree by a per-worktree sparse checkout; they stay in the index (skip-worktree), so they are neither shown as
            # deleted nor dropped from the sandbox's commits, and the main worktree is not affected
            git(repo, "worktree", "add", "--no-checkout", "-b", branch, str(path), base_sha)
            git(path, "sparse-checkout", "set", "--no-cone", "/*", *[f"!/{h.strip('/')}/" for h in (*hide, *omit)])
            git(path, "checkout", "-q", branch)
            for h in (*hide, *omit):
                if (path / h).exists():
                    raise SandboxError(f"hidden path {h} is still present in the sandbox")
        else:
            git(repo, "worktree", "add", "-b", branch, str(path), base_sha)
        (path / ".creator_sandbox.json").write_text(json.dumps({"id": sid, "base": base_sha, "branch": branch, "label": label,
                                                                "opened": stamp, "state": "OPEN"}), encoding="utf-8")
        _exclude_marker(path)
        sb = cls(repo, sid, base_sha, path, branch, scratch_dir)
        sb.omit = tuple(omit)
        return sb

    def reveal(self) -> None:
        """Bring hidden paths back (after the worker is done; the evaluation and the Creator's own tests need the full tree).
        Omitted paths stay out: nothing in the Creator's development or tests uses them."""
        if self.omit:
            git(self.path, "sparse-checkout", "set", "--no-cone", "/*", *[f"!/{h.strip('/')}/" for h in self.omit])
        else:
            git(self.path, "sparse-checkout", "disable")

    def close(self, delete_branch: bool = False) -> None:
        """Remove the worktree (idempotent). The branch is kept unless asked, so an adopted or audited change stays inspectable."""
        if self.closed:
            return
        if self.path.exists():
            git(self.repo, "worktree", "remove", "--force", str(self.path), check=False)
            if self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)
        git(self.repo, "worktree", "prune", check=False)
        if delete_branch:
            git(self.repo, "branch", "-D", self.branch, check=False)
        self.closed = True

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- changing files (firewalled)

    def _target(self, rel: str) -> Path:
        rel = _safe_rel(rel)
        if is_protected(rel):
            raise ProtectedPathError(f"refused: {rel} is protected (canon, contracts, locks, sealed benchmark, auditor, CI, "
                                     f"evaluator thresholds or sealed live state)")
        return self.path / rel

    def write(self, rel: str, text: str) -> None:
        t = self._target(rel)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(text.encode("utf-8"))

    def delete(self, rel: str) -> None:
        t = self._target(rel)
        if t.exists():
            t.unlink()

    def apply(self, files: Mapping[str, Optional[str]]) -> None:
        """Write (text) or delete (None) several files; ALL paths are checked before ANY is touched (no partial application)."""
        for rel in files:
            self._target(rel)
        for rel, text in files.items():
            if text is None:
                self.delete(rel)
            else:
                self.write(rel, text)

    def apply_patch(self, patch: str) -> None:
        """Apply a unified diff (from an implementer) after checking every path it touches."""
        touched = {line[6:].strip() for line in patch.splitlines() if line.startswith(("+++ b/", "--- a/"))}
        for rel in touched:
            if rel and rel != "/dev/null":
                self._target(rel)
        proc = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=self.path, input=patch, text=True,
                              capture_output=True, timeout=GIT_TIMEOUT)
        if proc.returncode != 0:
            raise SandboxError(f"patch does not apply: {proc.stderr.strip()[:500]}")

    def changes(self) -> ChangeSet:
        """Every path that differs from the base (staged, unstaged, untracked), excluding the sandbox marker."""
        git(self.path, "add", "-A")
        out = git(self.path, "diff", "--cached", "--name-status", self.base).stdout
        paths: dict[str, str] = {}
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                status, rel = parts[0][0], parts[-1]
                if rel != ".creator_sandbox.json":
                    paths[rel] = status
        for rel in paths:                                   # a change sneaked in by another route is still refused
            if is_protected(rel):
                raise ProtectedPathError(f"refused: the sandbox contains a change to protected {rel}")
        return ChangeSet(self.base, paths)

    def diff(self) -> str:
        git(self.path, "add", "-A")
        return git(self.path, "diff", "--cached", self.base, "--", ".", ":(exclude).creator_sandbox.json").stdout

    # ---------------------------------------------------------------- build + test vs base

    def _base_tree(self) -> Path:
        """A read-only worktree at the base commit, for the baseline build/test on the SAME selection."""
        p = self.scratch / f"{self.id}-base"
        if not p.exists():
            if self.omit:                                     # 2 Oct, CP0091: the base tree checked out all 47k files and timed out
                git(self.repo, "worktree", "add", "--detach", "--no-checkout", str(p), self.base)
                git(p, "sparse-checkout", "set", "--no-cone", "/*", *[f"!/{h.strip('/')}/" for h in self.omit])
                git(p, "checkout", "-q", "--detach", self.base)
            else:
                git(self.repo, "worktree", "add", "--detach", str(p), self.base)
        return p

    def evaluate(self, smoke: Iterable[str] = (), build_config: Optional[B.BuildConfig] = None,
                 pytest_config: Optional[T.PytestConfig] = None, run_base: bool = True,
                 flaky_reruns: int = 2, base_reuse: Optional[Mapping[str, str]] = None, widen_reason: str = "") -> Evaluation:
        """Build the candidate; select affected tests; run them at base and in the candidate; classify every case. Blocking or
        fixed cases are re-run `flaky_reruns` times per side (0 = off): a case whose outcome flips is FLAKY, never a
        regression by itself, and the verdict is then at best FLAKY (not CLEAN). `base_reuse` (test file -> junit xml) holds passing
        results of the byte-identical base tree (creator/treecache.py): those files are not run at base again. `widen_reason` (creator.decide:
        a trusted high-risk prediction) runs EVERY test instead of the selection: the selection only grows, and an empty selection is never
        widened (it ends the evaluation with no report, i.e. a rejection, exactly as before)."""
        for cache in list(self.path.rglob("__pycache__")):              # a worker's bytecode is never judged in place of its
            shutil.rmtree(cache, ignore_errors=True)                     # source (1 Oct: same-size same-second rewrite ran stale)
        change = self.changes()
        if not change.paths:
            raise SandboxError("nothing changed - there is nothing to evaluate")
        ev_dir = self.scratch / f"{self.id}-evidence"
        ev_dir.mkdir(parents=True, exist_ok=True)
        base_tree = self._base_tree() if run_base else None
        base_keys: frozenset = frozenset()
        cfg = build_config or B.BuildConfig()
        if base_tree is not None and cfg.run_typecheck:
            cmd = cfg.typecheck_command if cfg.typecheck_command is not None else B.ci_typecheck_command(base_tree, cfg.python)
            base_keys = B.baseline_type_keys(B.typecheck(base_tree, cmd, timeout=cfg.typecheck_timeout,
                                                         toplevels=B.repo_toplevels(base_tree)))
        graph = T.ImportGraph.build(self.path)
        build = B.run_build(self.path, [p for p, s in change.paths.items() if s != "D"], config=cfg, tree="candidate",
                            baseline_keys=base_keys)
        selection = T.select_tests(graph, change.files, smoke)
        if widen_reason:                                                # creator.decide: a trusted high risk only ever ADDS tests
            from creator import registry as REG
            selection = REG.get("decide").widen_selection(selection, graph, widen_reason)
        if not build.ok or selection.empty:
            return Evaluation(change, build, selection, None, None, None, str(ev_dir))
        targets = list(selection.tests)
        cand = T.run_pytest(self.path, targets, ev_dir / "candidate.xml", label="candidate", tree=self.id, config=pytest_config)
        base_run = None
        if base_tree is not None:
            base_targets = [t for t in targets if (base_tree / t).exists()]
            base_run = T.run_base(base_tree, base_targets, ev_dir / "base.xml", tree=self.base, config=pytest_config,
                                  reuse=base_reuse)
            report = T.compare_runs(base_run, cand, base_files=base_targets)
            if flaky_reruns > 0:
                def rerun(side: str, ids: list[str]) -> T.TestRun:
                    root, jp = (self.path, ev_dir / "rerun-candidate.xml") if side == "candidate" else (base_tree, ev_dir / "rerun-base.xml")
                    return T.run_pytest(root, ids, jp, label=f"rerun-{side}", tree=self.id if side == "candidate" else self.base,
                                        config=pytest_config)
                report = T.apply_flaky_reruns(report, rerun, reruns=flaky_reruns, baseline=base_run, candidate=cand)
        else:
            report = T.compare_runs(T.TestRun("base", self.base, (), T.RunStatus.NO_TESTS, {}, None, 0.0), cand)
        (ev_dir / "evaluation.json").write_text(json.dumps({"report": report.to_record(), "selection": selection.to_dict(),
                                                            "changed": dict(change.paths),
                                                            "base_reused": sorted(set(base_reuse or ()) & set(targets))}, indent=1), encoding="utf-8")
        return Evaluation(change, build, selection, base_run, cand, report, str(ev_dir))

    def cleanup_base(self) -> None:
        p = self.scratch / f"{self.id}-base"
        if p.exists():
            git(self.repo, "worktree", "remove", "--force", str(p), check=False)
            shutil.rmtree(p, ignore_errors=True)
        git(self.repo, "worktree", "prune", check=False)
        shutil.rmtree(self.scratch / f"{self.id}-evidence", ignore_errors=True)   # scratch copy; the kernel keeps its own record

    # ---------------------------------------------------------------- adopt / rollback

    def commit(self, message: str) -> str:
        change = self.changes()
        if not change.paths:
            raise SandboxError("nothing to commit")
        git(self.path, "-c", "user.name=creator", "-c", "user.email=creator@localhost", "commit", "-q", "-m", message)
        return head(self.path)


MARKER = ".creator_sandbox.json"
HANDOFF_FILES = (".creator_task.md", ".creator_done.json")      # kernel <-> Claude-session handoff, never committed


def _exclude_marker(path: Path) -> None:
    """Keep the sandbox marker file out of git. git reads info/exclude only from the COMMON dir, never a worktree's private
    gitdir (1 Oct: the marker was committed into main by the first adoption) - so write it where git reads it."""
    rel = git(path, "rev-parse", "--git-path", "info/exclude").stdout.strip()
    excl = Path(rel) if Path(rel).is_absolute() else path / rel
    excl.parent.mkdir(parents=True, exist_ok=True)
    text = excl.read_text(encoding="utf-8") if excl.is_file() else ""
    missing = [m for m in (MARKER, *HANDOFF_FILES) if m not in text.split()]
    if missing:
        with excl.open("a", encoding="utf-8") as fh:
            fh.write("\n" + "\n".join(missing) + "\n")


@contextlib.contextmanager
def _main_lock(repo: Path) -> Iterator[None]:
    lock = repo / ".git" / "creator_adopt.lock"
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() - t0 > 60:
                raise SandboxError("another adopt/rollback holds the main-tree lock") from None
            time.sleep(0.1)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(lock)


def adopt(sb: Sandbox, decision: Any, message: str) -> AdoptResult:
    """Merge a sandbox into the main tree - ONLY with an ADOPT decision object (the evaluator's, recorded in the ledger) whose
    subject is this change. The main tree must have no uncommitted changes to the files the change touches; a conflicting merge is
    aborted and refused (never forced). Returns the merge commit for a later rollback."""
    verdict = getattr(decision, "verdict", None)
    if getattr(verdict, "value", verdict) != "ADOPT":
        raise SandboxError("adopt needs an ADOPT decision; a change never becomes authoritative because it executed (C77 sec 44)")
    change = sb.changes()
    sb_commit = sb.commit(message)
    repo = sb.repo
    with _main_lock(repo):
        dirty = git(repo, "status", "--porcelain", "--", *change.files).stdout.strip() if change.files else ""
        if dirty:
            raise SandboxError(f"main tree has uncommitted changes to files this change touches: {dirty[:300]}")
        before = head(repo)
        proc = git(repo, "-c", "user.name=creator", "-c", "user.email=creator@localhost", "merge", "--no-ff", "--no-edit",
                   "-m", f"creator adopt {sb.id}: {message}", sb.branch, check=False)
        if proc.returncode != 0:
            git(repo, "merge", "--abort", check=False)
            raise SandboxError(f"merge of {sb.branch} conflicts with main - refused, main left at {before[:12]}: "
                               f"{proc.stdout.strip()[-300:]} {proc.stderr.strip()[-300:]}")
        merge = head(repo)
    return AdoptResult(sb.id, sb.branch, sb_commit, merge, before)


def rollback(repo: str | Path, merge_commit: str, reason: str) -> str:
    """Undo an adopted change with a revert commit (history is never rewritten). Returns the revert commit."""
    repo = Path(repo)
    if not commit_exists(repo, merge_commit):
        raise SandboxError(f"unknown merge commit {merge_commit}")
    parents = git(repo, "rev-list", "--parents", "-n", "1", merge_commit).stdout.split()
    with _main_lock(repo):
        args = ["revert", "--no-edit"] + (["-m", "1"] if len(parents) > 2 else []) + [merge_commit]
        proc = git(repo, "-c", "user.name=creator", "-c", "user.email=creator@localhost", *args, check=False)
        if proc.returncode != 0:
            git(repo, "revert", "--abort", check=False)
            raise SandboxError(f"revert of {merge_commit[:12]} failed - refused: {proc.stderr.strip()[:300]}")
        rev = head(repo)
        git(repo, "-c", "user.name=creator", "-c", "user.email=creator@localhost", "commit", "--amend", "-q", "-m",
            f"creator rollback {merge_commit[:12]}: {reason}", check=False)
        return head(repo) or rev


def discard(sb: Sandbox) -> None:
    """Reject a change that was never adopted: remove its worktree, base tree and branch."""
    sb.cleanup_base()
    sb.close(delete_branch=True)


def recover(repo: str | Path, scratch: str | Path | None = None) -> list[dict[str, Any]]:
    """After a crash (C77 sec 60): prune dead worktrees and report every sandbox that was left behind as INTERRUPTED - its
    change was never adopted unless a merge commit names it; nothing is assumed to have succeeded."""
    repo = Path(repo).resolve()
    git(repo, "worktree", "prune", check=False)
    scratch_dir = Path(scratch).resolve() if scratch else device.sandbox_root(repo)
    found: list[dict[str, Any]] = []
    branches = {b.strip().lstrip("*+ ").strip() for b in git(repo, "branch", "--list", f"{SANDBOX_PREFIX}*").stdout.splitlines()}
    merged = git(repo, "log", "--merges", "--format=%s", "-n", "500", check=False).stdout
    for br in sorted(b for b in branches if b):
        sid = br[len(SANDBOX_PREFIX):]
        adopted = f"creator adopt {sid}" in merged
        wt = scratch_dir / sid
        found.append({"id": sid, "branch": br, "worktree_exists": wt.exists(), "adopted": adopted,
                      "state": "ADOPTED" if adopted else "INTERRUPTED"})
    return found
