"""Creator K14 - the self-development kernel loop (C77 secs 10, 20-29, 44-46, 59-61, 64; package CR14) - IMPLEMENTED, NOT VALIDATED.

One CYCLE, every step recorded in the development ledger and in state/creator/cycles/<package>/:

    0 RECOVER    sandboxes left by a crash are discarded and reported INTERRUPTED; never assumed adopted (sec 60)
    1 ASSESS     fresh test evidence -> self-model -> requirement ladder -> gap sync -> AUDIT. A red audit (any CRITICAL finding)
                 stops the cycle: nothing is developed on a base whose own claims are not backed by evidence
    2 PLAN       planner.plan_next on WORKER steps only (validation is never a worker's job)
    3 SANDBOX    a git worktree from the current main commit, with the sealed answer keys hidden
    4 WORK       the worker (a budgeted, confined LLM call - or any Worker) changes files in the sandbox; its claim is recorded
                 and ignored
    5 EVALUATE   build + affected tests on base and candidate (sandbox.evaluate); test-weakening and hard-coded-answer audits on
                 the diff; the candidate's own self-model with fresh test evidence; every requirement re-checked on base and
                 candidate; Measurements + an ImprovementClaim computed by creator.model:improvement_verdict
    6 DECIDE     ADOPT only if the claim is IMPROVEMENT, the regression report is CLEAN and the diff audits are clean - the
                 Decision is validated by the ledger BEFORE the merge and appended after it, citing the merge commit
    7 VERIFY     main is re-assessed after the merge; any new audit finding or a failure of the adopted requirement rolls the
                 merge back (revert commit) with a ROLLBACK decision
    8 RECORD     planner outcome, StrategyOutcome (input to K13), ledger checkpoint, cycle report, kernel log

`run(max_cycles)` repeats until no gap is plannable, the budget refuses, an audit is red, or the cycle limit is reached. The kernel
never pushes to a remote and never touches protected paths (the sandbox refuses them)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from creator import agents as AG
from creator import build as B
from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import sandbox as S
from creator import selfmodel as SM
from creator import testrun as T
from creator.audit import checks as AUD
from creator.ledger import Ledger

HIDE = ("creator/devbench/sealed",)


class KernelError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ configuration

@dataclasses.dataclass(frozen=True)
class KernelConfig:
    repo: Path
    state: Path                                         # where the ledger, cycles and logs live (inside repo: evidence root)
    scratch: Optional[Path] = None                      # sandboxes (outside repo)
    capabilities: Optional[Sequence[SM.CapabilitySpec]] = None
    scope: tuple[str, ...] = ("creator", "tests", "scripts")   # scripts are real callers (integration), not just entry points
    hide: tuple[str, ...] = HIDE
    build: B.BuildConfig = dataclasses.field(default_factory=B.BuildConfig)
    pytest: T.PytestConfig = dataclasses.field(default_factory=T.PytestConfig)
    test_timeout: float = 900.0
    sealed_root: Optional[Path] = None                  # repository whose sealed keys the diff audit compares against
    steps: tuple[str, ...] = P.WORKER_STEPS             # which requirement steps workers may be planned for (never 'validated')

    @property
    def ledger_path(self) -> Path:
        return self.state / "ledger.jsonl"

    def specs(self) -> list[SM.CapabilitySpec]:
        return list(self.capabilities) if self.capabilities is not None else SM.load_capabilities()


# ------------------------------------------------------------------------------------------------ workers

@dataclasses.dataclass(frozen=True)
class WorkResult:
    claimed_done: bool
    notes: str
    calls: int = 0
    usd: float = 0.0
    refused: bool = False                               # the budget refused: nothing was attempted
    contaminated: tuple[str, ...] = ()                  # protected / answer-key references in what the worker said or wrote


class Worker(Protocol):
    name: str

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> WorkResult: ...


def git_changed_files(workdir: Path) -> list[Path]:
    """Files the worker added or modified in a sandbox (tracked changes + untracked, ignored files excluded)."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=workdir, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    files = []
    for ln in out.splitlines():
        rel = ln[3:].strip().strip('"')
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        if (workdir / rel).is_file():
            files.append(workdir / rel)
    return files


def render_package(plan: P.Plan, wp: M.WorkPackage, protected: Sequence[str] = S.PROTECTED) -> str:
    """The worker's prompt: the work package, verbatim fields only (no ledger internals, no answer keys)."""
    lines = [f"You are working in a git worktree of a Python project (the Creator). Package {wp.package_id}.",
             f"Objective: {wp.objective}", f"Why: {wp.why_it_exists}", "", "Do:"]
    lines += [f"- {s}" for s in wp.implementation_requirements]
    lines += ["", "Tests:"] + [f"- {s}" for s in wp.test_requirements]
    lines += ["", "Known interfaces:"] + [f"- {s}" for s in wp.interfaces[:20]]
    lines += ["", "Ways this commonly goes wrong (avoid them):"] + [f"- {s}" for s in wp.expected_failure_modes]
    lines += ["", "Done means (computed by the system, not by you):"] + [f"- {s}" for s in wp.completion_criteria]
    lines += ["", "Rules:", "- Python 3.11. Run tests exactly as `python -m pytest -q <files>` - no `cd`, pipes or redirects "
              "(other shell commands are refused).",
              "- Never edit these protected paths: " + ", ".join(p for p in protected if "*" not in p or p.endswith("/*")),
              "- Never weaken, skip or delete tests to make them pass.", "- Keep changes small and focused on this package."]
    return "\n".join(lines)


class HandoffWorker:
    """The ONLY real worker (owner, 1 Oct 2026: "you should be the only claude worker working on it"): the kernel writes the
    package into the sandbox as .creator_task.md and waits for the Claude session to implement it there and write
    .creator_done.json ({"claimed_done": bool, "notes": str}). The claim is recorded and ignored; the kernel still measures,
    decides, merges or rejects. A handoff that is not answered within `timeout_s` returns not-done (the kernel then sees an
    empty or partial change and rejects it)."""
    name = "claude-session"

    def __init__(self, poll_s: float = 10.0, timeout_s: float = 6 * 3600,
                 notify: Optional[Callable[[Path, str], None]] = None) -> None:
        self.poll_s, self.timeout_s, self.notify = poll_s, timeout_s, notify

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> WorkResult:
        task, done = workdir / S.HANDOFF_FILES[0], workdir / S.HANDOFF_FILES[1]
        done.unlink(missing_ok=True)
        task.write_text(render_package(plan, package), encoding="utf-8")
        if self.notify:
            self.notify(workdir, plan.package_id)
        t0 = time.monotonic()
        while not done.exists():
            if time.monotonic() - t0 > self.timeout_s:
                task.unlink(missing_ok=True)
                return WorkResult(False, f"handoff not answered within {self.timeout_s:.0f}s")
            time.sleep(self.poll_s)
        try:
            ans = json.loads(done.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            ans = {"claimed_done": False, "notes": f"unreadable .creator_done.json: {e}"}
        task.unlink(missing_ok=True)
        done.unlink(missing_ok=True)
        return WorkResult(bool(ans.get("claimed_done")), str(ans.get("notes", ""))[:2000], 0, 0.0)


class AgentWorker:
    """A budgeted LLM worker (creator.agents). NOT used for real work any more (owner, 1 Oct 2026): agent calls are disabled
    in creator.agents; kept so the runtime's tests and the history stay explicable."""

    def __init__(self, budget: AG.Budget, spec: AG.AgentSpec = AG.DEVELOPER, runner: AG.Runner = AG.subprocess_runner,
                 runs_dir: Path = AG.RUNS_DIR, name: str = "agent-implementer-v1") -> None:
        self.budget, self.spec, self.runner, self.runs_dir, self.name = budget, spec, runner, runs_dir, name

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> WorkResult:
        try:
            run = AG.run_agent(self.spec, f"kernel:{package.package_id}", render_package(plan, package), workdir, self.budget,
                               self.runner, self.runs_dir, markers=AG.CONTAMINATION_MARKERS, changed_files=git_changed_files)
        except AG.BudgetError as e:
            return WorkResult(False, f"budget refused: {e}", 0, 0.0, refused=True)
        return WorkResult(run.claimed_done, f"{run.outcome} run={run.run_id} turns={run.turns}", 1, run.usd,
                          contaminated=run.contamination)


# ------------------------------------------------------------------------------------------------ assessment

@dataclasses.dataclass(frozen=True)
class Assessed:
    model: SM.SelfModel
    rows: tuple[G.Assessment, ...]
    audit: AUD.AuditReport
    snapshot: Path


def assess_tree(cfg: KernelConfig, led: Ledger, root: Path, label: str, run_tests: bool = True,
                audit: bool = True) -> Assessed:
    """Fresh evidence for `root` (main or a sandbox): run the declared tests, build the self-model, re-check every requirement."""
    specs = cfg.specs()
    tests = sorted({t for s in specs for t in s.tests if (root / t).is_file()})
    evd = cfg.state / "evidence" / label
    evd.mkdir(parents=True, exist_ok=True)
    ev = SM.collect_test_evidence(root, tests, evd / "test_evidence.json", timeout=cfg.test_timeout, junit_dir=evd / "junit") \
        if run_tests and tests else {}
    model = SM.build(root, scope=cfg.scope, capabilities=specs, test_evidence=ev, ledger=led)
    snap = evd / f"selfmodel_{model.digest()}.json"
    SM.save(model, snap)
    rows = tuple(G.assess(led, model))
    rep = AUD.audit(led, model, repo=cfg.repo) if audit else AUD.AuditReport((), (), {})
    return Assessed(model, rows, rep, snap)


def red(report: AUD.AuditReport) -> bool:
    return bool(report.errors) or report.count("CRITICAL") > 0


# ------------------------------------------------------------------------------------------------ the gap-closure claim

def _measure(led: Ledger, ex: str, metric: str, value: float, split: M.Split, population: str, conditions: str,
             evidence: Sequence[M.EvidenceRef], inputs: Any) -> str:
    from creator.evaluate import computation
    return led.append(M.Measurement(created_by=M.Role.VALIDATOR, parents=(ex,), metric=metric, value=float(value), stderr=0.0,
                                    n=1, population=population, conditions=conditions, higher_is_better=True, split=split,
                                    computation=computation("creator.kernel:gap_closure_claim", inputs, value),
                                    evidence=tuple(evidence)))


def _pass_fraction(run: Optional[T.TestRun], default: float) -> float:
    """Passed / all cases; `default` when nothing ran (no affected tests at that tree = nothing there to regress)."""
    if run is None or not run.cases:
        return default
    return sum(1 for c in run.cases.values() if c.outcome is T.Outcome.PASSED) / len(run.cases)


def gap_closure_claim(led: Ledger, plan: P.Plan, base: Assessed, cand: Sequence[Assessed], ev: Optional[S.Evaluation],
                      evidence: Sequence[M.EvidenceRef]) -> tuple[str, M.Verdict, dict[str, Any]]:
    """C77 sec 30 for a development change. Deterministic checks, so stderr is 0 and replicates must AGREE (two independent
    candidate assessments = reproducibility):
        primary  (DEV)      the targeted check, base vs each candidate replicate
        guard    (DEV)      affected-tests pass fraction, base vs candidate (sandbox.evaluate)
        guard    (DEV)      requirements KEPT: the fraction of base-met requirements still met on the candidate (1 Oct: a
                            change that closed several K02 checks while breaking K01.no_stubs had a net holdout GAIN and was
                            adopted - the post-merge audit rolled it back; any loss is now a pre-merge regression)
        holdout  (HOLDOUT)  total requirements met, every check recomputed - a change that fixes one requirement and breaks
                            another shows no holdout gain and is not adopted"""
    key = plan.requirement_key
    conditions = f"check recomputation; base {plan.experiment_id}"
    pop = "creator self-model requirements"

    def met(a: Assessed, k: str) -> float:
        return float(next((r.met for r in a.rows if r.key == k), False))

    def total(a: Assessed) -> float:
        return sum(r.met for r in a.rows) / max(1, len(a.rows))
    metric = f"check:{plan.step}:{plan.component}"
    base_ids = [_measure(led, plan.experiment_id, metric, met(base, key), M.Split.DEV, pop, conditions, evidence,
                         {"tree": "base", "r": r}) for r in range(len(cand))]
    cand_ids = [_measure(led, plan.experiment_id, metric, met(c, key), M.Split.DEV, pop, conditions, evidence,
                         {"tree": "candidate", "r": r}) for r, c in enumerate(cand)]
    gb = _measure(led, plan.experiment_id, "affected_tests_pass", _pass_fraction(ev.base_run if ev else None, 1.0),
                  M.Split.DEV, pop, conditions, evidence, {"guard": "base"})
    gc = _measure(led, plan.experiment_id, "affected_tests_pass",
                  _pass_fraction(ev.candidate_run if ev else None, 1.0 if ev is not None and ev.builds else 0.0),
                  M.Split.DEV, pop, conditions, evidence, {"guard": "candidate"})
    base_met = {r.key for r in base.rows if r.met}

    def kept(a: Assessed) -> float:
        now = {r.key for r in a.rows if r.met}
        return len(base_met & now) / len(base_met) if base_met else 1.0
    kb = _measure(led, plan.experiment_id, "requirements_kept", 1.0, M.Split.DEV, pop, conditions, evidence, {"guard": "base"})
    kc = _measure(led, plan.experiment_id, "requirements_kept", min(kept(c) for c in cand), M.Split.DEV, pop, conditions,
                  evidence, {"guard": "candidate", "lost": sorted(base_met - {r.key for c in cand for r in c.rows if r.met})})
    hb = _measure(led, plan.experiment_id, "requirements_met", total(base), M.Split.HOLDOUT, pop, conditions, evidence,
                  {"holdout": "base"})
    hc = _measure(led, plan.experiment_id, "requirements_met", min(total(c) for c in cand), M.Split.HOLDOUT, pop, conditions,
                  evidence, {"holdout": "candidate"})
    def g(i: str) -> M.Measurement:
        rec = led.get(i)
        if not isinstance(rec, M.Measurement):
            raise KernelError(f"{i} is not a Measurement")
        return rec
    verdict, detail = M.improvement_verdict([g(i) for i in base_ids], [g(i) for i in cand_ids], [(g(gb), g(gc)), (g(kb), g(kc))],
                                            (g(hb), g(hc)))
    lost = sorted(base_met - {r.key for c in cand for r in c.rows if r.met})
    if lost:
        detail = dict(detail, requirements_lost=lost)
    from creator.evaluate import computation
    ids = base_ids + cand_ids + [gb, gc, kb, kc, hb, hc]
    cid = led.append(M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(plan.experiment_id,), subject_id=plan.experiment_id,
                                        baseline_ids=tuple(base_ids), candidate_ids=tuple(cand_ids), verdict=verdict,
                                        computation=computation(M.VERDICT_FUNCTION, ids, verdict.value, ids),
                                        regression_baseline_ids=(gb, kb), regression_candidate_ids=(gc, kc),
                                        holdout_baseline_id=hb, holdout_candidate_id=hc))
    return cid, verdict, detail


# ------------------------------------------------------------------------------------------------ one cycle

@dataclasses.dataclass
class CycleReport:
    cycle: int
    outcome: str                                        # ADOPTED / REJECTED / ROLLED_BACK / NOTHING_TO_DO / AUDIT_RED / BUDGET / ERROR
    package: str = ""
    requirement: str = ""
    reason: str = ""
    verdict: str = ""
    merge_commit: str = ""
    calls: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


def _test_sources(root: Path, base: str, paths: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for p in paths:
        if not Path(p).name.startswith("test_"):
            continue
        old = subprocess.run(["git", "show", f"{base}:{p}"], cwd=root, capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        if old.returncode == 0:
            before[p] = old.stdout
        if (root / p).is_file():
            after[p] = (root / p).read_text(encoding="utf-8", errors="replace")
    return before, after


def _evidence_file(cfg: KernelConfig, pkg: str, name: str, payload: Any) -> M.EvidenceRef:
    d = cfg.state / "cycles" / pkg
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return M.EvidenceRef.of(p, cfg.repo, "cycle")


def recover(cfg: KernelConfig) -> list[str]:
    """Discard every sandbox a crashed cycle left behind (never adopted unless a merge names it) - worktree and branch."""
    gone = []
    scratch = cfg.scratch.resolve() if cfg.scratch else cfg.repo.parent / f".{cfg.repo.name}_creator_sandboxes"
    for leftover in scratch.glob("*-evidence") if scratch.is_dir() else ():
        shutil.rmtree(leftover, ignore_errors=True)
    for r in S.recover(cfg.repo, cfg.scratch):
        if r.get("state") != "INTERRUPTED":
            continue
        wt = scratch / str(r["id"])
        S.git(cfg.repo, "worktree", "remove", "--force", str(wt), check=False)
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        base_wt = scratch / f"{r['id']}-base"
        if base_wt.exists():
            S.git(cfg.repo, "worktree", "remove", "--force", str(base_wt), check=False)
            shutil.rmtree(base_wt, ignore_errors=True)
        S.git(cfg.repo, "worktree", "prune", check=False)
        S.git(cfg.repo, "branch", "-D", str(r["branch"]), check=False)
        gone.append(str(r["id"]))
    return gone


class _KernelLock:
    """One kernel at a time per state directory (O_EXCL lock file; a stale lock older than `stale_s` is reported, not stolen)."""

    def __init__(self, state: Path, stale_s: float = 6 * 3600) -> None:
        self.path = state / "kernel.lock"
        self.stale_s = stale_s

    def __enter__(self) -> "_KernelLock":
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime
            raise KernelError(f"another kernel holds {self.path} ({age:.0f}s old"
                              + ("; looks stale - remove it only after checking no kernel runs" if age > self.stale_s else "")
                              + ")") from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.path.unlink(missing_ok=True)


def cycle(cfg: KernelConfig, worker: Worker, n: int = 1, led: Optional[Ledger] = None) -> CycleReport:
    with _KernelLock(cfg.state):
        return _cycle(cfg, worker, n, led)


def _cycle(cfg: KernelConfig, worker: Worker, n: int = 1, led: Optional[Ledger] = None) -> CycleReport:
    t0 = time.monotonic()
    led = led or Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    recovered = recover(cfg)                                            # 0 RECOVER
    oid = O.self_objective(led)                                         # 1 ASSESS
    O.compile_capabilities(led, oid, cfg.specs())
    main = assess_tree(cfg, led, cfg.repo, "main")
    G.sync(led, main.model)
    if red(main.audit):
        return CycleReport(n, "AUDIT_RED", reason="; ".join(f"{f.check}:{f.subject}" for f in main.audit.findings[:5])
                           or str(main.audit.errors), seconds=round(time.monotonic() - t0, 1))
    base_sha = S.head(cfg.repo)
    steps = tuple(s for s in cfg.steps if s in P.WORKER_STEPS)          # a validator step can never be handed to a worker
    plan = P.plan_next(led, main.model, base_sha, cfg.specs(), steps=steps)   # 2 PLAN
    if plan is None:
        return CycleReport(n, "NOTHING_TO_DO", reason="no unblocked worker gap", seconds=round(time.monotonic() - t0, 1))
    wp = led.get(plan.work_package_id)
    assert isinstance(wp, M.WorkPackage)
    led.transition(plan.work_package_id, M.Status.IN_PROGRESS, "cycle started", M.Role.KERNEL)
    rep = CycleReport(n, "ERROR", plan.package_id, plan.requirement_key, details={"recovered": recovered})
    sb = S.Sandbox.open(cfg.repo, base_sha, cfg.scratch, label=plan.package_id, hide=cfg.hide)   # 3 SANDBOX
    try:
        work = worker(plan, wp, sb.path)                                # 4 WORK
        rep.calls, rep.usd = work.calls, work.usd
        rep.details["worker"] = {"claimed_done": work.claimed_done, "notes": work.notes}
        if work.refused:
            S.discard(sb)
            led.transition(plan.work_package_id, M.Status.BLOCKED, work.notes, M.Role.KERNEL)
            led.transition(plan.gap_id, M.Status.FAILED, f"{plan.package_id} not attempted: {work.notes}", M.Role.KERNEL)
            rep.outcome, rep.reason = "BUDGET", work.notes
            return rep
        if work.contaminated:
            raise _Reject(f"worker run contaminated: {list(work.contaminated)[:5]}")
        change = sb.changes()                                           # 5 EVALUATE
        if not change.paths:
            raise _Reject("the worker changed nothing")
        frozen = hashlib.sha256(sb.diff().encode()).hexdigest()         # RESULT FREEZE (content) before anything runs in the tree
        if cfg.hide:
            sb.reveal()                                                 # the Creator's own tests need the sealed suite back
        ev = sb.evaluate(build_config=cfg.build, pytest_config=cfg.pytest)
        before, after = _test_sources(sb.path, base_sha, change.files)
        weak = AUD.check_test_weakening(before, after)
        planted = AUD.check_hardcoded_answers(repo=sb.path, sealed_root=cfg.sealed_root or cfg.repo)
        cand = [assess_tree(cfg, led, sb.path, f"{plan.package_id}_cand{r}", audit=False) for r in range(2)]
        evid = [_evidence_file(cfg, plan.package_id, "evaluation.json", ev.to_record()),
                M.EvidenceRef.of(main.snapshot, cfg.repo, "selfmodel"), M.EvidenceRef.of(cand[0].snapshot, cfg.repo, "selfmodel"),
                _evidence_file(cfg, plan.package_id, "diff.patch", sb.diff() if hasattr(sb, "diff") else "")]
        if hashlib.sha256(sb.diff().encode()).hexdigest() != frozen:
            raise _Reject("the change set moved during evaluation (something in the tree rewrote files)")
        cid, verdict, detail = gap_closure_claim(led, plan, main, cand, ev, evid)
        rep.verdict = verdict.value
        rep.details.update(claim=cid, detail=detail, regression=ev.report.verdict.value if ev.report else "NO_REPORT",
                           builds=ev.builds, weakening=[dataclasses.asdict(f) for f in weak],
                           planted=[dataclasses.asdict(f) for f in planted])
        reasons = []
        if verdict is not M.Verdict.IMPROVEMENT:
            reasons.append(f"claim {verdict.value}: {detail.get('why')}")
        if not ev.clean:
            reasons.append(f"sandbox evaluation not clean (builds={ev.builds}, "
                           f"report={ev.report.verdict.value if ev.report else None})")
        if weak:
            reasons.append(f"tests weakened: {[f.detail for f in weak]}")
        if planted:
            reasons.append(f"hard-coded answers: {[f.subject for f in planted]}")
        if reasons:
            raise _Reject("; ".join(reasons))
        decision = M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.ADOPT,
                              reason=f"{plan.requirement_key} closed with no regression", claim_id=cid)
        problems = led.problems(decision)                               # 6 DECIDE (validated before the merge)
        if problems:
            raise _Reject(f"ledger refuses the ADOPT decision: {problems}")
        res = S.adopt(sb, decision, f"{plan.package_id} {plan.requirement_key}")
        merge_ev = _evidence_file(cfg, plan.package_id, res.merge_commit, f"{res.merge_commit}\n")
        led.append(dataclasses.replace(decision, evidence=(dataclasses.replace(merge_ev, kind="merge_commit"),)))
        rep.merge_commit = res.merge_commit
        sb.cleanup_base()
        sb.close()
        after_main = assess_tree(cfg, led, cfg.repo, f"{plan.package_id}_main_after")   # 7 VERIFY
        G.sync(led, after_main.model)
        still = next((r.met for r in after_main.rows if r.key == plan.requirement_key), False)
        if red(after_main.audit) or not still:
            why = "requirement not met on main after merge" if not still else \
                f"audit red after merge: {[f.check for f in after_main.audit.findings[:5]]}"
            revert = S.rollback(cfg.repo, res.merge_commit, why)
            led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.ROLLBACK,
                                  reason=f"{why}; reverted by {revert[:12]}"))
            P.record_outcome(led, plan, False, f"rolled back: {why}")
            rep.outcome, rep.reason = "ROLLED_BACK", why
        else:
            P.record_outcome(led, plan, True, f"adopted as {res.merge_commit[:12]}")
            rep.outcome, rep.reason = "ADOPTED", f"merged {res.merge_commit[:12]}"
    except _Reject as r:
        S.discard(sb)
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.REJECT,
                              reason=str(r)[:2000]))
        P.record_outcome(led, plan, False, str(r)[:500])
        rep.outcome, rep.reason = "REJECTED", str(r)
    except Exception as e:                                              # noqa: BLE001 - recorded, sandbox discarded, never adopted
        if not sb.closed:
            S.discard(sb)
        P.record_outcome(led, plan, False, f"kernel error: {type(e).__name__}: {e}"[:500])
        rep.outcome, rep.reason = "ERROR", f"{type(e).__name__}: {e}"
    finally:
        rep.seconds = round(time.monotonic() - t0, 1)                   # 8 RECORD
        led.append(M.StrategyOutcome(created_by=M.Role.KERNEL, parents=(plan.work_package_id,),
                                     strategy_id=getattr(worker, "name", "worker"), problem_class=plan.step,
                                     subject_id=plan.work_package_id, success=rep.outcome == "ADOPTED", cost=rep.usd,
                                     duration_s=rep.seconds))
        led.checkpoint(f"cycle {n} {plan.package_id} {rep.outcome}")
        _evidence_file(cfg, plan.package_id, "cycle.json", dataclasses.asdict(rep))
        with (cfg.state / "kernel_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dataclasses.asdict(rep), default=str) + "\n")
    return rep


class _Reject(Exception):
    pass


def run(cfg: KernelConfig, worker: Worker, max_cycles: int = 1, on_cycle: Optional[Callable[[CycleReport], None]] = None
        ) -> list[CycleReport]:
    """Repeat cycles; stop on NOTHING_TO_DO, AUDIT_RED, BUDGET, or the cycle limit."""
    out = []
    for n in range(1, max_cycles + 1):
        r = cycle(cfg, worker, n)
        out.append(r)
        if on_cycle:
            on_cycle(r)
        if r.outcome in ("NOTHING_TO_DO", "AUDIT_RED", "BUDGET"):
            break
    return out


def summary(reports: Sequence[CycleReport]) -> dict[str, Any]:
    by: dict[str, int] = {}
    for r in reports:
        by[r.outcome] = by.get(r.outcome, 0) + 1
    return {"cycles": len(reports), "by_outcome": by, "usd": round(sum(r.usd for r in reports), 4),
            "calls": sum(r.calls for r in reports), "adopted": [r.package for r in reports if r.outcome == "ADOPTED"]}


def status(cfg: KernelConfig) -> Mapping[str, Any]:
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    return G.summary(led)
