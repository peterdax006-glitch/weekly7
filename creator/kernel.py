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

import contextlib
import dataclasses
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol, Sequence

from creator import build as B
from creator import device
from creator import fundamentals as FU
from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import sandbox as S
from creator import selfmodel as SM
from creator import testrun as T
from creator import treecache as TC
from creator.audit import checks as AUD
from creator.ledger import Ledger

HIDE = ("creator/devbench/sealed",)
OMIT = ("state/research",)                          # never checked out in a sandbox (44k of 47k tracked files; 2 Oct timeout)


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
    omit: tuple[str, ...] = OMIT
    build: B.BuildConfig = dataclasses.field(default_factory=B.BuildConfig)
    pytest: T.PytestConfig = dataclasses.field(default_factory=T.PytestConfig)
    test_timeout: float = 3600.0                        # a full suite under swarm contention took 1205 s (1 Oct)
    test_parallel: int = 1                              # test files run side by side when assessing a tree
    sealed_root: Optional[Path] = None                  # repository whose sealed keys the diff audit compares against
    steps: tuple[str, ...] = P.WORKER_STEPS             # which requirement steps workers may be planned for (never 'validated')
    mode: str = "auto"                                  # auto: gaps, then shrink when none | gaps | efficiency (shrink only)
    measure_memory: bool = True
    reuse_candidate_tests: bool = True                  # 2nd candidate replicate / post-merge check serve PASS files of cand0 (same reach digest)

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
    by: str = ""                                        # which worker actually made the change (self_share counts it)
    reasoning: str = ""                                 # free-text reasoning of the solver (curriculum lesson); optional
    deferred: bool = False                              # handed over, not attempted: no attempt is used up (planner.attempts_for)
    predicted: Optional[dict[str, float]] = None        # the solver's predicted effect of its change, e.g. {'size_delta': -12} (creator.reasoning)


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


def _tools_rule() -> str:
    """The tools a worker may call and the policy in plain words (creator.tools, loaded on demand - never imported at start)."""
    try:
        import importlib
        return "- " + importlib.import_module("creator.tools.toolbox").policy_text()
    except ImportError:
        return "- Run tests as `python -m pytest -q <files>`; the tools are not available in this tree."


def render_package(plan: P.Plan, wp: M.WorkPackage, protected: Sequence[str] = S.PROTECTED) -> str:
    """The worker's prompt: the work package, verbatim fields only (no ledger internals, no answer keys)."""
    lines = [f"You are working in a git worktree of a Python project (the Creator). Package {wp.package_id}.",
             f"Objective: {wp.objective}", f"Why: {wp.why_it_exists}", "", "Do:"]
    lines += [f"- {s}" for s in wp.implementation_requirements]
    lines += ["", "Tests:"] + [f"- {s}" for s in wp.test_requirements]
    lines += ["", "Known interfaces:"] + [f"- {s}" for s in wp.interfaces[:20]]
    from creator import registry as REG                                  # the sparse-activation rule reaches every package
    lines += ["", REG.sparse_rule_text()]
    lines += ["", "Ways this commonly goes wrong (avoid them):"] + [f"- {s}" for s in wp.expected_failure_modes]
    lines += ["", "Done means (computed by the system, not by you):"] + [f"- {s}" for s in wp.completion_criteria]
    lines += ["", "Rules:", "- Python 3.11.", _tools_rule(),
              "- Never edit these protected paths: " + ", ".join(p for p in protected if "*" not in p or p.endswith("/*")),
              "- Never weaken, skip or delete tests to make them pass.", "- Keep changes small and focused on this package."]
    return "\n".join(lines)


class HandoffWorker:
    """The ONLY real worker (owner, 1 Oct 2026: "you should be the only claude worker working on it"): the kernel writes the
    package into the sandbox as .creator_task.md and waits for the Claude session to implement it there and write
    .creator_done.json ({"claimed_done": bool, "notes": str, optional "reasoning": str}). The claim is recorded and ignored; the kernel still measures,
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
        return WorkResult(bool(ans.get("claimed_done")), str(ans.get("notes", ""))[:2000], 0, 0.0, by=self.name,
                          reasoning=str(ans.get("reasoning", ""))[:20000])


# ------------------------------------------------------------------------------------------------ assessment

@dataclasses.dataclass(frozen=True)
class Assessed:
    model: SM.SelfModel
    rows: tuple[G.Assessment, ...]
    audit: AUD.AuditReport
    snapshot: Path
    tests: Mapping[str, SM.TestEvidence] = dataclasses.field(default_factory=dict, compare=False)


def pass_reuse(a: Assessed) -> dict[str, tuple[str, str]]:
    """test file -> (reach digest, junit xml) for every file that PASSED in `a`; collect_test_evidence serves a file from it only
    when its reach digest on the tree being assessed equals the recorded one (failures always re-run)."""
    return {t: (e.source_digest, e.where) for t, e in a.tests.items() if e.outcome == "PASS" and e.where}


def assess_tree(cfg: KernelConfig, led: Ledger, root: Path, label: str, run_tests: bool = True,
                audit: bool = True, reuse: Optional[Mapping[str, tuple[str, str]]] = None) -> Assessed:
    """Fresh evidence for `root` (main or a sandbox): run the declared tests, build the self-model, re-check every requirement.
    Main's round assessment (label "main") serves passing files from an earlier assessment of the byte-identical clean tree;
    every other label (the candidate replicates, the post-merge check) runs everything."""
    specs = cfg.specs()
    tests = sorted({t for s in specs for t in s.tests if (root / t).is_file()})
    evd = cfg.state / "evidence" / label
    evd.mkdir(parents=True, exist_ok=True)
    cache = TC.TreeCache(cfg.state / "evidence" / "tree_cache")
    key = TC.tree_key(root, T.PytestConfig().python) if run_tests and tests else None   # None: dirty or sandbox tree -> all fresh
    ev = SM.collect_test_evidence(root, tests, evd / "test_evidence.json", timeout=cfg.test_timeout, junit_dir=evd / "junit",
                                  parallel=cfg.test_parallel, reuse=(cache.lookup(key) if label == "main" else reuse)) \
        if run_tests and tests else {}
    if key and TC.tree_key(root, T.PytestConfig().python) == key:       # the tests left the tree as they found it
        cache.store(key, {t: ev[t] for t in tests if t in ev})
    model = SM.build(root, scope=cfg.scope, capabilities=specs, test_evidence=ev, ledger=led)
    tmp = evd / "selfmodel.tmp.json"
    SM.save(model, tmp)
    snap = evd / f"selfmodel_{hashlib.sha256(tmp.read_bytes()).hexdigest()[:16]}.json"                            # (2 Oct: digest-named snapshots were overwritten each
    if snap.exists():                                                   # cycle with different non-digested fields, and the next
        tmp.unlink()                                                    # cycle's audit went red with evidence drift)
    else:
        tmp.replace(snap)
    rows = tuple(G.assess(led, model))
    rep = AUD.audit(led, model, repo=cfg.repo) if audit else AUD.AuditReport((), (), {})
    return Assessed(model, rows, rep, snap, {t: ev[t] for t in tests if t in ev})


def sandbox_pytest(cfg: KernelConfig) -> T.PytestConfig:
    """The sandbox comparison's pytest config. Left at its default, its timeout follows the suite timeout: CP0047 (1 Oct) touched
    kernel.py, its selected tests outran the 600 s default under swarm load on base AND candidate, and a sound change came back
    INCONCLUSIVE although main's own assessment allows test_timeout for the same tests."""
    if cfg.pytest.timeout == T.PytestConfig().timeout:
        return dataclasses.replace(cfg.pytest, timeout=max(cfg.pytest.timeout, cfg.test_timeout))
    return cfg.pytest


def red(report: AUD.AuditReport) -> bool:
    return bool(report.errors) or report.count("CRITICAL") > 0


# ------------------------------------------------------------------------------------------------ the gap-closure claim

def _measure(led: Ledger, ex: str, metric: str, value: float, split: M.Split, population: str, conditions: str,
             evidence: Sequence[M.EvidenceRef], inputs: Any, higher_is_better: bool = True, stderr: float = 0.0) -> str:
    from creator.evaluate import computation
    return led.append(M.Measurement(created_by=M.Role.VALIDATOR, parents=(ex,), metric=metric, value=float(value),
                                    stderr=float(stderr), n=1, population=population, conditions=conditions,
                                    higher_is_better=higher_is_better, split=split,
                                    computation=computation("creator.kernel:gap_closure_claim", inputs, value),
                                    evidence=tuple(evidence)))


def _pass_fraction(run: Optional[T.TestRun], default: float, ev: Optional[S.Evaluation] = None) -> float:
    """Passed / all cases; `default` when nothing ran (no affected tests at that tree = nothing there to regress). Cases the
    post-re-run report classed FLAKY are excluded: a coin-flip outcome must not move the guard either way."""
    flaky = set(ev.report.ids(T.CaseClass.FLAKY)) if ev is not None and ev.report is not None else set()
    cases = [c for k, c in run.cases.items() if k not in flaky] if run is not None else []
    if not cases:
        return default
    return sum(1 for c in cases if c.outcome is T.Outcome.PASSED) / len(cases)


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
    gb = _measure(led, plan.experiment_id, "affected_tests_pass", _pass_fraction(ev.base_run if ev else None, 1.0, ev),
                  M.Split.DEV, pop, conditions, evidence, {"guard": "base"})
    gc = _measure(led, plan.experiment_id, "affected_tests_pass",
                  _pass_fraction(ev.candidate_run if ev else None, 1.0 if ev is not None and ev.builds else 0.0, ev),
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


def efficiency_claim(led: Ledger, plan: P.Plan, base: Assessed, cand: Sequence[Assessed], ev: Optional[S.Evaluation],
                     evidence: Sequence[M.EvidenceRef], base_fp: Any, cand_fps: Sequence[Any]) -> tuple[str, M.Verdict, dict[str, Any]]:
    """C77 sec 30 for a SHRINK (owner directive 1 Oct 2026). Lower is better for size and memory.
        primary  (DEV)      AST nodes of the targeted module, base vs two candidate recomputations
        guards   (DEV)      affected tests pass, requirements kept, test count, peak memory (replicate spread as stderr)
        holdout  (HOLDOUT)  AST nodes of the whole creator package - moving code elsewhere shows no gain"""
    import statistics
    pop, cond, ex = "creator package", f"efficiency; base {plan.experiment_id}", plan.experiment_id
    activation_kind = plan.requirement_key == P.ACTIVATION_KEY
    coverage_kind = plan.requirement_key == P.COVERAGE_KEY              # test-gap work: fewer public names no test names
    metric = ("active_ast_nodes:kernel_start" if activation_kind else
              f"uncovered_public:{plan.component}" if coverage_kind else f"ast_nodes:{plan.component}")

    def primary(f: Any) -> float:
        return f.active_nodes if activation_kind else f.uncovered if coverage_kind else f.target_size
    base_ids = [_measure(led, ex, metric, primary(base_fp), M.Split.DEV, pop, cond, evidence, {"tree": "base", "r": r},
                         higher_is_better=False) for r in range(len(cand_fps))]
    cand_ids = [_measure(led, ex, metric, primary(f), M.Split.DEV, pop, cond, evidence, {"tree": "candidate", "r": r},
                         higher_is_better=False) for r, f in enumerate(cand_fps)]
    base_met = {r.key for r in base.rows if r.met}
    lost = sorted(k for k in base_met if not all(any(r.key == k and r.met for r in c.rows) for c in cand))

    def mem(fp: Any) -> tuple[float, float]:
        xs = list(fp.memory_mb) or [0.0]
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return statistics.fmean(xs), max(0.25, sd / max(1.0, len(xs) ** 0.5))
    bm, bse = mem(base_fp)
    cm, cse = mem(cand_fps[0])
    guards = [
        (_measure(led, ex, "affected_tests_pass", _pass_fraction(ev.base_run if ev else None, 1.0, ev), M.Split.DEV, pop, cond,
                  evidence, {"guard": "base"}),
         _measure(led, ex, "affected_tests_pass", _pass_fraction(ev.candidate_run if ev else None,
                                                                 1.0 if ev is not None and ev.builds else 0.0, ev),
                  M.Split.DEV, pop, cond, evidence, {"guard": "candidate"})),
        (_measure(led, ex, "requirements_kept", 1.0, M.Split.DEV, pop, cond, evidence, {"guard": "base"}),
         _measure(led, ex, "requirements_kept", (len(base_met) - len(lost)) / len(base_met) if base_met else 1.0,
                  M.Split.DEV, pop, cond, evidence, {"guard": "candidate", "lost": lost})),
        (_measure(led, ex, "test_functions", base_fp.test_cases, M.Split.DEV, pop, cond, evidence, {"guard": "base"}),
         _measure(led, ex, "test_functions", min(f.test_cases for f in cand_fps), M.Split.DEV, pop, cond, evidence,
                  {"guard": "candidate"})),
    ]
    if base_fp.memory_mb and cand_fps[0].memory_mb:
        guards.append((_measure(led, ex, "peak_memory_mb", bm, M.Split.DEV, pop, cond, evidence, {"guard": "base"},
                                higher_is_better=False, stderr=bse),
                       _measure(led, ex, "peak_memory_mb", cm, M.Split.DEV, pop, cond, evidence, {"guard": "candidate"},
                                higher_is_better=False, stderr=cse)))
    if not activation_kind and base_fp.active_nodes >= 0 and cand_fps[0].active_nodes >= 0:    # never load more than needed
        guards.append((_measure(led, ex, "active_ast_nodes", base_fp.active_nodes, M.Split.DEV, pop, cond, evidence,
                                {"guard": "base"}, higher_is_better=False),
                       _measure(led, ex, "active_ast_nodes", cand_fps[0].active_nodes, M.Split.DEV, pop, cond, evidence,
                                {"guard": "candidate"}, higher_is_better=False)))
    if activation_kind:                                                 # the holdout: eager-import load over EVERY module
        guards.append((_measure(led, ex, "package_ast_nodes", base_fp.package_size, M.Split.DEV, pop, cond, evidence,
                                {"guard": "base"}, higher_is_better=False, stderr=max(1.0, 0.02 * base_fp.package_size)),
                       _measure(led, ex, "package_ast_nodes", max(f.package_size for f in cand_fps), M.Split.DEV, pop, cond,
                                evidence, {"guard": "candidate"}, higher_is_better=False,
                                stderr=max(1.0, 0.02 * base_fp.package_size))))   # lazy imports may add a few nodes
        hb = _measure(led, ex, "static_eager_load", base_fp.static_load, M.Split.HOLDOUT, pop, cond, evidence,
                      {"holdout": "base"}, higher_is_better=False)
        hc = _measure(led, ex, "static_eager_load", max(f.static_load for f in cand_fps), M.Split.HOLDOUT, pop, cond, evidence,
                      {"holdout": "candidate"}, higher_is_better=False)
    elif coverage_kind:                                                 # the holdout: the whole package's test gap
        hb = _measure(led, ex, "uncovered_public:package", base_fp.uncovered_package, M.Split.HOLDOUT, pop, cond, evidence,
                      {"holdout": "base"}, higher_is_better=False)
        hc = _measure(led, ex, "uncovered_public:package", max(f.uncovered_package for f in cand_fps), M.Split.HOLDOUT, pop, cond,
                      evidence, {"holdout": "candidate"}, higher_is_better=False)
    else:
        hb = _measure(led, ex, "package_ast_nodes", base_fp.package_size, M.Split.HOLDOUT, pop, cond, evidence,
                      {"holdout": "base"}, higher_is_better=False)
        hc = _measure(led, ex, "package_ast_nodes", max(f.package_size for f in cand_fps), M.Split.HOLDOUT, pop, cond,
                      evidence, {"holdout": "candidate"}, higher_is_better=False)

    def g(i: str) -> M.Measurement:
        rec = led.get(i)
        if not isinstance(rec, M.Measurement):
            raise KernelError(f"{i} is not a Measurement")
        return rec
    verdict, detail = M.improvement_verdict([g(i) for i in base_ids], [g(i) for i in cand_ids],
                                            [(g(b), g(c)) for b, c in guards], (g(hb), g(hc)))
    detail = dict(detail, target=plan.component, size=(base_fp.target_size, cand_fps[0].target_size),
                  package=(base_fp.package_size, cand_fps[0].package_size), memory_mb=(round(bm, 2), round(cm, 2)),
                  tests=(base_fp.test_cases, cand_fps[0].test_cases), requirements_lost=lost,
                  active_nodes=(base_fp.active_nodes, cand_fps[0].active_nodes),
                  uncovered=(base_fp.uncovered, cand_fps[0].uncovered))
    from creator.evaluate import computation
    ids = base_ids + cand_ids + [i for pair in guards for i in pair] + [hb, hc]
    cid = led.append(M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(ex,), subject_id=ex,
                                        baseline_ids=tuple(base_ids), candidate_ids=tuple(cand_ids), verdict=verdict,
                                        computation=computation(M.VERDICT_FUNCTION, ids, verdict.value, ids),
                                        regression_baseline_ids=tuple(b for b, _ in guards),
                                        regression_candidate_ids=tuple(c for _, c in guards),
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
        if not (Path(p).name.startswith("test_") or Path(p).name in AUD.HARNESS_FILES):
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
    scratch = cfg.scratch.resolve() if cfg.scratch else device.sandbox_root(cfg.repo)
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


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return bool(psutil.pid_exists(pid)) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:                                                   # noqa: BLE001 - unknown means alive (never steal)
        return True


class _KernelLock:
    """One kernel at a time per state directory (O_EXCL lock file; a stale lock older than `stale_s` is reported, not stolen)."""

    def __init__(self, state: Path, stale_s: float = 6 * 3600) -> None:
        self.path = state / "kernel.lock"
        self.stale_s = stale_s

    def __enter__(self) -> "_KernelLock":
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():                                         # a lock whose holder is DEAD is taken over (1 Oct:
            try:                                                        # I deleted one by hand while its holder still ran)
                holder = int(self.path.read_text(encoding="utf-8").strip() or 0)
            except (ValueError, OSError):
                holder = -1
            if holder > 0 and not _pid_alive(holder):
                self.path.unlink(missing_ok=True)
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


def prepare(cfg: KernelConfig, led: Ledger) -> tuple[Optional[Assessed], list[str], Optional[str]]:
    """0 RECOVER + 1 ASSESS once: (main assessment, recovered sandboxes, why development must stop - or None)."""
    recovered = recover(cfg)
    oid = O.self_objective(led)
    O.compile_capabilities(led, oid, cfg.specs())
    main = assess_tree(cfg, led, cfg.repo, "main", audit=False)
    G.sync(led, main.model)                                             # sync FIRST: a newly failing check becomes FAILED,
    from creator import oversight as OV                                 # unexplained failures become KNOWLEDGE gaps (CR059)
    OV.knowledge_gaps(led)
    main = dataclasses.replace(main, audit=AUD.audit(led, main.model, repo=cfg.repo))   # then audit (1 Oct: audit-before-sync
    if red(main.audit):                                                 # reported it as a stale TESTED claim and stopped)
        return main, recovered, "; ".join(f"{f.check}:{f.subject}" for f in main.audit.findings[:5]) or str(main.audit.errors)
    return main, recovered, None


def plan_one(cfg: KernelConfig, led: Ledger, main: Assessed, base_sha: str, exclude_components: Sequence[str] = (),
             exclude_paths: Sequence[str] = ()) -> Optional[P.Plan]:
    """2 PLAN one package that touches none of the excluded components / paths (so parallel workers never collide)."""
    steps = tuple(s for s in cfg.steps if s in P.WORKER_STEPS)          # a validator step can never be handed to a worker
    specs = cfg.specs()
    by_id = {s.id: s for s in specs}
    held = set(exclude_paths) | {f for c in exclude_components if c in by_id for f in (*by_id[c].modules, *by_id[c].tests)}
    components = tuple(exclude_components) + tuple(s.id for s in specs if held & {*s.modules, *s.tests})   # same file = same owner
    plan = None
    if cfg.mode in ("auto", "gaps"):
        plan = P.plan_next(led, main.model, base_sha, specs, steps=steps, exclude_components=components)
    if plan is None and cfg.mode in ("auto", "efficiency"):
        plan = P.plan_efficiency(led, cfg.repo, base_sha, avoid=tuple(held))   # standing shrink / activation
    return plan


def _cycle(cfg: KernelConfig, worker: Worker, n: int = 1, led: Optional[Ledger] = None) -> CycleReport:
    t0 = time.monotonic()
    led = led or Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    main, recovered, stop = prepare(cfg, led)
    if stop is not None:
        return CycleReport(n, "AUDIT_RED", reason=stop, seconds=round(time.monotonic() - t0, 1))
    assert main is not None
    base_sha = S.head(cfg.repo)
    plan = plan_one(cfg, led, main, base_sha)
    if plan is None:
        return CycleReport(n, "NOTHING_TO_DO", reason="no unblocked worker gap and nothing to shrink",
                           seconds=round(time.monotonic() - t0, 1))
    return execute(cfg, worker, plan, main, base_sha, n, led, recovered, t0=t0)


def diagnose_rejection(led: Ledger, plan: P.Plan, ev: Optional[S.Evaluation]) -> Optional[dict[str, Any]]:
    """K11 in the loop: when a candidate's tests fail, diagnose the failure (creator.debug), record Failure + Diagnosis in the
    ledger, and return the diagnosis so the rejection reason - and through it the next attempt's work package - carries a
    root cause instead of a bare 'not clean'."""
    run = ev.candidate_run if ev is not None else None
    if run is None:
        return None
    bad = [c for c in run.cases.values() if c.outcome.bad]
    if not bad:
        return None
    from creator import debug as DBG                                    # loaded only when something failed
    text = "\n".join(f"{c.case_id}\n{c.message}" for c in bad)[:20000]
    d = DBG.diagnose(text, prefer="creator/")
    fid = led.append(M.Failure(created_by=M.Role.DEBUGGER, parents=(plan.work_package_id,), subject_id=plan.work_package_id,
                               symptom=f"{len(bad)} failing test case(s): {', '.join(c.case_id for c in bad[:3])}"[:500],
                               classification=d.classification, reproduction=f"pytest {' '.join(sorted(run.selection))}"[:500]))
    led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(fid,), failure_id=fid,
                           hypotheses=tuple(d.hypotheses) or ("unknown",), root_cause=d.root_cause or "undetermined"))
    return {"classification": d.classification, "root_cause": d.root_cause, "hypotheses": list(d.hypotheses[:3]),
            "failure_id": fid}


class _Stages:
    """Per-stage wall seconds of one cycle (time.monotonic stamps only; never changes what runs). Recorded as
    CycleReport.details["stages"]; constraints.cycle_time_metric names the dominant stage from it."""

    def __init__(self) -> None:
        self.sec: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t = time.monotonic()
        try:
            yield
        finally:
            self.sec[name] = round(self.sec.get(name, 0.0) + time.monotonic() - t, 3)


class Cancelled(Exception):
    """The swarm pulled this worker back (RAM tight): its sandbox is discarded, nothing is adopted."""


def execute(cfg: KernelConfig, worker: Worker, plan: P.Plan, main: Assessed, base_sha: str, n: int = 1,
            led: Optional[Ledger] = None, recovered: Sequence[str] = (), lock: Any = None, cancel: Any = None,
            checkpoint: bool = True, t0: Optional[float] = None) -> CycleReport:
    """3-8 for one planned package. `lock` serialises everything that changes the repository itself (sandbox creation, merge,
    post-merge verification, rollback) when several run in parallel; `cancel` (an Event) is checked between phases."""
    t_start = time.monotonic()
    t0 = t_start if t0 is None else t0
    stage = _Stages()
    stage.sec["before_execute"] = round(t_start - t0, 3)         # prepare + curriculum/student planning (the caller's stamp)
    led = led or Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    guard = lock if lock is not None else contextlib.nullcontext()

    def checkpoint_cancel(where: str) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"pulled back before {where} (RAM tight)")
    wp = led.get(plan.work_package_id)
    assert isinstance(wp, M.WorkPackage)
    led.transition(plan.work_package_id, M.Status.IN_PROGRESS, "cycle started", M.Role.KERNEL)
    rep = CycleReport(n, "ERROR", plan.package_id, plan.requirement_key, details={"recovered": list(recovered), "stages": stage.sec})
    with stage("sandbox_open"), guard:
        sb = S.Sandbox.open(cfg.repo, base_sha, cfg.scratch, label=plan.package_id, hide=cfg.hide,
                             omit=cfg.omit)   # 3 SANDBOX
    locked = False
    ev: Optional[S.Evaluation] = None
    try:
        checkpoint_cancel("work")
        with stage("worker"):
            work = worker(plan, wp, sb.path)                            # 4 WORK
        rep.calls, rep.usd = work.calls, work.usd
        rep.details["worker"] = {"claimed_done": work.claimed_done, "notes": work.notes, "by": work.by}
        if work.refused:
            S.discard(sb)
            led.transition(plan.work_package_id, M.Status.BLOCKED, work.notes, M.Role.KERNEL)
            led.transition(plan.gap_id, M.Status.FAILED, f"{plan.package_id} not attempted: {work.notes}", M.Role.KERNEL)
            rep.outcome, rep.reason = "BUDGET", work.notes
            return rep
        if work.deferred:                                               # handed over to its owner: nothing adopted, no attempt used
            S.discard(sb)
            P.record_outcome(led, plan, False, f"{P.DEFERRED_PREFIX} {work.notes}"[:500])
            rep.outcome, rep.reason = "DEFERRED", work.notes
            return rep
        if work.contaminated:
            raise _Reject(f"worker run contaminated: {list(work.contaminated)[:5]}")
        checkpoint_cancel("evaluation")
        change = sb.changes()                                           # 5 EVALUATE
        if not change.paths:
            raise _Reject("the worker changed nothing")
        frozen = hashlib.sha256(sb.diff().encode()).hexdigest()         # RESULT FREEZE (content) before anything runs in the tree
        if cfg.hide or cfg.omit:
            sb.reveal()                                                 # the Creator's own tests need the sealed suite back
        pcfg = sandbox_pytest(cfg)
        served = TC.TreeCache(cfg.state / "evidence" / "tree_cache").lookup(TC.tree_key(cfg.repo, pcfg.python, rev=base_sha,
                                                                                         require_clean=False))
        with stage("evaluation"):
            ev = sb.evaluate(build_config=cfg.build, pytest_config=pcfg, base_reuse={t: x for t, (_, x) in served.items()})
        _eval_stages(stage.sec, ev)
        with stage("audit_checks"):
            before, after = _test_sources(sb.path, base_sha, change.files)
            weak = AUD.check_test_weakening(before, after)
            planted = AUD.check_hardcoded_answers(repo=sb.path, sealed_root=cfg.sealed_root or cfg.repo)
        with stage("fundamentals"):
            fund = FU.evaluate_candidate(sb.path, base_sha, list(change.paths), plan.step, work.notes)   # advisory: recorded, never blocking
        cand: list[Assessed] = []
        for r in range(2):
            with stage(f"cand_assessment_{r}"):
                # replicate 1 re-derives every requirement check on the candidate tree; the declared test files that PASSED in
                # replicate 0 are served from it when their reach digest (test + every module it reaches) is unchanged.
                cand.append(assess_tree(cfg, led, sb.path, f"{plan.package_id}_cand{r}", audit=False,
                                        reuse=(pass_reuse(cand[0]) if r and cfg.reuse_candidate_tests else None)))
        evid = [_evidence_file(cfg, plan.package_id, "evaluation.json", ev.to_record()),
                M.EvidenceRef.of(main.snapshot, cfg.repo, "selfmodel"), M.EvidenceRef.of(cand[0].snapshot, cfg.repo, "selfmodel"),
                _evidence_file(cfg, plan.package_id, "diff.patch", sb.diff() if hasattr(sb, "diff") else "")]
        _evidence_file(cfg, plan.package_id, "fundamentals.json", fund.to_dict())
        if hashlib.sha256(sb.diff().encode()).hexdigest() != frozen:
            raise _Reject("the change set moved during evaluation (something in the tree rewrote files)")
        if plan.step == "coverage" and any(not p.startswith("tests/") for p in change.paths):
            raise _Reject("a coverage package may only add tests; changed: "
                          f"{[p for p in change.paths if not p.startswith('tests/')][:5]}")
        if plan.step in P.EFFICIENCY_STEPS:
            from creator import efficiency as E
            base_tree = sb.scratch / f"{sb.id}-base"
            base_root = base_tree if base_tree.is_dir() else cfg.repo
            need_act = cfg.measure_memory or plan.requirement_key == P.ACTIVATION_KEY
            base_fp = E.footprint(base_root, plan.component, memory=cfg.measure_memory, activation_=need_act)
            cand_fps = [E.footprint(sb.path, plan.component, memory=cfg.measure_memory and r == 0, activation_=need_act)
                        for r in range(2)]
            cand_fps[1] = dataclasses.replace(cand_fps[1], memory_mb=cand_fps[0].memory_mb)
            evid.append(_evidence_file(cfg, plan.package_id, "footprint.json",
                                       {"base": base_fp.to_dict(), "candidate": [f.to_dict() for f in cand_fps]}))
            cid, verdict, detail = efficiency_claim(led, plan, main, cand, ev, evid, base_fp, cand_fps)
        else:
            cid, verdict, detail = gap_closure_claim(led, plan, main, cand, ev, evid)
        rep.verdict = verdict.value
        rep.details.update(claim=cid, detail=detail, regression=ev.report.verdict.value if ev.report else "NO_REPORT",
                           builds=ev.builds, weakening=[dataclasses.asdict(f) for f in weak],
                           planted=[dataclasses.asdict(f) for f in planted], fundamentals=fund.to_dict())
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
        from creator import efficiency as E                              # SPARSE-ACTIVATION GUARD (like 'requirements kept'):
        base_tree = sb.scratch / f"{sb.id}-base"                         # an adopted change may not grow what start loads eagerly
        sparse = E.start_load_regression(base_tree if base_tree.is_dir() else cfg.repo, sb.path, plan.requirement_key)
        if sparse:
            rep.details["start_load"] = sparse
            reasons.append("start-time load grew: " + "; ".join(sparse))
        if reasons:
            raise _Reject("; ".join(reasons))
        decision = M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.ADOPT,
                              reason=f"{plan.requirement_key} closed with no regression", claim_id=cid)
        checkpoint_cancel("adoption")
        with stage("lock_wait"):
            guard.__enter__()                                           # one adoption + verification at a time
        locked = True
        problems = led.problems(decision)                               # 6 DECIDE (validated before the merge)
        if problems:
            raise _Reject(f"ledger refuses the ADOPT decision: {problems}")
        with stage("merge"):
            res = S.adopt(sb, decision, f"{plan.package_id} {plan.requirement_key}")
        merge_ev = _evidence_file(cfg, plan.package_id, res.merge_commit, f"{res.merge_commit}\n")
        led.append(dataclasses.replace(decision, evidence=(dataclasses.replace(merge_ev, kind="merge_commit"),)))
        rep.merge_commit = res.merge_commit
        sb.cleanup_base()
        sb.close()
        post_reuse: dict[str, tuple[str, str]] = {}
        if cfg.reuse_candidate_tests and _same_tree(cfg.repo, res.merge_commit, res.sandbox_commit):
            post_reuse = pass_reuse(cand[0])                            # merged tree is byte-identical to the one cand0 tested
        with stage("post_merge"):
            after_main = assess_tree(cfg, led, cfg.repo, f"{plan.package_id}_main_after",     # 7 VERIFY
                                     reuse=post_reuse or None)
        G.sync(led, after_main.model)
        lost_after = sorted({r.key for r in main.rows if r.met} - {r.key for r in after_main.rows if r.met})
        still = plan.step in P.EFFICIENCY_STEPS or next((r.met for r in after_main.rows if r.key == plan.requirement_key), False)
        if red(after_main.audit) or not still or lost_after:
            why = ("requirement not met on main after merge" if not still else
                   f"requirements lost on main after merge: {lost_after}" if lost_after else
                   f"audit red after merge: {[f.check for f in after_main.audit.findings[:5]]}")
            with stage("rollback"):
                revert = S.rollback(cfg.repo, res.merge_commit, why)
            led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.ROLLBACK,
                                  reason=f"{why}; reverted by {revert[:12]}"))
            fid = led.append(M.Failure(created_by=M.Role.DEBUGGER, parents=(plan.work_package_id,), subject_id=plan.experiment_id,
                                       symptom=f"post-merge rollback: {why}"[:500], classification="post-merge rollback",
                                       reproduction=f"merge {res.merge_commit[:12]}, re-assess main"))
            did = led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(fid,), failure_id=fid, hypotheses=(why[:300],),
                                         root_cause=why[:500], uncertainty=M.Uncertainty.LIKELY))
            led.append(M.Repair(created_by=M.Role.KERNEL, parents=(did,), diagnosis_id=did,       # CR204: a rollback is never silent
                                description=f"merge {res.merge_commit[:12]} reverted by {revert[:12]}"))
            P.record_outcome(led, plan, False, f"rolled back: {why}")
            rep.outcome, rep.reason = "ROLLED_BACK", why
        else:
            P.record_outcome(led, plan, True, f"adopted as {res.merge_commit[:12]}")
            if plan.step in P.EFFICIENCY_STEPS:                         # the shrink / coverage gap closes on the measured claim
                tr = led.append(M.TestRun(created_by=M.Role.KERNEL, command=f"efficiency:{plan.component}", passed=1, failed=0,
                                          errors=0, skipped=0, duration_s=0.0, subject_ids=(plan.gap_id,),
                                          selection=(plan.component,), evidence=tuple(evid)))
                G._advance_to_tested(led, plan.gap_id, tr, f"shrink adopted: {detail.get('size')} nodes")
            rep.outcome, rep.reason = "ADOPTED", f"merged {res.merge_commit[:12]}"
    except Cancelled as c:
        if not sb.closed:
            _save_pending(cfg, plan, sb, rep)
            S.discard(sb)
        P.record_outcome(led, plan, False, str(c))
        rep.outcome, rep.reason = "CANCELLED", str(c)
    except _Reject as r:
        S.discard(sb)
        reason = str(r)
        try:
            diag = diagnose_rejection(led, plan, ev)
        except Exception as e:                                          # noqa: BLE001 - a diagnosis is never worth a crash
            diag = {"classification": "UNKNOWN", "root_cause": f"diagnosis failed: {type(e).__name__}: {e}"}
        if diag:
            rep.details["diagnosis"] = diag
            reason += f"; diagnosis {diag['classification']}: {diag['root_cause']}"
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.REJECT,
                              reason=reason[:2000]))
        P.record_outcome(led, plan, False, reason[:500])
        rep.outcome, rep.reason = "REJECTED", reason
    except Exception as e:                                              # noqa: BLE001 - recorded, sandbox discarded, never adopted
        if not sb.closed:
            if cancel is not None and cancel.is_set():
                _save_pending(cfg, plan, sb, rep)
            S.discard(sb)
        if cancel is not None and cancel.is_set():                      # its processes were killed by the swarm's pull-back
            P.record_outcome(led, plan, False, "pulled back: RAM tight (processes stopped mid-phase)")
            rep.outcome, rep.reason = "CANCELLED", f"pulled back: RAM tight ({type(e).__name__} after its processes were stopped)"
        else:
            P.record_outcome(led, plan, False, f"kernel error: {type(e).__name__}: {e}"[:500])
            rep.outcome, rep.reason = "ERROR", f"{type(e).__name__}: {e}"
    finally:
        if locked:
            guard.__exit__(None, None, None)
        rep.seconds = round(time.monotonic() - t0, 1)                   # 8 RECORD
        t_rec = time.monotonic()
        led.append(M.StrategyOutcome(created_by=M.Role.KERNEL, parents=(plan.work_package_id,),
                                     strategy_id=(rep.details.get("worker", {}).get("by") or getattr(worker, "name", "worker")),
                                     problem_class=plan.step,
                                     subject_id=plan.work_package_id, success=rep.outcome == "ADOPTED", cost=rep.usd,
                                     duration_s=rep.seconds))
        if checkpoint:
            led.checkpoint(f"cycle {n} {plan.package_id} {rep.outcome}")
        stage.sec["record"] = round(time.monotonic() - t_rec, 3)
        _evidence_file(cfg, plan.package_id, "cycle.json", dataclasses.asdict(rep))
        _append_log_line(cfg.state / "kernel_log.jsonl", json.dumps(dataclasses.asdict(rep), default=str))
    return rep


def _same_tree(repo: Path, a: str, b: str) -> bool:
    """True when commits `a` and `b` have the identical git tree (so identical bytes in every tracked file)."""
    try:
        ta, tb = (S.git(repo, "rev-parse", f"{c}^{{tree}}", check=False) for c in (a, b))
    except Exception:                                                   # noqa: BLE001 - unknown means "run everything"
        return False
    return ta.returncode == 0 and tb.returncode == 0 and bool(ta.stdout.strip()) and ta.stdout.strip() == tb.stdout.strip()


def _eval_stages(sec: dict[str, float], ev: Optional[S.Evaluation]) -> None:
    """Split the sandbox evaluation's wall time into its timed parts; the remainder (selection, the base/candidate setup,
    flaky reruns, report assembly) stays in evaluation_other."""
    total = sec.get("evaluation", 0.0)
    try:
        build: float = sum((float(getattr(st, "seconds", 0.0) or 0.0) for st in (ev.build.steps if ev and ev.build else ())), 0.0)
        tm = (ev.report.timing if ev and ev.report else None) or {}
        base_s, cand_s = float(tm.get("baseline_s", 0.0) or 0.0), float(tm.get("candidate_s", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        return
    sec["evaluation_build"], sec["evaluation_base"], sec["evaluation_candidate"] = round(build, 3), round(base_s, 3), round(cand_s, 3)
    sec["evaluation_other"] = round(max(total - build - base_s - cand_s, 0.0), 3)


def _save_pending(cfg: KernelConfig, plan: P.Plan, sb: S.Sandbox, rep: CycleReport) -> None:
    """A pulled-back package keeps its work: the sandbox diff is saved to state/creator/pending/ before the sandbox goes (2 Oct:
    CP0065, a finished -136-node shrink, was pulled back for RAM and its sandbox deleted with nothing saved). Never raises."""
    try:
        diff = sb.diff()
        if diff.strip():
            out = cfg.state / "pending" / f"{plan.package_id}_{sb.id}.patch"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(diff, encoding="utf-8", newline="\n")
            rep.details["saved_diff"] = str(out)
    except Exception as e:                                              # noqa: BLE001 - saving is best effort, the cancel proceeds
        rep.details["saved_diff_error"] = f"{type(e).__name__}: {e}"[:300]


_LOG_LOCK = threading.Lock()                        # swarm workers (threads) share one kernel_log.jsonl


def _append_log_line(path: Path, text: str) -> None:
    """One whole line per binary write under a lock, so concurrent cycle records never interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (text + "\n").encode("utf-8")
    with _LOG_LOCK, path.open("ab") as fh:
        fh.write(data)
        fh.flush()


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
        if r.outcome in ("NOTHING_TO_DO", "AUDIT_RED", "BUDGET", "DEFERRED"):
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
    return {**G.summary(led), "fundamentals": FU.status_line(cfg.state)}
