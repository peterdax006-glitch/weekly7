"""Creator K24 - recursive self-improvement of the DEVELOPMENT PROCESS (C77 sec 53-54; CR191-198) - IMPLEMENTED, NOT VALIDATED.

One recursion step, entirely from ledger evidence and computed verdicts:

    1. weakness  - find_weaknesses(ledger): the most frequent REJECT-reason class, the (strategy, problem class) with the lowest
                   success rate (StrategyOutcome), the most frequent Failure classification. Counts, never opinions.
    2. design    - design_change(weakness, process, tried): ONE concrete change to a process parameter, as data (Change). The
                   process is a ProcessConfig (research_budget, design_breadth, reviewer_depth, max_retries; bounds from meta).
    3. test      - ab_test(): baseline vs candidate process on a FIXED deterministic workload (dev replicates + guard + holdout),
                   every number a Measurement in the ledger, the verdict computed by creator.model.improvement_verdict.
    4. adopt     - only on IMPROVEMENT (ImprovementClaim + ADOPT Decision citing it); otherwise a REJECT Decision.
    5. record    - a Finding (parent = the Experiment) carries the step as JSON: weakness, change, verdict, resulting process.
    6. repeat    - current_process(ledger) is rebuilt from the recorded steps, so iteration n+1 starts from what n adopted and
                   never re-proposes a change already tried.

The workload is a stand-in for development tasks: each task needs some retries and some review depth to be solved, so a process
parameter has a measurable effect that no one typed in. It is cheap and deterministic; the real devbench measurement is
creator.evaluate.compare and plugs in through the `workload` argument (any callable process -> Scores)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import meta as ME
from creator import model as M
from creator.ledger import Ledger, sha256_text

PRIMARY = "process_solve_rate"
GUARD = "within_budget_rate"
BUDGET = 12.0
PARAMS = ("research_budget", "design_breadth", "reviewer_depth", "max_retries")
STEP_TAG = "RECURSION_STEP "
# weakness kind -> process parameters worth trying, in order (a design rule, itself open to later improvement)
REMEDIES: dict[str, tuple[str, ...]] = {
    "rejection:REGRESSION": ("reviewer_depth", "max_retries"),
    "rejection:NO_EFFECT": ("research_budget", "design_breadth"),
    "rejection:INSUFFICIENT_EVIDENCE": ("max_retries", "reviewer_depth"),
    "rejection:OTHER": ("max_retries", "reviewer_depth"),
    "worker": ("max_retries", "reviewer_depth"),
    "failure": ("max_retries", "design_breadth"),
}


@dataclasses.dataclass(frozen=True)
class ProcessConfig:
    """The development process as data: the parameters a recursion step may change."""
    research_budget: int = 3
    design_breadth: int = 2
    reviewer_depth: int = 1
    max_retries: int = 2

    def with_(self, param: str, value: int) -> "ProcessConfig":
        return dataclasses.replace(self, **{param: value})

    def digest(self) -> str:
        return sha256_text(json.dumps(dataclasses.asdict(self), sort_keys=True))[:16]


@dataclasses.dataclass(frozen=True)
class Weakness:
    kind: str                      # key of REMEDIES
    key: str                       # the class / strategy that is weak
    count: int
    share: float                   # of its population (rejections, outcomes of that strategy, failures)
    evidence_ids: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Change:
    param: str
    old: int
    new: int
    rationale: str


@dataclasses.dataclass(frozen=True)
class Scores:
    """What a workload returns for one process: per-replicate dev rates, per-replicate guard rates, and the holdout rate."""
    n_dev: int
    dev: tuple[float, ...]
    guard: tuple[float, ...]
    n_holdout: int
    holdout: float
    holdout_guard: float = 1.0
    guard_metric: str = GUARD            # name of the guard metric this workload computes
    confirm_label: str = "holdout"       # what the confirmation split really is (the real workload never touches the holdout)


Workload = Callable[[ProcessConfig], Scores]


@dataclasses.dataclass(frozen=True)
class StepReport:
    iteration: int
    weakness: Optional[Weakness]
    change: Optional[Change]
    verdict: Optional[str]
    adopted: bool
    process_before: ProcessConfig
    process_after: ProcessConfig
    experiment_id: Optional[str] = None
    claim_id: Optional[str] = None
    detail: Optional[dict[str, Any]] = None


# ------------------------------------------------------------------------------------------------ 1. weakness

def rejection_class(reason: str) -> str:
    """The verdict name a REJECT reason starts with (autotune writes '<VERDICT>: why'); anything else is OTHER."""
    m = re.match(r"\s*([A-Z_]+)\b", reason)
    return m.group(1) if m and m.group(1) in {v.value for v in M.Verdict} else "OTHER"


def find_weaknesses(led: Ledger, min_n: int = 3) -> list[Weakness]:
    """Weaknesses of the development process, worst first, computed only from ledger records."""
    out: list[Weakness] = []
    rej: dict[str, list[str]] = {}
    for e in led.of_type("Decision"):
        d = e.record
        if isinstance(d, M.Decision) and d.verdict is M.DecisionVerdict.REJECT and not d.reason.startswith(STEP_TAG):
            rej.setdefault(rejection_class(d.reason), []).append(e.id)
    total = sum(len(v) for v in rej.values())
    if rej and total >= min_n:
        cls, ids = max(sorted(rej.items()), key=lambda kv: len(kv[1]))
        out.append(Weakness(f"rejection:{cls}", cls, len(ids), len(ids) / total, tuple(ids)))
    by: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for e in led.of_type("StrategyOutcome"):
        o = e.record
        if isinstance(o, M.StrategyOutcome):
            by.setdefault((o.strategy_id, o.problem_class), []).append((e.id, o.success))
    worst: Optional[tuple[float, str, list[tuple[str, bool]]]] = None
    for (sid, pc), rows in sorted(by.items()):
        if len(rows) < min_n:
            continue
        rate = sum(1 for _, s in rows if s) / len(rows)
        if rate < 0.5 and (worst is None or rate < worst[0]):
            worst = (rate, f"{sid}/{pc}", rows)
    if worst is not None:
        rate, key, rows = worst
        out.append(Weakness("worker", key, sum(1 for _, s in rows if not s), 1.0 - rate, tuple(i for i, s in rows if not s)))
    fails: dict[str, list[str]] = {}
    for e in led.of_type("Failure"):
        f = e.record
        if isinstance(f, M.Failure):
            fails.setdefault(f.classification, []).append(e.id)
    nf = sum(len(v) for v in fails.values())
    if fails and nf >= min_n:
        cls, ids = max(sorted(fails.items()), key=lambda kv: len(kv[1]))
        out.append(Weakness("failure", cls, len(ids), len(ids) / nf, tuple(ids)))
    out.sort(key=lambda w: (-w.share * w.count, w.kind, w.key))
    return out


# ------------------------------------------------------------------------------------------------ 2. design

def design_change(w: Weakness, process: ProcessConfig, tried: set[tuple[str, int]]) -> Optional[Change]:
    """One step up of the first remedy parameter that has room and whose (param, value) was never tried; when every remedy of this
    weakness was tried, the first untried step of any other parameter (the remedy table is a prior, not a wall)."""
    for param in REMEDIES.get(w.kind, ()):
        cur = getattr(process, param)
        lo, hi = ME.BOUNDS[param]
        new = cur + 1
        if new > hi or new < lo or (param, new) in tried:
            continue
        return Change(param, cur, new, f"{w.kind} {w.key!r}: {w.count} cases ({w.share:.0%}) -> raise {param} {cur} -> {new}")
    for param in PARAMS:                       # the remedies for this weakness are tried or at their bound: widen to any untried parameter
        cur = getattr(process, param)
        new = cur + 1
        if param not in REMEDIES.get(w.kind, ()) and ME.BOUNDS[param][0] <= new <= ME.BOUNDS[param][1] and (param, new) not in tried:
            return Change(param, cur, new, f"{w.kind} {w.key!r}: {w.count} cases ({w.share:.0%}); remedies exhausted -> fallback: raise {param} {cur} -> {new}")
    return None


# ------------------------------------------------------------------------------------------------ the fixed workload

def _h(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def _solve(process: ProcessConfig, seed: int, split: str, i: int, rep: int) -> tuple[bool, float]:
    """One development task: needs `r` retries and `q` review depth (fixed by its id); a replicate adds an occasional flake."""
    r = _h(seed, split, i, "r") % 5
    q = (0, 0, 0, 0, 0, 0, 1, 1, 1, 2)[_h(seed, split, i, "q") % 10]
    flake = 1 if _h(seed, split, i, rep, "f") % 7 == 0 else 0
    ok = process.max_retries >= r + flake and process.reviewer_depth >= q
    cost = 1.0 + min(process.max_retries, r + flake) + 0.5 * process.reviewer_depth
    return ok, cost


def make_workload(seed: int = 7, n_dev: int = 120, n_holdout: int = 60, replicates: int = 3) -> Workload:
    def run(p: ProcessConfig) -> Scores:
        def rate(split: str, n: int, rep: int) -> tuple[float, float]:
            res = [_solve(p, seed, split, i, rep) for i in range(n)]
            return (sum(1 for ok, _ in res if ok) / n, sum(1 for ok, c in res if ok and c <= BUDGET) / n)
        dev = [rate("dev", n_dev, r) for r in range(replicates)]
        hold = rate("holdout", n_holdout, 0)
        return Scores(n_dev, tuple(d[0] for d in dev), tuple(d[1] for d in dev), n_holdout, hold[0], hold[1])
    return run


# ------------------------------------------------------------------------------------------------ 3-5. test, adopt, record

def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def _computation(function: str, inputs: Any, output: Any, ids: Sequence[str] = ()) -> M.ComputationRef:
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return M.ComputationRef(function=function, code_hash=code, inputs_sha256=sha256_text(json.dumps(inputs, sort_keys=True, default=str)),
                            output_sha256=sha256_text(json.dumps(output, sort_keys=True, default=str)), record_ids=tuple(ids))


def _objective(led: Ledger) -> str:
    for e in led.of_type("Objective"):
        if isinstance(e.record, M.Objective) and e.record.statement.startswith("Improve the development process"):
            return e.id
    return led.append(M.Objective(created_by=M.Role.OWNER, statement="Improve the development process itself, by measurement",
                                  acceptance_criteria=("each adopted process change is an IMPROVEMENT by creator.model.improvement_verdict",)))


def ab_test(led: Ledger, experiment_id: str, base: ProcessConfig, cand: ProcessConfig, workload: Workload,
            min_effect: float = 0.0) -> tuple[str, M.Verdict, dict[str, Any]]:
    """Append the Measurements and the ImprovementClaim for base vs cand; returns (claim id, computed verdict, detail)."""
    sb, sc = workload(base), workload(cand)
    pop = f"process-workload:{sha256_text(json.dumps([sb.n_dev, sb.n_holdout, len(sb.dev)]))[:12]}"
    cond = f"fixed deterministic workload; budget {BUDGET}; replicates {len(sb.dev)}"
    if sb.confirm_label != "holdout":
        cond += f"; confirmation split = {sb.confirm_label}"

    def meas(metric: str, v: float, n: int, split: M.Split, who: str, inputs: Any) -> str:
        return led.append(M.Measurement(
            created_by=M.Role.VALIDATOR, parents=(experiment_id,), metric=metric, value=v, stderr=_se(v, n), n=n,
            population=f"{pop}:{split.value}", conditions=cond, higher_is_better=True, split=split,
            computation=_computation("creator.recursion:make_workload", {"arm": who, **inputs}, v)))

    def ids(s: Scores, who: str, cfg: ProcessConfig) -> tuple[list[str], str, str]:
        d = [meas(PRIMARY, v, s.n_dev, M.Split.DEV, who, {"cfg": dataclasses.asdict(cfg), "rep": r}) for r, v in enumerate(s.dev)]
        g = meas(s.guard_metric, sum(s.guard) / len(s.guard), s.n_dev, M.Split.DEV, who, {"cfg": dataclasses.asdict(cfg), "guard": 1})
        h = meas(PRIMARY, s.holdout, s.n_holdout, M.Split.HOLDOUT, who, {"cfg": dataclasses.asdict(cfg), "holdout": 1})
        return d, g, h
    bd, bg, bh = ids(sb, "baseline", base)
    cd, cg, ch = ids(sc, "candidate", cand)
    get = led.get
    verdict, detail = M.improvement_verdict([get(i) for i in bd], [get(i) for i in cd], [(get(bg), get(cg))],  # type: ignore[misc,list-item]
                                            (get(bh), get(ch)), min_effect=min_effect)  # type: ignore[arg-type]
    allids = bd + cd + [bg, cg, bh, ch]
    cid = led.append(M.ImprovementClaim(
        created_by=M.Role.VALIDATOR, parents=(experiment_id,), subject_id=experiment_id, baseline_ids=tuple(bd), candidate_ids=tuple(cd),
        verdict=verdict, computation=_computation(M.VERDICT_FUNCTION, allids, verdict.value, allids),
        regression_baseline_ids=(bg,), regression_candidate_ids=(cg,), holdout_baseline_id=bh, holdout_candidate_id=ch,
        min_effect=min_effect))
    return cid, verdict, detail


def recorded_steps(led: Ledger) -> list[dict[str, Any]]:
    out = []
    for e in led.of_type("Finding"):
        f = e.record
        if isinstance(f, M.Finding) and f.statement.startswith(STEP_TAG):
            out.append(json.loads(f.statement[len(STEP_TAG):]))
    return out


def current_process(led: Ledger, initial: Optional[ProcessConfig] = None) -> ProcessConfig:
    """The process in force: the initial one with every ADOPTED recorded step applied, in order."""
    p = initial or ProcessConfig()
    for s in recorded_steps(led):
        if s["adopted"]:
            p = p.with_(s["change"]["param"], int(s["change"]["new"]))
    return p


def tried_changes(led: Ledger) -> set[tuple[str, int]]:
    return {(s["change"]["param"], int(s["change"]["new"])) for s in recorded_steps(led)}


def step(led: Ledger, workload: Workload, initial: Optional[ProcessConfig] = None, min_n: int = 3,
         forced: Optional[Change] = None, min_effect: float = 0.0) -> StepReport:
    """One recursion step. `forced` replaces steps 1-2 (tests of the gate itself); the adoption rule is never bypassed."""
    iteration = len(recorded_steps(led)) + 1
    process = current_process(led, initial)
    weakness: Optional[Weakness] = None
    change = forced
    if change is None:
        tried = tried_changes(led)
        for w in find_weaknesses(led, min_n):
            change = design_change(w, process, tried)
            if change is not None:
                weakness = w
                break
    if change is None:
        return StepReport(iteration, None, None, None, False, process, process, detail={"why": "no weakness with an untried change"})
    cand = process.with_(change.param, change.new)
    obj = _objective(led)
    gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(obj,), kind=M.GapKind.CAPABILITY, importance=0.5,
                           description=f"process weakness: {change.rationale}"))
    ex = led.append(M.Experiment(
        created_by=M.Role.KERNEL, parents=(gap,), hypothesis=f"{change.param} {change.old} -> {change.new} raises the process solve rate",
        design="fixed deterministic workload: dev replicates + within-budget guard + holdout", metrics=(PRIMARY, GUARD), seed=0,
        baseline_ref=process.digest(), candidate_ref=cand.digest(), uses_holdout=True))
    claim, verdict, detail = ab_test(led, ex, process, cand, workload, min_effect)
    adopted = verdict is M.Verdict.IMPROVEMENT
    if adopted:
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, claim_id=claim,
                              reason=f"process improved: {change.param} {change.old} -> {change.new}"))
    else:
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.REJECT,
                              reason=f"{STEP_TAG}{verdict.value}: {detail.get('why')}"))
    after = cand if adopted else process
    rec = {"iteration": iteration, "change": dataclasses.asdict(change), "verdict": verdict.value, "adopted": adopted,
           "weakness": dataclasses.asdict(weakness) if weakness else None, "process_before": dataclasses.asdict(process),
           "process_after": dataclasses.asdict(after), "claim": claim, "experiment": ex}
    led.append(M.Finding(created_by=M.Role.KERNEL, parents=(ex,), uncertainty=M.Uncertainty.LIKELY,
                         statement=STEP_TAG + json.dumps(rec, sort_keys=True, default=str)))
    return StepReport(iteration, weakness, change, verdict.value, adopted, process, after, ex, claim, detail)


def run(led: Ledger, workload: Workload, iterations: int = 3, initial: Optional[ProcessConfig] = None,
        min_n: int = 3) -> list[StepReport]:
    """Repeat `step`, each iteration starting from the process the previous ones adopted. Stops when nothing is left to try."""
    reps = []
    for _ in range(iterations):
        r = step(led, workload, initial, min_n)
        reps.append(r)
        if r.change is None:
            break
    return reps
