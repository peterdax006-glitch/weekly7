"""Creator K20b - the development PROCESS as real levers (C77 sec 53-54; CR196-198) - IMPLEMENTED, NOT VALIDATED.

creator.recursion changes a ProcessConfig. This module makes that data act on the system:

    levers(process)        ProcessConfig -> Levers: what the own workers really run with
                             research_budget -> search budget (candidates tried)        40 x research_budget   (3 -> 120)
                             design_breadth  -> mutation candidates kept per family     20 x design_breadth    (2 -> 40)
                             reviewer_depth  -> width of the pair-mutation stage        12 x reviewer_depth    (1 -> 12; 0 = no pairs)
                             max_retries     -> extra search attempts (each wider than the last) and the RuleWorker's round cap
    build_worker(...)      the SelfFirst worker the swarm constructs, made from the levers (scripts/creator_swarm.py make_worker)
    load_process / save_process   the adopted process, in state/creator/process.json (defaults when the file is missing)
    DevWorkload            the recursion workload on the REAL devbench: DEV split only. The primary replicates are disjoint
                             chunks of dev tasks; the confirmation arm is a further disjoint chunk of DEV tasks, recorded as
                             `dev-confirm` - the devbench holdout is never read here (it is reserved for the final check)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from creator import meta as ME
from creator import recursion as R

GUARD_NO_HARM = "no_false_completion_rate"
CONFIRM_LABEL = "dev-confirm (disjoint dev tasks; devbench holdout untouched)"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "state" / "creator" / "process.json"


@dataclasses.dataclass(frozen=True)
class Levers:
    search_budget: int
    per_family: int
    pair_width: int
    retries: int
    rule_rounds: int


def levers(p: R.ProcessConfig) -> Levers:
    return Levers(search_budget=40 * p.research_budget, per_family=20 * p.design_breadth, pair_width=12 * p.reviewer_depth,
                  retries=p.max_retries, rule_rounds=5 * (p.max_retries + 3))


class ProcessSolver:
    """A devbench Solver (own search only, no model) whose behaviour is the process: attempt 0 with the levers, each retry with
    twice as many candidates kept per family. Stops at the first attempt that passes the visible tests."""
    name = "self-search-process"

    def __init__(self, process: R.ProcessConfig) -> None:
        self.process, self.lv = process, levers(process)

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> Any:
        from creator import devbench as D
        from creator import generator as G
        tried = 0
        for attempt in range(self.lv.retries + 1):
            ok, _, n = G.solve_with_search(task, workdir, self.lv.search_budget, pairs=self.lv.pair_width > 0,
                                           per_family=self.lv.per_family * (attempt + 1), pair_width=self.lv.pair_width)
            tried += n
            if ok:
                return D.SolverResult(True, 0, f"search passed after {tried} candidates (attempt {attempt})")
        return D.SolverResult(False, 0, f"search failed after {tried} candidates")


def tuned_search_budget(process: R.ProcessConfig, worker_config: Optional[Path] = None) -> int:
    """K19 -> the process: the search budget the swarm's worker really uses. The process lever (research_budget) is the default; once
    creator.autotune has ADOPTED a worker configuration by measured trial (state/creator/worker_config.json exists), that
    configuration's search_budget replaces it - the Creator's own tuning reaches the worker the swarm builds."""
    lv = levers(process)
    path = worker_config or DEFAULT_PATH.with_name("worker_config.json")
    if not path.is_file():
        return lv.search_budget
    from creator import autotune as AT
    try:
        budget = int(AT.load_active(path).search_budget)
    except (OSError, ValueError, KeyError, TypeError):             # a damaged file must not stop the swarm building its worker
        return lv.search_budget
    return budget if budget >= 1 else lv.search_budget


def build_worker(process: R.ProcessConfig, session: Any = None, worker_config: Optional[Path] = None) -> Any:
    """The swarm's worker for this process: own workers first (rules, generated tests, search), the session last."""
    from creator import selfworkers as SW
    from creator import testgen as TG
    lv = levers(process)
    return SW.SelfFirst([SW.RuleWorker(max_rounds=lv.rule_rounds), TG.TestGenWorker(),
                         SW.SearchWorker(budget=tuned_search_budget(process, worker_config))], session)


def load_process(path: Path = DEFAULT_PATH) -> R.ProcessConfig:
    """The adopted process; the default one when nothing was adopted yet or the file is unreadable (never a crash in a kernel)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        vals = {k: int(d[k]) for k in R.PARAMS if k in d}
        if any(not ME.BOUNDS[k][0] <= v <= ME.BOUNDS[k][1] for k, v in vals.items()):
            return R.ProcessConfig()                                # a hand-edited / corrupt file must not set unbounded budgets
        return R.ProcessConfig(**vals)
    except (OSError, ValueError, TypeError):
        return R.ProcessConfig()


def save_process(process: R.ProcessConfig, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(process), sort_keys=True, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the real workload

Evaluate = Callable[[R.ProcessConfig, str], tuple[bool, bool]]     # (process, task id) -> (solved, no false completion/regression)


def pick_tasks(ids: Sequence[str], seed: int, n: int) -> list[str]:
    """A seed-dependent, deterministic selection of n task ids (different seed -> different tasks)."""
    key = lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest()   # noqa: E731
    return sorted(ids, key=key)[:n]


class DevWorkload:
    """recursion.Workload on the real devbench DEV split. Per call: `reps` disjoint chunks of `chunk` dev tasks (primary), then
    `confirm` more disjoint dev tasks (confirmation). Results are cached per (process, task) so the baseline is run once."""

    def __init__(self, seed: int = 1, chunk: int = 4, reps: int = 3, confirm: int = 4, evaluate: Optional[Evaluate] = None,
                 dev_ids: Optional[Sequence[str]] = None) -> None:
        self.seed, self.chunk, self.reps, self.confirm = seed, chunk, reps, confirm
        self._tasks: dict[str, Any] = {}
        self._manifest: Optional[Mapping[str, Any]] = None
        self.evaluate = evaluate or self._real
        if dev_ids is None:
            from creator import devbench as D
            self._tasks = {t.id: t for t in D.load_tasks() if t.split == "dev"}     # DEV only: holdout tasks are filtered out
            dev_ids = list(self._tasks)
        need = chunk * reps + confirm
        if len(dev_ids) < need:
            raise ValueError(f"DevWorkload needs {need} dev tasks ({reps} chunks of {chunk} + {confirm} confirm), got {len(dev_ids)}")
        picked = pick_tasks(dev_ids, seed, need)
        self.chunks = [picked[i * chunk:(i + 1) * chunk] for i in range(reps)]
        self.confirm_ids = picked[chunk * reps:]
        self._cache: dict[tuple[str, str], tuple[bool, bool]] = {}
        self.runs = 0

    def _real(self, p: R.ProcessConfig, task_id: str) -> tuple[bool, bool]:
        from creator import devbench as D
        if self._manifest is None:
            self._manifest = D.load_manifest()
        t = self._tasks[task_id]
        assert t.split == "dev", "the real workload never runs a non-dev task"
        s = D.run_task(t, ProcessSolver(p), self._manifest)
        return s.outcome == "SOLVED", s.outcome not in ("FALSE_COMPLETION", "REGRESSION")

    def _one(self, p: R.ProcessConfig, tid: str) -> tuple[bool, bool]:
        k = (p.digest(), tid)
        if k not in self._cache:
            self.runs += 1
            self._cache[k] = self.evaluate(p, tid)
        return self._cache[k]

    def _rates(self, p: R.ProcessConfig, ids: Sequence[str]) -> tuple[float, float]:
        res = [self._one(p, t) for t in ids]
        n = max(len(res), 1)
        return sum(1 for s, _ in res if s) / n, sum(1 for _, h in res if h) / n

    def __call__(self, p: R.ProcessConfig) -> R.Scores:
        rows = [self._rates(p, c) for c in self.chunks]
        conf = self._rates(p, self.confirm_ids)
        return R.Scores(self.chunk, tuple(r[0] for r in rows), tuple(r[1] for r in rows), len(self.confirm_ids), conf[0], conf[1],
                        guard_metric=GUARD_NO_HARM, confirm_label=CONFIRM_LABEL)
