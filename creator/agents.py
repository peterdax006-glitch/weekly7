"""Creator K05 - the agent runtime: budgeted, confined, recorded LLM workers (C77 secs 14-16, 44-46, 66-68; package CR04)
- IMPLEMENTED, NOT VALIDATED.

A worker is one headless Claude Code call (`claude -p`) doing one bounded job inside one directory. The runtime owns:

    BUDGET       a persistent call/cost ledger (state/creator/agent_budget.json). Calls are refused - fail closed - when the
                 daily call cap, the daily dollar cap or the per-job cap would be exceeded, and when agent calls are not ENABLED
                 at all (owner, 30 Sep: "usage credits is to much of a limiting factor"; the default is off).
    CONFINEMENT  cwd = the job directory (outside the repository), --permission-mode dontAsk with an explicit tool allow-list
                 (no web, no subagents, shell limited to running pytest), only the job directory's own project settings (--setting-sources project; no user settings) and no MCP servers loaded,
                 --max-budget-usd per call. What the CLI cannot enforce (a test file the worker writes can read any path) is
                 checked afterwards: CONTAMINATION scans the transcript and the job directory for protected paths and answer-key
                 names; a contaminated run is reported as such and must never be scored as a success.
    RECORD       prompt, command line, full JSON output, cost, turns, duration and outcome are written to a run directory and
                 returned as an AgentRun; nothing a worker says is trusted as a fact - its claim is one field, the evaluator
                 decides.

`AgentSolver` adapts the runtime to the devbench Solver protocol so the Creator's own development ability can be measured.
A `runner` can be injected (tests use a fake CLI script, so the test suite spends no credits)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from creator import devbench as D
from creator.ledger import REPO_ROOT

STATE = REPO_ROOT / "state" / "creator"
BUDGET_FILE = STATE / "agent_budget.json"
RUNS_DIR = STATE / "agent_runs"
DEFAULT_MODEL = "sonnet"                                                # memory: Sonnet by default, pass the model explicitly
DEFAULT_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "Bash(python -m pytest:*)", "Bash(pytest:*)")
DENIED_TOOLS = ("WebFetch", "WebSearch", "Agent", "Task", "NotebookEdit")
CONTAMINATION_MARKERS = ("devbench/sealed", "devbench\\sealed", "MANIFEST.json", "_hidden_acceptance", "/reference/",
                         "\\reference\\", "state/livesim", "state\\livesim")
STATUS_RE = re.compile(r"^\s*STATUS:\s*(DONE|NOT_DONE)\s*$", re.MULTILINE)


class BudgetError(RuntimeError):
    """The call was refused before it was made: no credits were spent."""


# ------------------------------------------------------------------------------------------------ budget

@dataclasses.dataclass(frozen=True)
class BudgetPolicy:
    enabled: bool = False
    daily_calls: int = 12
    daily_usd: float = 5.0
    per_call_usd: float = 0.75
    per_job_calls: int = 2

    @classmethod
    def from_env(cls) -> "BudgetPolicy":
        """Agent calls are OFF, and no environment variable turns them on (owner, 1 Oct 2026: "the creator shouldn't hire a
        claude worker, you should be the only claude worker working on it"). Real work goes through the kernel's
        HandoffWorker to the Claude session; this runtime remains for fake-runner tests and explicit BudgetPolicy objects."""
        def num(name: str, default: float, ceiling: float) -> float:
            try:
                return min(float(os.environ.get(name, default)), ceiling)
            except ValueError:
                return default
        return cls(enabled=False,
                   daily_calls=int(num("CREATOR_AGENT_DAILY_CALLS", 12, 50)),
                   daily_usd=num("CREATOR_AGENT_DAILY_USD", 5.0, 20.0),
                   per_call_usd=num("CREATOR_AGENT_CALL_USD", 0.75, 2.0),
                   per_job_calls=int(num("CREATOR_AGENT_JOB_CALLS", 2, 5)))


class Budget:
    """Append-only record of every call; the caps are checked against it before each call (C77 sec 45 resource limits)."""

    def __init__(self, path: Path = BUDGET_FILE, policy: Optional[BudgetPolicy] = None,
                 clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)) -> None:
        self.path = path
        self.policy = policy or BudgetPolicy.from_env()
        self.clock = clock

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return list(json.loads(self.path.read_text(encoding="utf-8")).get("calls", []))

    def today(self) -> tuple[int, float]:
        day = self.clock().date().isoformat()
        calls = [c for c in self._load() if c["at"][:10] == day]
        return len(calls), round(sum(float(c.get("usd", 0.0)) for c in calls), 4)

    def job_calls(self, job: str) -> int:
        return sum(1 for c in self._load() if c.get("job") == job)

    def check(self, job: str) -> None:
        p = self.policy
        if not p.enabled:
            raise BudgetError("agent calls are disabled (hard-disabled by the owner: the Creator uses no outside agent)")
        n, usd = self.today()
        if n + 1 > p.daily_calls:
            raise BudgetError(f"daily call cap reached ({n}/{p.daily_calls})")
        if usd + p.per_call_usd > p.daily_usd:
            raise BudgetError(f"daily spend cap would be exceeded (${usd:.2f} + ${p.per_call_usd:.2f} > ${p.daily_usd:.2f})")
        if self.job_calls(job) + 1 > p.per_job_calls:
            raise BudgetError(f"per-job call cap reached for {job}")

    def record(self, job: str, usd: float, run_id: str, outcome: str) -> None:
        calls = self._load()
        calls.append({"at": self.clock().isoformat(), "job": job, "usd": round(float(usd), 4), "run": run_id, "outcome": outcome})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"calls": calls}, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


# ------------------------------------------------------------------------------------------------ the call

@dataclasses.dataclass(frozen=True)
class AgentSpec:
    role: str                                   # e.g. developer, researcher, critic
    instructions: str                           # the role's standing instructions (system prompt append)
    model: str = DEFAULT_MODEL
    tools: tuple[str, ...] = DEFAULT_TOOLS
    timeout_s: int = 900


@dataclasses.dataclass(frozen=True)
class AgentRun:
    run_id: str
    job: str
    role: str
    outcome: str                                # OK / CLI_ERROR / TIMEOUT / CONTAMINATED / REFUSED
    claimed_done: bool
    final_text: str
    usd: float
    turns: int
    seconds: float
    contamination: tuple[str, ...]
    run_dir: str


Runner = Callable[[Sequence[str], Path, int, str], tuple[int, str, str]]


def worker_env() -> dict[str, str]:
    """The worker's environment: this interpreter's directory FIRST on PATH, so the worker's `python -m pytest` is the project's
    venv (30 Sep: the first real worker fixed D01 but could not run the tests - system python had no pytest - and said NOT_DONE)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE_CODE_", "ANTHROPIC_LOG", "PYTHON"))}
    import sys
    bindir = str(Path(sys.executable).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    venv = Path(sys.prefix)
    if (venv / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(venv)
    return env


def subprocess_runner(cmd: Sequence[str], cwd: Path, timeout: int, stdin_text: str) -> tuple[int, str, str]:
    env = worker_env()
    p = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, encoding="utf-8",
                       errors="replace", input=stdin_text)
    return p.returncode, p.stdout, p.stderr


def resolve_cli(cli: str = "claude") -> str:
    """The CLI's real path: on Windows `claude` is a .cmd/.exe shim that subprocess cannot launch by bare name (30 Sep)."""
    return shutil.which(cli) or cli


def build_command(spec: AgentSpec, instructions_file: Path, per_call_usd: float, cli: str = "claude") -> list[str]:
    """The prompt travels on STDIN and the instructions in a FILE: on Windows the CLI is a .CMD shim run through cmd.exe, which
    cuts the command line at the first newline - every flag after a multi-line prompt was silently dropped (30 Sep: the worker
    ran with default permissions and text output). No argument may contain a newline."""
    cmd = [cli, "-p", "--output-format", "json", "--model", spec.model, "--permission-mode", "dontAsk",
           "--allowedTools", ",".join(spec.tools), "--disallowedTools", ",".join(DENIED_TOOLS), "--setting-sources", "project",
           "--strict-mcp-config", "--no-session-persistence", "--max-budget-usd", f"{per_call_usd:.2f}",
           "--append-system-prompt-file", str(instructions_file)]
    bad = [a for a in cmd if "\n" in a or "\r" in a]
    if bad:
        raise ValueError(f"command-line argument contains a newline: {bad[0][:60]!r}")
    return cmd


def scan_contamination(texts: Sequence[str], workdir: Path, markers: Sequence[str] = CONTAMINATION_MARKERS,
                       files: Optional[Sequence[Path]] = None) -> tuple[str, ...]:
    """Protected-path or answer-key references in what the worker said or wrote. Any hit voids the run. `files` limits the file
    scan to what the worker changed (1 Oct: in a full-repo sandbox, 46 pre-existing files mention state/livesim legitimately)."""
    hits: set[str] = set()
    for t in texts:
        for mk in markers:
            if mk in t:
                hits.add(f"transcript mentions {mk!r}")
    for p in (files if files is not None else workdir.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".txt", ".json", ".md", ".cfg", ".ini", ".toml") and p.stat().st_size < 2_000_000:
            body = p.read_text(encoding="utf-8", errors="replace")
            for mk in markers:
                if mk in body:
                    hits.add(f"{p.relative_to(workdir).as_posix()} contains {mk!r}")
    return tuple(sorted(hits))


def run_agent(spec: AgentSpec, job: str, prompt: str, workdir: Path, budget: Budget, runner: Runner = subprocess_runner,
              runs_dir: Path = RUNS_DIR, cli: str = "claude",
              markers: Sequence[str] = CONTAMINATION_MARKERS,
              changed_files: Optional[Callable[[Path], Sequence[Path]]] = None) -> AgentRun:
    """One budgeted, confined, recorded worker call. Raises BudgetError BEFORE any spend when the budget refuses."""
    wd = workdir.resolve()
    if REPO_ROOT.resolve() in (wd, *wd.parents):
        raise BudgetError(f"refusing to run a worker inside the repository: {wd}")
    budget.check(job)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha256(
        (job + prompt + str(time.time_ns())).encode()).hexdigest()[:8]
    rd = runs_dir / run_id
    rd.mkdir(parents=True, exist_ok=True)
    full_prompt = (prompt.rstrip() + "\n\nWork only inside the current directory. When finished, end your reply with exactly one "
                   "line: 'STATUS: DONE' if you believe the job is complete, otherwise 'STATUS: NOT_DONE'.")
    (rd / "prompt.txt").write_text(full_prompt, encoding="utf-8")
    (rd / "instructions.txt").write_text(spec.instructions, encoding="utf-8")
    cmd = build_command(spec, (rd / "instructions.txt").resolve(), budget.policy.per_call_usd,
                        resolve_cli(cli) if runner is subprocess_runner else cli)
    (rd / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
    t0 = time.monotonic()
    try:
        rc, out, err = runner(cmd, wd, spec.timeout_s, full_prompt)
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, out, err, timed_out = -1, "", "timeout", True
    except OSError as e:                                                # the CLI could not even start: recorded, never a crash
        rc, out, err, timed_out = -2, "", f"{type(e).__name__}: {e}", False
    secs = round(time.monotonic() - t0, 1)
    (rd / "stdout.json").write_text(out, encoding="utf-8")
    (rd / "stderr.txt").write_text(err, encoding="utf-8")
    try:
        data = json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, list):                                          # stream form: take the final result event
        data = next((x for x in reversed(data) if isinstance(x, dict) and x.get("type") == "result"), {})
    if not isinstance(data, dict):                                      # valid JSON that is not an object (a bare string, a number)
        data = {}
    text = str(data.get("result", ""))
    usd = float(data.get("total_cost_usd", 0.0) or 0.0)
    turns = int(data.get("num_turns", 0) or 0)
    m = STATUS_RE.findall(text)
    claimed = bool(m) and m[-1] == "DONE"
    contamination = scan_contamination([text, out], wd, markers, changed_files(wd) if changed_files else None)
    if timed_out:
        outcome = "TIMEOUT"
    elif contamination:
        outcome = "CONTAMINATED"
    elif rc != 0 or data.get("is_error") or not data:
        outcome = "CLI_ERROR"
    else:
        outcome = "OK"
    run = AgentRun(run_id, job, spec.role, outcome, claimed and outcome == "OK", text, usd, turns, secs, contamination, str(rd))
    (rd / "run.json").write_text(json.dumps(dataclasses.asdict(run), indent=1), encoding="utf-8")
    # an unknown cost is charged at the per-call cap: a call that ran but reported nothing must not look free (30 Sep: the first
    # real call printed text, not JSON, and was booked at $0)
    started = rc != -2
    budget.record(job, usd if usd > 0 else (budget.policy.per_call_usd if started else 0.0), run_id, outcome)
    return run


# ------------------------------------------------------------------------------------------------ devbench adapter

DEVELOPER = AgentSpec(
    role="developer",
    instructions=("You are a careful software developer working on a small Python repository in the current directory. Read the "
                  "code and tests, make the smallest correct change that achieves the objective, add or update tests where the "
                  "objective implies new behaviour, and run the tests with `python -m pytest -q`. Never edit tests just to make "
                  "them pass. Do not look outside the current directory."))


class AgentSolver:
    """A devbench Solver backed by the agent runtime: one budgeted call per task (a second only if the policy allows it and the
    first did not claim done). The prompt contains the public task only."""

    def __init__(self, budget: Budget, spec: AgentSpec = DEVELOPER, runner: Runner = subprocess_runner,
                 runs_dir: Path = RUNS_DIR, attempts: int = 1, name: str = "agent-developer-v1") -> None:
        self.budget, self.spec, self.runner, self.runs_dir = budget, spec, runner, runs_dir
        self.attempts = max(1, attempts)
        self.name = name
        self.runs: list[AgentRun] = []

    def config(self) -> dict[str, Any]:
        return {"solver": self.name, "model": self.spec.model, "tools": list(self.spec.tools), "attempts": self.attempts,
                "instructions_sha": hashlib.sha256(self.spec.instructions.encode()).hexdigest()[:16]}

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> D.SolverResult:
        calls = 0
        last: Optional[AgentRun] = None
        for attempt in range(self.attempts):
            prompt = f"Objective ({task['category']}): {task['objective']}"
            if last is not None:
                prompt += f"\n\nA previous attempt ended without claiming completion. Its last message was:\n{last.final_text[-1500:]}"
            try:
                last = run_agent(self.spec, f"devbench:{task['id']}:{self.name}", prompt, workdir, self.budget, self.runner,
                                 self.runs_dir)
            except BudgetError as e:
                return D.SolverResult(False, calls, f"budget refused: {e}")
            calls += 1
            self.runs.append(last)
            if last.outcome == "CONTAMINATED":
                raise RuntimeError(f"contaminated worker run {last.run_id}: {last.contamination}")   # scored ERROR, never SOLVED
            if last.claimed_done:
                break
        assert last is not None
        return D.SolverResult(last.claimed_done, calls, f"{last.outcome} ${last.usd:.3f} {last.turns} turns run={last.run_id}")


def clean_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dst
