"""Creator K18 - the Creator's OWN worker: generates code with no other AI (owner, 1 Oct 2026: "the goal is to make it so it is
fully independent of any other ai where it just permanently develops itself forever one day") - IMPLEMENTED, NOT VALIDATED.

Two strategies, both local and offline:

    model   a local open-weights code model (llama.cpp server on 127.0.0.1, weights in C:\\Users\\Peter\\creator_runtime) is
            shown the task and the code, rewrites whole files, sees the visible tests' failures, and retries. The weights are the
            Creator's own file: no service is called, and the Creator may retrain or replace them.
    search  home-grown test-guided program repair: single and paired AST mutations (operator swaps, comparison flips,
            constants +-1, abs()/negation wrappers, argument swaps, return-expression substitutions) applied to the code and kept
            only when every visible test passes.

LEARNING (from development experience only - never from the holdout split):
    * an experience log (state/creator/generator_memory.jsonl): task category, objective, which strategy worked, the files it
      wrote; successful solutions are retrieved as worked examples for similar objectives in later prompts;
    * the strategy order per task category is learned from success rates (Laplace-smoothed), so the Creator spends its effort
      where it has worked before.

Visible tests guide the worker; the sealed hidden tests judge it (creator.devbench). Passing only the visible tests is reported as
claimed_done and scored by the evaluator - overfitting to visible tests shows up as FALSE_COMPLETION."""
from __future__ import annotations

import ast
import copy
import dataclasses
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from creator import devbench as D

RUNTIME = Path(os.environ.get("CREATOR_RUNTIME", str(Path.home() / "creator_runtime")))
SERVER_EXE = RUNTIME / "llama" / "llama-server.exe"
DEFAULT_MODEL = RUNTIME / "models" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MEMORY_FILE = Path(__file__).resolve().parents[1] / "state" / "creator" / "generator_memory.jsonl"
STRATEGIES = ("model", "search")


# ------------------------------------------------------------------------------------------------ the local model

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LocalModel:
    """A llama.cpp server bound to 127.0.0.1 for the life of the context. Nothing leaves this machine."""

    def __init__(self, model: Path = DEFAULT_MODEL, exe: Path = SERVER_EXE, ctx: int = 8192, threads: int = 6,
                 startup_s: float = 120.0) -> None:
        self.model, self.exe, self.ctx, self.threads, self.startup_s = model, exe, ctx, threads, startup_s
        self.port = 0
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.calls = 0
        self.seconds = 0.0

    def __enter__(self) -> "LocalModel":
        if not self.exe.is_file() or not self.model.is_file():
            raise FileNotFoundError(f"local model runtime missing: {self.exe} / {self.model}")
        self.port = free_port()
        self.proc = subprocess.Popen([str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                                      "-c", str(self.ctx), "-t", str(self.threads), "--log-disable"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.startup_s:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("local model server exited during start-up")
            time.sleep(0.5)
        self.__exit__()
        raise TimeoutError("local model server did not become healthy")

    def __exit__(self, *exc: Any) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 1500, temperature: float = 0.2,
             seed: int = 0) -> str:
        body = json.dumps({"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature,
                           "seed": seed}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
        self.calls += 1
        self.seconds += time.monotonic() - t0
        return str(data["choices"][0]["message"]["content"])


# ------------------------------------------------------------------------------------------------ reading and writing code

def read_code(workdir: Path, max_chars: int = 12000) -> dict[str, str]:
    files: dict[str, str] = {}
    total = 0
    for p in sorted(workdir.rglob("*.py")):
        rel = p.relative_to(workdir).as_posix()
        if "__pycache__" in rel or rel.endswith("__init__.py") or rel == "conftest.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if total + len(text) > max_chars:
            break
        files[rel] = text
        total += len(text)
    return files


FILE_RE = re.compile(r"FILE:\s*(?P<path>[\w./-]+\.py)\s*\n```(?:python)?\n(?P<body>.*?)```", re.S)


def parse_files(reply: str) -> dict[str, str]:
    """Whole-file outputs in the format 'FILE: path' + a fenced block. Paths must stay inside the task (no .., no absolute)."""
    out = {}
    for m in FILE_RE.finditer(reply):
        path = m.group("path").strip()
        if path.startswith(("/", "\\")) or ".." in Path(path).parts:
            continue
        out[path] = m.group("body")
    return out


def apply_files(workdir: Path, files: Mapping[str, str]) -> None:
    for rel, body in files.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def visible_tests(workdir: Path) -> D.TestCounts:
    return D.run_pytest(workdir, "tests", timeout=120)


def failure_text(workdir: Path, limit: int = 2500) -> str:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                       cwd=workdir, capture_output=True, text=True, timeout=120)
    return (p.stdout + p.stderr)[-limit:]


# ------------------------------------------------------------------------------------------------ experience (learning)

@dataclasses.dataclass(frozen=True)
class Experience:
    category: str
    objective: str
    strategy: str
    solved_visible: bool
    files: Mapping[str, str]
    seconds: float


class ExperienceMemory:
    """Append-only experience log. Learns only from what it is given - the caller never records holdout tasks."""

    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self.path = path

    def load(self) -> list[Experience]:
        if not self.path.is_file():
            return []
        out = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(Experience(**json.loads(ln)))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def add(self, e: Experience) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dataclasses.asdict(e)) + "\n")

    def strategy_order(self, category: str) -> list[str]:
        """Strategies ranked by Laplace-smoothed success rate in this category (ties keep the default order)."""
        exp = [e for e in self.load() if e.category == category]

        def rate(s: str) -> float:
            xs = [e for e in exp if e.strategy == s]
            return (sum(e.solved_visible for e in xs) + 1) / (len(xs) + 2)
        return sorted(STRATEGIES, key=lambda s: (-rate(s), STRATEGIES.index(s)))

    def examples(self, objective: str, k: int = 2) -> list[Experience]:
        """The k most similar SUCCESSFUL past solutions (word-overlap similarity) - worked examples for the model."""
        words = set(re.findall(r"[a-z]+", objective.lower()))
        good = [e for e in self.load() if e.solved_visible and e.strategy == "model" and e.files]
        scored = sorted(good, key=lambda e: -len(words & set(re.findall(r"[a-z]+", e.objective.lower()))))
        return [e for e in scored[:k] if words & set(re.findall(r"[a-z]+", e.objective.lower()))]


# ------------------------------------------------------------------------------------------------ strategy: model

SYSTEM_STEPWISE = ("You are a careful Python developer. First find the exact line that makes the tests or the objective fail, "
                   "then change as little as possible. Keep every existing behaviour the objective does not mention. "
                   "Reply ONLY with the complete new content of every file you change, each as:\nFILE: <path>\n```python\n<code>\n```\n"
                   "Never edit existing tests unless the objective asks for new tests.")
SYSTEM = ("You are a careful Python developer. You change code so that the objective is met and all tests pass. "
          "Reply ONLY with the complete new content of every file you change, each as:\nFILE: <path>\n```python\n<code>\n```\n"
          "Never edit existing tests unless the objective asks for new tests.")


def model_prompt(task: Mapping[str, Any], code: Mapping[str, str], examples: Sequence[Experience]) -> str:
    parts = []
    for ex in examples:
        shown = "\n".join(f"FILE: {p}\n```python\n{b}```" for p, b in list(ex.files.items())[:2])
        parts.append(f"Example of a solved task. Objective: {ex.objective}\nSolution:\n{shown}")
    parts.append(f"Objective ({task.get('category', 'task')}): {task['objective']}")
    parts.append("Current files:\n" + "\n".join(f"FILE: {p}\n```python\n{b}```" for p, b in code.items()))
    return "\n\n".join(parts)


def solve_with_model(task: Mapping[str, Any], workdir: Path, llm: Any, memory: Optional[ExperienceMemory],
                     attempts: int = 3, temperature: float = 0.2, examples_k: int = 2,
                     system: str = SYSTEM) -> tuple[bool, dict[str, str], int]:
    """Generate -> run visible tests -> feed failures back. Returns (visible tests pass, files written, model calls)."""
    examples = memory.examples(str(task["objective"]), k=examples_k) if memory and examples_k > 0 else []
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": model_prompt(task, read_code(workdir), examples)}]
    written: dict[str, str] = {}
    calls = 0
    for _ in range(attempts):
        reply = llm.chat(messages, seed=calls, temperature=temperature)
        calls += 1
        files = parse_files(reply)
        if not files:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Reply with complete files in the FILE: format only."}]
            continue
        apply_files(workdir, files)
        written.update(files)
        if visible_tests(workdir).ok:
            return True, written, calls
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "The tests fail:\n" + failure_text(workdir) + "\nFix it. Complete files only."}]
    return False, written, calls


# ------------------------------------------------------------------------------------------------ strategy: search

SWAP_BIN: dict[type, tuple[type, ...]] = {ast.Add: (ast.Sub, ast.Mult), ast.Sub: (ast.Add,), ast.Mult: (ast.Add, ast.FloorDiv),
                                         ast.FloorDiv: (ast.Div, ast.Mult), ast.Div: (ast.FloorDiv,)}
SWAP_CMP: dict[type, tuple[type, ...]] = {ast.Lt: (ast.LtE, ast.Gt), ast.LtE: (ast.Lt,), ast.Gt: (ast.GtE, ast.Lt),
                                         ast.GtE: (ast.Gt,), ast.Eq: (ast.NotEq,), ast.NotEq: (ast.Eq,)}


Edit = Callable[[Any], None]


def _set_op(alt: type) -> Edit:
    def f(m: Any) -> None:
        m.op = alt()
    return f


def _set_cmp(alt: type) -> Edit:
    def f(m: Any) -> None:
        m.ops[0] = alt()
    return f


def _shift(d: int) -> Edit:
    def f(m: Any) -> None:
        m.value = m.value + d
    return f


def _wrap_abs(m: Any) -> None:
    m.value = ast.Call(ast.Name("abs", ast.Load()), [m.value], [])


def _swap_args(m: Any) -> None:
    m.args = [m.args[1], m.args[0]]


def _none_default(j: int) -> Edit:
    def f(m: Any) -> None:
        m.args.defaults[j] = ast.Constant(None)
    return f


def mutations(tree: ast.Module) -> Iterator[ast.Module]:
    """Single-edit variants of a module, most plausible bug fixes first."""
    for i, n in enumerate(list(ast.walk(tree))):
        if isinstance(n, ast.BinOp):
            for alt in SWAP_BIN.get(type(n.op), ()):
                yield _edit(tree, i, _set_op(alt))
        elif isinstance(n, ast.Compare) and n.ops:
            for alt in SWAP_CMP.get(type(n.ops[0]), ()):
                yield _edit(tree, i, _set_cmp(alt))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            for d in (1, -1):
                yield _edit(tree, i, _shift(d))
        elif isinstance(n, ast.Return) and n.value is not None:
            yield _edit(tree, i, _wrap_abs)
        elif isinstance(n, ast.Call) and len(n.args) == 2:
            yield _edit(tree, i, _swap_args)
        elif isinstance(n, ast.FunctionDef):
            for j, dflt in enumerate(n.args.defaults):
                if isinstance(dflt, (ast.List, ast.Dict)):
                    yield _edit(tree, i, _none_default(j))


def _edit(tree: ast.Module, index: int, fn: Callable[[Any], None]) -> ast.Module:
    t = copy.deepcopy(tree)
    target = list(ast.walk(t))[index]
    fn(target)
    return ast.fix_missing_locations(t)


def solve_with_search(task: Mapping[str, Any], workdir: Path, budget: int = 120, pairs: bool = True) -> tuple[bool, dict[str, str], int]:
    """Test-guided mutation repair over the non-test code. Returns (visible tests pass, files written, candidates tried)."""
    targets = [p for p in sorted(workdir.rglob("*.py")) if "tests" not in p.relative_to(workdir).parts
               and p.name not in ("__init__.py", "conftest.py") and "__pycache__" not in p.parts]
    tried = 0
    for path in targets:
        original = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(original)
        except SyntaxError:
            continue
        singles = list(mutations(tree))
        cands: Iterator[ast.Module] = iter(singles)
        if pairs:
            cands = itertools.chain(singles, (m2 for m1 in singles[:12] for m2 in itertools.islice(mutations(m1), 12)))
        for cand in cands:
            if tried >= budget:
                path.write_text(original, encoding="utf-8")
                return False, {}, tried
            tried += 1
            src = ast.unparse(cand) + "\n"
            path.write_text(src, encoding="utf-8")
            if visible_tests(workdir).ok:
                return True, {path.relative_to(workdir).as_posix(): src}, tried
        path.write_text(original, encoding="utf-8")
    return False, {}, tried


# ------------------------------------------------------------------------------------------------ the solver

@dataclasses.dataclass(frozen=True)
class WorkerConfig:
    """Everything about how the worker works that the Creator may change by itself (creator.autotune), one measured step at
    a time. The defaults are the first version a human wrote; every later version is the Creator's own choice."""
    model_attempts: int = 3
    temperature: float = 0.2
    examples_k: int = 2
    search_budget: int = 120
    order: str = "learned"                      # learned | model_first | search_first
    prompt: str = "plain"                       # plain | stepwise

    def digest(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:12]


PROMPTS = {"plain": SYSTEM, "stepwise": SYSTEM_STEPWISE}


class GeneratorSolver:
    """A devbench Solver made only of the Creator's own parts. Learns from dev tasks; holdout tasks are never recorded."""

    def __init__(self, llm: Any, memory: Optional[ExperienceMemory] = None, learn: bool = True, model_attempts: int = 3,
                 search_budget: int = 120, name: str = "creator-generator-v1",
                 config: Optional[WorkerConfig] = None) -> None:
        self.cfg = config or WorkerConfig(model_attempts=model_attempts, search_budget=search_budget)
        self.llm, self.memory, self.learn = llm, memory, learn
        self.model_attempts, self.search_budget, self.name = self.cfg.model_attempts, self.cfg.search_budget, name
        self.log: list[dict[str, Any]] = []

    def config(self) -> dict[str, Any]:
        return {"solver": self.name, "model": str(getattr(self.llm, "model", "")), "learn": self.learn,
                **dataclasses.asdict(self.cfg)}

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> D.SolverResult:
        cat = str(task.get("category", ""))
        if self.cfg.order == "model_first":
            order = ["model", "search"]
        elif self.cfg.order == "search_first":
            order = ["search", "model"]
        else:
            order = self.memory.strategy_order(cat) if self.memory else list(STRATEGIES)
        calls = 0
        for strategy in order:
            t0 = time.monotonic()
            if strategy == "model":
                ok, files, n = solve_with_model(task, workdir, self.llm, self.memory, self.cfg.model_attempts,
                                                self.cfg.temperature, self.cfg.examples_k, PROMPTS[self.cfg.prompt])
                calls += n
            else:
                ok, files, n = solve_with_search(task, workdir, self.cfg.search_budget)
            secs = round(time.monotonic() - t0, 1)
            self.log.append({"task": task["id"], "strategy": strategy, "visible_ok": ok, "n": n, "seconds": secs})
            if self.memory is not None and self.learn:
                self.memory.add(Experience(cat, str(task["objective"]), strategy, ok, files if ok else {}, secs))
            if ok:
                return D.SolverResult(True, calls, f"{strategy} passed the visible tests after {n} tries")
        return D.SolverResult(False, calls, f"no strategy passed the visible tests ({order})")
