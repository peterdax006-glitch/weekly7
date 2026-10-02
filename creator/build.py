"""K06 build stage (CR03; canon C77 sec 23 "SOURCE EXISTS is not SYSTEM BUILDS", sec 29 version comparison, sec 61 integrity).

A change builds only if every changed Python file compiles, every changed module (and, optionally, its importers) imports from
the SANDBOX tree (an import that resolves to another tree is an environment fault, not a pass), and CI's type-check command shows
no NEW issues against the same command on the base commit. Every failure is classified SYNTAX / IMPORT / TYPE / ENVIRONMENT /
TIMEOUT and the raw output is kept as evidence. Process launching lives here and is shared by testrun.py and sandbox.py.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EVIDENCE_TAIL = 20_000          # characters of raw stdout/stderr kept per process in a record
_SCRUB_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONSAFEPATH", "PYTEST_ADDOPTS", "MYPYPATH", "GIT_DIR",
              "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR", "GIT_ALTERNATE_OBJECT_DIRECTORIES")


class BuildFailureKind(str, Enum):
    SYNTAX = "SYNTAX"
    IMPORT = "IMPORT"
    TYPE = "TYPE"
    ENVIRONMENT = "ENVIRONMENT"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class ProcResult:
    """One external process: argv, where, how it ended, and its (tail-trimmed) output as evidence."""
    argv: tuple[str, ...]
    cwd: str
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    launch_error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.launch_error

    def evidence(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "cwd": self.cwd, "returncode": self.returncode, "seconds": round(self.seconds, 3),
                "timed_out": self.timed_out, "launch_error": self.launch_error,
                "stdout": self.stdout[-EVIDENCE_TAIL:], "stderr": self.stderr[-EVIDENCE_TAIL:]}


def clean_env(root: str | Path | None = None, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Child environment: inherited variables that could redirect imports, pytest or git to ANOTHER tree are removed;
    the sandbox root (if given) is the only PYTHONPATH entry; no bytecode is written into the worktree."""
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV}
    if root is not None:
        env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.update(extra or {})
    return env


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def run_cmd(argv: Sequence[str], cwd: str | Path, *, timeout: float, env: Mapping[str, str] | None = None,
            stdin: str | None = None, budget: bool = False) -> ProcResult:
    """Run one child process to completion (or its own timeout). A timeout or a launch failure is a result, never an
    exception: callers classify it. Only the child started here (and its descendants) is ever stopped. budget=True: a TEST launch,
    it waits for a machine-wide test slot (creator.testslots) and measures its memory; verdicts are unaffected."""
    t0 = time.monotonic()
    args = tuple(str(a) for a in argv)
    try:
        if budget:
            from creator import testslots
            cp = testslots.run(list(args), cwd=str(cwd), text=True, encoding="utf-8", errors="replace", timeout=timeout,
                               env=dict(env) if env is not None else None, input=stdin)
        else:
            cp = subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, env=dict(env) if env is not None else None, input=stdin)
    except subprocess.TimeoutExpired as e:
        return ProcResult(args, str(cwd), None, _text(e.stdout), _text(e.stderr), time.monotonic() - t0, timed_out=True)
    except OSError as e:
        return ProcResult(args, str(cwd), None, "", "", time.monotonic() - t0, launch_error=f"{type(e).__name__}: {e}")
    return ProcResult(args, str(cwd), cp.returncode, cp.stdout or "", cp.stderr or "", time.monotonic() - t0)


# ---------------------------------------------------------------------------------------------------------------- modules
def module_name_for(rel_path: str) -> str | None:
    """'pkg/sub/mod.py' -> 'pkg.sub.mod'; 'pkg/__init__.py' -> 'pkg'; None for non-Python or non-importable paths."""
    p = rel_path.replace("\\", "/").strip("/")
    if not p.endswith(".py"):
        return None
    parts = p[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(x.isidentifier() for x in parts):
        return None
    return ".".join(parts)


def repo_toplevels(root: str | Path) -> frozenset[str]:
    """Top-level importable names that belong to the tree (packages, namespace dirs holding .py files, top-level modules)."""
    out: set[str] = set()
    rootp = Path(root)
    if not rootp.is_dir():
        return frozenset()
    for child in rootp.iterdir():
        if child.name.startswith(".") or not child.name.replace("-", "_").isidentifier():
            continue
        if child.is_file() and child.suffix == ".py" and child.stem.isidentifier():
            out.add(child.stem)
        elif child.is_dir() and child.name.isidentifier():
            if (child / "__init__.py").exists() or any(child.glob("*.py")):
                out.add(child.name)
    return frozenset(out)


# ------------------------------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class BuildIssue:
    kind: BuildFailureKind
    path: str
    message: str
    line: int | None = None
    code: str = ""
    module: str = ""

    def key(self) -> tuple[str, str, str, str]:
        """Identity used for new-vs-base comparison: line numbers move when code above is edited, so they are excluded."""
        return (self.kind.value, self.path.replace("\\", "/").lower(), self.code, re.sub(r"\s+", " ", self.message).strip())

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "path": self.path, "line": self.line, "code": self.code, "module": self.module,
                "message": self.message[:2000]}


@dataclass(frozen=True)
class BuildStep:
    name: str                               # compile | import | typecheck
    ran: bool
    issues: tuple[BuildIssue, ...] = ()
    proc: ProcResult | None = None
    skipped_reason: str = ""
    seconds: float = 0.0
    checked: tuple[str, ...] = ()           # files or modules the step actually covered

    @property
    def ok(self) -> bool:
        return self.ran and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ran": self.ran, "ok": self.ok, "skipped_reason": self.skipped_reason,
                "seconds": round(self.seconds, 3), "checked": list(self.checked),
                "issues": [i.to_dict() for i in self.issues], "proc": self.proc.evidence() if self.proc else None}


@dataclass(frozen=True)
class BuildResult:
    """Outcome of building one tree. `baseline_keys` holds type issues already present at the base commit: those are reported
    but do not fail this change (a fault that predates the change is not caused by it)."""
    tree: str
    steps: tuple[BuildStep, ...]
    baseline_keys: frozenset[tuple[str, str, str, str]] = frozenset()
    require_typecheck: bool = True
    notes: tuple[str, ...] = ()

    def step(self, name: str) -> BuildStep | None:
        return next((s for s in self.steps if s.name == name), None)

    @property
    def all_issues(self) -> tuple[BuildIssue, ...]:
        return tuple(i for s in self.steps for i in s.issues)

    @property
    def new_issues(self) -> tuple[BuildIssue, ...]:
        return tuple(i for i in self.all_issues if i.kind is not BuildFailureKind.TYPE or i.key() not in self.baseline_keys)

    @property
    def preexisting_issues(self) -> tuple[BuildIssue, ...]:
        return tuple(i for i in self.all_issues if i.kind is BuildFailureKind.TYPE and i.key() in self.baseline_keys)

    @property
    def kinds(self) -> frozenset[BuildFailureKind]:
        return frozenset(i.kind for i in self.new_issues)

    @property
    def ok(self) -> bool:
        """Fail closed: compile and import must have run; the type check must have run unless explicitly not required."""
        comp, imp, typ = self.step("compile"), self.step("import"), self.step("typecheck")
        if comp is None or not comp.ran or imp is None or not imp.ran:
            return False
        if self.require_typecheck and (typ is None or not typ.ran):
            return False
        return not self.new_issues

    def to_record(self) -> dict[str, Any]:
        return {"tree": self.tree, "ok": self.ok, "kinds": sorted(k.value for k in self.kinds),
                "new_issues": len(self.new_issues), "preexisting_type_issues": len(self.preexisting_issues),
                "require_typecheck": self.require_typecheck, "notes": list(self.notes),
                "steps": [s.to_dict() for s in self.steps]}


# ------------------------------------------------------------------------------------------------------------- compile
def compile_check(root: str | Path, rel_paths: Iterable[str]) -> BuildStep:
    """Byte-compile every changed .py file in-process (compile() executes nothing). Encoding errors, null bytes and syntax
    errors are all SYNTAX: the source cannot become code."""
    t0 = time.monotonic()
    issues: list[BuildIssue] = []
    checked: list[str] = []
    for rel in sorted(set(rel_paths)):
        if not rel.endswith(".py"):
            continue
        path = Path(root) / rel
        if not path.is_file():
            continue                        # a deletion: nothing to compile; importers are checked in the import step
        checked.append(rel)
        try:
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except SyntaxError as e:
            issues.append(BuildIssue(BuildFailureKind.SYNTAX, rel, f"{type(e).__name__}: {e.msg}", e.lineno,
                                     module=module_name_for(rel) or ""))
        except (ValueError, UnicodeDecodeError) as e:
            issues.append(BuildIssue(BuildFailureKind.SYNTAX, rel, f"{type(e).__name__}: {e}", None,
                                     module=module_name_for(rel) or ""))
    return BuildStep("compile", True, tuple(issues), None, seconds=time.monotonic() - t0, checked=tuple(checked))


# -------------------------------------------------------------------------------------------------------------- import
_IMPORT_MARK = "@@CREATOR_IMPORT@@"
_IMPORT_SCRIPT = r"""
import importlib, json, os, sys, traceback
root = os.path.normcase(os.path.realpath(sys.argv[1]))
out = []
for name in sys.argv[2:]:
    rec = {"module": name}
    try:
        m = importlib.import_module(name)
        f = getattr(m, "__file__", None)
        rec["ok"] = True
        rec["file"] = f
        if f is None:
            locs = [os.path.normcase(os.path.realpath(p)) for p in getattr(m, "__path__", [])]
            if locs and not any(p == root or p.startswith(root + os.sep) for p in locs):
                rec.update(ok=False, etype="Shadowed", message="namespace package resolved outside the sandbox: %r" % locs)
        else:
            rf = os.path.normcase(os.path.realpath(f))
            if not rf.startswith(root + os.sep):
                rec.update(ok=False, etype="Shadowed", message="module resolved outside the sandbox: %s" % f)
    except BaseException as e:
        rec.update(ok=False, etype=type(e).__name__, message=str(e)[:2000], missing=getattr(e, "name", None),
                   lineno=getattr(e, "lineno", None), filename=getattr(e, "filename", None),
                   tb=traceback.format_exc()[-4000:])
    out.append(rec)
sys.stdout.write("\n@@CREATOR_IMPORT@@" + json.dumps(out) + "\n")
"""


def classify_import_failure(rec: Mapping[str, Any], toplevels: frozenset[str]) -> BuildFailureKind:
    """A missing THIRD-PARTY module is an environment fault (the code may be right; the machine lacks a package). A missing
    module of the tree itself, an exception raised while executing the module, or a name that no longer exists is IMPORT."""
    etype = str(rec.get("etype", ""))
    if etype in ("SyntaxError", "IndentationError", "TabError"):
        return BuildFailureKind.SYNTAX
    if etype == "Shadowed":
        return BuildFailureKind.ENVIRONMENT
    if etype == "ModuleNotFoundError":
        missing = str(rec.get("missing") or "")
        top = missing.split(".")[0]
        if top and top not in toplevels:
            return BuildFailureKind.ENVIRONMENT
    return BuildFailureKind.IMPORT


def import_check(root: str | Path, modules: Iterable[str], *, python: str = sys.executable, timeout: float = 120.0,
                 toplevels: frozenset[str] | None = None, path_of: Mapping[str, str] | None = None) -> BuildStep:
    """Import each module in ONE fresh child interpreter rooted at the sandbox, so a module's import-time side effects never
    reach this process and a module that silently resolves to another tree (e.g. the main checkout) is caught."""
    mods = sorted(set(m for m in modules if m))
    if not mods:
        return BuildStep("import", True, (), None, skipped_reason="no importable modules changed")
    tops = toplevels if toplevels is not None else repo_toplevels(root)
    t0 = time.monotonic()
    proc = run_cmd([python, "-c", _IMPORT_SCRIPT, str(root), *mods], root, timeout=timeout, env=clean_env(root))
    paths = dict(path_of or {})
    if proc.timed_out:
        return BuildStep("import", True, (BuildIssue(BuildFailureKind.TIMEOUT, "", f"import check exceeded {timeout}s"),),
                         proc, seconds=time.monotonic() - t0, checked=tuple(mods))
    payload = _parse_import_payload(proc.stdout)
    if proc.launch_error or payload is None:
        msg = proc.launch_error or f"import checker produced no result (exit {proc.returncode})"
        return BuildStep("import", True, (BuildIssue(BuildFailureKind.ENVIRONMENT, "", msg),), proc,
                         seconds=time.monotonic() - t0, checked=tuple(mods))
    issues: list[BuildIssue] = []
    seen = {str(r.get("module")) for r in payload}
    for rec in payload:
        if rec.get("ok"):
            continue
        kind = classify_import_failure(rec, tops)
        name = str(rec.get("module"))
        line = rec.get("lineno") if isinstance(rec.get("lineno"), int) else None
        issues.append(BuildIssue(kind, paths.get(name, name.replace(".", "/") + ".py"),
                                 f"{rec.get('etype')}: {rec.get('message')}", line, module=name))
    for missing in sorted(set(mods) - seen):     # the child died part-way: modules never reported are not assumed to pass
        issues.append(BuildIssue(BuildFailureKind.ENVIRONMENT, paths.get(missing, missing), "import result missing",
                                 module=missing))
    return BuildStep("import", True, tuple(issues), proc, seconds=time.monotonic() - t0, checked=tuple(mods))


def _parse_import_payload(stdout: str) -> list[dict[str, Any]] | None:
    idx = stdout.rfind(_IMPORT_MARK)
    if idx < 0:
        return None
    line = stdout[idx + len(_IMPORT_MARK):].splitlines()[0] if stdout[idx + len(_IMPORT_MARK):] else ""
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else None


# ----------------------------------------------------------------------------------------------------------- typecheck
_MYPY_LINE = re.compile(r"^(?P<path>(?:[A-Za-z]:)?[^:\n]+?):(?P<line>\d+)(?::\d+)?: (?P<sev>error|warning|note): "
                        r"(?P<msg>.*?)(?:\s+\[(?P<code>[a-z0-9-]+)\])?\s*$")
_MISSING_MOD = re.compile(r'(?:Cannot find implementation or library stub for module named|No module named) "?\'?'
                          r'(?P<mod>[A-Za-z0-9_.]+)')


def ci_typecheck_command(root: str | Path, python: str = sys.executable) -> list[str]:
    """CI's type-check command (the `run:` step of .github/workflows/ci.yml that calls mypy), with `python` mapped to the
    interpreter in use. Falls back to `python -m mypy` (which reads [tool.mypy] in pyproject.toml) if CI names none."""
    ci = Path(root) / ".github" / "workflows" / "ci.yml"
    if ci.is_file():
        for raw in ci.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if not line.startswith("run:") or "mypy" not in line:
                continue
            cmd = line[4:].strip()
            if any(ch in cmd for ch in "|&;><"):
                continue                    # shell pipelines cannot be run without a shell; keep looking
            try:
                argv = shlex.split(cmd)
            except ValueError:
                continue
            invokes = bool(argv) and (Path(argv[0]).stem == "mypy" or (Path(argv[0]).stem in ("python", "python3")
                                                                        and argv[1:3] == ["-m", "mypy"]))
            if not invokes:
                continue                    # e.g. `pip install ... mypy` (1 Oct: the kernel ran a pip install as 'type check')
            if Path(argv[0]).stem in ("python", "python3"):
                argv[0] = python
            return argv
    return [python, "-m", "mypy"]


def parse_typecheck_output(text: str, toplevels: frozenset[str]) -> list[BuildIssue]:
    issues: list[BuildIssue] = []
    for raw in text.splitlines():
        m = _MYPY_LINE.match(raw.strip())
        if not m or m.group("sev") != "error":
            continue
        code = m.group("code") or ""
        msg = m.group("msg")
        kind = BuildFailureKind.TYPE
        if code == "syntax":
            kind = BuildFailureKind.SYNTAX
        elif code in ("import-not-found", "import-untyped", "import"):
            mm = _MISSING_MOD.search(msg)
            top = mm.group("mod").split(".")[0] if mm else ""
            kind = BuildFailureKind.IMPORT if top and top in toplevels else BuildFailureKind.ENVIRONMENT
        issues.append(BuildIssue(kind, m.group("path").strip(), msg, int(m.group("line")), code))
    return issues


def typecheck(root: str | Path, command: Sequence[str] | None, *, timeout: float = 600.0, cache_dir: str | Path | None = None,
              toplevels: frozenset[str] | None = None) -> BuildStep:
    """Run the type-check command in the tree. mypy exit 0 = clean, 1 = type errors, 2 = crash/usage/config: a 2 with no
    parsed errors, a missing mypy, or a timeout is an ENVIRONMENT/TIMEOUT fault, never a pass."""
    if not command:
        return BuildStep("typecheck", False, (), None, skipped_reason="no type-check command configured")
    argv = list(command)
    if cache_dir is not None and not any(a.startswith("--cache-dir") for a in argv):
        argv.append(f"--cache-dir={cache_dir}")
    tops = toplevels if toplevels is not None else repo_toplevels(root)
    t0 = time.monotonic()
    proc = run_cmd(argv, root, timeout=timeout, env=clean_env(root))
    secs = time.monotonic() - t0
    if proc.timed_out:
        return BuildStep("typecheck", True, (BuildIssue(BuildFailureKind.TIMEOUT, "", f"type check exceeded {timeout}s"),),
                         proc, seconds=secs)
    if proc.launch_error:
        return BuildStep("typecheck", True, (BuildIssue(BuildFailureKind.ENVIRONMENT, "", proc.launch_error),), proc,
                         seconds=secs)
    issues = parse_typecheck_output(proc.stdout + "\n" + proc.stderr, tops)
    combined = (proc.stdout + proc.stderr).lower()
    if proc.returncode != 0 and not issues:
        why = "type checker is not installed" if "no module named mypy" in combined else \
            f"type checker failed without reporting errors (exit {proc.returncode})"
        issues.append(BuildIssue(BuildFailureKind.ENVIRONMENT, "", why))
    elif proc.returncode == 0 and issues:
        issues = [i for i in issues if i.kind is not BuildFailureKind.TYPE]   # warnings-as-text on a clean exit are not errors
    return BuildStep("typecheck", True, tuple(issues), proc, seconds=secs)


# --------------------------------------------------------------------------------------------------------------- build
@dataclass
class BuildConfig:
    python: str = sys.executable
    typecheck_command: list[str] | None = None      # None -> derived from the tree's CI file at build time
    run_typecheck: bool = True
    require_typecheck: bool = True
    import_dependents: bool = False                 # also import modules that import a changed module
    import_timeout: float = 120.0
    typecheck_timeout: float = 600.0
    extra_modules: list[str] = field(default_factory=list)


def run_build(root: str | Path, changed_paths: Iterable[str], *, config: BuildConfig | None = None, tree: str = "",
              cache_dir: str | Path | None = None, dependents: Iterable[str] = (),
              baseline_keys: Iterable[tuple[str, str, str, str]] = ()) -> BuildResult:
    """Build one tree for a change: compile -> import -> type check. The import step is skipped only for modules whose source
    did not compile (their SYNTAX issue already fails the build); the type check runs only if compile passed."""
    cfg = config or BuildConfig()
    changed = sorted({p.replace("\\", "/") for p in changed_paths})
    tops = repo_toplevels(root)
    comp = compile_check(root, changed)
    bad = {i.path for i in comp.issues}
    path_of: dict[str, str] = {}
    for p in changed:
        name = module_name_for(p)
        if name and p not in bad and (Path(root) / p).is_file():
            path_of[name] = p
    mods = set(path_of) | set(cfg.extra_modules)
    if cfg.import_dependents:
        mods |= {d for d in dependents if d}
    imp = import_check(root, mods, python=cfg.python, timeout=cfg.import_timeout, toplevels=tops, path_of=path_of)
    notes: list[str] = []
    if not cfg.run_typecheck:
        typ = BuildStep("typecheck", False, (), None, skipped_reason="type check disabled by configuration")
        notes.append("typecheck disabled")
    elif comp.issues:
        typ = BuildStep("typecheck", False, (), None, skipped_reason="sources do not compile")
    else:
        cmd = cfg.typecheck_command if cfg.typecheck_command is not None else ci_typecheck_command(root, cfg.python)
        typ = typecheck(root, cmd, timeout=cfg.typecheck_timeout, cache_dir=cache_dir, toplevels=tops)
    return BuildResult(tree, (comp, imp, typ), frozenset(baseline_keys), cfg.require_typecheck, tuple(notes))


def baseline_type_keys(step: BuildStep | None) -> frozenset[tuple[str, str, str, str]]:
    """Issue keys of a base-commit type check, for new-vs-preexisting separation. A base check that itself failed to run
    (environment/timeout) yields no keys, so every candidate issue then counts as new (fail closed)."""
    if step is None or not step.ran:
        return frozenset()
    return frozenset(i.key() for i in step.issues if i.kind is BuildFailureKind.TYPE)
