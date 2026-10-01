"""Creator K02 - the self-model, derived from evidence (C77 secs 11, 41, 42, 58; package CR02) - IMPLEMENTED, NOT VALIDATED.

The Creator's machine-readable model of itself, built by SCANNING the tree, never by asking anything to describe itself:

    components   every module: path, dotted name, first docstring line, meaningful lines (the one ruler), public interface
                 (classes, functions, signatures), source hash, stub markers (NotImplementedError / pass-only bodies / TODO)
    dependencies the static import graph (creator.testrun.ImportGraph)
    tests        test files, which modules each one reaches, and the latest recorded result per test file
    capabilities each declared capability (creator/capabilities.json: id -> modules, tests, depth floor) with a state COMPUTED from
                 evidence: NOT_STARTED (no module) / IMPLEMENTED (code, no passing tests) / TESTED (its tests ran and passed on the
                 current source) / FAILED (its tests ran and failed on the current source) - never higher; VALIDATED comes only from
                 the ledger via an independent role
    limitations  below-floor depth, stubs, untested modules, failing tests, unparsable files, unreached modules
    resources    CPU count, free RAM, free disk, python version
    versions     git HEAD (+dirty), creator and engine tree hashes
    active work  open ledger items and live sandboxes

`diagnose()` compares CLAIMS (a checklist or the ledger saying a capability is TESTED/VALIDATED) with this evidence; where they
disagree the evidence wins and a Contradiction is reported (C77 sec 42). The model is deterministic: the same tree and the same
evidence give the same digest."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from creator import testrun as T
from creator.build import module_name_for

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_FILE = Path(__file__).resolve().parent / "capabilities.json"
SKIP_DIRS = frozenset({".git", ".venv", "venv", "__pycache__", "node_modules", "state", "data", ".mypy_cache", ".pytest_cache"})
STUB_TODO_MARKERS = ("TODO", "FIXME", "XXX")


# ------------------------------------------------------------------------------------------------ the ruler

def _ruler() -> Any:
    """scripts/contract_lines.meaningful(path): the repository's ONE ruler (blanks, comments and docstrings excluded)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("contract_lines", REPO_ROOT / "scripts" / "contract_lines.py")
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "meaningful", None)


_RULER = _ruler()


def meaningful_lines(path: Path) -> int:
    """Meaningful lines of one file by the repository ruler; a file the ruler cannot parse counts its non-blank, non-comment lines."""
    if _RULER is not None:
        try:
            return int(_RULER(path))
        except (SyntaxError, ValueError, UnicodeDecodeError):
            pass
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))


# ------------------------------------------------------------------------------------------------ components

@dataclasses.dataclass(frozen=True)
class Interface:
    kind: str                       # class | function | method
    name: str
    signature: str
    doc: str


@dataclasses.dataclass(frozen=True)
class Component:
    path: str
    module: str
    doc: str
    lines: int
    meaningful: int
    sha256: str
    interfaces: tuple[Interface, ...]
    stubs: tuple[str, ...]          # "name:line: why"
    todos: int
    parse_error: str = ""


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        return f"{fn.name}({ast.unparse(fn.args)})" + (f" -> {ast.unparse(fn.returns)}" if fn.returns else "")
    except Exception:                                                   # noqa: BLE001 - unparse is best-effort
        return fn.name


def _first_doc(node: Any) -> str:
    d = ast.get_docstring(node) or ""
    return d.strip().splitlines()[0][:200] if d.strip() else ""


def _is_stub(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]                                                  # ignore the docstring
    if not body:
        return "docstring only"
    if all(isinstance(s, ast.Pass) for s in body):
        return "pass only"
    for s in body:
        if isinstance(s, ast.Raise) and s.exc is not None:
            exc = s.exc.func if isinstance(s.exc, ast.Call) else s.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return "raises NotImplementedError"
    only = body[0] if len(body) == 1 else None
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) and only.value.value is Ellipsis:
        return "ellipsis only"
    return None


def scan_component(root: Path, rel: str) -> Component:
    full = root / rel
    raw = full.read_bytes()
    source = raw.decode("utf-8", errors="replace")
    module = module_name_for(rel) or rel[:-3].replace("/", ".")
    sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    todos = sum(source.count(m) for m in STUB_TODO_MARKERS)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return Component(rel, module or "", "", source.count("\n") + 1, meaningful_lines(full), sha, (), (), todos,
                         parse_error=f"{e.msg} at line {e.lineno}")
    ifaces: list[Interface] = []
    stubs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ifaces.append(Interface("function", node.name, _signature(node), _first_doc(node)))
            why = _is_stub(node)
            if why and not node.name.startswith("_"):
                stubs.append(f"{node.name}:{node.lineno}: {why}")
        elif isinstance(node, ast.ClassDef):
            ifaces.append(Interface("class", node.name, node.name, _first_doc(node)))
            is_protocol = any(isinstance(b, ast.Name) and b.id in ("Protocol", "ABC") for b in node.bases)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("__"):
                    ifaces.append(Interface("method", f"{node.name}.{sub.name}", _signature(sub), _first_doc(sub)))
                    why = _is_stub(sub)
                    abstract = any(isinstance(d, ast.Name) and d.id == "abstractmethod" for d in sub.decorator_list)
                    if why and not is_protocol and not abstract and not sub.name.startswith("_"):
                        stubs.append(f"{node.name}.{sub.name}:{sub.lineno}: {why}")
    return Component(rel, module or "", _first_doc(tree), source.count("\n") + 1, meaningful_lines(full), sha,
                     tuple(ifaces), tuple(stubs), todos)


def python_files(root: Path, under: Sequence[str] = (), skip: Iterable[str] = SKIP_DIRS) -> list[str]:
    skip_set = set(skip)
    roots = [root / u for u in under] if under else [root]
    out: list[str] = []
    for base in roots:
        if base.is_file() and base.suffix == ".py":
            out.append(base.relative_to(root).as_posix())
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in skip_set and not d.startswith("."))
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    out.append((Path(dirpath) / fn).relative_to(root).as_posix())
    return sorted(set(out))


# ------------------------------------------------------------------------------------------------ capabilities

@dataclasses.dataclass(frozen=True)
class CapabilitySpec:
    """A declared capability: which modules implement it, which tests exercise it, its depth floor. Declared in
    creator/capabilities.json; the STATE is never declared - it is computed."""
    id: str
    name: str
    modules: tuple[str, ...]
    tests: tuple[str, ...]
    floor: int = 0


@dataclasses.dataclass(frozen=True)
class CapabilityState:
    id: str
    name: str
    state: str                      # NOT_STARTED / IMPLEMENTED / TESTED / FAILED (computed)
    uncertainty: str                # sec 71
    meaningful: int
    floor: int
    below_floor: bool
    present_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]
    tests_present: tuple[str, ...]
    tests_missing: tuple[str, ...]
    last_results: Mapping[str, str]
    stubs: tuple[str, ...]
    why: str


def load_capabilities(path: Path = CAPABILITIES_FILE) -> list[CapabilitySpec]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [CapabilitySpec(c["id"], c["name"], tuple(c.get("modules", ())), tuple(c.get("tests", ())), int(c.get("floor", 0)))
            for c in data.get("capabilities", [])]


@dataclasses.dataclass(frozen=True)
class TestEvidence:
    """The latest recorded result for one test file, valid only for the source hash it ran against."""
    test_file: str
    outcome: str                    # PASS / FAIL / ERROR / EMPTY
    source_digest: str              # digest of (test file + the modules it reaches) when it ran
    where: str                      # evidence path (junit xml, log, or ledger id)


def compute_capability(spec: CapabilitySpec, comps: Mapping[str, Component], test_ev: Mapping[str, TestEvidence],
                       current_digest: Mapping[str, str]) -> CapabilityState:
    present = tuple(m for m in spec.modules if m in comps)
    missing = tuple(m for m in spec.modules if m not in comps)
    meaningful = sum(comps[m].meaningful for m in present)
    stubs = tuple(f"{m}:{s}" for m in present for s in comps[m].stubs)
    tests_present = tuple(t for t in spec.tests if t in comps)
    tests_missing = tuple(t for t in spec.tests if t not in comps)
    results: dict[str, str] = {}
    for t in tests_present:
        ev = test_ev.get(t)
        if ev is None:
            results[t] = "NOT_RUN"
        elif ev.source_digest != current_digest.get(t, ""):
            results[t] = "STALE"                                        # ran against different source: not evidence any more
        else:
            results[t] = ev.outcome
    below = spec.floor > 0 and meaningful < spec.floor
    if not present:
        state, unc, why = "NOT_STARTED", "KNOWN", "no implementing module exists"
    elif not tests_present:
        state, unc, why = "IMPLEMENTED", "UNTESTED", "code exists but no test file exists"
    elif any(v in ("FAIL", "ERROR") for v in results.values()):
        state, unc, why = "FAILED", "FAILED", "a test file failed on the current source"
    elif results and all(v == "PASS" for v in results.values()) and not missing and not tests_missing:
        state, unc, why = "TESTED", ("LIKELY" if not below and not stubs else "UNCERTAIN"), "every test file passed on the current source"
    else:
        state, unc, why = "IMPLEMENTED", "UNTESTED", "tests not run (or stale) on the current source"
    if below:
        why += f"; depth {meaningful} < floor {spec.floor}"
    if stubs:
        why += f"; {len(stubs)} stub(s)"
    return CapabilityState(spec.id, spec.name, state, unc, meaningful, spec.floor, below, present, missing, tests_present,
                           tests_missing, results, stubs, why)


# ------------------------------------------------------------------------------------------------ the model

@dataclasses.dataclass(frozen=True)
class Limitation:
    kind: str                       # BELOW_FLOOR / STUB / UNTESTED_MODULE / FAILING_TESTS / UNPARSABLE / UNREACHED / TODO
    subject: str
    detail: str


@dataclasses.dataclass(frozen=True)
class SelfModel:
    root: str
    scope: tuple[str, ...]
    components: Mapping[str, Component]
    dependencies: Mapping[str, tuple[str, ...]]
    dependents: Mapping[str, tuple[str, ...]]
    tests_of: Mapping[str, tuple[str, ...]]
    capabilities: tuple[CapabilityState, ...]
    limitations: tuple[Limitation, ...]
    resources: Mapping[str, Any]
    versions: Mapping[str, Any]
    active: Mapping[str, Any]

    def digest(self) -> str:
        """Deterministic digest of the evidence-derived content (resources and timestamps excluded)."""
        body = {"components": {k: v.sha256 for k, v in sorted(self.components.items())},
                "capabilities": [(c.id, c.state, c.meaningful) for c in self.capabilities],
                "limitations": [(x.kind, x.subject) for x in self.limitations]}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    def capability(self, cid: str) -> CapabilityState:
        for c in self.capabilities:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "scope": list(self.scope), "digest": self.digest(),
                "components": {k: dataclasses.asdict(v) for k, v in sorted(self.components.items())},
                "dependencies": {k: list(v) for k, v in sorted(self.dependencies.items())},
                "tests_of": {k: list(v) for k, v in sorted(self.tests_of.items())},
                "capabilities": [dataclasses.asdict(c) for c in self.capabilities],
                "limitations": [dataclasses.asdict(x) for x in self.limitations],
                "resources": dict(self.resources), "versions": dict(self.versions), "active": dict(self.active)}


def resources() -> dict[str, Any]:
    out: dict[str, Any] = {"cpus": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform()}
    try:
        import psutil
        out["free_ram_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
    except ImportError:
        out["free_ram_gb"] = None
    try:
        out["free_disk_gb"] = round(shutil.disk_usage(REPO_ROOT).free / 1e9, 1)
    except OSError:
        out["free_disk_gb"] = None
    return out


def versions(root: Path) -> dict[str, Any]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    from creator.ledger import tree_hash
    return {"git_head": run("rev-parse", "HEAD") or "unknown", "dirty_files": len(run("status", "--porcelain").splitlines()),
            "creator_tree": tree_hash(root / "creator") if (root / "creator").is_dir() else "absent",
            "engine_tree": tree_hash(root / "engine") if (root / "engine").is_dir() else "absent"}


def reach_digest(graph: T.ImportGraph, test_file: str, comps: Mapping[str, Component]) -> str:
    """Digest of a test file plus every module it reaches (transitively) inside the tree: a recorded result is valid only while
    this digest is unchanged."""
    mod_of_path = {p: m for m, p in graph.modules.items()}
    start = mod_of_path.get(test_file)
    seen: set[str] = set()
    frontier = [start] if start else []
    while frontier:
        nxt: list[str] = []
        for m in frontier:
            if m in seen or m not in graph.modules:
                continue
            seen.add(m)
            for imp in graph.imports.get(m, set()):
                for cand in (imp, imp.rsplit(".", 1)[0]):
                    if cand in graph.modules and cand not in seen:
                        nxt.append(cand)
        frontier = nxt
    files = sorted({graph.modules[m] for m in seen} | {test_file})
    root = Path(graph.root)
    h = hashlib.sha256()
    for f in files:                       # hashed from disk, so the digest does not depend on which folders were scanned (30 Sep)
        h.update(f.encode())
        fp = root / f
        h.update(fp.read_bytes().replace(b"\r\n", b"\n") if fp.is_file() else b"<missing>")
    return h.hexdigest()[:16]


def build(root: str | Path = REPO_ROOT, scope: Sequence[str] = ("creator", "tests"),
          capabilities: Optional[Sequence[CapabilitySpec]] = None, test_evidence: Mapping[str, TestEvidence] = (),  # type: ignore[assignment]
          ledger: Any = None, include_versions: bool = True) -> SelfModel:
    """Scan `scope` under `root` into a SelfModel. Pure reading: nothing is written."""
    rootp = Path(root).resolve()
    files = python_files(rootp, scope)
    comps = {rel: scan_component(rootp, rel) for rel in files}
    graph = T.ImportGraph.build(rootp)
    mod_to_path = dict(graph.modules)
    deps: dict[str, tuple[str, ...]] = {}
    rdeps: dict[str, set[str]] = {}
    for m, path in mod_to_path.items():
        if path not in comps:
            continue
        targets = sorted({mod_to_path[i] for i in graph.imports.get(m, set()) if i in mod_to_path and mod_to_path[i] in comps}
                         | {mod_to_path[i.rsplit(".", 1)[0]] for i in graph.imports.get(m, set())
                            if "." in i and i.rsplit(".", 1)[0] in mod_to_path and mod_to_path[i.rsplit(".", 1)[0]] in comps})
        targets = [t for t in targets if t != path]
        deps[path] = tuple(targets)
        for t in targets:
            rdeps.setdefault(t, set()).add(path)
    tests_of: dict[str, set[str]] = {}
    for path in comps:
        if T.is_test_file(path):
            for d in deps.get(path, ()):
                if not T.is_test_file(d):
                    tests_of.setdefault(d, set()).add(path)
    current_digest = {p: reach_digest(graph, p, comps) for p in comps if T.is_test_file(p)}
    specs = list(capabilities) if capabilities is not None else load_capabilities()
    tev = dict(test_evidence or {})
    caps = tuple(compute_capability(s, comps, tev, current_digest) for s in specs)
    lims: list[Limitation] = []
    for c in caps:
        if c.below_floor:
            lims.append(Limitation("BELOW_FLOOR", c.id, f"{c.meaningful} meaningful lines < floor {c.floor}"))
        if c.state == "FAILED":
            lims.append(Limitation("FAILING_TESTS", c.id, ", ".join(k for k, v in c.last_results.items() if v in ("FAIL", "ERROR"))))
    for path, comp in sorted(comps.items()):
        if comp.parse_error:
            lims.append(Limitation("UNPARSABLE", path, comp.parse_error))
        for s in comp.stubs:
            lims.append(Limitation("STUB", path, s))
        if not T.is_test_file(path) and comp.interfaces and path not in tests_of and not path.endswith("__init__.py"):
            lims.append(Limitation("UNTESTED_MODULE", path, "no test file imports it"))
        if not T.is_test_file(path) and not rdeps.get(path) and path not in tests_of and not path.endswith("__init__.py") \
                and comp.interfaces:
            lims.append(Limitation("UNREACHED", path, "nothing in scope imports it"))
    active: dict[str, Any] = {}
    if ledger is not None:
        try:
            snap = ledger.snapshot()
            active["open_ledger_items"] = len(snap.get("open", []))
            active["ledger_head"] = snap.get("head")
        except Exception as e:                                          # noqa: BLE001 - recorded, not hidden
            active["ledger_error"] = str(e)
    return SelfModel(str(rootp), tuple(scope), comps, deps, {k: tuple(sorted(v)) for k, v in rdeps.items()},
                     {k: tuple(sorted(v)) for k, v in tests_of.items()}, caps, tuple(lims), resources(),
                     versions(rootp) if include_versions else {}, active)


# ------------------------------------------------------------------------------------------------ self-diagnosis

@dataclasses.dataclass(frozen=True)
class Contradiction:
    capability: str
    claimed: str
    evidenced: str
    source: str
    why: str


ORDER = {"NOT_STARTED": 0, "IN_PROGRESS": 0, "IMPLEMENTED": 1, "FAILED": 1, "TESTED": 2, "INTENDED_BEHAVIOR_VERIFIED": 3,
         "VALIDATED": 4}


def diagnose(model: SelfModel, claims: Mapping[str, tuple[str, str]]) -> list[Contradiction]:
    """claims: capability id -> (claimed state, where the claim comes from). A claim that is AHEAD of the evidence (says TESTED or
    VALIDATED while the evidence says IMPLEMENTED / FAILED / NOT_STARTED) is a contradiction: the evidence wins (C77 sec 42).
    A claim BEHIND the evidence is reported too (stale bookkeeping), as a weaker finding."""
    out: list[Contradiction] = []
    by_id = {c.id: c for c in model.capabilities}
    for cid, (claimed, source) in sorted(claims.items()):
        c = by_id.get(cid)
        if c is None:
            out.append(Contradiction(cid, claimed, "UNKNOWN", source, "claimed capability is not declared or not scanned"))
            continue
        ev = c.state
        if c.state == "FAILED" and claimed in ("TESTED", "INTENDED_BEHAVIOR_VERIFIED", "VALIDATED"):
            out.append(Contradiction(cid, claimed, ev, source, f"tests fail on the current source: {c.why}"))
        elif ORDER.get(claimed, 0) > ORDER.get(ev, 0) and claimed != "VALIDATED":
            out.append(Contradiction(cid, claimed, ev, source, f"claim ahead of evidence: {c.why}"))
        elif claimed == "VALIDATED" and ev != "TESTED":
            out.append(Contradiction(cid, claimed, ev, source, f"VALIDATED but not even TESTED on the current source: {c.why}"))
        elif ORDER.get(claimed, 0) < ORDER.get(ev, 0) and claimed not in ("FAILED",):
            out.append(Contradiction(cid, claimed, ev, source, "claim behind the evidence (stale bookkeeping)"))
    return out


def save(model: SelfModel, path: Path) -> str:
    """Write the model as JSON (atomic) and return its digest; the file is evidence, the digest names it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_dict(), indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return model.digest()


# ------------------------------------------------------------------------------------------------ test evidence (sec 25)

def load_test_evidence(path: Path) -> dict[str, TestEvidence]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: TestEvidence(**v) for k, v in raw.items()}


def collect_test_evidence(root: str | Path, test_files: Sequence[str], store: Path, timeout: float = 600.0,
                          junit_dir: Optional[Path] = None) -> dict[str, TestEvidence]:
    """Run each test file once, record its outcome against the digest of (the file + every module it reaches), and merge into
    `store`. A later edit to any reached module changes the digest, so the old result reads STALE - it is never reused as
    evidence for different code."""
    rootp = Path(root).resolve()
    graph = T.ImportGraph.build(rootp)
    comps = {p: scan_component(rootp, p) for p in python_files(rootp, ("creator", "tests", "engine", "scripts"))}
    jdir = junit_dir or (store.parent / "junit")
    jdir.mkdir(parents=True, exist_ok=True)
    known = load_test_evidence(store)
    for tf in test_files:
        jp = jdir / (tf.replace("/", "__") + ".xml")
        run = T.run_pytest(rootp, [tf], jp, label="selfmodel", config=T.PytestConfig(timeout=timeout))
        status = run.status.value if hasattr(run.status, "value") else str(run.status)
        outcome = {"PASSED": "PASS", "FAILED": "FAIL", "NO_TESTS": "EMPTY"}.get(status, "ERROR")
        known[tf] = TestEvidence(tf, outcome, reach_digest(graph, tf, comps), str(jp))
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: dataclasses.asdict(v) for k, v in sorted(known.items())}, indent=1), encoding="utf-8")
    os.replace(tmp, store)
    return known
