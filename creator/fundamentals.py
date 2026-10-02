"""Creator: burned-in software-engineering FUNDAMENTALS (owner, 2 Oct 2026: "theres a lot of fundamentals we need to burn into the
system and we shouldnt leave it to figure them out on its own ... speed up time by letting it know so we dont have to wait until
millions of trial and error").

A curated, versioned CATALOGUE of principles Nupen applies by default. Each principle carries: id, name, statement, one-line why,
where it applies in the loop (WHERE: planner package text / student candidate filter / kernel evaluation guard / audit / test
generation), the package kinds it matters for, and - where feasible - a COMPUTED CHECK on a candidate change: deterministic,
AST/diff based, cheap, no model, no network.

ADVISORY: the kernel records a 'fundamentals' report per candidate (cycle.json details, cycles/<pkg>/fundamentals.json) and never
rejects on it. Rules that already block elsewhere (test weakening, coverage) are referenced, not duplicated. The constraint loop
ranks the violation rate as the 'fundamentals_violations' constraint (creator.constraints). principles_for(kind) is the short
checklist text for a package kind (for planner package text)."""
from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

CATALOGUE_VERSION = "1.0"

# where a principle applies in Nupen's loop
PLANNER, STUDENT, GUARD, AUDIT, TESTGEN = "planner", "student", "guard", "audit", "testgen"
ALL_KINDS = ("shrink", "efficiency", "coverage", "feature", "repair", "research")

DIFF_CAP = {"shrink": 400, "efficiency": 300, "coverage": 700, "feature": 500, "repair": 200, "research": 500}   # changed lines
COMPLEXITY_CAP = 15                  # cyclomatic complexity of a changed function
FUNCTION_LINES_CAP = 60
NESTING_CAP = 4
PARAMS_CAP = 8
DUP_MIN_STATEMENTS = 3
NAME_CORPUS_MAX = 800                # files read for the cross-reference corpus
INJECT_PARAMS = {"now", "clock", "rng", "seed", "random_state", "t0", "timer", "sleep", "time_fn", "now_fn", "rand"}
NONDETERMINISTIC = {"time.time", "time.monotonic", "time.perf_counter", "datetime.now", "datetime.datetime.now", "datetime.utcnow",
                    "datetime.datetime.utcnow", "datetime.today", "uuid.uuid4", "os.urandom", "secrets.token_hex"}
CLAIM_WORDS = re.compile(r"\b(faster|speed[- ]?up|speeds up|optimi[sz]e[sd]?|more efficient|performance gain|cheaper)\b", re.I)
SECRET_RES = (
    ("absolute personal path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/")),
    ("api key", re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{30,})")),
    ("hard-coded credential", re.compile(r"(?i)\b(password|passwd|secret|api_?key|token)\w*\s*=\s*[\"'][^\"'\s]{8,}[\"']")),
)


@dataclasses.dataclass(frozen=True)
class Change:
    path: str
    old: Optional[str]                  # None = new file
    new: Optional[str]                  # None = deleted file


@dataclasses.dataclass(frozen=True)
class Violation:
    principle: str
    path: str
    subject: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


Check = Callable[["Context"], list[Violation]]


@dataclasses.dataclass
class Context:
    changes: Sequence[Change]
    kind: str = "feature"
    corpus: Mapping[str, str] = dataclasses.field(default_factory=dict)      # path -> source of the rest of the tree
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class Principle:
    id: str
    name: str
    statement: str
    why: str
    where: tuple[str, ...]
    kinds: tuple[str, ...] = ALL_KINDS
    check: Optional[Check] = None
    no_check_reason: str = ""           # required when there is no computed check
    enforced_elsewhere: str = ""        # where a blocking rule already exists

    def checklist_line(self) -> str:
        return f"- {self.name}: {self.statement}"


# ------------------------------------------------------------------------------------------------ ast helpers
def _parse(src: Optional[str]) -> Optional[ast.Module]:
    if src is None:
        return None
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError):
        return None


def _py(path: str) -> bool:
    return path.endswith(".py")


def _is_test(path: str) -> bool:
    p = path.replace("\\", "/")
    return p.startswith("tests/") or "/tests/" in p or Path(p).name.startswith("test_") or Path(p).name == "conftest.py"


def functions(tree: Optional[ast.Module]) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """qualname -> function node (methods as Class.name, nested as outer.inner)."""
    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{ch.name}"
                out[q] = ch
                walk(ch, q + ".")
            elif isinstance(ch, ast.ClassDef):
                walk(ch, f"{prefix}{ch.name}.")
            else:
                walk(ch, prefix)
    if tree is not None:
        walk(tree, "")
    return out


def _body_dump(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = fn.body
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        body = body[1:]
    return "|".join(ast.dump(s, annotate_fields=False, include_attributes=False) for s in body)


def changed_functions(c: Change) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Functions that are new or whose body/signature differs from the old version."""
    new_f, old_f = functions(_parse(c.new)), functions(_parse(c.old))
    out = {}
    for q, fn in new_f.items():
        o = old_f.get(q)
        if o is None or ast.dump(o) != ast.dump(fn):
            out[q] = fn
    return out


def complexity(fn: ast.AST) -> int:
    """McCabe cyclomatic complexity: 1 + decision points (if/for/while/except/with-less, boolean operators, ternaries, comprehension
    clauses, match cases)."""
    n = 1
    for x in ast.walk(fn):
        if isinstance(x, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.Assert)):
            n += 1
        elif isinstance(x, ast.BoolOp):
            n += len(x.values) - 1
        elif isinstance(x, ast.comprehension):
            n += 1 + len(x.ifs)
        elif hasattr(ast, "match_case") and isinstance(x, ast.match_case):
            n += 1
    return n


def nesting(fn: ast.AST) -> int:
    def depth(node: ast.AST, d: int) -> int:
        best = d
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                best = max(best, depth(ch, d + 1))
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            else:
                best = max(best, depth(ch, d))
        return best
    return depth(fn, 0)


def _length(fn: ast.AST) -> int:
    return int(getattr(fn, "end_lineno", 0) or 0) - int(getattr(fn, "lineno", 0) or 0) + 1


def added_lines(c: Change) -> list[str]:
    old = (c.old or "").splitlines()
    new = (c.new or "").splitlines()
    return [ln[2:] for ln in difflib.ndiff(old, new) if ln.startswith("+ ")]


def changed_line_count(c: Change) -> int:
    old = (c.old or "").splitlines()
    new = (c.new or "").splitlines()
    return sum(1 for ln in difflib.ndiff(old, new) if ln[:2] in ("+ ", "- "))


def _public_names(tree: Optional[ast.Module]) -> set[str]:
    out: set[str] = set()
    if tree is None:
        return out
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out.update(t.id for t in n.targets if isinstance(t, ast.Name) and not t.id.startswith("_") and not t.id.isupper())
    return out


def _referenced(sources: Sequence[str]) -> set[str]:
    refs: set[str] = set()
    for s in sources:
        t = _parse(s)
        if t is None:
            continue
        for x in ast.walk(t):
            if isinstance(x, ast.Name):
                refs.add(x.id)
            elif isinstance(x, ast.Attribute):
                refs.add(x.attr)
            elif isinstance(x, ast.alias):
                refs.add(x.name.split(".")[-1])
            elif isinstance(x, ast.Constant) and isinstance(x.value, str) and len(x.value) < 80:
                refs.add(x.value)                                       # __all__, getattr("name"), registries
    return refs


# ------------------------------------------------------------------------------------------------ the computed checks
def check_small_change(ctx: Context) -> list[Violation]:
    cap = DIFF_CAP.get(ctx.kind, DIFF_CAP["feature"])
    total = sum(changed_line_count(c) for c in ctx.changes)
    if total > cap:
        return [Violation("F01", "*", f"{total} changed lines", f"a {ctx.kind} package should change <= {cap} lines; split it")]
    return []


def check_complexity(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path):
            continue
        old_f = functions(_parse(c.old))
        for q, fn in changed_functions(c).items():
            cx = complexity(fn)
            prior = complexity(old_f[q]) if q in old_f else 0
            if cx > COMPLEXITY_CAP and cx > prior:
                out.append(Violation("F02", c.path, q, f"cyclomatic complexity {cx} (was {prior}) exceeds {COMPLEXITY_CAP}: split into functions"))
            n = len(fn.args.args) + len(fn.args.kwonlyargs)
            if n > PARAMS_CAP and q not in old_f:
                out.append(Violation("F02", c.path, q, f"{n} parameters (> {PARAMS_CAP}): group them or split the responsibility"))
    return out


def check_dry(ctx: Context) -> list[Violation]:
    """A new function whose body duplicates another function (any file in the tree or another new function) is flagged."""
    seen: dict[str, str] = {}
    changed_paths = {c.path for c in ctx.changes}
    for p, src in ctx.corpus.items():
        if p in changed_paths or _is_test(p) or not _py(p):
            continue
        for q, fn in functions(_parse(src)).items():
            if len(fn.body) >= DUP_MIN_STATEMENTS:
                seen.setdefault(_body_dump(fn), f"{p}:{q}")
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path):
            continue
        old_names = set(functions(_parse(c.old)))
        for q, fn in functions(_parse(c.new)).items():
            if q in old_names or len(fn.body) < DUP_MIN_STATEMENTS:
                continue
            d = _body_dump(fn)
            if d in seen and seen[d] != f"{c.path}:{q}":
                out.append(Violation("F03", c.path, q, f"body duplicates {seen[d]}: reuse it"))
            seen.setdefault(d, f"{c.path}:{q}")
    return out


def check_dead_public(ctx: Context) -> list[Violation]:
    """New public top-level names that nothing in the tree (and no test) references."""
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path) or Path(c.path).name in ("__init__.py", "conftest.py"):
            continue
        fresh = _public_names(_parse(c.new)) - _public_names(_parse(c.old))
        if not fresh:
            continue
        others = [s for p, s in ctx.corpus.items() if p != c.path] + [x.new or "" for x in ctx.changes if x.path != c.path]
        refs = _referenced(others)
        for name in sorted(fresh):
            if name == "main" or name.startswith("test") or name in refs or _count_uses(c.new or "", name) > 0:
                continue
            out.append(Violation("F04", c.path, name, "new public name nothing references (dead code / speculative): remove it or use it"))
    return out


def _count_uses(src: str, name: str) -> int:
    t = _parse(src)
    if t is None:
        return 0
    n = 0
    for x in ast.walk(t):
        if (isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Load)) or (isinstance(x, ast.Attribute) and x.attr == name):
            n += 1
    return n


def check_test_present(ctx: Context) -> list[Violation]:
    src = [c for c in ctx.changes if _py(c.path) and not _is_test(c.path) and c.new is not None and changed_functions(c)]
    if src and not any(_is_test(c.path) for c in ctx.changes):
        return [Violation("F05", src[0].path, "*", "behaviour changed in source but no test changed: add a test that fails without it "
                          "(coverage is enforced by the kernel claim; this is the early warning)")]
    return []


def check_test_weakening(ctx: Context) -> list[Violation]:
    """Reference to the blocking audit rule (creator.audit.checks.check_test_weakening); re-run here only so the report is whole."""
    try:
        from creator.audit import checks as AUD
        before = {c.path: c.old for c in ctx.changes if _is_test(c.path) and c.old is not None}
        after = {c.path: c.new for c in ctx.changes if _is_test(c.path) and c.new is not None}
        return [Violation("F06", str(getattr(f, "subject", "")), str(getattr(f, "check", "")), str(getattr(f, "detail", ""))[:300])
                for f in AUD.check_test_weakening(before, after)]
    except Exception:                                                    # noqa: BLE001 - an advisory check never raises
        return []


def _swallowers(tree: Optional[ast.Module]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if tree is None:
        return out
    for q, fn in {"<module>": tree, **functions(tree)}.items():
        for x in ast.walk(fn):
            if isinstance(x, ast.ExceptHandler):
                broad = x.type is None or (isinstance(x.type, ast.Name) and x.type.id in ("Exception", "BaseException"))
                silent = all(isinstance(s, (ast.Pass, ast.Continue)) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                             for s in x.body)
                if x.type is None or (broad and silent):
                    out.append((q, x.lineno))
    return out


def check_swallowed(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path):
            continue
        before = len(_swallowers(_parse(c.old)))
        now = _swallowers(_parse(c.new))
        if len(now) > before:
            q, ln = now[-1]
            out.append(Violation("F07", c.path, q, f"{len(now) - before} new bare/silent broad except (line {ln}): record or re-raise the error"))
    return out


def _nondet_calls(tree: Optional[ast.Module]) -> dict[str, int]:
    out: dict[str, int] = {}
    for q, fn in functions(tree).items():
        params = {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs)}
        if params & INJECT_PARAMS:
            continue
        for x in ast.walk(fn):
            if isinstance(x, ast.Call):
                name = _dotted(x.func)
                if name in NONDETERMINISTIC or (name.startswith("random.") and name != "random.Random") or name.startswith("np.random.") \
                        or name.startswith("numpy.random."):
                    out[f"{q}:{name}"] = out.get(f"{q}:{name}", 0) + 1
    return out


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check_determinism(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path):
            continue
        old, new = _nondet_calls(_parse(c.old)), _nondet_calls(_parse(c.new))
        for key, n in sorted(new.items()):
            if n > old.get(key, 0):
                q, call = key.split(":", 1)
                out.append(Violation("F08", c.path, q, f"new {call}() with no injected clock/rng/seed parameter: logic becomes untestable and irreproducible"))
    return out


def check_secrets(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    for c in ctx.changes:
        if _is_test(c.path) and not _py(c.path):
            continue
        for ln in added_lines(c):
            for what, rx in SECRET_RES:
                if rx.search(ln):
                    out.append(Violation("F09", c.path, what, f"added line: {ln.strip()[:80]}"))
                    break
    return out


def check_unmeasured_claim(ctx: Context) -> list[Violation]:
    """Efficiency wording in added comments/docstrings (or the worker's notes) outside an efficiency package: unmeasured claim. An
    efficiency package's claim is measured by the kernel (creator.kernel.efficiency_claim)."""
    if ctx.kind == "efficiency":
        return []
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path):
            continue
        for ln in added_lines(c):
            s = ln.strip()
            if (s.startswith("#") or s.startswith('"""') or s.startswith("'''")) and CLAIM_WORDS.search(s):
                out.append(Violation("F10", c.path, "claim", f"efficiency claim without a measurement: {s[:80]}"))
                break
    if CLAIM_WORDS.search(ctx.notes or "") and not out:
        out.append(Violation("F10", "*", "notes", "the worker claims an efficiency gain in a package that is not an efficiency package"))
    return out


def check_readability(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path):
            continue
        old_f = functions(_parse(c.old))
        for q, fn in changed_functions(c).items():
            ln, dp = _length(fn), nesting(fn)
            po, do = (_length(old_f[q]), nesting(old_f[q])) if q in old_f else (0, 0)
            if ln > FUNCTION_LINES_CAP and ln > po:
                out.append(Violation("F11", c.path, q, f"{ln} lines (cap {FUNCTION_LINES_CAP}, was {po}): extract steps with names"))
            if dp > NESTING_CAP and dp > do:
                out.append(Violation("F11", c.path, q, f"nesting depth {dp} (cap {NESTING_CAP}, was {do}): use early returns/helpers"))
    return out


def check_interface_stability(ctx: Context) -> list[Violation]:
    """A removed/renamed public top-level name that another file in the tree still uses."""
    out: list[Violation] = []
    changed = {c.path: c for c in ctx.changes}
    for c in ctx.changes:
        if not _py(c.path) or _is_test(c.path) or c.old is None:
            continue
        gone = _public_names(_parse(c.old)) - (_public_names(_parse(c.new)) if c.new is not None else set())
        if not gone:
            continue
        other = [s for p, s in ctx.corpus.items() if p != c.path and p not in changed] + [x.new or "" for x in ctx.changes if x.path != c.path]
        refs = _referenced(other)
        for name in sorted(gone):
            if name in refs:
                out.append(Violation("F12", c.path, name, "public name removed but still referenced elsewhere: keep it or update every caller"))
    return out


# ------------------------------------------------------------------------------------------------ the catalogue
CATALOGUE: tuple[Principle, ...] = (
    Principle("F01", "Small, reversible changes", "Keep each change small enough to review and revert as one unit; split larger work.",
              "Small diffs are easy to verify and cheap to roll back, so more of them survive evaluation.",
              (PLANNER, STUDENT, GUARD), check=check_small_change),
    Principle("F02", "One responsibility per function/module", "A function does one thing; complexity must not grow past the cap without justification.",
              "Low complexity is testable and changeable; complex functions hide the defects the loop pays to find.",
              (PLANNER, STUDENT, GUARD), check=check_complexity),
    Principle("F03", "DRY", "Reuse an existing function instead of adding a copy of its body.",
              "Duplicates drift apart: one copy gets fixed, the other keeps the bug.",
              (STUDENT, GUARD), ("shrink", "feature", "repair"), check=check_dry),
    Principle("F04", "YAGNI / no dead code", "Do not add public names nothing uses.",
              "Unused code is cost with no benefit, and it is the first thing a shrink package must delete again.",
              (STUDENT, GUARD), check=check_dead_public),
    Principle("F05", "Every behaviour change has a test that fails without it", "Change behaviour only together with a test that would fail on the old code.",
              "An unproven change cannot be told from a harmless one; the test is the proof.",
              (PLANNER, TESTGEN, GUARD), check=check_test_present,
              enforced_elsewhere="creator.kernel gap_closure_claim / coverage claim (blocking); this check is the early warning"),
    Principle("F06", "Never weaken a test", "Do not delete tests, loosen assertions or lower thresholds to make a change pass.",
              "A weakened test turns the evidence into noise; every later decision rests on it.",
              (GUARD, AUDIT), check=check_test_weakening, enforced_elsewhere="creator.audit.checks.check_test_weakening (blocking in the kernel)"),
    Principle("F07", "Errors are not swallowed silently", "No bare except and no 'except Exception: pass'; record, narrow or re-raise.",
              "A swallowed error is a failure that never reaches the diagnosis loop.",
              (STUDENT, GUARD), check=check_swallowed),
    Principle("F08", "Determinism", "Time, randomness and ids enter logic through an injected clock/rng/seed parameter.",
              "Reproducibility is what lets a claim be re-derived; hidden clocks make tests flaky and replays impossible.",
              (STUDENT, GUARD, TESTGEN), ("feature", "repair", "research"), check=check_determinism),
    Principle("F09", "No secrets or personal absolute paths", "Never write keys, passwords or machine-specific absolute paths into code.",
              "They leak, and they break on every other machine.",
              (STUDENT, GUARD, AUDIT), check=check_secrets),
    Principle("F10", "Measure before optimising", "An efficiency claim needs a measurement of base and candidate; do not assert speed-ups in comments.",
              "Unmeasured optimisation adds complexity for a gain that may not exist.",
              (PLANNER, GUARD), check=check_unmeasured_claim, enforced_elsewhere="creator.kernel.efficiency_claim measures efficiency packages"),
    Principle("F11", "Naming and readability", "Short functions, shallow nesting, early returns.",
              "Code is read far more often than written; the next worker is a model with a small context window.",
              (STUDENT, GUARD), check=check_readability),
    Principle("F12", "Public interface stability", "Do not remove or rename a public name that callers still use.",
              "Breaking callers turns a local change into a repository-wide regression.",
              (STUDENT, GUARD), ("shrink", "feature", "repair", "efficiency"), check=check_interface_stability),
    Principle("F13", "Fail loudly at boundaries, trust inside", "Validate input where it enters the system; internal functions assume validated data.",
              "Validation scattered everywhere is noise; validation nowhere lets bad data travel far from its cause.",
              (PLANNER, STUDENT), ("feature", "repair"),
              no_check_reason="needs semantic knowledge of what is a boundary; not decidable from an AST diff"),
    Principle("F14", "Reproduce before fixing", "Write the failing test or command that shows the defect, then fix it.",
              "A fix without a reproduction is a guess that cannot be verified.",
              (PLANNER, TESTGEN), ("repair",),
              no_check_reason="ordering of actions inside the worker's session is not visible in the final diff; F05 checks the outcome"),
    Principle("F15", "Change one thing at a time", "A package changes one concern; unrelated cleanups go in their own package.",
              "A mixed change cannot be attributed: when it fails nobody knows which part was wrong.",
              (PLANNER, STUDENT), no_check_reason="'one concern' is semantic; F01 bounds the size as a proxy"),
)

BY_ID = {p.id: p for p in CATALOGUE}


def catalogue() -> list[dict[str, Any]]:
    return [{"id": p.id, "name": p.name, "statement": p.statement, "why": p.why, "where": list(p.where), "kinds": list(p.kinds),
             "computed": p.check is not None, "reason": p.no_check_reason, "enforced_elsewhere": p.enforced_elsewhere} for p in CATALOGUE]


def norm_kind(task_kind: str) -> str:
    k = (task_kind or "").lower()
    for name in ALL_KINDS:
        if name in k:
            return name
    return {"gap": "feature", "capability": "feature", "fix": "repair", "failure": "repair", "test": "coverage"}.get(k, "feature")


def principles_for(task_kind: str) -> str:
    """The short checklist text for a package kind (planner package text)."""
    kind = norm_kind(task_kind)
    lines = [f"Engineering fundamentals for this {kind} package (catalogue v{CATALOGUE_VERSION}; checked by the kernel, advisory):"]
    lines += [p.checklist_line() for p in CATALOGUE if kind in p.kinds]
    lines.append(f"Size bound: at most {DIFF_CAP.get(kind, DIFF_CAP['feature'])} changed lines; cyclomatic complexity <= {COMPLEXITY_CAP} per changed function.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ running
@dataclasses.dataclass
class Report:
    kind: str
    violations: list[Violation]
    checked: int = 0
    version: str = CATALOGUE_VERSION
    errors: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.principle] = out.get(v.principle, 0) + 1
        return out

    @property
    def total(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "kind": self.kind, "advisory": True, "files": self.checked, "total": self.total,
                "counts": self.counts, "violations": [v.to_dict() for v in self.violations[:50]], "errors": self.errors}


def run_checks(changes: Sequence[Change], kind: str = "feature", corpus: Optional[Mapping[str, str]] = None, notes: str = "") -> Report:
    ctx = Context(list(changes), norm_kind(kind), corpus or {}, notes)
    found: list[Violation] = []
    errors: dict[str, str] = {}
    for p in CATALOGUE:
        if p.check is None or ctx.kind not in p.kinds:
            continue
        try:
            found.extend(p.check(ctx))
        except Exception as e:                                          # noqa: BLE001 - advisory: a broken check is a note, never a failure
            errors[p.id] = f"{type(e).__name__}: {e}"[:200]
    return Report(ctx.kind, found, len(ctx.changes), errors=errors)


def _git_show(root: Path, base: str, path: str) -> Optional[str]:
    r = subprocess.run(["git", "show", f"{base}:{path}"], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None


def load_corpus(root: Path, skip: Sequence[str] = ()) -> dict[str, str]:
    out: dict[str, str] = {}
    for top in ("creator", "engine", "scripts", "tests"):
        for p in sorted((Path(root) / top).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if rel in skip or len(out) >= NAME_CORPUS_MAX:
                continue
            try:
                out[rel] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return out


def evaluate_candidate(root: Path, base: str, paths: Sequence[str], kind: str, notes: str = "") -> Report:
    """Fundamentals report for the candidate tree at `root` against commit `base` (python files only). Never raises."""
    try:
        root = Path(root)
        changes = []
        for p in paths:
            if not _py(p):
                continue
            f = root / p
            new = f.read_text(encoding="utf-8", errors="replace") if f.is_file() else None
            changes.append(Change(p, _git_show(root, base, p), new))
        return run_checks(changes, kind, load_corpus(root), notes)
    except Exception as e:                                               # noqa: BLE001
        return Report(norm_kind(kind), [], 0, errors={"evaluate": f"{type(e).__name__}: {e}"[:200]})


# ------------------------------------------------------------------------------------------------ status / constraint feed
def read_reports(state: Path) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    try:
        dirs = sorted(p for p in (Path(state) / "cycles").iterdir() if p.is_dir())
    except OSError:
        return out
    for d in dirs:
        f = d / "fundamentals.json"
        try:
            out.append((d.name, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def status_line(state: Path) -> str:
    """STATUS line: 'fundamentals: N violations over M candidates (F02:3 F07:1), advisory'."""
    reps = read_reports(state)
    counts: dict[str, int] = {}
    for _, r in reps:
        for k, v in (r.get("counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    detail = " ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"
    return f"fundamentals: {sum(counts.values())} violations over {len(reps)} candidates ({detail}), advisory"
