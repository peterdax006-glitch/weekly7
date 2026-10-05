"""P0.5 bug pinpoint: traceback parser, spectrum-based fault localization (Ochiai) and a planted-bug data generator.

Loaded on demand (light imports: stdlib only at import time; creator.testrun / creator.build are imported inside the generator).
  parse_traceback(text, root)   -> Traceback(exc_type, message, assert_line, frames innermost first)
  collect(root, tests)          -> per-test outcome + line coverage (sys.settrace plugin in a subprocess; coverage.py not needed)
  rank_lines(runs, ...)         -> [(file, line, score)] best first: Ochiai, boosted for diff lines and traceback frames
  generate(...)                 -> labelled JSONL rows of planted bugs; run it on a temp COPY of the tree, never on the worktree
CLI:  python -m creator.tools.pinpoint gen --root <tree> --target creator/x.py [--target ...] --out rows.jsonl --max-rows N
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import zip_longest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PLUGIN = "creator.tools.pinpoint_plugin"


# ------------------------------------------------------------------------------------------------------- traceback parser
@dataclass(frozen=True)
class Frame:
    file: str
    line: int
    func: str = ""
    is_test: bool = False


@dataclass(frozen=True)
class Traceback:
    exc_type: str = ""
    message: str = ""
    assert_line: str = ""                 # the failing `assert ...` source line, when pytest printed one
    frames: tuple[Frame, ...] = ()        # inside the repo only, innermost first

    def repo_frames(self, include_tests: bool = False) -> list[Frame]:
        return [f for f in self.frames if include_tests or not f.is_test]

    def to_dict(self) -> dict[str, Any]:
        return {"exc_type": self.exc_type, "message": self.message, "assert_line": self.assert_line,
                "frames": [[f.file, f.line, f.func] for f in self.frames]}


_PY_FRAME = re.compile(r'^\s*File "(?P<f>[^"]+)", line (?P<l>\d+)(?:, in (?P<fn>.+))?$')
_PT_FRAME = re.compile(r"^(?P<f>[^\s:][^:\n]*\.py):(?P<l>\d+): (?:in (?P<fn>\S.*)|(?P<exc>[A-Za-z_][\w.]*)(?::.*)?)?$")
_EXC_LINE = re.compile(r"^(?:E\s+)?(?P<t>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failed|Timeout|Stop\w*|Skipped|Fail))"
                       r"(?::\s?(?P<m>.*))?$")
_IS_TEST = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")


def _to_rel(path: str, root: str | Path | None) -> str | None:
    """Path relative to the repo root (posix), or None when the file is outside the repo or in a virtualenv/site-packages."""
    p = path.replace("\\", "/")
    if "site-packages" in p or "/.venv/" in p or "/lib/python" in p.lower() or "<" in p:
        return None
    if root is not None:
        r = str(root).replace("\\", "/").rstrip("/") + "/"
        if p.lower().startswith(r.lower()):
            return p[len(r):]
        if re.match(r"^([A-Za-z]:)?/", p):
            return None                                    # absolute and not under the root
    return p.lstrip("./") if p.startswith("./") else p


def parse_traceback(text: str, root: str | Path | None = None) -> Traceback:
    """Parse pytest failure text (any --tb style) or a plain Python traceback. Frames are returned innermost first."""
    frames: list[Frame] = []
    exc_type = message = assert_line = ""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        ln = raw.rstrip()
        m = _PY_FRAME.match(ln)
        mp = None if m else _PT_FRAME.match(ln)
        if m or mp:
            mm = m or mp
            rel = _to_rel(mm.group("f"), root)
            if rel is not None:
                func = (mm.group("fn") or "").strip()
                frames.append(Frame(rel, int(mm.group("l")), func, bool(_IS_TEST.search(rel))))
            if mp and mp.group("exc") and not exc_type:
                exc_type = mp.group("exc")
            continue
        s = ln.lstrip()
        if s.startswith(">") and "assert" in s and not assert_line:
            assert_line = s.lstrip("> ").strip()
        em = _EXC_LINE.match(s[1:].lstrip() if s.startswith("E ") else s)
        if em and ln.startswith(("E ", "E\t")) or (em and not ln.startswith((" ", ">"))):
            exc_type, message = em.group("t"), (em.group("m") or "").strip()      # last one wins: the final exception
        elif ln.startswith("E ") and not message and exc_type:
            message = ln[1:].strip()
        if ln.startswith("E ") and not em and not exc_type:
            body = ln[1:].strip()
            if body.startswith("assert "):
                exc_type, message = "AssertionError", body
    # plain python tracebacks list OUTERMOST first; pytest prints outermost first too. Innermost first = reverse.
    frames.reverse()
    dedup: list[Frame] = []
    for f in frames:
        if not dedup or dedup[-1] != f:
            dedup.append(f)
    if not exc_type and "assert" in (assert_line or ""):
        exc_type = "AssertionError"
    return Traceback(exc_type, message, assert_line, tuple(dedup))


# ----------------------------------------------------------------------------------------------------- coverage collection
def clean_child_env(root: str | Path, out_json: str | Path) -> dict[str, str]:
    from creator.build import clean_env
    return clean_env(root, {"PINPOINT_ROOT": str(root), "PINPOINT_OUT": str(out_json)})


def collect(root: str | Path, tests: Sequence[str], *, python: str = sys.executable, timeout: float = 300.0,
            extra_args: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Run `tests` (files or node ids) under the settrace plugin inside `root`; returns {nodeid: {outcome, lines, text, tail}}.
    Raises RuntimeError when pytest produced no result file (crash, timeout)."""
    if not tests:
        return {}
    fd, out = tempfile.mkstemp(prefix="pinpoint_", suffix=".json")
    os.close(fd)
    os.unlink(out)
    argv = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", PLUGIN, "--tb=short", "--rootdir", str(root),
            "-o", "addopts=", *extra_args, "--", *tests]
    try:
        subprocess.run(argv, cwd=str(root), env=clean_child_env(root, out), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("pytest timed out") from e
    try:
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"no coverage result: {e}") from e
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------- ranking
def parse_diff_lines(diff: str) -> dict[str, set[int]]:
    """Added/changed line numbers (new side) per file from a unified diff."""
    out: dict[str, set[int]] = {}
    cur = None
    new_ln = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            p = raw[4:].strip()
            cur = None if p == "/dev/null" else (p[2:] if p.startswith(("b/", "a/")) else p)
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            new_ln = int(m.group(1))
            continue
        if cur is None or raw.startswith(("---", "\\")):
            continue
        if raw.startswith("+"):
            out.setdefault(cur, set()).add(new_ln)
            new_ln += 1
        elif raw.startswith("-"):
            continue
        else:
            new_ln += 1
    return out


def _line_prior(root: Path, rel: str, cache: dict[str, dict[int, float]]) -> dict[int, float]:
    """Small structural prior per line: lines holding comparisons, arithmetic, returns, conditions are likelier bug sites than
    `def`/`class`/import/decorator lines (which every call executes). Only breaks ties between equal spectrum scores."""
    if rel in cache:
        return cache[rel]
    pri: dict[int, float] = {}
    try:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        cache[rel] = pri
        return pri
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if ln is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            pri[ln] = min(pri.get(ln, 0.0), -0.5)
            for d in getattr(node, "decorator_list", []):
                pri[d.lineno] = min(pri.get(d.lineno, 0.0), -0.5)
        elif isinstance(node, (ast.Compare, ast.BinOp, ast.BoolOp)):
            pri[ln] = max(pri.get(ln, 0.0), 1.0)
        elif isinstance(node, (ast.Return, ast.If, ast.While, ast.Assign, ast.AugAssign)):
            pri[ln] = max(pri.get(ln, 0.0), 0.6)
    cache[rel] = pri
    return pri


def ochiai(ef: int, ep: int, total_failed: int) -> float:
    d = math.sqrt(total_failed * (ef + ep))
    return ef / d if d else 0.0


def rank_lines(runs: Mapping[str, Mapping[str, Any]], *, root: str | Path | None = None, diff: str | Mapping[str, Iterable[int]] = "",
               traceback_text: str | None = None, diff_boost: float = 0.25, frame_boost: float = 0.2,
               top: int | None = None) -> list[tuple[str, int, float]]:
    """Ranked (file, line, score), best first. `runs` is collect()'s result. Score = Ochiai(line) + boosts for lines in the
    supplied diff and for traceback frames (innermost repo frame most) + small structural/recency tie-breakers (< 0.05 total)."""
    failing = [k for k, r in runs.items() if r["outcome"] == "failed"]
    passing = [k for k, r in runs.items() if r["outcome"] == "passed"]
    if not failing:
        return []
    ef: dict[tuple[str, int], int] = {}
    ep: dict[tuple[str, int], int] = {}
    for k in failing:
        for f, lns in runs[k]["lines"].items():
            for ln in lns:
                ef[(f, ln)] = ef.get((f, ln), 0) + 1
    for k in passing:
        for f, lns in runs[k]["lines"].items():
            for ln in lns:
                if (f, ln) in ef:
                    ep[(f, ln)] = ep.get((f, ln), 0) + 1
    dl = parse_diff_lines(diff) if isinstance(diff, str) else {k: set(v) for k, v in diff.items()}
    frames: dict[tuple[str, int], float] = {}
    text = traceback_text if traceback_text is not None else "\n".join(runs[k].get("text", "") for k in failing)
    for k in failing:
        tb = parse_traceback(runs[k].get("text", ""), root) if traceback_text is None else parse_traceback(text, root)
        repo = tb.repo_frames()
        for i, fr in enumerate(repo):
            w = frame_boost * (1.0 if i == 0 else 0.5)
            key = (fr.file, fr.line)
            frames[key] = max(frames.get(key, 0.0), w)
        if traceback_text is not None:
            break
    recency: dict[tuple[str, int], float] = {}
    for k in failing:
        tail = runs[k].get("tail") or []
        for i, (f, ln, *_d) in enumerate(tail):
            recency[(f, ln)] = max(recency.get((f, ln), 0.0), (i + 1) / max(1, len(tail)))
    rootp = Path(root) if root is not None else None
    pcache: dict[str, dict[int, float]] = {}
    nf = len(failing)
    out: list[tuple[str, int, float]] = []
    for (f, ln), e in ef.items():
        s = ochiai(e, ep.get((f, ln), 0), nf)
        if ln in dl.get(f, ()):
            s += diff_boost
        s += frames.get((f, ln), 0.0)
        tie = 0.01 * recency.get((f, ln), 0.0)
        if rootp is not None:
            tie += 0.02 * _line_prior(rootp, f, pcache).get(ln, 0.0)
        out.append((f, ln, round(s + tie, 6)))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out[:top] if top else out


def pinpoint(root: str | Path, tests: Sequence[str], *, diff: str = "", top: int = 20, **kw: Any) -> list[tuple[str, int, float]]:
    """Convenience: run `tests` traced in `root` and return the top-ranked suspicious lines."""
    return rank_lines(collect(root, tests, **kw), root=root, diff=diff, top=top)


# ------------------------------------------------------------------------------------------------------------- mutations
MUTATIONS = ("cmp_flip", "off_by_one", "swap_var", "drop_return", "negate_cond", "arith_flip")
_CMP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"), ast.Eq: ("==", "!="),
        ast.NotEq: ("!=", "==")}
_ARITH = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "/"), ast.Div: ("/", "*"), ast.FloorDiv: ("//", "*"),
          ast.Mod: ("%", "*")}


@dataclass(frozen=True)
class Mutant:
    kind: str
    line: int
    start: int            # absolute char offsets in the source
    end: int
    new_text: str


def _offsets(src: str) -> list[int]:
    out, pos = [0], 0
    for ln in src.splitlines(keepends=True):
        pos += len(ln)
        out.append(pos)
    return out


def _abs(src_b: bytes, offs_b: list[int], line: int, col: int) -> int:
    return offs_b[line - 1] + col                      # col_offset is a UTF-8 BYTE offset


def enumerate_mutants(src: str, covered: set[int] | None = None) -> list[Mutant]:
    """All single-site mutations inside function bodies whose line is in `covered` (None = any). Edits are in-place text
    replacements on one line, so line numbers never shift and `line` is the exact ground-truth bug line."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    b = src.encode("utf-8")
    offs = _offsets(src)
    boffs = [len(src[:o].encode("utf-8")) for o in offs]       # byte offsets of line starts

    def span(n: ast.AST) -> tuple[int, int] | None:
        if getattr(n, "end_lineno", None) != n.lineno:           # type: ignore[attr-defined]
            return None
        return _abs(b, boffs, n.lineno, n.col_offset), _abs(b, boffs, n.lineno, n.end_col_offset)   # type: ignore[attr-defined]

    def seg(a: ast.AST, c: ast.AST) -> tuple[int, int] | None:
        if a.end_lineno != c.lineno:                             # type: ignore[attr-defined]
            return None
        return _abs(b, boffs, a.end_lineno, a.end_col_offset), _abs(b, boffs, c.lineno, c.col_offset)   # type: ignore[attr-defined]

    def text(s: int, e: int) -> str:
        return b[s:e].decode("utf-8")

    out: list[Mutant] = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    seen: set[tuple[int, int, str]] = set()
    for fn in funcs:
        names = sorted({a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")}
                       | {t.id for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension))
                          for t in ast.walk(n.targets[0] if isinstance(n, ast.Assign) else n.target if hasattr(n, "target") else n)
                          if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)})
        for n in ast.walk(fn):
            ln = getattr(n, "lineno", None)
            if ln is None or (covered is not None and ln not in covered):
                continue

            def add(kind: str, s: int, e: int, new: str) -> None:
                if (s, e, new) in seen or text(s, e) == new:
                    return
                seen.add((s, e, new))
                out.append(Mutant(kind, ln, s, e, new))

            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
                sg = seg(n.left, n.comparators[0])
                if sg:
                    old, new = _CMP[type(n.ops[0])]
                    mid = text(*sg)
                    i = mid.find(old)
                    if i >= 0:
                        add("cmp_flip", sg[0] + len(mid[:i].encode()), sg[0] + len(mid[:i].encode()) + len(old.encode()), new)
            elif isinstance(n, ast.BinOp) and type(n.op) in _ARITH:
                sg = seg(n.left, n.right)
                if sg:
                    old, new = _ARITH[type(n.op)]
                    mid = text(*sg)
                    i = mid.find(old)
                    if i >= 0:
                        add("arith_flip", sg[0] + len(mid[:i].encode()), sg[0] + len(mid[:i].encode()) + len(old.encode()), new)
            elif isinstance(n, ast.Constant) and type(n.value) is int and abs(n.value) < 10 ** 6:
                sp = span(n)
                if sp and re.fullmatch(r"\d+", text(*sp)):
                    add("off_by_one", sp[0], sp[1], str(n.value + 1) if n.value % 2 == 0 else str(max(n.value - 1, 0)))
            elif isinstance(n, ast.Return) and n.value is not None and not (isinstance(n.value, ast.Constant) and n.value.value is None):
                sp = span(n.value)
                if sp:
                    add("drop_return", sp[0], sp[1], "None")
            elif isinstance(n, (ast.If, ast.While)):
                sp = span(n.test)
                if sp:
                    add("negate_cond", sp[0], sp[1], f"not ({text(*sp)})")
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and len(names) > 1 and n.id in names:
                sp = span(n)
                if sp:
                    others = [x for x in names if x != n.id]
                    add("swap_var", sp[0], sp[1], others[(n.col_offset + ln) % len(others)])
    return out


def apply_mutant(src: str, m: Mutant) -> str:
    b = src.encode("utf-8")
    return (b[:m.start] + m.new_text.encode("utf-8") + b[m.end:]).decode("utf-8")


def snippet(src: str, line: int, ctx: int = 0) -> str:
    lines = src.splitlines()
    return "\n".join(lines[max(0, line - 1 - ctx):line + ctx])


# ------------------------------------------------------------------------------------------------------------- generator
COPY_DIRS = ("creator", "engine", "tests", "scripts")
COPY_FILES = ("pyproject.toml",)


def make_copy(root: str | Path, dest: str | Path) -> Path:
    """Copy only what the tests need (never state/, .git, .venv) into a scratch tree."""
    root, dest = Path(root), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for d in COPY_DIRS:
        if (root / d).is_dir():
            shutil.copytree(root / d, dest / d, ignore=ign, dirs_exist_ok=True)
    for f in COPY_FILES:
        if (root / f).is_file():
            shutil.copy2(root / f, dest / f)
    return dest


@dataclass
class GenStats:
    tried: int = 0
    kept: int = 0
    survived: int = 0           # mutants no test noticed
    broken: int = 0             # could not run / unparsable
    seconds: float = 0.0
    kinds: dict[str, int] = field(default_factory=dict)


def _tests_for_target(tree: Path, target: str, limit: int = 4) -> list[str]:
    """Test files that import the target module DIRECTLY (not transitively: a transitive selection runs hundreds of files),
    via creator.testrun's import graph; at most `limit`."""
    from creator import testrun as TR
    from creator.build import module_name_for
    g = TR.ImportGraph.build(tree)
    mod = module_name_for(target)
    out = []
    for key, names in list(g.imports.items()) + list(g.nonmodule_tests.items()):
        path = g.path_of(key)
        if path.startswith("tests/") and TR.is_test_file(path) and any(n == mod or n.startswith(mod + ".") for n in names if mod):
            out.append(path)
    return sorted(out)[:limit]


def _run_mutant(w: Any, target: str, orig: str, m: Mutant, files: Sequence[str], *, timeout: float,
                evaluate: bool, base: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """Plant one mutant through warm worker `w`, run the covering test files. A labelled row, or None when no test fails.
    With `base` (the traced green baseline of these files) the run is two-phase: an untraced pass finds the failing tests (survivors
    stop here, several times cheaper), then ONLY the failing tests are re-run traced; passing tests keep their baseline coverage."""
    from creator.tools import pinpoint_feat as PF
    mutated = apply_mutant(orig, m)
    t0 = time.monotonic()
    if base is None:
        res = w.run(files, {target: mutated}, timeout=timeout)
    else:
        quick = w.run(files, {target: mutated}, timeout=timeout, trace=False)
        failing = sorted(k for k, r in quick.items() if r["outcome"] == "failed")
        if not failing:
            return None
        traced = w.run(failing, {target: mutated}, timeout=timeout)
        res = {k: r for k, r in traced.items() if r["outcome"] == "failed"}
        for k, r in quick.items():
            if r["outcome"] == "passed" and k in base:
                res[k] = {"outcome": "passed", "lines": base[k]["lines"], "text": "", "tail": []}
    bad = sorted(k for k, r in res.items() if r["outcome"] == "failed")
    if not bad:
        return None
    keys, xs = PF.line_features(res, w.tree, sources={target: mutated})
    label = next((i for i, (f, ln) in enumerate(keys) if f == target and ln == m.line), -1)
    row: dict[str, Any] = {"file": target, "true_line": m.line, "mutation": m.kind, "failing_tests": bad,
                           "traceback": res[bad[0]]["text"][:4000],
                           "original_snippet": snippet(orig, m.line), "mutated_snippet": snippet(mutated, m.line),
                           "mutant": {"kind": m.kind, "start": m.start, "end": m.end, "new": m.new_text},
                           "tests": list(files), "label": label, "seconds": round(time.monotonic() - t0, 2),
                           "cand": {"cols": PF.BASE_FEATS, "keys": [list(k) for k in keys], "x": xs}}
    if evaluate:
        ranked = rank_lines(res, root=w.tree, top=None)
        row["rank"] = next((i + 1 for i, (f, ln, _s) in enumerate(ranked) if f == target and ln == m.line), None)
    return row


def generate(root: str | Path, targets: Sequence[str], out_path: str | Path, *, tests: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
             max_rows: int = 200, seed: int = 0, per_function: int = 3, workdir: str | Path | None = None,
             python: str = sys.executable, timeout: float = 120.0, evaluate: bool = True, workers: int = 1,
             progress: Any = None, pool: Any = None, fast: bool = True) -> GenStats:
    """Plant bugs in TEMP COPIES of `root` (one per warm worker; the real tree is never touched) and write labelled rows. Per target
    file: a traced baseline run finds the lines covered by passing tests; each mutant is run on the test files covering its line;
    mutants no test notices are dropped. Row: {file, true_line, mutation, failing_tests, traceback, original_snippet,
    mutated_snippet} + mutant offsets, the candidate feature table (pinpoint_feat.BASE_FEATS) and `label` (index of the true line in
    it, -1 when missing). per_function caps tried mutants per (kind, 15-line region). `tests` may map target -> test files.
    max_rows is per call (across targets)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from creator.tools.pinpoint_worker import WarmPool
    rnd = random.Random(seed)
    base_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="pinpoint_gen_"))
    own_pool = pool is None
    pool = pool or WarmPool(root, max(1, workers), base_dir, python)
    workers = len(pool.workers)
    lock = threading.Lock()
    base_runs: dict[str, Any] = {}
    stats = GenStats()
    t0 = time.monotonic()
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def work(job: tuple[str, str, Mutant, list[str]]) -> None:
        target, orig, m, files = job
        with lock:
            if stats.kept >= max_rows:
                return
        w = pool.acquire()
        try:
            row = _run_mutant(w, target, orig, m, files, timeout=timeout, evaluate=evaluate, base=base_runs.get(target) if fast else None)
            err = False
        except RuntimeError:
            row, err = None, True
        finally:
            pool.release(w)
        with lock:
            stats.tried += 1
            if err:
                stats.broken += 1
            elif row is None:
                stats.survived += 1
            elif stats.kept < max_rows:
                with open(outp, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                stats.kept += 1
                stats.kinds[m.kind] = stats.kinds.get(m.kind, 0) + 1
                if progress:
                    progress(stats)

    try:
        for target in targets:
            if stats.kept >= max_rows:
                break
            if isinstance(tests, Mapping):
                tfiles = list(tests.get(target, []))
            else:
                tfiles = list(tests) if tests else _tests_for_target(pool.workers[0].tree, target)
            if not tfiles:
                continue
            w0 = pool.acquire()
            try:
                base = w0.run(tfiles, timeout=timeout * 3)
            except RuntimeError:
                continue
            finally:
                pool.release(w0)
            if any(r["outcome"] == "failed" for r in base.values()):
                continue                                              # the baseline must be green
            base_runs[target] = base
            cov: dict[int, set[str]] = {}                              # line -> test files covering it (passing tests)
            for nid, r in base.items():
                if r["outcome"] == "passed":
                    for ln in r["lines"].get(target, []):
                        cov.setdefault(ln, set()).add(nid.split("::", 1)[0])
            orig = (pool.workers[0].tree / target).read_text(encoding="utf-8")
            muts = enumerate_mutants(orig, set(cov))
            rnd.shuffle(muts)
            caps: dict[tuple[str, int], int] = {}
            jobs = []
            for m in muts:
                k = (m.kind, m.line // 15)
                if caps.get(k, 0) >= per_function:
                    continue
                try:
                    ast.parse(apply_mutant(orig, m))
                except SyntaxError:
                    stats.broken += 1
                    continue
                caps[k] = caps.get(k, 0) + 1
                jobs.append((target, orig, m, sorted(cov[m.line])))
            by_kind: dict[str, list[Any]] = {}
            for j in jobs:
                by_kind.setdefault(j[2].kind, []).append(j)
            jobs = [j for grp in zip_longest(*by_kind.values()) for j in grp if j is not None]      # interleave kinds: balanced mix
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(work, jobs))
    finally:
        stats.seconds = time.monotonic() - t0
        if own_pool:
            pool.close()
            if workdir is None:
                shutil.rmtree(base_dir, ignore_errors=True)
    return stats


def summarize(rows_path: str | Path) -> dict[str, Any]:
    rows = [json.loads(x) for x in Path(rows_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    n = len(rows)
    def frac(k: int) -> float:
        return round(sum(1 for r in rows if r.get("rank") and r["rank"] <= k) / n, 4) if n else 0.0
    by: dict[str, list[int]] = {}
    for r in rows:
        by.setdefault(r["mutation"], []).append(1 if r.get("rank") and r["rank"] <= 5 else 0)
    return {"rows": n, "top1": frac(1), "top3": frac(3), "top5": frac(5), "top10": frac(10),
            "top5_by_mutation": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())},
            "counts_by_mutation": {k: len(v) for k, v in sorted(by.items())}}


def _main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pinpoint")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--root", default=".")
    g.add_argument("--target", action="append", required=True)
    g.add_argument("--tests", action="append")
    g.add_argument("--out", required=True)
    g.add_argument("--max-rows", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--per-function", type=int, default=3)
    g.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summary")
    s.add_argument("rows")
    a = ap.parse_args(list(argv))
    if a.cmd == "summary":
        print(json.dumps(summarize(a.rows), indent=1))
        return 0
    st = generate(a.root, a.target, a.out, tests=a.tests, max_rows=a.max_rows, seed=a.seed, per_function=a.per_function, workers=a.workers)
    print(json.dumps({"tried": st.tried, "kept": st.kept, "survived": st.survived, "broken": st.broken,
                      "seconds": round(st.seconds, 1), "rows_per_hour": round(st.kept / st.seconds * 3600) if st.seconds else 0,
                      "kinds": st.kinds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
