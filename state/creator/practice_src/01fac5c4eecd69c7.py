"""Creator K19 - deterministic program synthesis for function stubs (no LLM, no network).

A stub is `def f(...): "docstring"; raise NotImplementedError`. The solver collects EXAMPLES (visible tests plus `f(x) == y` and
`'a' -> 'b'` patterns in the docstring) and proposes candidate bodies from two sources:

    idioms   a library of general-purpose small programs (clamp, running max, dedupe, gcd, ...) each guarded by docstring
             vocabulary regexes and an arity; candidates are ordered by how many vocabulary patterns the docstring matches,
             then by size. Task ids, function names and answer values are never consulted.
    enum     bottom-up enumeration of expressions over the parameters (arithmetic, comparisons, len/sum/min/max/sorted/set,
             slicing, str methods) with observational-equivalence pruning; smallest first (Occam).

A candidate is accepted only if it passes EVERY available example. One visible example under-determines a program, so
idioms that are general by construction and docstring-gated are tried before blind enumeration. Hidden tests judge the result
(creator.devbench): a pass on the visible tests alone that fails hidden ones is reported there as FALSE_COMPLETION."""
from __future__ import annotations

import ast
import copy
import dataclasses
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from creator import devbench as D

Example = tuple[tuple[Any, ...], Any]


# ------------------------------------------------------------------------------------------------ the idiom library

@dataclasses.dataclass(frozen=True)
class Idiom:
    name: str
    nargs: int
    vocab: tuple[str, ...]          # regexes over the lower-cased docstring; at least one must match
    body: str                       # function body; {0} {1} ... are the parameter names


def _i(name: str, nargs: int, vocab: str, body: str) -> Idiom:
    return Idiom(name, nargs, (vocab,), body)


IDIOMS: tuple[Idiom, ...] = (
    _i("clamp", 3, r"limited to|clamp|closed range", "return max({1}, min({0}, {2}))"),
    _i("running_max", 1, r"largest of .*\[0|running max|maximum so far",
       "out = []\nfor v in {0}:\n    out.append(v if not out or v > out[-1] else out[-1])\nreturn out"),
    _i("running_min", 1, r"smallest of .*\[0|running min|minimum so far",
       "out = []\nfor v in {0}:\n    out.append(v if not out or v < out[-1] else out[-1])\nreturn out"),
    _i("prefix_sums", 1, r"running total|prefix sum|cumulative sum",
       "out = []\ntotal = 0\nfor v in {0}:\n    total += v\n    out.append(total)\nreturn out"),
    _i("count_vowels", 1, r"vowel", "return sum(1 for c in {0}.lower() if c in 'aeiou')"),
    _i("count_consonants", 1, r"consonant", "return sum(1 for c in {0}.lower() if c.isalpha() and c not in 'aeiou')"),
    _i("dedupe", 1, r"remove repeated|dedup|duplicates|first occurrence",
       "seen = set()\nout = []\nfor v in {0}:\n    if v not in seen:\n        seen.add(v)\n        out.append(v)\nreturn out"),
    _i("palindrome_alnum", 1, r"reads the same backwards|palindrome",
       "t = [c.lower() for c in {0} if c.isalnum()]\nreturn t == t[::-1]"),
    _i("fizzbuzz", 1, r"fizz",
       "out = []\nfor i in range(1, {0} + 1):\n    s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n"
       "    out.append(s or str(i))\nreturn out"),
    _i("gcd", 2, r"greatest common divisor|\bgcd\b",
       "a, b = abs({0}), abs({1})\nwhile b:\n    a, b = b, a % b\nreturn a"),
    _i("lcm", 2, r"least common multiple|\blcm\b",
       "a, b = abs({0}), abs({1})\nif a == 0 or b == 0:\n    return 0\nx, y = a, b\nwhile y:\n    x, y = y, x % y\nreturn a * b // x"),
    _i("rotate_right", 2, r"rotate .*right",
       "if not {0}:\n    return []\nk = {1} % len({0})\nreturn list({0}[-k:] + {0}[:-k]) if k else list({0})"),
    _i("rotate_left", 2, r"rotate .*left",
       "if not {0}:\n    return []\nk = {1} % len({0})\nreturn list({0}[k:] + {0}[:k])"),
    _i("caesar", 2, r"shift letters|caesar",
       "out = []\nfor c in {0}:\n    if 'a' <= c <= 'z':\n        out.append(chr((ord(c) - 97 + {1}) % 26 + 97))\n"
       "    elif 'A' <= c <= 'Z':\n        out.append(chr((ord(c) - 65 + {1}) % 26 + 65))\n    else:\n        out.append(c)\n"
       "return ''.join(out)"),
    _i("second_largest", 1, r"second largest",
       "u = sorted(set({0}))\nreturn u[-2] if len(u) >= 2 else None"),
    _i("second_smallest", 1, r"second smallest",
       "u = sorted(set({0}))\nreturn u[1] if len(u) >= 2 else None"),
    _i("pairs_sum", 2, r"index pairs",
       "n = 0\nfor i in range(len({0})):\n    for j in range(i + 1, len({0})):\n        if {0}[i] + {0}[j] == {1}:\n            n += 1\nreturn n"),
    _i("digits_sum", 1, r"sum of the .*digits|digit sum|sum of digits",
       "return sum(int(c) for c in str(abs({0})) if c.isdigit())"),
    _i("median", 1, r"median",
       "s = sorted({0})\nn = len(s)\nreturn s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2"),
    _i("rle", 1, r"run-length|run length",
       "out = []\ni = 0\nwhile i < len({0}):\n    j = i\n    while j < len({0}) and {0}[j] == {0}[i]:\n        j += 1\n"
       "    out.append({0}[i] + str(j - i))\n    i = j\nreturn ''.join(out)"),
    _i("transpose", 1, r"transpose", "return [list(r) for r in zip(*{0})]"),
    _i("roman", 1, r"roman numeral",
       "vals = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), "
       "(9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\nout = ''\nn = {0}\nfor v, s in vals:\n    while n >= v:\n        out += s\n        n -= v\nreturn out"),
    _i("max_window", 2, r"consecutive items|sliding window",
       "if not 1 <= {1} <= len({0}):\n    return None\nreturn max(sum({0}[i:i + {1}]) for i in range(len({0}) - {1} + 1))"),
    _i("leap_year", 1, r"leap year", "return {0} % 4 == 0 and ({0} % 100 != 0 or {0} % 400 == 0)"),
    _i("normalize_spaces", 1, r"collapse runs of whitespace|collapse whitespace|single spaces", "return ' '.join({0}.split())"),
    _i("slugify", 1, r"slug", "return re.sub(r'[^a-z0-9]+', '-', {0}.lower()).strip('-')"),
    _i("factorial", 1, r"factorial", "r = 1\nfor i in range(2, {0} + 1):\n    r *= i\nreturn r"),
    _i("fibonacci", 1, r"fibonacci", "a, b = 0, 1\nfor _ in range({0}):\n    a, b = b, a + b\nreturn a"),
    _i("is_prime", 1, r"\bprime\b",
       "if {0} < 2:\n    return False\ni = 2\nwhile i * i <= {0}:\n    if {0} % i == 0:\n        return False\n    i += 1\nreturn True"),
    _i("flatten", 1, r"flatten", "return [v for row in {0} for v in row]"),
    _i("word_count", 1, r"number of words|count the words|word count", "return len({0}.split())"),
    _i("reverse_words", 1, r"reverse the (order of )?words", "return ' '.join(reversed({0}.split()))"),
    _i("reverse", 1, r"reverse", "return {0}[::-1]"),
    _i("is_anagram", 2, r"anagram", "return sorted({0}.lower().replace(' ', '')) == sorted({1}.lower().replace(' ', ''))"),
    _i("sum_even", 1, r"sum of .*even", "return sum(v for v in {0} if v % 2 == 0)"),
    _i("sum_odd", 1, r"sum of .*odd", "return sum(v for v in {0} if v % 2 != 0)"),
    _i("evens", 1, r"even (numbers|items|values)", "return [v for v in {0} if v % 2 == 0]"),
    _i("odds", 1, r"odd (numbers|items|values)", "return [v for v in {0} if v % 2 != 0]"),
    _i("mean", 1, r"^(?!.*median).*(\bmean\b|average)", "return sum({0}) / len({0}) if {0} else 0.0"),
    _i("count_char", 2, r"occurrences of", "return {0}.count({1})"),
    _i("most_common", 1, r"most (common|frequent)",
       "best = None\nbc = 0\nfor v in {0}:\n    c = {0}.count(v)\n    if c > bc:\n        best, bc = v, c\nreturn best"),
    _i("is_sorted", 1, r"is sorted|non-decreasing",
       "return all({0}[i] <= {0}[i + 1] for i in range(len({0}) - 1))"),
    _i("binary", 1, r"binary (string|representation)", "return bin({0})[2:]"),
)


# ------------------------------------------------------------------------------------------------ spec extraction

@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    params: tuple[str, ...]
    doc: str
    examples: tuple[Example, ...]
    probes: tuple[tuple[Any, ...], ...] = ()    # extra inputs (testgen.candidate_calls) used to detect ambiguity


def _lit(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def examples_from_tests(src: str, fname: str) -> list[Example]:
    out: list[Example] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            continue
        call = node.left
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == fname):
            continue
        try:
            if len(call.args) == 1 and isinstance(call.args[0], ast.Starred):
                args = tuple(_lit(call.args[0].value))
            else:
                args = tuple(_lit(a) for a in call.args)
            out.append((args, _lit(node.comparators[0])))
        except (ValueError, SyntaxError, TypeError):
            continue
    return out


_ATOM = r"""'[^'\n]*'|"[^"\n]*"|\[[^\]\n]*\]|-?\d+"""


def examples_from_doc(doc: str, fname: str, nparams: int) -> list[Example]:
    out: list[Example] = []
    for m in re.finditer(re.escape(fname) + r"\(([^\n]*?)\)\s*==\s*(" + _ATOM + ")", doc):
        try:
            args = _lit(ast.parse(f"({m.group(1)},)", mode="eval").body)
            out.append((tuple(args), _lit(ast.parse(m.group(2), mode="eval").body)))
        except (ValueError, SyntaxError, TypeError):
            continue
    if nparams == 1:
        for m in re.finditer("(" + _ATOM + r")\s*->\s*(" + _ATOM + ")", doc):
            try:
                out.append(((_lit(ast.parse(m.group(1), mode="eval").body),), _lit(ast.parse(m.group(2), mode="eval").body)))
            except (ValueError, SyntaxError, TypeError):
                continue
    return out


# ------------------------------------------------------------------------------------------------ checking

def _same(got: Any, want: Any) -> bool:
    if isinstance(got, bool) != isinstance(want, bool):
        return False
    return bool(got == want)


# ------------------------------------------------------------------------------------------------ evaluation guards
# Candidate programs run in-process on the examples' values. A huge example (or a candidate that loops on one) must not hang
# the solver: every evaluation runs under a step budget (traced line events), and examples beyond a size cap are refused.
STEP_BUDGET = 100_000               # traced line/call events per guarded evaluation; normal examples use a few hundred
BIG_INT = 10_000                    # an example int beyond this (fibonacci(10**9), pow(10, 10**9)) makes loops/powers risky
BIG_INT_STEPS = 20_000              # ... so the step budget is tighter and C-level pow/repeat expressions are not tried
MAX_EXAMPLE_SIZE = 200_000          # approximate total elements/characters/bytes across all example inputs and outputs


class BudgetExceeded(Exception):
    pass


class _StepGuard:
    """Context manager: raises BudgetExceeded once the traced Python code under it has run STEP_BUDGET events."""

    def __init__(self, steps: int = STEP_BUDGET) -> None:
        self.left = steps
        self.prev: Any = None

    def _trace(self, frame: Any, event: str, arg: Any) -> Any:
        self.left -= 1
        if self.left < 0:
            raise BudgetExceeded("step budget exhausted")
        return self._trace

    def __enter__(self) -> "_StepGuard":
        self.prev = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc: Any) -> None:
        sys.settrace(self.prev)


def _size(v: Any, cap: int) -> int:
    """Approximate size of a value, stopping once it exceeds cap."""
    n, stack = 0, [v]
    while stack and n <= cap:
        x = stack.pop()
        if isinstance(x, bool) or x is None:
            n += 1
        elif isinstance(x, int):
            n += 1 + x.bit_length() // 8
        elif isinstance(x, (str, bytes)):
            n += 1 + len(x)
        elif isinstance(x, dict):
            n += 1 + len(x)
            stack.extend(list(x.keys())[:cap])
            stack.extend(list(x.values())[:cap])
        elif isinstance(x, (list, tuple, set, frozenset)):
            n += 1 + len(x)
            stack.extend(list(x)[:cap])
        else:
            n += 1
    return n


def _has_big_int(v: Any, depth: int = 3) -> bool:
    if isinstance(v, int) and not isinstance(v, bool):
        return abs(v) > BIG_INT
    if depth and isinstance(v, (list, tuple, set, frozenset)):
        return any(_has_big_int(x, depth - 1) for x in list(v)[:1000])
    return False


def _risky(spec: "Spec") -> bool:
    return any(_has_big_int(a) for args, _ in spec.examples for a in args)


def _steps(spec: "Spec") -> int:
    return BIG_INT_STEPS if _risky(spec) else STEP_BUDGET


def examples_too_big(spec: "Spec", cap: int = MAX_EXAMPLE_SIZE) -> bool:
    total = 0
    for args, want in spec.examples:
        total += _size(args, cap) + _size(want, cap)
        if total > cap:
            return True
    return False


def _build(spec: Spec, body: str) -> Optional[Callable[..., Any]]:
    src = f"def {spec.name}({', '.join(spec.params)}):\n" + "".join("    " + ln + "\n" for ln in body.splitlines())
    ns: dict[str, Any] = {"re": re}
    try:
        exec(compile(src, "<synth>", "exec"), ns)
    except Exception:
        return None
    fn: Callable[..., Any] = ns[spec.name]
    return fn


def passes(spec: Spec, body: str) -> bool:
    fn = _build(spec, body)
    if fn is None or not spec.examples:
        return False
    if examples_too_big(spec):
        return False
    with _StepGuard(_steps(spec)):
        for args, want in spec.examples:
            try:
                got = fn(*[copy.deepcopy(a) for a in args])
            except Exception:                       # includes BudgetExceeded
                return False
            if not _same(got, want):
                return False
    return True


# ------------------------------------------------------------------------------------------------ enumeration

_UNARY = ("len({})", "sum({})", "min({})", "max({})", "sorted({})", "list(reversed({}))", "set({})", "abs({})", "{}.lower()",
          "{}.upper()", "{}.strip()", "{}.split()", "str({})", "int({})", "{}[::-1]", "{}[1:]", "{}[:-1]", "(not {})",
          "list({})", "''.join({})", "' '.join({})", "{}.isalpha()", "{}.isdigit()", "all({})", "any({})")
_BINARY = ("({} + {})", "({} - {})", "({} * {})", "({} // {})", "({} % {})", "({} == {})", "({} < {})", "({} <= {})",
           "({} > {})", "({} >= {})", "({} != {})", "min({}, {})", "max({}, {})", "({} and {})", "({} or {})", "{}[{}]",
           "({} in {})", "{}.count({})", "{}.join({})", "{}.split({})", "{}[:{}]", "{}[{}:]", "pow({}, {})")


def enumerate_exprs(spec: Spec, max_size: int = 5, limit: int = 40000) -> Optional[str]:
    """Smallest expression (AST-node count) passing every example: bottom-up enumeration, pruned by value signature."""
    if examples_too_big(spec):
        return None
    risky, steps = _risky(spec), _steps(spec)
    names = list(spec.params)
    envs = [dict(zip(names, [copy.deepcopy(a) for a in args])) for args, _ in spec.examples]
    wants = [w for _, w in spec.examples]
    by_size: dict[int, list[str]] = {}
    seen: set[str] = set()

    def ev(code: str) -> Optional[tuple[Any, ...]]:
        if risky and ("pow(" in code or "*" in code):      # C-level power / repeat on a huge number cannot be interrupted
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = compile(code, "<e>", "eval")
            with _StepGuard(steps):
                return tuple(eval(c, {}, dict(e)) for e in envs)
        except Exception:                           # includes BudgetExceeded
            return None

    def add(size: int, code: str, vals: tuple[Any, ...]) -> bool:
        try:
            s = repr(vals) + repr([type(v).__name__ for v in vals])
        except Exception:
            return False
        if s in seen:
            return False
        seen.add(s)
        by_size.setdefault(size, []).append(code)
        return all(_same(v, w) for v, w in zip(vals, wants))

    for lf in names + ["0", "1", "2", "''", "' '", "[]", "True", "False"]:
        vals = ev(lf)
        if vals is not None and add(1, lf, vals):
            return lf
    count = 0
    for size in range(2, max_size + 1):
        for code0 in list(by_size.get(size - 1, [])):
            for u in _UNARY:
                code = u.format(code0)
                v = ev(code)
                if v is not None and add(size, code, v):
                    return code
        for ls in range(1, size - 1):
            for a in list(by_size.get(ls, [])):
                for b in list(by_size.get(size - 1 - ls, [])):
                    for t in _BINARY:
                        count += 1
                        if count > limit:
                            return None
                        code = t.format(a, b)
                        v = ev(code)
                        if v is not None and add(size, code, v):
                            return code
    return None


# ------------------------------------------------------------------------------------------------ driver

def _cues(idm: Idiom, doc: str) -> int:
    """Independent docstring cues: distinct top-level alternatives of the vocab regexes that match."""
    n = 0
    for v in idm.vocab:
        for alt in re.split(r"\|(?![^(]*\))", v):
            if alt and re.search(alt, doc, re.S):
                n += 1
    return n


def candidates(spec: Spec, disabled: frozenset[str] = frozenset(), gated: bool = True) -> list[tuple[str, str]]:
    """(label, body) for every idiom whose arity and vocabulary fit, best vocabulary match first."""
    doc = spec.doc.lower()
    scored = []
    for idm in IDIOMS:
        if idm.nargs != len(spec.params) or idm.name in disabled:
            continue
        hits = sum(1 for v in idm.vocab if re.search(v, doc, re.S))
        if hits or not gated:
            body = idm.body.format(*spec.params)
            scored.append((-hits, len(body), idm.name, body))
    scored.sort()
    return [(f"idiom:{n}", b) for _, _, n, b in scored]


def _probe_inputs(spec: Spec) -> list[tuple[Any, ...]]:
    """Probe argument tuples: the examples' own inputs, each argument position swapped in from other examples, plus testgen's."""
    out: list[tuple[Any, ...]] = [tuple(a) for a, _ in spec.examples]
    out += [tuple(p) for p in spec.probes]
    for i in range(len(spec.params)):
        for a, _ in spec.examples:
            for b, _ in spec.examples:
                if i < len(a) and i < len(b):
                    out.append(tuple(b[i] if j == i else v for j, v in enumerate(a)))
    return out


def _outputs(spec: Spec, body: str, probes: Sequence[tuple[Any, ...]]) -> list[Any]:
    fn = _build(spec, body)
    res: list[Any] = []
    for args in probes:
        try:
            with _StepGuard(_steps(spec)):
                res.append(("ok", repr(fn(*[copy.deepcopy(a) for a in args]))) if fn else ("err",))
        except Exception:                           # includes BudgetExceeded
            res.append(("err",))
    return res


def _disagree(spec: Spec, a: str, b: str) -> bool:
    probes = _probe_inputs(spec)
    for x, y in zip(_outputs(spec, a, probes), _outputs(spec, b, probes)):
        if x[0] == "ok" and y[0] == "ok" and x != y:
            return True
    return False


def synthesize(spec: Spec, disabled: frozenset[str] = frozenset()) -> Optional[tuple[str, str]]:
    """(label, body) of the first candidate passing every example, or None. Abstains (None) unless the evidence is strong:
    at least two distinct examples or two independent docstring cues, and no other passing idiom disagreeing on probe inputs."""
    if examples_too_big(spec):                      # refuse before repr()/evaluating anything on a huge example
        return None
    distinct = len({repr(a) for a, _ in spec.examples})
    doc = spec.doc.lower()
    cues = {i.name: _cues(i, doc) for i in IDIOMS}
    passing = [(lbl, b) for lbl, b in candidates(spec, disabled) if passes(spec, b)]
    if passing:
        lbl, body = passing[0]
        if distinct < 2 and cues[lbl.split(":", 1)[1]] < 2:
            return None
        rivals = [b for _, b in candidates(spec, disabled, gated=False) if b != body and passes(spec, b)]
        if any(_disagree(spec, body, r) for r in rivals):
            return None
        return lbl, body
    if spec.examples and distinct >= 2:
        expr = enumerate_exprs(spec)
        if expr:
            body = f"return {expr}"
            rivals = [b for _, b in candidates(spec, disabled, gated=False) if passes(spec, b)]
            if any(_disagree(spec, body, r) for r in rivals):
                return None
            return "enum", body
    return None


def _stub_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            stmts = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if len(stmts) == 1 and isinstance(stmts[0], ast.Raise):
                exc = stmts[0].exc
                nm = exc.func if isinstance(exc, ast.Call) else exc
                if isinstance(nm, ast.Name) and nm.id == "NotImplementedError":
                    out.append(node)
    return out


def solve_file(path: Path, test_srcs: Sequence[str], disabled: frozenset[str] = frozenset()) -> tuple[bool, str]:
    """Fill every stub in `path` that can be synthesized; (all stubs filled, note). Writes only on success."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    stubs = _stub_functions(tree)
    if not stubs:
        return False, "no stubs"
    notes = []
    for fn in stubs:
        doc = ast.get_docstring(fn) or ""
        params = tuple(a.arg for a in fn.args.args)
        ex: list[Example] = []
        for ts in test_srcs:
            ex += examples_from_tests(ts, fn.name)
        ex += examples_from_doc(doc, fn.name, len(params))
        uniq: list[Example] = []
        for e in ex:
            if e not in uniq:
                uniq.append(e)
        try:
            from creator import testgen
            probes = tuple(testgen.candidate_calls(fn))
        except Exception:
            probes = ()
        got = synthesize(Spec(fn.name, params, doc, tuple(uniq), probes), disabled)
        if got is None:
            return False, f"no program for {fn.name} ({len(uniq)} examples)"
        label, body = got
        keep = [s for s in fn.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)]
        fn.body = keep + ast.parse(body).body
        notes.append(f"{fn.name}<-{label}")
    out = ast.unparse(tree) + "\n"
    if "re." in out and not re.search(r"^import re$", out, re.M):
        out = "import re\n" + out
    path.write_text(out, encoding="utf-8")
    return True, ", ".join(notes)


class SynthSolver:
    """A devbench Solver made only of deterministic program synthesis: no model, no network."""
    name = "self-synth"

    def __init__(self, disabled: frozenset[str] = frozenset()) -> None:
        self.disabled = disabled

    def __call__(self, task: Mapping[str, Any], workdir: Path) -> D.SolverResult:
        tests = [p.read_text(encoding="utf-8") for p in sorted(workdir.rglob("test*.py"))]
        targets = [p for p in sorted(workdir.rglob("*.py")) if "tests" not in p.relative_to(workdir).parts
                   and p.name not in ("__init__.py", "conftest.py") and not p.name.startswith("test")]
        for p in targets:
            ok, note = solve_file(p, tests, self.disabled)
            if ok:
                return D.SolverResult(True, 0, f"synth: {note}")
        return D.SolverResult(False, 0, "synth found no program")
