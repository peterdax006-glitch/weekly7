"""Grow the sealed devbench so improvement becomes MEASURABLE (C77 secs 31-32, 54-56): 15 tasks cannot show a gain.

A library of small reference functions, each with a spec (docstring) and input cases. From each reference the factory makes:
    bugfix   the reference with ONE injected bug from a family the Creator's own search does not undo by construction
             (wrong variable, off-by-one in a range/slice bound, dropped edge-case guard, swapped branches, wrong accumulator
             reset, wrong built-in) - visible tests: 1 case that exposes the bug; hidden tests: all cases
    feature  the spec only (body = raise NotImplementedError) - visible tests: 1 case; hidden tests: all cases
Hidden expected values are computed by running the reference (the oracle). Splits are by FUNCTION: holdout functions never appear
in the dev split. Every generated task is calibrated before sealing: the reference solution must pass visible + hidden tests,
the starting code must fail the hidden tests. Output goes to a staging directory; scripts/devbench_install.py seals it in."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Optional

REFS: list[tuple[str, str, list[tuple[Any, ...]]]] = [
    ("clamp", '''
def clamp(x, lo, hi):
    """Return x limited to the closed range [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
''', [(5, 0, 10), (-3, 0, 10), (12, 0, 10), (0, 0, 10), (10, 0, 10)]),
    ("running_max", '''
def running_max(xs):
    """Return a list whose i-th item is the largest of xs[0..i]. Empty input gives an empty list."""
    out = []
    best = None
    for x in xs:
        if best is None or x > best:
            best = x
        out.append(best)
    return out
''', [([3, 1, 4, 1, 5],), ([],), ([2, 2, 1],), ([-1, -5, 0],)]),
    ("chunk", '''
def chunk(xs, n):
    """Split xs into consecutive lists of length n; the last one may be shorter."""
    return [xs[i:i + n] for i in range(0, len(xs), n)]
''', [([1, 2, 3, 4, 5], 2), ([], 3), ([1, 2, 3], 3), ([1], 5)]),
    ("count_vowels", '''
def count_vowels(s):
    """Count the vowels a, e, i, o, u in s, ignoring case."""
    total = 0
    for ch in s.lower():
        if ch in "aeiou":
            total += 1
    return total
''', [("Hello",), ("",), ("AEIOU xyz",), ("rhythm",)]),
    ("dedupe", '''
def dedupe(xs):
    """Remove repeated items, keeping the first occurrence and the original order."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''', [([1, 2, 1, 3, 2],), ([],), (["a", "a"],), ([3, 2, 1],)]),
    ("is_palindrome", '''
def is_palindrome(s):
    """True if s reads the same backwards, considering only letters and digits and ignoring case."""
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
''', [("A man, a plan, a canal: Panama",), ("abc",), ("",), ("No lemon, no melon",), ("ab",)]),
    ("fizzbuzz", '''
def fizzbuzz(n):
    """Return the FizzBuzz strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' of 5, 'FizzBuzz' of both, else the number."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
''', [(15,), (1,), (0,), (5,)]),
    ("gcd", '''
def gcd(a, b):
    """Greatest common divisor of two non-negative integers (gcd(0, 0) is 0)."""
    while b:
        a, b = b, a % b
    return a
''', [(12, 18), (0, 5), (7, 0), (0, 0), (17, 13)]),
    ("merge_sorted", '''
def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
''', [([1, 3, 5], [2, 4]), ([], [1]), ([1, 1], [1]), ([], [])]),
    ("word_freq", '''
def word_freq(text):
    """Map each lower-cased word (runs of letters) to how often it occurs."""
    counts = {}
    word = ""
    for ch in text.lower() + " ":
        if ch.isalpha():
            word += ch
        elif word:
            counts[word] = counts.get(word, 0) + 1
            word = ""
    return counts
''', [("the cat the hat",), ("",), ("Hi, hi!",), ("a1b",)]),
    ("rotate", '''
def rotate(xs, k):
    """Rotate xs right by k places (k may exceed the length; empty input stays empty)."""
    if not xs:
        return []
    k = k % len(xs)
    return xs[-k:] + xs[:-k] if k else list(xs)
''', [([1, 2, 3, 4], 1), ([1, 2, 3], 5), ([], 3), ([1, 2], 0)]),
    ("flatten", '''
def flatten(xs):
    """Flatten arbitrarily nested lists into one flat list."""
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
''', [([1, [2, [3, 4]], 5],), ([],), ([[[]]],), ([1, 2],)]),
    ("binary_search", '''
def binary_search(xs, target):
    """Index of target in the sorted list xs, or -1 if absent."""
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''', [([1, 3, 5, 7], 5), ([1, 3, 5, 7], 4), ([], 1), ([2], 2), ([1, 2, 3, 4, 5, 6], 6)]),
    ("caesar", '''
def caesar(s, k):
    """Shift letters of s by k places in the alphabet, keeping case; other characters unchanged."""
    out = []
    for ch in s:
        if ch.isupper():
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
        elif ch.islower():
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)
''', [("abc", 1), ("XYZ", 3), ("Hello, World!", 13), ("", 4)]),
    ("second_largest", '''
def second_largest(xs):
    """The second largest DISTINCT value in xs, or None if there is none."""
    distinct = sorted(set(xs))
    if len(distinct) < 2:
        return None
    return distinct[-2]
''', [([3, 1, 4, 4],), ([5],), ([],), ([2, 2],), ([-1, -2, -3],)]),
    ("pairs_sum", '''
def pairs_sum(xs, target):
    """Count index pairs i < j with xs[i] + xs[j] == target."""
    count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target:
                count += 1
    return count
''', [([1, 2, 3, 4], 5), ([], 1), ([1, 1, 1], 2), ([5], 10)]),
    ("title_words", '''
def title_words(s):
    """Capitalise the first letter of every space-separated word and lower-case the rest."""
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))
''', [("hello wORLD",), ("",), ("a b  c",), ("x",)]),
    ("digits_sum", '''
def digits_sum(n):
    """Sum of the decimal digits of an integer (sign ignored)."""
    n = abs(n)
    total = 0
    while n:
        total += n % 10
        n //= 10
    return total
''', [(123,), (0,), (-45,), (9999,)]),
    ("median_of", '''
def median_of(xs):
    """Median of a non-empty list of numbers (mean of the two middle values for even length)."""
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2
''', [([3, 1, 2],), ([4, 1, 3, 2],), ([5],), ([1, 1, 2, 2],)]),
    ("compress", '''
def compress(s):
    """Run-length encode s: 'aaab' -> 'a3b1'. Empty input gives an empty string."""
    if not s:
        return ""
    out = []
    cur, n = s[0], 1
    for ch in s[1:]:
        if ch == cur:
            n += 1
        else:
            out.append(f"{cur}{n}")
            cur, n = ch, 1
    out.append(f"{cur}{n}")
    return "".join(out)
''', [("aaab",), ("",), ("abc",), ("zzzz",)]),
    ("transpose", '''
def transpose(m):
    """Transpose a rectangular matrix given as a list of rows."""
    if not m:
        return []
    return [[row[i] for row in m] for i in range(len(m[0]))]
''', [([[1, 2, 3], [4, 5, 6]],), ([],), ([[1]],), ([[1, 2]],)]),
    ("anagrams", '''
def anagrams(a, b):
    """True if a and b contain the same letters with the same counts, ignoring case and spaces."""
    norm = lambda s: sorted(s.replace(" ", "").lower())
    return norm(a) == norm(b)
''', [("Listen", "Silent"), ("abc", "abd"), ("", ""), ("a gentleman", "elegant man")]),
    ("prefix_sums", '''
def prefix_sums(xs):
    """List of running totals: prefix_sums([1, 2, 3]) == [1, 3, 6]."""
    out = []
    total = 0
    for x in xs:
        total += x
        out.append(total)
    return out
''', [([1, 2, 3],), ([],), ([5, -5, 5],), ([0],)]),
    ("roman", '''
def roman(n):
    """Roman numeral for 1 <= n <= 3999."""
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym
            n -= v
    return out
''', [(1,), (4,), (1994,), (3999,), (58,)]),
    ("safe_div", '''
def safe_div(a, b, default=0):
    """a / b, or `default` when b is zero."""
    if b == 0:
        return default
    return a / b
''', [(6, 3), (1, 0), (1, 0, -1), (0, 5)]),
    ("max_window", '''
def max_window(xs, k):
    """Largest sum of any k consecutive items; None if k is not between 1 and len(xs)."""
    if k < 1 or k > len(xs):
        return None
    best = cur = sum(xs[:k])
    for i in range(k, len(xs)):
        cur += xs[i] - xs[i - k]
        best = max(best, cur)
    return best
''', [([1, 2, 3, 4], 2), ([5], 1), ([1, 2], 3), ([-1, -2, -3], 2), ([1, 2], 0)]),
    ("split_even_odd", '''
def split_even_odd(xs):
    """Return (evens, odds) keeping the original order within each list."""
    evens, odds = [], []
    for x in xs:
        if x % 2 == 0:
            evens.append(x)
        else:
            odds.append(x)
    return evens, odds
''', [([1, 2, 3, 4],), ([],), ([0, -1],), ([7],)]),
    ("leap_year", '''
def leap_year(y):
    """True for Gregorian leap years."""
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
''', [(2000,), (1900,), (2024,), (2023,)]),
    ("normalize_spaces", '''
def normalize_spaces(s):
    """Collapse runs of whitespace into single spaces and strip both ends."""
    return " ".join(s.split())
''', [("  a   b  ",), ("",), ("x",), ("a\\tb\\nc",)]),
    ("top_n", '''
def top_n(scores, n):
    """Names of the n highest scores from a dict name->score, highest first, ties by name."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:n]]
''', [({"a": 3, "b": 5, "c": 3}, 2), ({}, 1), ({"x": 1}, 3), ({"b": 2, "a": 2}, 1)]),
]


# ------------------------------------------------------------------------------------------------ bug injection

def _fn(tree: ast.Module) -> ast.FunctionDef:
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef))


def bug_variants(src: str) -> list[tuple[str, str]]:
    """(description, buggy source) - families the Creator's own mutation search does not apply (it swaps operators, shifts
    constants by one, wraps abs, swaps call args, fixes mutable defaults)."""
    out: list[tuple[str, str]] = []
    tree = ast.parse(src)
    fn = _fn(tree)
    names = sorted({n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id not in dir(__builtins__)} | {a.arg for a in fn.args.args})
    # wrong variable: one Load of a local replaced by another local
    loads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names]
    for i, n in enumerate(loads):
        for other in names:
            if other != n.id:
                t = copy.deepcopy(tree)
                tl = [m for m in ast.walk(_fn(t)) if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load) and m.id in names]
                tl[i].id = other
                out.append((f"uses {other} where {n.id} is needed", ast.unparse(t)))
                break
        if len(out) >= 3:
            break
    # dropped guard: remove the first `if ...: return ...` statement
    for i, st in enumerate(fn.body):
        if isinstance(st, ast.If) and any(isinstance(s, ast.Return) for s in st.body):
            t = copy.deepcopy(tree)
            del _fn(t).body[i]
            out.append(("an edge-case guard is missing", ast.unparse(t)))
            break
    # swapped if/else branches
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and n.orelse and not (len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If)):
            t = copy.deepcopy(tree)
            idx = list(ast.walk(fn)).index(n)
            m = list(ast.walk(_fn(t)))[idx - list(ast.walk(fn)).index(fn)] if False else None
            for mm in ast.walk(_fn(t)):
                if isinstance(mm, ast.If) and ast.dump(mm) == ast.dump(n):
                    mm.body, mm.orelse = mm.orelse, mm.body
                    break
            out.append(("two branches are swapped", ast.unparse(t)))
            break
    # wrong slice bound / range end: drop the upper bound of the first slice or range(a, b)
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" and len(n.args) >= 2:
            t = copy.deepcopy(tree)
            for mm in ast.walk(_fn(t)):
                if isinstance(mm, ast.Call) and isinstance(mm.func, ast.Name) and mm.func.id == "range" and ast.dump(mm) == ast.dump(n):
                    mm.args[1] = ast.BinOp(mm.args[1], ast.Sub(), ast.Constant(1))
                    break
            out.append(("a range stops one item early", ast.unparse(t)))
            break
    # wrong built-in: max<->min, sorted->list, sum->len
    swaps = {"max": "min", "min": "max", "sorted": "list", "sum": "len", "extend": "append", "append": "extend"}
    for n in ast.walk(fn):
        name = n.func.id if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) else \
            n.func.attr if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else None
        if name in swaps:
            t = copy.deepcopy(tree)
            for mm in ast.walk(_fn(t)):
                if isinstance(mm, ast.Call) and ast.dump(mm) == ast.dump(n):
                    if isinstance(mm.func, ast.Name):
                        mm.func.id = swaps[name]
                    else:
                        mm.func.attr = swaps[name]
                    break
            out.append((f"{name} is used where {swaps[name]} belongs", ast.unparse(t)))
            break
    seen, uniq = set(), []
    for d, s in out:
        if s not in seen and s != ast.unparse(tree):
            seen.add(s)
            uniq.append((d, s + "\n"))
    return uniq


def stub(src: str) -> str:
    tree = ast.parse(src)
    fn = _fn(tree)
    doc = ast.get_docstring(fn)
    fn.body = ([ast.Expr(ast.Constant(doc))] if doc else []) + [ast.Raise(ast.Call(ast.Name("NotImplementedError", ast.Load()),
                                                                                       [], []), None)]
    return ast.unparse(tree) + "\n"


# ------------------------------------------------------------------------------------------------ oracle tests

class _StepLimit(Exception):
    pass


def run(src: str, name: str, args: tuple[Any, ...], max_steps: int = 200_000) -> tuple[str, Any]:
    """Run one case under a line-step limit (an injected bug can loop forever): a timeout is an observable behaviour too."""
    env: dict[str, Any] = {}
    steps = [0]

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event == "line":
            steps[0] += 1
            if steps[0] > max_steps:
                raise _StepLimit()
        return tracer
    try:
        exec(src, env)
        sys.settrace(tracer)
        try:
            return "ok", env[name](*copy.deepcopy(args))
        finally:
            sys.settrace(None)
    except _StepLimit:
        return "err", "Timeout"
    except Exception as e:                                          # noqa: BLE001 - a crash is an observable behaviour
        return "err", type(e).__name__


def test_file(name: str, cases: list[tuple[Any, ...]], expected: list[Any]) -> str:
    lines = [f"from app.solution import {name}", ""]
    for i, (args, exp) in enumerate(zip(cases, expected)):
        lines += ["", f"def test_case_{i}():", f"    assert {name}(*{args!r}) == {exp!r}"]
    return "\n".join(lines) + "\n"


def build(out: Path, holdout_every: int = 3, seed: int = 7) -> dict[str, Any]:
    rng = random.Random(seed)
    order = list(range(len(REFS)))
    rng.shuffle(order)
    holdout_fns = {REFS[i][0] for k, i in enumerate(order) if k % holdout_every == 0}
    made, skipped = [], []
    n = 0
    for name, src, cases in REFS:
        src = textwrap.dedent(src).strip() + "\n"
        expected = [run(src, name, c)[1] for c in cases]
        split = "holdout" if name in holdout_fns else "dev"
        variants: list[tuple[str, str, str, str]] = []
        for desc, bug in bug_variants(src)[:3]:
            variants.append(("bugfix", f"{name}() in app/solution.py does not do what its docstring says for some inputs. "
                             f"Find the bug and fix it.", bug, desc))         # the symptom, never the cause
        variants.append(("feature", f"Implement {name}() in app/solution.py so it meets its docstring.", stub(src), "stub"))
        for cat, objective, start, desc in variants:
            got = [run(start, name, c) for c in cases]
            if any(g == ("err", "Timeout") for g in got):
                skipped.append(f"{name}:{desc}: the starting code can loop forever (would stall every unsolved run)")
                continue
            failing = [i for i, (g, e) in enumerate(zip(got, expected)) if g != ("ok", e)]
            if not failing:
                skipped.append(f"{name}:{desc}: the starting code already passes every case")
                continue
            vis = [failing[0]]
            n += 1
            tid = f"{'G' if split == 'dev' else 'GH'}{n:03d}"
            root = out / "tasks" / tid
            (root / "repo" / "app").mkdir(parents=True, exist_ok=True)
            (root / "repo" / "tests").mkdir(parents=True, exist_ok=True)
            (root / "repo" / "app" / "__init__.py").write_text("", encoding="utf-8")
            (root / "repo" / "tests" / "__init__.py").write_text("", encoding="utf-8")
            (root / "repo" / "app" / "solution.py").write_text(start, encoding="utf-8")
            (root / "repo" / "tests" / "test_solution.py").write_text(
                test_file(name, [cases[i] for i in vis], [expected[i] for i in vis]), encoding="utf-8")
            (root / "task.json").write_text(json.dumps({"id": tid, "category": cat, "split": split, "objective": objective,
                                                        "time_limit_s": 600, "call_limit": 10}, indent=1), encoding="utf-8")
            hid = out / "sealed" / tid / "hidden"
            hid.mkdir(parents=True, exist_ok=True)
            (hid / "__init__.py").write_text("", encoding="utf-8")
            (hid / "test_hidden.py").write_text(test_file(name, cases, expected), encoding="utf-8")
            ref = out / "sealed" / tid / "reference" / "app"
            ref.mkdir(parents=True, exist_ok=True)
            (ref / "solution.py").write_text(src, encoding="utf-8")
            made.append({"id": tid, "fn": name, "category": cat, "split": split, "bug": desc})
    return {"tasks": made, "skipped": skipped, "holdout_functions": sorted(holdout_fns),
            "digest": hashlib.sha256(json.dumps(made, sort_keys=True).encode()).hexdigest()[:16]}


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "creator_runtime" / "devbench_staging"
    info = build(dest)
    from collections import Counter
    print(len(info["tasks"]), "tasks", dict(Counter((t["split"], t["category"]) for t in info["tasks"])), "skipped",
          len(info["skipped"]))
    (dest / "factory.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
