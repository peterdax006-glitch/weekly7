"""Teacher's contrastive lessons for the ActionStudent: small modules where TWO confusable actions are both available.

Each lesson is built from a module in which both actions apply, the objective selects exactly one, and files_after is produced by
creator.action_student.apply_action (so it is exactly what the action does). A lesson is marked adopted only when before/after are
exec-compared on sample calls (every surviving function must return the same values). Writes state/creator/lessons_contrast.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creator import action_student as A  # noqa: E402
from creator.curriculum import Lesson  # noqa: E402

# lib -> (expression over x, sample x)
LIBS: dict[str, tuple[str, Any]] = {
    "json": ("json.dumps(x)", [1, 2]), "textwrap": ("textwrap.dedent(x)", "  a\n  b"), "statistics": ("statistics.mean(x)", [1, 2, 6]),
    "string": ("string.capwords(x)", "ab cd"), "html": ("html.escape(x)", "<a>"), "shlex": ("shlex.split(x)", "a 'b c'"),
    "fnmatch": ("fnmatch.fnmatch(x, '*.py')", "m.py"), "math": ("math.sqrt(x)", 16), "base64": ("base64.b64encode(x.encode()).decode()", "hi"),
    "hashlib": ("hashlib.md5(x.encode()).hexdigest()", "a"), "zlib": ("zlib.crc32(x.encode())", "zz"), "csv": ("next(csv.reader([x]))", "a,b"),
}
# temp shapes: (assignment rhs over xs, return expr over v)
TEMPS = [("sum(xs)", "v * 2"), ("len(xs)", "v > 3"), ("max(xs)", "v + 1"), ("min(xs)", "-v"), ("sum(xs)", "v, 'ok'"), ("len(xs)", "v - 1")]

STYLES = ("plain", "const", "pair", "doc")


def build(style: int, lazy: Optional[str], unused: Optional[str], top: Optional[str], dead: Optional[str], tmp: Optional[tuple[str, str, str]],
          f: list[str]) -> tuple[str, list[tuple[str, tuple[Any, ...]]]]:
    """(source, sample calls). lazy: lib used by ONE function; unused: lib never used; top: lib used at module level; dead: uncalled fn;
    tmp: (function, variable, shape index as str)."""
    lines: list[str] = []
    calls: list[tuple[str, tuple[Any, ...]]] = []
    for lib in (lazy, top, unused):
        if lib:
            lines.append(f"import {lib}")
    lines.append("")
    if top:
        lines += ["", f"LIMIT = len({top}.__name__) + {style}"]
    elif style % 2:
        lines += ["", f"LIMIT = {10 + style}"]
    lines.append("")
    doc = style == 3
    if lazy:
        expr, sample = LIBS[lazy]
        lines += ["", f"def {f[0]}(x):"] + ([f'    """Apply {lazy} to x."""'] if doc else []) + [f"    return {expr}", ""]
        calls.append((f[0], (sample,)))
    if dead:
        lines += ["", f"def {dead}():"] + ([f'    """Leftover; nothing calls this."""'] if doc else []) + ["    return 42", ""]
    if tmp:
        fn, var, idx = tmp
        rhs, ret = TEMPS[int(idx)]
        lines += ["", f"def {fn}(xs):"] + ([f'    """Summarise xs."""'] if doc else []) + [f"    {var} = {rhs}", f"    return {ret.replace('v', var)}", ""]
        calls.append((fn, ([4, 1, 6, 3, 9],)))
    base = "a + b + LIMIT" if (top or style % 2) else "a + b"
    lines += ["", f"def {f[3]}(a, b):", f"    return {base}", ""]
    calls.append((f[3], (2, 5)))
    if style == 2:
        lines += ["", f"def {f[3]}_twice(a, b):", f"    return {f[3]}(a, b) * 2", ""]
        calls.append((f"{f[3]}_twice", (1, 2)))
    return "\n".join(lines).rstrip() + "\n", calls


def run_calls(src: str, calls: list[tuple[str, tuple[Any, ...]]]) -> dict[str, Any]:
    ns: dict[str, Any] = {"__name__": "contrast_mod"}
    exec(compile(src, "<mod>", "exec"), ns)
    return {fn: ns[fn](*args) for fn, args in calls}


# (pair, wanted, objective, reasoning-template). {u}=unused import, {d}=dead function, {l}=lazily-used lib, {t}/{fn}=temp var/function
OBJ = {
    "drop_unused_import": [
        "Tidy the import block: one module imported at the top is never referenced anywhere in the file, so take that import out.",
        "Delete the import {u}; no line of this module uses it.",
        "A top-level import here is dead weight (nothing in the module touches it). Remove only that import line.",
        "Fix the linter warning about an unused import in this module.",
        "One of the imports is never used. Drop it and leave every function alone.",
        "Cleanup: the module imports {u} but never mentions it again. Remove that import.",
    ],
    "remove_unused": [
        "Delete the dead function {d}: no code calls or references it.",
        "There is a leftover function that nothing calls (its name is {d}). Remove that function.",
        "Remove the unreferenced helper from this module; no caller exists for it.",
        "Dead code sweep: delete the function that is defined but never called anywhere.",
        "Get rid of the obsolete function {d}; it has no callers.",
        "Remove the function nobody uses. The imports are fine as they are.",
    ],
    "inline_temp": [
        "In {fn}, the temporary variable {t} is assigned once and read once; replace it with its expression.",
        "Inline the single-use local variable in {fn} so the function returns the expression directly.",
        "Simplify {fn}: drop the throwaway variable {t} by substituting its value where it is read.",
        "The local {t} in {fn} only exists to be returned from an expression; fold it in.",
    ],
    "lazy_import": [
        "Make startup lighter: {l} is used by one function only, so import it inside that function instead of at the top.",
        "Defer the {l} import into the function that needs it; the module should not load it until that function runs.",
        "Move the top-level import of {l} into its single user function (the library is still needed, just later).",
        "Reduce import time: load {l} lazily, inside the function that uses it.",
    ],
}

WHY = {
    ("drop_unused_import", "remove_unused"): "The import name `{u}` appears nowhere else in the module, so the thing to delete is an import: drop_unused_import({u}). "
                                              "The function `{d}` is also never called, but the objective is about imports; deleting an uncalled function is remove_unused, a different action.",
    ("remove_unused", "drop_unused_import"): "`{d}` is a function that is defined and never called or referenced: remove_unused({d}). "
                                              "The import `{u}` is also unused, but removing an import line is drop_unused_import; this objective asks for the function.",
    ("lazy_import", "inline_temp"): "`{l}` IS used (by `{f0}` only), so it is moved into that function: lazy_import({l}), nothing is deleted. "
                                     "Folding the single-use variable `{t}` in `{fn}` is inline_temp, a different action on different code.",
    ("inline_temp", "lazy_import"): "`{t}` is assigned once and read once inside `{fn}`, so inline it: inline_temp({fn},{t}). "
                                     "The `{l}` import is used by one function, which would be lazy_import, but the objective is about a local variable.",
    ("remove_unused", "inline_temp"): "`{d}` has no callers anywhere in the module, so the whole function goes: remove_unused({d}). "
                                       "The local `{t}` in `{fn}` is a different kind of cleanup (inline_temp) and the function `{fn}` is still wanted.",
    ("inline_temp", "remove_unused"): "`{t}` in `{fn}` is assigned once and read once: inline_temp({fn},{t}). "
                                       "The uncalled function `{d}` could be deleted (remove_unused) but the objective asks to simplify a variable, not delete a function.",
    ("lazy_import", "drop_unused_import"): "`{l}` is still used (by `{f0}`), so it must not be deleted; it only moves into that function: lazy_import({l}). "
                                            "The import `{u}` is never used anywhere and would be drop_unused_import, which the objective does not ask for.",
    ("drop_unused_import", "lazy_import"): "`{u}` appears nowhere else in the module, so the import can simply be deleted: drop_unused_import({u}). "
                                            "The import `{l}` is used by `{f0}`, so it is not unused and moving it would be lazy_import, a different action.",
    ("lazy_import", "remove_unused"): "`{l}` is used by `{f0}` and stays in use, it just loads later: lazy_import({l}). "
                                       "The dead function `{d}` would be remove_unused; the objective is about loading cost, not deleting code.",
    ("remove_unused", "lazy_import"): "`{d}` is never called, so delete the whole function: remove_unused({d}). "
                                       "The `{l}` import is live (used by `{f0}`); making it lazy is lazy_import and is not what was asked.",
}

PAIRS: list[tuple[str, str, int]] = [                       # (wanted, other, count)
    ("drop_unused_import", "remove_unused", 6), ("remove_unused", "drop_unused_import", 6),
    ("lazy_import", "inline_temp", 3), ("inline_temp", "lazy_import", 3),
    ("remove_unused", "inline_temp", 3), ("inline_temp", "remove_unused", 3),
    ("lazy_import", "drop_unused_import", 3), ("drop_unused_import", "lazy_import", 3),
    ("lazy_import", "remove_unused", 2), ("remove_unused", "lazy_import", 2),
]
LIBL = list(LIBS)
FN = ["parse", "render", "load", "scan", "emit", "build", "merge", "pack", "split", "route", "check", "fold", "audit", "encode", "stage", "probe"]
VARS = ["tmp", "acc", "res", "val", "buf", "out", "cur", "tot", "n", "size", "peak"]
DEADN = ["{w}_helper", "old_{w}", "legacy_{w}", "{w}_v1", "debug_{w}", "unused_{w}", "{w}_compat"]


def make(scale: int = 1) -> list[Lesson]:
    out: list[Lesson] = []
    i = 0
    for wanted, other, count in PAIRS:
        for j in range(count * scale):
            i += 1
            L = [LIBL[(i * 3 + k * 5) % len(LIBL)] for k in range(3)]
            while len(set(L)) < 3:
                L = [LIBL[(LIBL.index(x) + 1) % len(LIBL)] if L.count(x) > 1 else x for x in L]
            f = [FN[(i * 5 + k * 3) % len(FN)] for k in range(4)]
            if len(set(f)) < 4:
                f = [FN[(i + k) % len(FN)] for k in range(4)]
            kinds = {wanted, other}
            lazy = L[0] if "lazy_import" in kinds else None
            unused = L[2] if "drop_unused_import" in kinds else None
            top = L[1] if (i % 3 == 0 or "lazy_import" in kinds and i % 2) else None
            dead = DEADN[i % len(DEADN)].format(w=f[1]) if "remove_unused" in kinds else None
            tmp = (f[2], VARS[i % len(VARS)], str(i % len(TEMPS))) if "inline_temp" in kinds else None
            src, calls = build(i % 4, lazy, unused, top, dead, tmp, f)
            cands = A.enumerate_actions(src, "mod.py")
            want = {"drop_unused_import": f"drop_unused_import({unused})", "remove_unused": f"remove_unused({dead})",
                    "inline_temp": f"inline_temp({tmp[0]},{tmp[1]})" if tmp else "", "lazy_import": f"lazy_import({lazy})"}[wanted]
            oth = {"drop_unused_import": f"drop_unused_import({unused})", "remove_unused": f"remove_unused({dead})",
                   "inline_temp": f"inline_temp({tmp[0]},{tmp[1]})" if tmp else "", "lazy_import": f"lazy_import({lazy})"}[other]
            shorts = [c.short() for c in cands]
            assert want in shorts and oth in shorts, (wanted, other, shorts, src)
            act = cands[shorts.index(want)]
            after = A.apply_action(src, act)
            assert after is not None
            before_r, after_r = run_calls(src, calls), run_calls(after, [c for c in calls if c[0] in after])
            ok = all(before_r[k] == v for k, v in after_r.items()) and set(after_r) == set(before_r)
            ctx = dict(u=unused, d=dead, l=lazy, t=tmp[1] if tmp else None, fn=tmp[0] if tmp else None, f0=f[0])
            objs = OBJ[wanted]
            obj = objs[(j * 2 + i) % len(objs)].format(**ctx)
            reasoning = WHY[(wanted, other)].format(**ctx)
            out.append(Lesson(lesson_id=f"contrast-{i:02d}-{wanted}-vs-{other}", package_id=f"contrast-{i:02d}", component="contrast",
                              task_kind="shrink", objective=obj, context=f"Confusable pair: {wanted} (wanted) vs {other}. Candidates: " + "; ".join(shorts),
                              files_before={"mod.py": src}, files_after={"mod.py": after}, reasoning=reasoning, solver="claude", adopted=ok,
                              verdict="behaviour-preserving (exec-compared sample calls)" if ok else "exec comparison FAILED", claimed_done=True,
                              at="2026-10-02T01:10:00"))
    return out


def main() -> int:
    les = make()
    path = ROOT / "state/creator/lessons_contrast.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for l in les:
            fh.write(json.dumps(l.to_dict(), sort_keys=True) + "\n")
    from collections import Counter
    print(len(les), "lessons,", sum(1 for l in les if l.adopted), "adopted")
    print(Counter(l.lesson_id.split("-", 2)[2] for l in les))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
