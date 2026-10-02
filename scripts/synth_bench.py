"""Benchmark creator.synth on the dev feature tasks: before (legacy first-passing-candidate), after (abstaining), and the
leave-one-out versions (idioms that fire on exactly one dev task are disabled for that task). Usage: synth_bench.py [--json]"""
from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from creator import devbench as D  # noqa: E402
from creator import synth as S  # noqa: E402


def legacy_synthesize(spec: S.Spec, disabled: frozenset[str] = frozenset()) -> Optional[tuple[str, str]]:
    for label, body in S.candidates(spec, disabled):
        if S.passes(spec, body):
            return label, body
    if spec.examples:
        expr = S.enumerate_exprs(spec)
        if expr:
            return "enum", f"return {expr}"
    return None


def feature_tasks() -> list[D.Task]:
    return [t for t in D.load_tasks() if t.split == "dev" and t.category == "feature"]


def fire_map(tasks: list[D.Task]) -> dict[str, set[str]]:
    """idiom name -> ids of dev tasks whose docstring gate matches (arity-fit and vocabulary)."""
    import ast
    fires: dict[str, set[str]] = {}
    for t in tasks:
        for p in sorted(Path(t.visible_dir).rglob("*.py")):
            if "tests" in p.relative_to(t.visible_dir).parts:
                continue
            for fn in S._stub_functions(ast.parse(p.read_text(encoding="utf-8"))):
                spec = S.Spec(fn.name, tuple(a.arg for a in fn.args.args), ast.get_docstring(fn) or "", ())
                for lbl, _ in S.candidates(spec):
                    fires.setdefault(lbl.split(":", 1)[1], set()).add(t.id)
    return fires


def run(tasks: list[D.Task], solver_for: Any, legacy: bool) -> dict[str, str]:
    m = D.load_manifest()
    orig = S.synthesize
    if legacy:
        S.synthesize = legacy_synthesize  # type: ignore[assignment]
    try:
        with ThreadPoolExecutor(2) as ex:
            futs = {t.id: ex.submit(D.run_task, t, solver_for(t), m, None, "self-synth") for t in tasks}
            return {k: f.result().outcome for k, f in futs.items()}
    finally:
        S.synthesize = orig  # type: ignore[assignment]


def main() -> int:
    tasks = feature_tasks()
    fires = fire_map(tasks)
    only = {t.id: frozenset(n for n, ids in fires.items() if ids == {t.id}) for t in tasks}
    rows: dict[str, dict[str, str]] = {}
    rows["before"] = run(tasks, lambda t: S.SynthSolver(), True)
    rows["after"] = run(tasks, lambda t: S.SynthSolver(), False)
    rows["loo_before"] = run(tasks, lambda t: S.SynthSolver(only[t.id]), True)
    rows["loo_after"] = run(tasks, lambda t: S.SynthSolver(only[t.id]), False)
    print(f"n={len(tasks)} feature dev tasks")
    for k, r in rows.items():
        c = Counter(r.values())
        print(f"{k:11s} solved={c['SOLVED']:2d} false_completion={c['FALSE_COMPLETION']:2d} unsolved={c['UNSOLVED']:2d} other="
              f"{sum(v for o, v in c.items() if o not in ('SOLVED', 'FALSE_COMPLETION', 'UNSOLVED'))}")
    for t in tasks:
        print(t.id, {k: rows[k][t.id][:5] for k in rows}, sorted(only[t.id]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
