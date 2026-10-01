"""Honest choosing-skill benchmark for ActionStudent: seeded synthetic tasks, shuffled candidate order, baselines.

Each task is a small module with several valid actions and an objective that names exactly one of them. The real prompting pieces of
creator.action_student (SYSTEM, enumerate_actions, Action.label, parse_choice) are reused; only the objective text and the candidate
order are the bench's. Position bias shows up as accuracy that tracks the 'always choose 1' baseline instead of beating it."""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creator import action_student as A  # noqa: E402

LIBS = ["csv", "json", "textwrap", "string", "statistics", "base64", "hashlib", "shlex", "html", "fnmatch"]
ATTR = {"csv": "reader", "json": "dumps", "textwrap": "dedent", "string": "capwords", "statistics": "mean", "base64": "b64encode",
        "hashlib": "md5", "shlex": "split", "html": "escape", "fnmatch": "fnmatch"}
NAMES = ["parse", "render", "load", "scan", "emit", "build", "merge", "pack", "split", "route", "check", "fold"]
VARS = ["tmp", "acc", "res", "val", "buf", "out", "cur", "tot"]
KINDS = ("lazy_import", "remove_unused", "inline_temp", "drop_unused_import")


@dataclasses.dataclass
class Task:
    tid: int
    src: str
    objective: str
    kind: str
    candidates: list[A.Action]            # already shuffled
    correct: int                           # 1-based index into candidates


def _make_source(rng: random.Random, kind: str) -> tuple[str, str, str]:
    """(source, objective, short() of the one correct action). Distractor actions of other kinds appear at random."""
    lazy, top, unused = rng.sample(LIBS, 3)
    fn = rng.sample(NAMES, 4)
    var = rng.choice(VARS)
    lines = [f"import {lazy}", f"import {top}"]
    if kind == "drop_unused_import" or rng.random() < 0.5:
        lines.append(f"import {unused}")
    has_dead = kind == "remove_unused" or rng.random() < 0.5
    has_tmp = kind == "inline_temp" or rng.random() < 0.5
    has_lazy = kind == "lazy_import" or rng.random() < 0.6
    lines += ["", f"LIMIT = len({top}.{ATTR[top]}.__name__)", ""]
    if has_lazy:
        lines += ["", f"def {fn[0]}(x):", f"    return {lazy}.{ATTR[lazy]}(x)", ""]
    if has_dead:
        lines += ["", f"def {fn[1]}_helper():", "    return 42", ""]
    if has_tmp:
        lines += ["", f"def {fn[2]}(xs):", f"    {var} = sum(xs)", f"    return {var} * 2", ""]
    lines += ["", f"def {fn[3]}(a, b):", "    return a + b + LIMIT", ""]
    src = "\n".join(lines).rstrip() + "\n"
    if kind == "lazy_import":
        return src, f"Load less code at start: defer the {lazy} import into the function that needs it.", f"lazy_import({lazy})"
    if kind == "remove_unused":
        return src, f"Remove the dead helper function {fn[1]}_helper; nothing calls it.", f"remove_unused({fn[1]}_helper)"
    if kind == "inline_temp":
        return src, f"Inline the single-use temporary variable {var} in {fn[2]}.", f"inline_temp({fn[2]},{var})"
    return src, f"Delete the unused import {unused}; the module never uses it.", f"drop_unused_import({unused})"


def generate_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    tid = 0
    while len(tasks) < n:
        kind = KINDS[tid % len(KINDS)]
        tid += 1
        src, objective, want = _make_source(rng, kind)
        cands = A.enumerate_actions(src, "mod.py")
        hits = [i for i, a in enumerate(cands) if a.short() == want]
        if len(hits) != 1 or len(cands) < 2:
            continue                                      # exactly one correct action among >= 2 candidates
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuffled = [cands[i] for i in order]
        correct = [i for i, a in enumerate(shuffled, 1) if a.short() == want][0]
        tasks.append(Task(len(tasks), src, objective, kind, shuffled, correct))
    return tasks


LESSONS = [
    ("Load less code at start: defer the json import into the function that needs it.", "lazy_import(json)"),
    ("Remove the dead helper function old_helper; nothing calls it.", "remove_unused(old_helper)"),
]


def build_prompt(task: Task, lessons: bool) -> str:
    parts: list[str] = []
    if lessons:
        parts += [f"Example task: {o}\nChosen actions: {c}" for o, c in LESSONS]
    parts.append(f"Task (code_change):\n{task.objective}\n\n```python\n{task.src}```")
    parts.append("Candidate actions:\n" + "\n".join(f"{i}. [{a.path}] {a.label()}" for i, a in enumerate(task.candidates, 1)))
    return "\n\n".join(parts)


def classify(reply: str, n: int) -> tuple[str, Optional[list[int]]]:
    """('ok'|'none'|'invalid', picks). 'none' is an explicit CHOICE: none; anything unparseable or out of range is invalid."""
    picks = A.parse_choice(reply, n)
    if picks:
        return "ok", picks
    m = re.search(r"CHOICE:\s*(.*)", reply, re.I)
    if m and re.fullmatch(r"none\.?", m.group(1).strip(), re.I):
        return "none", None
    return "invalid", None


def _score(tasks: list[Task], answers: list[Optional[list[int]]], statuses: list[str]) -> dict[str, Any]:
    n = len(tasks)
    right = [a == [t.correct] for t, a in zip(tasks, answers)]
    bypos: dict[int, list[bool]] = {}
    for t, r in zip(tasks, right):
        bypos.setdefault(t.correct, []).append(r)
    chosen_pos: dict[str, int] = {}
    for a in answers:
        if a:
            chosen_pos[str(a[0])] = chosen_pos.get(str(a[0]), 0) + 1
    return {"accuracy": sum(right) / n if n else 0.0, "correct": sum(right), "n": n,
            "none_rate": statuses.count("none") / n if n else 0.0, "invalid_rate": statuses.count("invalid") / n if n else 0.0,
            "accuracy_by_correct_position": {str(p): {"n": len(v), "accuracy": sum(v) / len(v)} for p, v in sorted(bypos.items())},
            "chosen_position_counts": dict(sorted(chosen_pos.items()))}


def baselines(tasks: list[Task], seed: int) -> dict[str, Any]:
    rng = random.Random(seed + 1)
    n = len(tasks)
    return {"always_choose_1": sum(t.correct == 1 for t in tasks) / n if n else 0.0,
            "uniform_random_expected": sum(1 / len(t.candidates) for t in tasks) / n if n else 0.0,
            "uniform_random_sampled": sum(rng.randint(1, len(t.candidates)) == t.correct for t in tasks) / n if n else 0.0}


def run_condition(tasks: list[Task], llm: Any, lessons: bool, max_tokens: int = 80) -> dict[str, Any]:
    answers: list[Optional[list[int]]] = []
    statuses: list[str] = []
    replies: list[str] = []
    t0 = time.monotonic()
    for t in tasks:
        reply = str(llm.chat([{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": build_prompt(t, lessons)}],
                             max_tokens=max_tokens, temperature=0.0))
        st, picks = classify(reply, len(t.candidates))
        statuses.append(st)
        answers.append(picks)
        replies.append(reply[:80])
    out = _score(tasks, answers, statuses)
    out["seconds"] = round(time.monotonic() - t0, 1)
    out["by_kind"] = {k: sum(1 for t, a in zip(tasks, answers) if t.kind == k and a == [t.correct]) for k in KINDS}
    out["sample_replies"] = replies[:5]
    return out


def run_bench(llm: Any, n: int = 30, seed: int = 7, with_lessons: bool = True) -> dict[str, Any]:
    tasks = generate_tasks(n, seed)
    res: dict[str, Any] = {"n_tasks": n, "seed": seed,
                           "correct_position_distribution": {str(p): sum(t.correct == p for t in tasks) for p in range(1, 6)},
                           "mean_candidates": sum(len(t.candidates) for t in tasks) / n if n else 0.0,
                           "baselines": baselines(tasks, seed), "no_lessons": run_condition(tasks, llm, False)}
    if with_lessons:
        res["with_lessons"] = run_condition(tasks, llm, True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-lessons", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "state/creator/action_choice_bench.json"))
    args = ap.parse_args()
    from creator import generator as G
    t0 = time.monotonic()
    with G.LocalModel(startup_s=120.0) as llm:
        res = run_bench(llm, args.n, args.seed, not args.no_lessons)
    res["wall_seconds"] = round(time.monotonic() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
