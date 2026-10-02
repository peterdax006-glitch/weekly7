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


def _make_source(rng: random.Random, kind: str, hard: bool = False) -> tuple[str, str, str]:
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
    if hard:                                       # same sources; the objective describes the role and never names the target
        return src, {"lazy_import": "Make startup lighter: the one library that only a single function needs should be loaded there, not at the top.",
                     "remove_unused": "Remove the dead helper function (its name ends in _helper); nothing calls it.",
                     "inline_temp": "Inline the temporary variable that is used only once.",
                     "drop_unused_import": "Delete the import that nothing in the module uses."}[kind],             {"lazy_import": f"lazy_import({lazy})", "remove_unused": f"remove_unused({fn[1]}_helper)",
             "inline_temp": f"inline_temp({fn[2]},{var})", "drop_unused_import": f"drop_unused_import({unused})"}[kind]
    if kind == "lazy_import":
        return src, f"Load less code at start: defer the {lazy} import into the function that needs it.", f"lazy_import({lazy})"
    if kind == "remove_unused":
        return src, f"Remove the dead helper function {fn[1]}_helper; nothing calls it.", f"remove_unused({fn[1]}_helper)"
    if kind == "inline_temp":
        return src, f"Inline the single-use temporary variable {var} in {fn[2]}.", f"inline_temp({fn[2]},{var})"
    return src, f"Delete the unused import {unused}; the module never uses it.", f"drop_unused_import({unused})"


def generate_tasks(n: int, seed: int, hard: bool = False) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    tid = 0
    while len(tasks) < n:
        kind = KINDS[tid % len(KINDS)]
        tid += 1
        src, objective, want = _make_source(rng, kind, hard)
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


def _evidence(task: Task, a: A.Action) -> str:
    """Where the candidate's name is used in the task source (computed from the AST, never from the objective)."""
    import ast
    tree = ast.parse(task.src)
    p = a.p
    name = p.get("alias") or p.get("name") or p.get("variable")
    if a.kind == "inline_temp":
        fn = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == p["function"])
        n = sum(1 for x in ast.walk(fn) if isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Load))
        return f"'{name}' is read {n} time(s) in {p['function']}"
    users = [f.name for f in tree.body if isinstance(f, ast.FunctionDef) and f.name != name
             and any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(f))]
    top = any(isinstance(x, ast.Name) and x.id == name for st in tree.body if not isinstance(st, (ast.FunctionDef, ast.Import, ast.ImportFrom))
              for x in ast.walk(st))
    if a.kind == "remove_unused":
        return f"called by: {', '.join(users) or 'nobody'}" + ("; also used at module level" if top else "")
    where = users + (["module-level code"] if top else [])
    return f"'{name}' is used in: {', '.join(where) if where else 'nowhere'}"


def _question(task: Task, rich: bool, evidence: bool) -> str:
    cl = []
    for i, a in enumerate(task.candidates, 1):
        cl.append(f"{i}. [{a.path}] {a.label(rich)}" + (f"  (evidence: {_evidence(task, a)})" if evidence else ""))
    return (f"Task (code_change):\n{task.objective}\n\n```python\n{task.src}```\n\nCandidate actions:\n" + "\n".join(cl))


def shots(k: int, rich: bool, evidence: bool) -> list[dict[str, str]]:
    """k worked examples in EXACTLY the real question layout (own seed, never a dev/held-out seed), as chat turns; kinds rotate."""
    msgs: list[dict[str, str]] = []
    for t in generate_tasks(k, 9999):
        msgs += [{"role": "user", "content": _question(t, rich, evidence)},
                 {"role": "assistant", "content": f"CHOICE: {t.correct}\nWHY: that action does what the task asks."}]
    return msgs


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


def run_variant(tasks: list[Task], llm: Any, rich: bool = False, evidence: bool = False, nshot: int = 0, votes: int = 1,
                combined: bool = False, max_tokens: int = 60) -> dict[str, Any]:
    """One prompt variant over `tasks`. votes>1 = self-consistency (temperature 0.7, distinct seeds, majority). Also scores the lexical
    baseline and (combined=True) the lexical-prior+model policy from the same replies."""
    pre = shots(nshot, rich, evidence) if nshot else []
    model_p: list[Optional[int]] = []
    t0 = time.monotonic()
    for t in tasks:
        msgs = [{"role": "system", "content": A.SYSTEM}, *pre, {"role": "user", "content": _question(t, rich, evidence)}]
        ps = []
        for v in range(votes):
            r = str(llm.chat(msgs, max_tokens=max_tokens, temperature=0.0 if votes == 1 else 0.7, seed=v)) if votes > 1 else                 str(llm.chat(msgs, max_tokens=max_tokens, temperature=0.0))
            st, pk = classify(r, len(t.candidates))
            ps.append(pk[0] if pk else None)
        model_p.append(A.majority_vote(ps))
    n = len(tasks)
    acc = lambda picks: round(sum(p == t.correct for t, p in zip(tasks, picks)) / n, 3)  # noqa: E731
    lex = [A.lexical_pick(t.objective, t.candidates) for t in tasks]
    out: dict[str, Any] = {"model": acc(model_p), "lexical": acc(lex), "seconds": round(time.monotonic() - t0, 1), "n": n,
                           "by_kind": {k: sum(1 for t, p in zip(tasks, model_p) if t.kind == k and p == t.correct) for k in KINDS}}
    out["combined"] = acc([A.combine_choice(p, t.objective, t.candidates) for t, p in zip(tasks, model_p)])
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


VARIANTS: dict[str, dict[str, Any]] = {
    "base": {}, "rich": {"rich": True}, "evidence": {"rich": True, "evidence": True},
    "shots4": {"nshot": 4}, "rich_shots4": {"rich": True, "nshot": 4}, "evidence_shots4": {"rich": True, "evidence": True, "nshot": 4},
    "rich_shots4_vote3": {"rich": True, "nshot": 4, "votes": 3},
    "rich_vote3": {"rich": True, "votes": 3}, "evidence_vote3": {"rich": True, "evidence": True, "votes": 3},
}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def _picks(llm: Any, tasks: list[Task], rich: bool, nshot: int, call_timeout: float, tag: str) -> list[Optional[int]]:
    """Model picks per task (temperature 0). Every call has its own timeout (a hung/slow call counts as no pick) and prints progress."""
    pre = shots(nshot, rich, False) if nshot else []
    out: list[Optional[int]] = []
    for i, t in enumerate(tasks):
        msgs = [{"role": "system", "content": A.SYSTEM}, *pre, {"role": "user", "content": _question(t, rich, False)}]
        t1 = time.monotonic()
        try:
            r = str(llm.chat(msgs, max_tokens=60, temperature=0.0, timeout=call_timeout))
            pk = classify(r, len(t.candidates))[1]
            out.append(pk[0] if pk else None)
        except Exception as e:                        # noqa: BLE001 - a timeout is a miss, not a hang
            print(f"  [{tag}] task {i} call failed: {type(e).__name__}", flush=True)
            out.append(None)
        print(f"  [{tag}] {i + 1}/{len(tasks)} {time.monotonic() - t1:.1f}s pick={out[-1]} want={t.correct}", flush=True)
    return out


def _kind_of(t: Task, pick: Optional[int]) -> str:
    return "none" if pick is None else t.candidates[pick - 1].kind


def main_heldout(seeds: list[int], n: int, hard: bool, call_timeout: float, out_path: Path) -> dict[str, Any]:
    """The one-shot held-out measurement: best dev config (rich labels + 4 shots), base prompt, lexical, combined, random. Writes JSON
    after every stage so a stall never loses finished work."""
    from creator import generator as G
    tasks = [t for sd in seeds for t in generate_tasks(n, sd, hard)]
    N = len(tasks)
    res: dict[str, Any] = {"seeds": seeds, "n_per_seed": n, "hard": hard, "N": N, "policies": {}}
    pols: dict[str, list[Optional[int]]] = {"lexical": [A.lexical_pick(t.objective, t.candidates) for t in tasks]}
    rng = random.Random(4242)
    pols["random_sampled"] = [rng.randint(1, len(t.candidates)) for t in tasks]
    exp_rand = sum(1 / len(t.candidates) for t in tasks) / N

    def dump() -> None:
        res["policies"] = {}
        for name, pk in pols.items():
            k = sum(p == t.correct for t, p in zip(tasks, pk))
            ent: dict[str, Any] = {"correct": k, "n": N, "acc": round(k / N, 3), "ci95": _wilson(k, N),
                                   "per_seed": {str(sd): sum(p == t.correct for t, p in zip(tasks[i * n:(i + 1) * n], pk[i * n:(i + 1) * n]))
                                                for i, sd in enumerate(seeds)},
                                   "by_kind": {kd: sum(1 for t, p in zip(tasks, pk) if t.kind == kd and p == t.correct) for kd in KINDS}}
            ent["confusion"] = {}
            for t, p in zip(tasks, pk):
                if p != t.correct:
                    key = f"{t.kind}->{_kind_of(t, p)}"
                    ent["confusion"][key] = ent["confusion"].get(key, 0) + 1
            res["policies"][name] = ent
        res["random_expected"] = round(exp_rand, 3)
        res["tasks_per_kind"] = {kd: sum(t.kind == kd for t in tasks) for kd in KINDS}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")

    dump()
    with G.LocalModel(startup_s=300.0) as llm:
        for name, (rich, ns) in {"model_rich_shots4": (True, 4), "model_base": (False, 0)}.items():
            print(name, flush=True)
            pk = _picks(llm, tasks, rich, ns, call_timeout, name)
            pols[name] = pk
            pols["combined_" + name] = [A.combine_choice(p, t.objective, t.candidates) for t, p in zip(tasks, pk)]
            dump()
            print(name, res["policies"][name]["acc"], res["policies"]["combined_" + name]["acc"], flush=True)
    return res


def main_variants(seeds: list[int], n: int, configs: dict[str, dict[str, Any]], hard: bool = False) -> dict[str, Any]:
    from creator import generator as G
    res: dict[str, Any] = {}
    with G.LocalModel(startup_s=120.0) as llm:
        for name, cfg in configs.items():
            res[name] = {str(sd): run_variant(generate_tasks(n, sd, hard), llm, **cfg) for sd in seeds}
            print(name, {k: (v["model"], v["lexical"], v["combined"], v["seconds"]) for k, v in res[name].items()}, flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-lessons", action="store_true")
    ap.add_argument("--variants", help="comma list of variant names (dev tuning) from VARIANTS; needs --seeds")
    ap.add_argument("--hard", action="store_true", help="name-free objectives (role descriptions only)")
    ap.add_argument("--seeds", default="101,102,103")
    ap.add_argument("--heldout", action="store_true", help="one-shot held-out measurement of all policies with CIs (use with --seeds, --hard)")
    ap.add_argument("--call-timeout", type=float, default=90.0)
    ap.add_argument("--out", default=str(ROOT / "state/creator/action_choice_bench.json"))
    args = ap.parse_args()
    from creator import generator as G
    t0 = time.monotonic()
    if args.heldout:
        main_heldout([int(x) for x in args.seeds.split(",")], args.n, args.hard, args.call_timeout, Path(args.out))
        return 0
    if args.variants:
        res = main_variants([int(x) for x in args.seeds.split(",")], args.n, {k: VARIANTS[k] for k in args.variants.split(",")}, args.hard)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
        return 0
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
