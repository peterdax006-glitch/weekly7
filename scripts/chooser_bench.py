"""Held-out comparison of choosing policies on the hard action-choice bench (fresh seeds), with Wilson CIs.

Stage 1 (--picks): the 1.5B model's raw picks (rich labels + 4 shots, temperature 0, through creator.generator.LocalModel only) are
cached per seed in a JSON file. Stage 2 (--eval): every policy is computed offline from the cached picks:
lexical, model+lexical (the current ActionStudent default), chooser alone, chooser+model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import action_choice_bench as B  # noqa: E402
from creator import action_student as A  # noqa: E402
from creator import chooser as C  # noqa: E402


def get_picks(seeds: list[int], n: int, cache: Path, call_timeout: float = 120.0) -> dict[str, list[Optional[int]]]:
    have: dict[str, list[Optional[int]]] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [s for s in seeds if str(s) not in have]
    if todo:
        from creator import generator as G
        with G.LocalModel(startup_s=300.0) as llm:
            for s in todo:
                got = B._picks(llm, B.generate_tasks(n, s, True), True, 4, call_timeout, f"seed{s}")
                if got.count(None) > n // 5:                      # a dead server is not a model that guessed nothing: do not cache it
                    raise RuntimeError(f"seed {s}: {got.count(None)} of {n} model calls failed or gave no choice")
                have[str(s)] = got
                cache.write_text(json.dumps(have), encoding="utf-8", newline="\n")
    return have


def policies(tasks: list[B.Task], mp: list[Optional[int]], ch: C.Chooser, weight: float) -> dict[str, list[Optional[int]]]:
    srcs = [{"mod.py": t.src} for t in tasks]
    return {
        "lexical": [A.lexical_pick(t.objective, t.candidates) for t in tasks],
        "model": list(mp),
        "model+lexical (current default)": [A.combine_choice(p, t.objective, t.candidates) for t, p in zip(tasks, mp)],
        "chooser alone": [ch.pick(t.objective, t.candidates, s) for t, s in zip(tasks, srcs)],
        f"chooser+model (w={weight})": [ch.pick(t.objective, t.candidates, s, p, weight) for t, s, p in zip(tasks, srcs, mp)],
    }


def score(tasks: list[B.Task], pols: dict[str, list[Optional[int]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, pk in pols.items():
        c = sum(p == t.correct for t, p in zip(tasks, pk))
        out[k] = {"correct": c, "n": len(tasks), "acc": round(c / len(tasks), 3), "ci95": B._wilson(c, len(tasks))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="401,402,403")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--cache", default=str(ROOT / "state/creator/chooser_bench_picks.json"))
    ap.add_argument("--chooser", default=str(C.DEFAULT_PATH))
    ap.add_argument("--weights", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--picks-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "state/creator/chooser_bench.json"))
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    picks = get_picks(seeds, a.n, Path(a.cache))
    if a.picks_only:
        return 0
    ch = C.Chooser.load(Path(a.chooser))
    assert ch is not None and ch.trained, "no trained chooser"
    tasks = [t for s in seeds for t in B.generate_tasks(a.n, s, True)]
    mp = [p for s in seeds for p in picks[str(s)]]
    res: dict[str, Any] = {"seeds": seeds, "N": len(tasks), "chooser_meta": ch.meta, "by_weight": {}}
    for w in [float(x) for x in a.weights.split(",")]:
        res["by_weight"][str(w)] = score(tasks, {"chooser+model": [ch.pick(t.objective, t.candidates, {"mod.py": t.src}, p, w) for t, p in zip(tasks, mp)]})
    res["policies"] = score(tasks, policies(tasks, mp, ch, C.MODEL_WEIGHT))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    for k, v in res["policies"].items():
        print(f"{k:40s} {v['correct']}/{v['n']} acc={v['acc']} ci={v['ci95']}")
    for w, v in res["by_weight"].items():
        print("w", w, v["chooser+model"]["acc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
