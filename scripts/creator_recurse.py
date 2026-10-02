"""Run N recursion steps of the Creator on the REAL workload (devbench DEV split only; C77 sec 53-54, CR196-198).

    python scripts/creator_recurse.py --steps 2 --seed 1
Each step finds a weakness in the recursion ledger (or, with --probe, tries the next untried process parameter), A/B tests the
changed process on a bounded subset of dev tasks, adopts it only on IMPROVEMENT, and writes the process in force to
state/creator/process.json, which scripts/creator_swarm.py make_worker() reads. A later step uses seed+iteration, so it runs under
different conditions. The devbench holdout is never touched. Never pushes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import power as PW  # noqa: E402
from creator import process_levers as PL  # noqa: E402
from creator import synth_tasks as ST  # noqa: E402
from creator import recursion as R  # noqa: E402
from creator.ledger import Ledger  # noqa: E402

STATE = ROOT / "state" / "creator"


def probe_change(process: R.ProcessConfig, tried: set[tuple[str, int]]) -> R.Change | None:
    for param in R.PARAMS:
        cur = getattr(process, param)
        if cur + 1 <= R.ME.BOUNDS[param][1] and (param, cur + 1) not in tried:
            return R.Change(param, cur, cur + 1, f"probe: raise {param} {cur} -> {cur + 1}")
    return None


def summarize(rep: R.StepReport, wl: PL.DevWorkload, seed: int, min_effect: float = 0.0) -> dict:
    """The step as one JSON line: the choice arm (primary chunks) and the disjoint confirmation arm, both with their intervals."""
    d = rep.detail or {}
    out: dict = {"iteration": rep.iteration, "seed": seed, "weakness": rep.weakness and [rep.weakness.kind, rep.weakness.key, rep.weakness.count],
                 "change": rep.change and rep.change.rationale, "verdict": rep.verdict, "adopted": rep.adopted,
                 "process": rep.process_after.__dict__, "task_runs": wl.runs, "tasks_per_arm": wl.chunk * wl.reps + wl.confirm,
                 "reason": d.get("why")}
    if rep.change is not None:
        sb, sc = wl(rep.process_before), wl(rep.process_before.with_(rep.change.param, rep.change.new))
        out["choice_arm"] = {"n": wl.chunk * wl.reps, "base": sb.dev, "cand": sc.dev, "diff": d.get("diff"), "se": d.get("se"),
                             "ci": [d.get("lo"), d.get("hi")]}
        pb = sum(sb.dev) / len(sb.dev)
        n = wl.chunk                  # improvement_verdict does not shrink its SE with replicates: the chunk is the n that counts
        if 0 < pb < 1:
            out["power"] = {"p_base": pb, "n_per_chunk": n, "n_per_arm": wl.chunk * wl.reps, "detectable_effect_80pct": PW.detectable_effect(pb, n),
                            "n_needed_for_min_effect": PW.required_n(pb, min_effect) if min_effect > 0 else None}
        hd, hse = d.get("holdout_diff"), d.get("holdout_se")
        out["confirm_arm"] = {"n": wl.confirm, "base": sb.holdout, "cand": sc.holdout, "diff": hd, "se": hse,
                              "ci": [hd - 2 * hse, hd + 2 * hse] if hd is not None and hse is not None else None,
                              "disjoint_from_choice": not set(wl.confirm_ids) & {t for c in wl.chunks for t in c}}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--confirm", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="when the ledger shows no weakness, try the next untried parameter")
    ap.add_argument("--synth", action="store_true", help="use the cheap in-process synthetic repair tasks (creator.synth_tasks) instead of devbench dev tasks")
    ap.add_argument("--min-effect", type=float, default=0.0, help="smallest gain that matters (also lets NO_EFFECT be concluded)")
    ap.add_argument("--initial", default=None, help="starting process as research_budget,design_breadth,reviewer_depth,max_retries (a different operating point)")
    ap.add_argument("--nodes", default="14,90", help="synthetic tasks: size window of the functions, AST nodes lo,hi")
    ap.add_argument("--cache", type=Path, default=None, help="jsonl of per-(process, task) results: resume after an interrupted run")
    ap.add_argument("--ledger", type=Path, default=STATE / "recursion_ledger.jsonl")
    ap.add_argument("--process-file", type=Path, default=PL.DEFAULT_PATH)
    a = ap.parse_args(argv)
    led = Ledger(a.ledger, evidence_root=a.ledger.parent)
    initial = R.ProcessConfig(*map(int, a.initial.split(","))) if a.initial else None
    bench = ST.SynthBench(lo=int(a.nodes.split(",")[0]), hi=int(a.nodes.split(",")[1])) if a.synth else None
    for _ in range(a.steps):
        it = len(R.recorded_steps(led)) + 1
        if a.synth:
            wl = ST.workload(seed=a.seed + it - 1, chunk=a.chunk, reps=a.reps, confirm=a.confirm, cache_path=a.cache, bench=bench)
        else:
            wl = PL.DevWorkload(seed=a.seed + it - 1, chunk=a.chunk, reps=a.reps, confirm=a.confirm, cache_path=a.cache)
        forced = None
        if a.probe and not any(R.design_change(w, R.current_process(led, initial), R.tried_changes(led)) for w in R.find_weaknesses(led)):
            forced = probe_change(R.current_process(led, initial), R.tried_changes(led))
        rep = R.step(led, wl, forced=forced, initial=initial, min_effect=a.min_effect)
        PL.save_process(R.current_process(led, initial), a.process_file)
        print(json.dumps(summarize(rep, wl, a.seed + it - 1, a.min_effect)), flush=True)
        if rep.change is None:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
