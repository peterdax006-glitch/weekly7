"""Section-87 acceptance in miniature (contract C62 sections 3, 4, 87; canon C55, C63). IMPLEMENTED - NOT VALIDATED.

Runs learner.run_acceptance on the planted world for several seeds (default 3, 4, 5) and writes
state/research/acceptance_mini/{report.md,summary.json}. Every number comes from the run; nothing is typed. The planted world is
synthetic, so this is a wiring/behaviour check of the learner, not evidence about markets.

usage: acceptance_mini.py [--seeds 3 4 5] [--weeks 50] [--stocks 40] [--out DIR]"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np

from engine import provenance
from engine.learning import learner as LN
from engine.learning import planted_world as PW


def spec_and_config(weeks, stocks):
    """The miniature the learner's own tests use (tests/test_learning_learner.py): one source of truth for its relaxed gates."""
    import test_learning_learner as T
    return T.mini_spec(weeks, stocks), T.make_cfg


def score_row(s):
    f = lambda x: None if x is None else float(x)
    return {"n_long": int(s.n_long), "n_with_knowledge": int(s.n_with_knowledge), "mean_edge": f(s.mean_edge),
            "t_stat": f(s.t_stat), "hit_rate": f(s.hit_rate)}


def run_seed(seed, weeks, stocks):
    spec, make_cfg = spec_and_config(weeks, stocks)
    t0 = time.time()
    rep = LN.run_acceptance(spec, seed, lambda: LN.LegitimateLearner(make_cfg(seed=seed), code_hash_fn=lambda: "acceptance-mini"))
    return {"seed": seed, "seconds": round(time.time() - t0, 1), "improved": bool(rep.improved()), "verdict": rep.protocol.verdict,
            "items": rep.items, "production": rep.production, "learned_weeks": rep.learned_weeks, "probe_weeks": rep.probe_weeks,
            "lesson": score_row(rep.lesson), "control": score_row(rep.control), "improvement_vs_none": rep.improvement_vs_none,
            "improvement_vs_control": rep.improvement_vs_control, "identity_invariant": rep.identity_invariant,
            "changed": rep.protocol.n_changed, "changed_with_knowledge": rep.protocol.n_changed_with_knowledge,
            "changed_without_knowledge": rep.protocol.n_changed_without_knowledge, "truth": dict(rep.truth),
            "notes": list(rep.notes), "text": rep.render()}


def summarise(rows):
    imp = [r["improvement_vs_none"] for r in rows if r["improvement_vs_none"] is not None]
    return {"n_seeds": len(rows), "n_improved": sum(r["improved"] for r in rows),
            "n_invalid": sum(r["verdict"].startswith("INVALID") for r in rows),
            "n_identity_invariant": sum(r["identity_invariant"] is True for r in rows),
            "mean_improvement_vs_none": float(np.mean(imp)) if imp else None,
            "total_changed_without_knowledge": sum(r["changed_without_knowledge"] for r in rows)}


def markdown(rows, summ, prov, args):
    out = ["# Section-87 acceptance, miniature", "", "Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by "
           "`scripts/acceptance_mini.py`, none are typed.", "",
           f"- seeds {args.seeds}, {args.weeks} weeks, {args.stocks} stocks; code hash `{prov.get('code_hash')}`; git `{prov.get('git_commit')}`",
           f"- improved in {summ['n_improved']} of {summ['n_seeds']} seeds; INVALID verdicts {summ['n_invalid']}; identity-invariant "
           f"{summ['n_identity_invariant']}; decisions changed without knowledge behind them: {summ['total_changed_without_knowledge']}",
           f"- mean improvement vs no lesson: {summ['mean_improvement_vs_none']}", "", "| seed | verdict | items | production | picks | "
           "mean edge | t | vs none | vs control | identity-invariant |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x: "n/a" if x is None else f"{x:+.5f}"
        out.append(f"| {r['seed']} | {r['verdict']} | {r['items']} | {r['production']} | {r['lesson']['n_long']} | {f(r['lesson']['mean_edge'])} | "
                   f"{f(r['lesson']['t_stat'])} | {f(r['improvement_vs_none'])} | {f(r['improvement_vs_control'])} | {r['identity_invariant']} |")
    for r in rows:
        out += ["", f"## seed {r['seed']} ({r['seconds']} s)", "", "```", r["text"], "```"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--weeks", type=int, default=50)
    ap.add_argument("--stocks", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "state" / "research" / "acceptance_mini"))
    args = ap.parse_args(argv)
    rows = [run_seed(s, args.weeks, args.stocks) for s in args.seeds]
    summ = summarise(rows)
    prov = provenance.stamp({"seeds": args.seeds, "weeks": args.weeks, "stocks": args.stocks}, seed=args.seeds[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"label": LN.LABEL, "summary": summ, "seeds": [{k: v for k, v in r.items() if k != "text"} for r in rows],
                                                  "provenance": {k: v for k, v in prov.items() if k != "code_files"}}, indent=1, default=str),
                                      encoding="utf-8")
    (out / "report.md").write_text(markdown(rows, summ, prov, args), encoding="utf-8")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
