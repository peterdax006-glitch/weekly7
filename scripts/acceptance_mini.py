"""Section-87 acceptance in miniature (contract C62 sections 3, 4, 87; canon C55, C63; C69 ledger W-07). IMPLEMENTED - NOT VALIDATED.

Runs learner.run_acceptance on the planted world for several seeds (default 3, 4, 5) and writes
state/research/acceptance_mini/{report.md,summary.json}. Every number comes from the run; nothing is typed. The planted world is
synthetic, so this is a wiring/behaviour check of the learner, not evidence about markets.

F07 additions: `--null` runs the same protocol on a world with no planted truth (two noise cells only) beside every seed; an
improvement there is a FALSE improvement and must stay 0.  `--learning-claim` sets the PromotionGate's learning-claim mode (the
learner's default is 'enforce').  Every seed carries the truth trace (learner.truth_trace): the first stage at which each planted
item's signal was lost.  `--diagnose A B ...` renders state/research/acceptance_mini/diagnosis.md from saved summary.json files.

usage: acceptance_mini.py [--seeds 3 4 5] [--weeks 50] [--stocks 40] [--null] [--learning-claim enforce|record] [--out DIR]
       acceptance_mini.py --diagnose LABEL=DIR [LABEL=DIR ...] [--out DIR]"""
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

OUT = ROOT / "state" / "research" / "acceptance_mini"


def spec_and_config(weeks, stocks, noise=False):
    """The miniature the learner's own tests use (tests/test_learning_learner.py): one source of truth for its relaxed gates."""
    import test_learning_learner as T
    return T.mini_spec(weeks, stocks, noise=noise), T.make_cfg


def score_row(s):
    f = lambda x: None if x is None else float(x)
    return {"n_long": int(s.n_long), "n_with_knowledge": int(s.n_with_knowledge), "mean_edge": f(s.mean_edge),
            "t_stat": f(s.t_stat), "hit_rate": f(s.hit_rate)}


def run_seed(seed, weeks, stocks, noise=False, learning_claim="enforce"):
    spec, make_cfg = spec_and_config(weeks, stocks, noise)
    t0 = time.time()
    holder = {}

    def make():
        L = LN.LegitimateLearner(make_cfg(seed=seed, learning_claim=learning_claim), code_hash_fn=lambda: "acceptance-mini")
        holder.setdefault("first_trained", L)
        return L

    rep = LN.run_acceptance(spec, seed, make)
    lesson = holder["first_trained"]                   # run_acceptance builds the lesson learner first
    return {"seed": seed, "world": "null" if noise else "planted", "learning_claim": learning_claim, "seconds": round(time.time() - t0, 1),
            "improved": bool(rep.improved()), "verdict": rep.protocol.verdict,
            "items": rep.items, "production": rep.production, "learned_weeks": rep.learned_weeks, "probe_weeks": rep.probe_weeks,
            "lesson": score_row(rep.lesson), "control": score_row(rep.control), "improvement_vs_none": rep.improvement_vs_none,
            "improvement_vs_control": rep.improvement_vs_control, "identity_invariant": rep.identity_invariant,
            "changed": rep.protocol.n_changed, "changed_with_knowledge": rep.protocol.n_changed_with_knowledge,
            "changed_without_knowledge": rep.protocol.n_changed_without_knowledge, "truth": dict(rep.truth),
            "skill": {k: rep.skill.get(k) for k in ("status", "n", "mean_edge", "p")}, "trace": list(getattr(rep, "trace", ())),
            "counters": {k: v for k, v in lesson.counters.items() if k in ("predictions", "shadow_only_predictions", "recovered_from_degraded")},
            "gate_blockers": sorted({f for _, ok, fails in lesson._gate_log if not ok for f in fails}),
            "notes": list(rep.notes), "text": rep.render()}


def summarise(rows):
    real = [r for r in rows if r["world"] == "planted"]
    null = [r for r in rows if r["world"] == "null"]
    imp = [r["improvement_vs_none"] for r in real if r["improvement_vs_none"] is not None]
    imp_c = [r["improvement_vs_control"] for r in real if r["improvement_vs_control"] is not None]
    lost: dict[str, int] = {}
    for r in real:
        for t in r["trace"]:
            if t["planted_sign"]:
                lost[t["stage"]] = lost.get(t["stage"], 0) + 1
    return {"n_seeds": len(real), "n_improved": sum(r["improved"] for r in real),
            "n_invalid": sum(r["verdict"].startswith("INVALID") for r in rows),
            "n_identity_invariant": sum(r["identity_invariant"] is True for r in real),
            "mean_improvement_vs_none": float(np.mean(imp)) if imp else None,
            "mean_improvement_vs_none_all_seeds": float(np.sum(imp) / len(real)) if real else None,
            "mean_improvement_vs_control": float(np.mean(imp_c)) if imp_c else None,
            "total_changed_without_knowledge": sum(r["changed_without_knowledge"] for r in rows),
            "null_seeds": len(null), "null_false_improvements": sum(r["improved"] for r in null),
            "null_changed_decisions": sum(r["changed"] for r in null), "null_production": sum(r["production"] for r in null),
            "planted_true_items_by_stage": lost}


def fmt(x, spec="+.5f"):
    return "n/a" if x is None else format(x, spec)


def markdown(rows, summ, prov, args):
    out = ["# Section-87 acceptance, miniature", "", "Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world; numbers are produced by "
           "`scripts/acceptance_mini.py`, none are typed.", "",
           f"- seeds {args.seeds}, {args.weeks} weeks, {args.stocks} stocks, learning_claim `{args.learning_claim}`; code hash "
           f"`{prov.get('code_hash')}`; git `{prov.get('git_commit')}`",
           f"- improved in {summ['n_improved']} of {summ['n_seeds']} seeds; INVALID verdicts {summ['n_invalid']}; identity-invariant "
           f"{summ['n_identity_invariant']}; decisions changed without knowledge behind them: {summ['total_changed_without_knowledge']}",
           f"- mean improvement vs no lesson (seeds that picked): {summ['mean_improvement_vs_none']}; over all seeds (0 where no pick): "
           f"{summ['mean_improvement_vs_none_all_seeds']}; vs shuffled-outcome control: {summ['mean_improvement_vs_control']}",
           f"- null world (no planted truth): {summ['null_seeds']} seeds, false improvements {summ['null_false_improvements']}, "
           f"changed decisions {summ['null_changed_decisions']}, items in production {summ['null_production']}",
           f"- planted true items by the stage their signal was lost at: {summ['planted_true_items_by_stage']}", "",
           "| seed | world | verdict | items | production | picks | mean edge | t | vs none | vs control | identity-invariant |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['seed']} | {r['world']} | {r['verdict']} | {r['items']} | {r['production']} | {r['lesson']['n_long']} | "
                   f"{fmt(r['lesson']['mean_edge'])} | {fmt(r['lesson']['t_stat'], '+.2f')} | {fmt(r['improvement_vs_none'])} | "
                   f"{fmt(r['improvement_vs_control'])} | {r['identity_invariant']} |")
    for r in rows:
        out += ["", f"## seed {r['seed']}, {r['world']} world ({r['seconds']} s)", "", "```", r["text"], "```"]
    return "\n".join(out) + "\n"


def run(args):
    rows = []
    for s in args.seeds:
        rows.append(run_seed(s, args.weeks, args.stocks, False, args.learning_claim))
        print(json.dumps({k: rows[-1][k] for k in ("seed", "world", "verdict", "production", "improvement_vs_none")}), flush=True)
        if args.null:
            rows.append(run_seed(s, args.weeks, args.stocks, True, args.learning_claim))
            print(json.dumps({k: rows[-1][k] for k in ("seed", "world", "verdict", "production", "improvement_vs_none")}), flush=True)
    summ = summarise(rows)
    prov = provenance.stamp({"seeds": args.seeds, "weeks": args.weeks, "stocks": args.stocks, "null": args.null,
                             "learning_claim": args.learning_claim}, seed=args.seeds[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"label": LN.LABEL, "summary": summ, "seeds": [{k: v for k, v in r.items() if k != "text"} for r in rows],
                                                  "provenance": {k: v for k, v in prov.items() if k != "code_files"}}, indent=1, default=str),
                                      encoding="utf-8")
    (out / "report.md").write_text(markdown(rows, summ, prov, args), encoding="utf-8")
    print(json.dumps(summ))


# ------------------------------------------------------------------------------------------------ diagnosis.md renderer

def load(paths) -> dict:
    """One run from one or several comma-separated output dirs (parallel shards of one configuration); shards are merged and
    re-summarised by the same `summarise`, old-format summaries (no 'world' field) count as planted."""
    parts = [json.loads((Path(p) / "summary.json").read_text(encoding="utf-8")) for p in str(paths).split(",")]
    if len(parts) == 1:
        return parts[0]
    rows = [dict(r, world=r.get("world", "planted")) for js in parts for r in js["seeds"]]
    return {"label": parts[0]["label"], "summary": summarise(rows), "seeds": rows, "provenance": [js["provenance"] for js in parts]}


def trace_table(runs: dict) -> list[str]:
    out = ["| run | seed | item | kind | pattern | sign | belief | role | lifecycle | skill | weighted rows | LONG rows | outcome | lost at | why |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, js in runs.items():
        for r in js["seeds"]:
            if r.get("world", "planted") != "planted":
                continue
            for t in r.get("trace", []):
                out.append(f"| {label} | {r['seed']} | {t['item']} | {t['kind']} | {t['pattern']} | {t['planted_sign']:+d} | {t['belief_mean']} | "
                           f"{t['role']} | {t['lifecycle']} | {t['skill']} | {t['weighted_rows']} | {t['long_rows']} | {t['outcome']} | "
                           f"{t['stage']} | {t['why'][:90]} |")
    return out


def before_after(runs: dict) -> list[str]:
    out = ["| run | seeds | improved | mean vs none (picking seeds) | mean vs none (all seeds) | mean vs control | null seeds | "
           "null false improvements | null changed decisions | changed without knowledge |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label, js in runs.items():
        s = js["summary"]
        imp = [r["improvement_vs_none"] for r in js["seeds"] if r.get("world", "planted") == "planted" and r["improvement_vs_none"] is not None]
        n = s["n_seeds"]
        out.append(f"| {label} | {n} | {s['n_improved']} | {fmt(s['mean_improvement_vs_none'])} | {fmt(float(np.sum(imp)) / n if n else None)} | "
                   f"{fmt(s.get('mean_improvement_vs_control'))} | {s.get('null_seeds', 'not run')} | {s.get('null_false_improvements', 'not run')} | "
                   f"{s.get('null_changed_decisions', 'not run')} | {s['total_changed_without_knowledge']} |")
    return out


def per_seed(runs: dict) -> list[str]:
    out = ["| run | seed | world | verdict | production | picks | vs none | skill at probe | gate blockers |", "|---|---|---|---|---|---|---|---|---|"]
    for label, js in runs.items():
        for r in js["seeds"]:
            sk = r.get("skill", {})
            out.append(f"| {label} | {r['seed']} | {r.get('world', 'planted')} | {r['verdict']} | {r['production']} | {r['lesson']['n_long']} | "
                       f"{fmt(r['improvement_vs_none'])} | {sk.get('status', 'n/a')} (n={sk.get('n', 'n/a')}) | "
                       f"{', '.join(r.get('gate_blockers', [])) or '-'} |")
    return out


def diagnose(pairs, out_dir, narrative_path=None):
    runs = {}
    for p in pairs:
        label, _, path = p.partition("=")
        runs[label] = load(path)
    lines = ["# F07 learner diagnosis (C69 W-07; C62 C03, E04, I16, I18)", "",
             "Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world only (C63). Every table below is rendered by "
             "`scripts/acceptance_mini.py --diagnose` from the summary.json files named in its header; no number is typed.", "",
             "Runs: " + ", ".join(f"`{k}` = {Path(p.partition('=')[2]).as_posix()}" for k, p in zip(runs, pairs)), ""]
    if narrative_path is not None and Path(narrative_path).exists():
        lines += [Path(narrative_path).read_text(encoding="utf-8").rstrip(), ""]
    lines += ["## Before / after", ""] + before_after(runs) + ["", "## Per seed", ""] + per_seed(runs)
    lines += ["", "## Truth trace: each planted item through store -> retrieval -> knowledge -> decision -> outcome", "",
              "Stages, in order: " + " -> ".join(LN.TRACE_STAGES) + ". `outcome` = planted sign x mean realised excess return of the probe "
              "rows the item carried (> 0: the probe world paid the call).", ""] + trace_table(runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'diagnosis.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--weeks", type=int, default=50)
    ap.add_argument("--stocks", type=int, default=40)
    ap.add_argument("--null", action="store_true", help="also run the no-truth world beside every seed")
    ap.add_argument("--learning-claim", default="enforce", choices=("off", "record", "enforce"))
    ap.add_argument("--diagnose", nargs="+", metavar="LABEL=DIR", help="render diagnosis.md from saved runs instead of running")
    ap.add_argument("--narrative", default=None, help="markdown file inserted above the rendered tables of diagnosis.md")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    if args.diagnose:
        diagnose(args.diagnose, args.out, args.narrative)
    else:
        run(args)


if __name__ == "__main__":
    main()
