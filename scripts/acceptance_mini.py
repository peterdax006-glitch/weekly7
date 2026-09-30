"""Section-87 acceptance in miniature (contract C62 sections 3, 4, 87; canon C55, C63; C69 ledger W-07). IMPLEMENTED - NOT VALIDATED.

Runs learner.run_acceptance on the planted world for several seeds (default 3, 4, 5) and writes
state/research/acceptance_mini/{report.md,summary.json}. Every number comes from the run; nothing is typed. The planted world is
synthetic, so this is a wiring/behaviour check of the learner, not evidence about markets.

F07 additions: `--null` runs the same protocol on a world with no planted truth (two noise cells only) beside every seed; an
improvement there is a FALSE improvement and must stay 0.  `--learning-claim` sets the PromotionGate's learning-claim mode (the
learner's default is 'enforce').  Every seed carries the truth trace (learner.truth_trace): the first stage at which each planted
item's signal was lost.  `--diagnose A B ...` renders state/research/acceptance_mini/diagnosis.md from saved summary.json files.

F10 additions: `--years N` runs the multi-year planted world (planted_world.multi_year_spec: truths that persist, one that decays;
the null world the same length) so the learner's evidence card has forward-year folds; `--probe-weeks` bounds the scored probe;
`--null-only` makes a null shard; every seed records the degrade audit (each DEGRADE against the planted truth), the evidence-card
log (valid cards, first card the claim gate allowed, the blockers of the last one) and the last promotion attempt per pattern.
`--degrade-study` runs retirement.degrade_study per planted item and window; `--f10-report` renders f10_enforce_evidence.md.

usage: acceptance_mini.py [--seeds 3 4 5] [--weeks 50] [--stocks 40] [--years 5] [--probe-weeks 52] [--null | --null-only]
                          [--learning-claim enforce|record] [--out DIR]
       acceptance_mini.py --diagnose LABEL=DIR [LABEL=DIR ...] [--out DIR]
       acceptance_mini.py --degrade-study [--years 5] [--seeds 3] [--n-sims 100] [--out DIR]
       acceptance_mini.py --f10-report LABEL=DIR[,DIR] [...] [--narrative FILE] [--out DIR]"""
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


def spec_and_config(weeks, stocks, noise=False, years=0):
    """The miniature the learner's own tests use (tests/test_learning_learner.py): one source of truth for its relaxed gates.
    years > 0 (F10): planted_world.multi_year_spec - persisting truths plus one decaying item over `years` years, so the evidence
    card's forward-year folds exist; the null world has the same length."""
    import test_learning_learner as T
    if years:
        return PW.multi_year_spec(years, stocks, noise=noise), T.make_cfg
    return T.mini_spec(weeks, stocks, noise=noise), T.make_cfg


def score_row(s):
    f = lambda x: None if x is None else float(x)
    return {"n_long": int(s.n_long), "n_with_knowledge": int(s.n_with_knowledge), "mean_edge": f(s.mean_edge),
            "t_stat": f(s.t_stat), "hit_rate": f(s.hit_rate)}


def last_attempts(learner) -> dict:
    """pattern -> critical failures of its LAST promotion attempt (earlier attempts were made on less evidence)."""
    out = {}
    for kid, ok, fails in learner._gate_log:
        out[learner._pid_of.get(kid, kid)] = [] if ok else sorted(fails)
    return dict(sorted(out.items()))


def evidence_summary(log) -> dict:
    """The lesson learner's evidence-card log reduced to what the report needs: how many cards were valid, when the claim gate first
    allowed one, and the blockers of the last valid card (or the refusal reason of the last card when none was valid)."""
    valid = [e for e in log if e["valid"]]
    allowed = [e for e in valid if e["allowed"]]
    last = valid[-1] if valid else (log[-1] if log else None)
    return {"cards": len(log), "valid": len(valid), "allowed": len(allowed), "first_allowed": allowed[0]["now"] if allowed else None,
            "first_valid": valid[0]["now"] if valid else None, "last_now": None if last is None else last["now"],
            "last_blockers": [] if last is None else last["blockers"], "last_refusal": None if last is None or last["valid"] else last["why"],
            "last_untested": [] if last is None else last["untested"]}


def run_seed(seed, weeks, stocks, noise=False, learning_claim="enforce", years=0, probe_weeks=None):
    spec, make_cfg = spec_and_config(weeks, stocks, noise, years)
    t0 = time.time()
    holder = {}

    def make():
        L = LN.LegitimateLearner(make_cfg(seed=seed, learning_claim=learning_claim), code_hash_fn=lambda: "acceptance-mini")
        holder.setdefault("first_trained", L)
        return L

    rep = LN.run_acceptance(spec, seed, make, probe_weeks=probe_weeks)
    lesson = holder["first_trained"]                   # run_acceptance builds the lesson learner first
    return {"seed": seed, "world": "null" if noise else "planted", "learning_claim": learning_claim, "seconds": round(time.time() - t0, 1),
            "years": years, "weeks": spec.weeks, "retire_window": lesson.cfg.retire_window,
            "degrades": dict(rep.degrades), "evidence": evidence_summary(rep.evidence), "last_attempts": last_attempts(lesson),
            "improved": bool(rep.improved()), "verdict": rep.protocol.verdict,
            "items": rep.items, "production": rep.production, "learned_weeks": rep.learned_weeks, "probe_weeks": rep.probe_weeks,
            "lesson": score_row(rep.lesson), "control": score_row(rep.control), "improvement_vs_none": rep.improvement_vs_none,
            "improvement_vs_control": rep.improvement_vs_control, "identity_invariant": rep.identity_invariant,
            "changed": rep.protocol.n_changed, "changed_with_knowledge": rep.protocol.n_changed_with_knowledge,
            "changed_without_knowledge": rep.protocol.n_changed_without_knowledge, "truth": dict(rep.truth),
            "skill": {k: rep.skill.get(k) for k in ("status", "n", "mean_edge", "p")}, "trace": list(getattr(rep, "trace", ())),
            "counters": {k: v for k, v in lesson.counters.items() if k in ("predictions", "shadow_only_predictions", "recovered_from_degraded",
                                                                          "recovered_to_probation", "recovered_after_probation", "evidence_cards",
                                                                          "evidence_cards_refused") or k.startswith("retirement_")},
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
            "planted_true_items_by_stage": lost, **degrade_totals(rows)}


def degrade_totals(rows) -> dict:
    """F10: DEGRADE calls against the planted truth, summed over seeds: false ones (the item still worked), correct ones (noise or a
    decayed item), per planted kind; and on the null world (every item held there is noise, so every degrade there is correct)."""
    out = {"false_degrades": 0, "correct_degrades": 0, "null_degrades": 0, "false_degrades_by_kind": {}, "weeks_not_active_by_kind": {}}
    for r in rows:
        d = r.get("degrades") or {}
        if r["world"] == "null":
            out["null_degrades"] += d.get("correct_degrades", 0) + d.get("false_degrades", 0)
            continue
        out["false_degrades"] += d.get("false_degrades", 0)
        out["correct_degrades"] += d.get("correct_degrades", 0)
        for it in (d.get("items") or {}).values():
            k = it["kind"]
            out["false_degrades_by_kind"][k] = out["false_degrades_by_kind"].get(k, 0) + it["false_degrades"]
            out["weeks_not_active_by_kind"].setdefault(k, []).append(it["weeks_not_active"])
    return out


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
    """Every finished seed is written at once (summary.json, report.md, and one line of seeds.jsonl), so a killed shard loses only
    the seed in flight, and a restarted one skips the seeds it already has."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance.stamp({"seeds": args.seeds, "weeks": args.weeks, "stocks": args.stocks, "null": args.null,
                             "learning_claim": args.learning_claim, "years": args.years, "probe_weeks": args.probe_weeks,
                             "null_only": args.null_only}, seed=args.seeds[0])
    done = out / "seeds.jsonl"
    rows = [json.loads(l) for l in done.read_text(encoding="utf-8").splitlines() if l.strip()] if done.exists() else []
    have = {(r["seed"], r["world"]) for r in rows}
    jobs = [(s, w) for s in args.seeds for w in ((() if args.null_only else ("planted",)) + (("null",) if args.null or args.null_only else ()))]
    for s, world in jobs:
        if (s, world) in have:
            continue
        r = run_seed(s, args.weeks, args.stocks, world == "null", args.learning_claim, args.years, args.probe_weeks)
        rows.append(r)
        with done.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(json.dumps({k: r[k] for k in ("seed", "world", "verdict", "production", "improvement_vs_none")}), flush=True)
        write_outputs(out, rows, prov, args)
    write_outputs(out, rows, prov, args)


def write_outputs(out, rows, prov, args):
    summ = summarise(rows)
    (out / "summary.json").write_text(json.dumps({"label": LN.LABEL, "summary": summ, "seeds": [{k: v for k, v in r.items() if k != "text"} for r in rows],
                                                  "provenance": {k: v for k, v in prov.items() if k != "code_files"}}, indent=1, default=str),
                                      encoding="utf-8")
    (out / "report.md").write_text(markdown(rows, summ, prov, args), encoding="utf-8")
    print(json.dumps(summ), flush=True)


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


# ------------------------------------------------------------------------------------------------ F10: false-degrade study and report

STUDY_WINDOWS = (8, 13, 16, 26)


def degrade_study(args):
    """retirement.degrade_study for every planted item of the multi-year world, on the item's own weekly noise (planted_world.
    cell_weekly over the first two years, where every item is at full strength), for each retirement window at the unchanged
    degrade_t.  The decaying item is replayed with its planted multipliers: a DEGRADE before its change week is false, the first exit
    from ACTIVE after it is the detection.  Writes f10_degrade_study.{json,md}."""
    from engine.learning import retirement as RT
    years = args.years or 5
    world = PW.make_world(PW.multi_year_spec(years, args.stocks), seed=args.seeds[0])
    W = world.spec.weeks
    rows = []
    for it in world.spec.items:
        v = world.cell_weekly(it.item_id, (0, 2 * 52))
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        for win in STUDY_WINDOWS:
            pol = RT.RetirementPolicy(min_n=win, recover_min_n=2 * win)
            kw = dict(multipliers=it.multipliers(W), change_week=it.change_week(W)) if it.change_week(W) is not None else {}
            s = RT.degrade_study(mean if it.kind != PW.NOISE else 0.0, sd, W, pol, win, n_sims=args.n_sims, seed=args.seeds[0], **kw)
            rows.append({"item": it.item_id, "kind": it.kind, "weekly_mean": round(mean, 5), "weekly_sd": round(sd, 5), **s.as_dict()})
            print(json.dumps({k: rows[-1][k] for k in ("item", "window", "any_degrade", "share_weeks_not_active", "false_before_change")}), flush=True)
    prov = provenance.stamp({"years": years, "stocks": args.stocks, "seed": args.seeds[0], "n_sims": args.n_sims, "windows": STUDY_WINDOWS},
                            seed=args.seeds[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f10_degrade_study.json").write_text(json.dumps({"label": LN.LABEL, "rows": rows, "provenance": {k: v for k, v in prov.items()
                                                                                                          if k != "code_files"}}, indent=1, default=str), encoding="utf-8")
    (out / "f10_degrade_study.md").write_text("\n".join(degrade_study_table(rows)) + "\n", encoding="utf-8")
    print(f"wrote {out / 'f10_degrade_study.md'}")


def degrade_study_table(rows) -> list[str]:
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.2f}" if isinstance(x, float) else str(x))
    out = ["| item | kind | weekly mean | weekly sd | window | degrade_t | share of checks t < degrade_t | histories with a DEGRADE | "
           "DEGRADEs per year | share of weeks not ACTIVE | ever DORMANT | false DEGRADE before the change | decay detected | "
           "detection delay (weeks) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['item']} | {r['kind']} | {r['weekly_mean']:+.4f} | {r['weekly_sd']:.4f} | {r['window']} | {r['degrade_t']} | "
                   f"{f(r['per_check'])} | {f(r['any_degrade'])} | {f(r['degrades_per_year'])} | {f(r['share_weeks_not_active'])} | "
                   f"{f(r['ever_dormant'])} | {f(r['false_before_change'])} | {f(r['detected_after_change'])} | {f(r['detect_delay_median'])} |")
    return out


def f10_seed_table(runs: dict) -> list[str]:
    out = ["| run | seed | world | verdict | production | vs none | vs control | cards valid / built | first valid card | first card the "
           "claim gate allowed | claim-gate blockers (last card) | last promotion attempt per pattern |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, js in runs.items():
        for r in js["seeds"]:
            ev = r.get("evidence") or {}
            la = "; ".join(f"{p}: {','.join(f) or 'PROMOTED'}" for p, f in (r.get("last_attempts") or {}).items()) or "no attempt"
            blk = ", ".join(ev.get("last_blockers") or []) or (f"refused: {ev.get('last_refusal')}" if ev.get("last_refusal") else "-")
            out.append(f"| {label} | {r['seed']} | {r.get('world', 'planted')} | {r['verdict']} | {r['production']} | {fmt(r['improvement_vs_none'])} | "
                       f"{fmt(r['improvement_vs_control'])} | {ev.get('valid', 0)} / {ev.get('cards', 0)} | {ev.get('first_valid') or '-'} | "
                       f"{ev.get('first_allowed') or 'never'} | {blk} | {la[:200]} |")
    return out


def f10_degrade_table(runs: dict) -> list[str]:
    out = ["| run | seed | world | item | kind | registered (week) | DEGRADEs | of them FALSE | full recoveries | DORMANT | weeks not ACTIVE |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, js in runs.items():
        for r in js["seeds"]:
            for iid, it in ((r.get("degrades") or {}).get("items") or {}).items():
                out.append(f"| {label} | {r['seed']} | {r.get('world', 'planted')} | {iid} | {it['kind']} | {it['registered_week']} | {it['degrades']} | "
                           f"{it['false_degrades']} | {it['recoveries']} | {it['dormant']} | {it['weeks_not_active']} |")
    return out


def f10_report(pairs, out_dir, narrative_path=None):
    """f10_enforce_evidence.md from saved runs (each LABEL=DIR[,DIR...]): before/after, per seed with the gate that still blocks it,
    and every DEGRADE call against the planted truth.  Every number is read from summary.json; nothing is typed."""
    runs = {}
    for p in pairs:
        label, _, path = p.partition("=")
        runs[label] = load(path)
    lines = ["# F10: the learning claim under `enforce`, with the evidence supplied (C69 W-07; C62 C03, E04, I16, I18)", "",
             "Status: IMPLEMENTED - NOT VALIDATED. Planted synthetic world only (C63). Every table is rendered by "
             "`scripts/acceptance_mini.py --f10-report` from the summary.json files named here; no number is typed.", "",
             "Runs: " + ", ".join(f"`{k}` = {Path(p.partition('=')[2]).as_posix()}" for k, p in zip(runs, pairs)), ""]
    if narrative_path is not None and Path(narrative_path).exists():
        lines += [Path(narrative_path).read_text(encoding="utf-8").rstrip(), ""]
    lines += ["## Before / after", ""] + before_after(runs)
    lines += ["", "## Degrade calls against the planted truth (totals)", "", "| run | false DEGRADEs | correct DEGRADEs | null-world DEGRADEs | "
              "false by kind | weeks not ACTIVE by kind (per seed) |", "|---|---|---|---|---|---|"]
    for label, js in runs.items():
        s = js["summary"]
        lines.append(f"| {label} | {s.get('false_degrades', 'n/a')} | {s.get('correct_degrades', 'n/a')} | {s.get('null_degrades', 'n/a')} | "
                     f"{s.get('false_degrades_by_kind', {})} | {s.get('weeks_not_active_by_kind', {})} |")
    lines += ["", "## Per seed: what the claim gate saw and what still blocks", ""] + f10_seed_table(runs)
    lines += ["", "## Every planted item's lifecycle", ""] + f10_degrade_table(runs)
    lines += ["", "## Truth trace", "", "Stages, in order: " + " -> ".join(LN.TRACE_STAGES) + ".", ""] + trace_table(runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f10_enforce_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'f10_enforce_evidence.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--weeks", type=int, default=50)
    ap.add_argument("--stocks", type=int, default=40)
    ap.add_argument("--null", action="store_true", help="also run the no-truth world beside every seed")
    ap.add_argument("--learning-claim", default="enforce", choices=("off", "record", "enforce"))
    ap.add_argument("--years", type=int, default=0, help="F10: multi-year planted world of this many years (0 = the one-year miniature)")
    ap.add_argument("--probe-weeks", type=int, default=None, help="weeks of the disguised probe episode that are scored (default: all)")
    ap.add_argument("--null-only", action="store_true", help="run only the null world (one shard of a parallel run)")
    ap.add_argument("--degrade-study", action="store_true", help="F10: the false-degrade study on the multi-year world, then exit")
    ap.add_argument("--n-sims", type=int, default=100, help="histories per cell of the degrade study")
    ap.add_argument("--f10-report", nargs="+", metavar="LABEL=DIR", help="F10: render f10_enforce_evidence.md from saved runs")
    ap.add_argument("--diagnose", nargs="+", metavar="LABEL=DIR", help="render diagnosis.md from saved runs instead of running")
    ap.add_argument("--narrative", default=None, help="markdown file inserted above the rendered tables of diagnosis.md")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    if args.diagnose:
        diagnose(args.diagnose, args.out, args.narrative)
    elif args.degrade_study:
        degrade_study(args)
    elif args.f10_report:
        f10_report(args.f10_report, args.out, args.narrative)
    else:
        run(args)


if __name__ == "__main__":
    main()
