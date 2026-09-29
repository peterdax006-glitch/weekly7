"""Bible Phase 25 runner: every calibration scenario x many seeds, in parallel, through the FULL PatternMiner.
Writes state/research/algorithm/planted/{summary.json, report.md} and logs a registry record with the verdict.

Usage: python scripts/planted_calibration.py [seeds=8] [workers=4] [--quick]"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import config as K
from engine import planted as PL

MINER = {"max_pairs": 400, "max_unless": 80, "null_reps": 2, "min_n": 200, "half_life_years": 50}


def one(job):
    sc_name, seed = job
    from engine.patterns import PatternMiner
    sc = next(s for s in PL.scenarios() if s.name == sc_name)
    X, y, _ = PL.generate(sc, seed=seed)
    M = PatternMiner(MINER).fit(X, y, now=X.index.get_level_values(0).max())
    r = PL.score_run(M.patterns, sc)
    r["seed"] = seed
    return r


def report_md(summary, verdict, secs):
    L = ["# Planted-pattern calibration (Bible Phase 25)", "",
         f"**Verdict: {'VALIDATED' if verdict['validated'] else 'NOT VALIDATED'}** ({summary['n_runs']} runs, {secs:.0f}s)", "",
         "| criterion | value | rule | pass |", "|---|---|---|---|"]
    for k, v in verdict["criteria"].items():
        val = "n/a" if v["value"] is None else f"{v['value']:.3f}"
        L.append(f"| {k} | {val} | {v['rule']} | {'yes' if v['pass'] else 'NO'} |")
    for scn, s in summary["by_scenario"].items():
        L += ["", f"## {scn} ({s['runs']} runs)", "",
              f"active/run {s['active_per_run']:.1f}, false active/run {s['false_active_per_run']:.2f}, "
              f"FDR {s['false_discovery_rate']:.1%}, P(real) Brier {s['calibration']['brier']}, ECE {s['calibration']['ece']}", "",
              "| plant | kind | should admit | rate | via child | effect ratio | sign ok | statuses |", "|---|---|---|---|---|---|---|---|"]
        for name, p in s["plants"].items():
            rate = p.get("detection_rate", p.get("false_admission_rate"))
            L.append(f"| {name} | {p['kind']} | {p['should_admit']} | {rate:.2f} | {p['admitted_via_child_rate']:.2f} | "
                     f"{p['median_effect_ratio'] if p['median_effect_ratio'] is None else round(p['median_effect_ratio'], 2)} | "
                     f"{p['sign_accuracy']} | {p['status_counts']} |")
        L += ["", "P(real) reliability:", ""]
        for b, v in s["calibration"]["bins"].items():
            L.append(f"- {b}: mean P {v['mean_p']:.2f} -> truly real {v['share_truly_real']:.0%} (n={v['n']})")
    return "\n".join(L) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = int(args[0]) if args else 8
    workers = int(args[1]) if len(args) > 1 else 4
    names = [s.name for s in PL.scenarios()]
    if "--quick" in sys.argv:
        names = ["standard", "noise_only"]
    jobs = [(n, 100 + i) for n in names for i in range(seeds)]
    t0 = time.perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, jobs):
            runs.append(r)
            print(f"{r['scenario']:>14s} seed {r['seed']}: active {r['n_active']}, false {r['n_false_active']}", flush=True)
    summary = PL.summarise(runs)
    v = PL.verdict(summary)
    secs = time.perf_counter() - t0
    out = K.STATE / "research" / "algorithm" / "planted"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "verdict": v, "miner": MINER}, indent=1, default=str))
    (out / "report.md").write_text(report_md(summary, v, secs), encoding="utf-8")
    print(report_md(summary, v, secs))
    from engine.improve import log_experiment
    log_experiment({"event": "planted_calibration"}, cfg=MINER, seed=100,
                   window_ids=[f"{n}:{s}" for n, s in jobs], metrics={k: c["value"] for k, c in v["criteria"].items()},
                   gates={k: c["pass"] for k, c in v["criteria"].items()},
                   outcome="adopt" if v["validated"] else "continue_testing",
                   reason="Phase 25 verdict: " + ("validated" if v["validated"] else
                          "NOT validated: " + ", ".join(k for k, c in v["criteria"].items() if not c["pass"])),
                   train_range="synthetic", validation_range="synthetic", test_range="synthetic")


if __name__ == "__main__":
    main()
