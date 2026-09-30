"""F19 real-vs-noise benchmark runner (canon C70-C75). IMPLEMENTED - NOT VALIDATED.

    python scripts/pattern_benchmark.py heldout --upto 1000                     fix and record the held-out seed list (once)
    python scripts/pattern_benchmark.py run --seeds 0-499 --part 0 --parts 2   the current system on every world of this part
    python scripts/pattern_benchmark.py run --seeds 0-499 --mode screen        candidate generation only (no gate), all worlds
    python scripts/pattern_benchmark.py memorization --seeds 1 4               C75 3M: the same world renamed / shifted
    python scripts/pattern_benchmark.py mutation --name weaker --seeds 0-9      C75 3N: one generator axis changed
    python scripts/pattern_benchmark.py report                                  tables, ranked failures, learning curve -> SUMMARY.md
    python scripts/pattern_benchmark.py freeze                                  Firewall 10: the record a final holdout must match

Every world: SEED -> WORLD -> TRUTH -> TRUTH SEAL (key + sha256 in manifest.jsonl) -> MARKET RELEASE (frame only) -> LEARNER ->
RESULT FREEZE (answers + sha256) -> EVALUATOR (opens the key, fails closed) -> SCORE. Worlds are checkpointed one by one (a killed
part resumes at the next unscored world). Results go to state/research/pattern_benchmark/ with provenance (never git)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd                                                              # noqa: E402

from engine.research import pattern_benchmark as PB                             # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "state" / "research" / "pattern_benchmark"


def seeds_of(spec: list[str]) -> list[int]:
    out = []
    for s in spec:
        if "-" in s:
            a, b = s.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return list(dict.fromkeys(out))


def provenance(args: dict, cfg: PB.BenchConfig) -> dict:
    """Code and config stamp WITHOUT git (engine.provenance.stamp shells out to git; builders never run git)."""
    import dataclasses
    from engine import provenance as PV
    st = PV.code_stamp()
    return {"code_hash": st["code_hash"], "code_mixed": st["code_mixed"], "config": dataclasses.asdict(cfg), "args": args,
            "config_hash": PV.config_hash(dataclasses.asdict(cfg)), "freeze": PB.freeze_record(cfg), "label": PB.LABEL,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _log(path: Path):
    def f(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return f


def cmd_run(a) -> int:
    cfg = PB.BenchConfig(gate=a.mode == "full")
    if a.mutation:
        cfg = PB.mutate(cfg, a.mutation)
    errs = PB.full_scale_ok(cfg)
    if errs:
        raise SystemExit("; ".join(errs))
    out = Path(a.out) if a.out else OUT / ("worlds" if a.mode == "full" else "screen") / (a.mutation or "")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in seeds_of(a.seeds) if s % a.parts == a.part]
    log = _log(out / f"log_part{a.part}.txt")
    (out / f"provenance_part{a.part}.json").write_text(json.dumps(provenance(vars(a), cfg), indent=1, default=str), encoding="utf-8")
    log(f"part {a.part}/{a.parts}: {len(seeds)} worlds, mode {a.mode}, pid {os.getpid()}")
    code = PB.freeze_record(cfg)["code_hash"]
    for s in seeds:
        if a.stop_file and Path(a.stop_file).exists():
            log("stop file present: stopping cleanly")
            break
        t0 = time.monotonic()
        try:
            res = PB.run_world(s, out, cfg, code_hash=code, log=log if a.verbose else None)
            m = res["summary"]
            log(f"W{s:05d} {m['split']} tier {m.get('tier')} era {m['eval_era']}: real {m['real_right']}/{m['real_detectable']} detectable "
                f"({m['real_total']} planted), FP {m['false_positives']} {m['fp_by_kind']}, {time.monotonic() - t0:.0f}s")
        except Exception as e:                                                   # noqa: BLE001 - one world's failure is logged, never silent
            log(f"W{s:05d} FAILED {type(e).__name__}: {e}")
            (out / "errors").mkdir(exist_ok=True)
            (out / "errors" / f"W{s:05d}.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return 0


def cmd_report(a) -> int:
    out = Path(a.out) if a.out else OUT / "worlds"
    rows, summ = PB.load_scores(out)
    if not summ:
        raise SystemExit(f"no scored worlds under {out}")
    T = PB.aggregate(rows, summ)
    R, S = pd.DataFrame(rows), pd.DataFrame(summ)
    T["tier"] = PB.tier_table(R, S)
    T["world"] = S[["world_id", "split", "tier", "eval_era", "shift", "real_right", "real_detectable", "real_total", "noise_rejected",
                    "noise_total", "false_positives", "expected_fp_stated_alpha", "seconds"]].sort_values("world_id")
    T["learning_curve"] = PB.learning_curve(summ, order=sorted(S["world_id"]))
    secs = S["seconds"].dropna()
    meta = {"note": f"Cost: {secs.mean():.0f} s per world (median {secs.median():.0f}), {secs.sum() / 3600:.1f} CPU-hours in total."}
    md = PB.report_markdown(T, summ, meta)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    R.to_csv(out / "pattern_rows.csv", index=False)
    json.dump({k: v.to_dict("records") for k, v in T.items()}, open(out / "tables.json", "w", encoding="utf-8"), indent=1, default=str)
    print(md[:6000])
    return 0


def cmd_heldout(a) -> int:
    rec = PB.write_heldout_record(OUT / "heldout_seeds.json", a.upto)
    print(f"{len(rec['heldout'])} held-out seeds below {rec['upto']}; sha256 {rec['sha256'][:16]}")
    return 0


def cmd_memorization(a) -> int:
    """C75 3M: every world is run as generated and renamed (tickers renamed and reordered, planted columns renamed), and again with the
    dates shifted by 52 weeks; the answer sheets must agree after mapping the names back."""
    cfg = PB.BenchConfig()
    out = OUT / "memorization"
    out.mkdir(parents=True, exist_ok=True)
    log = _log(out / "log.txt")
    for s in seeds_of(a.seeds):
        w = PB.make_world(s, cfg)
        base = PB.run_system(w.frame, f"W{s:05d}", s, cfg)
        res = {}
        for lab, shift in (("renamed", 0), ("renamed_shifted", 52)):
            G, back, _ = PB.disguise(w.frame, s + 1000, shift_weeks=shift)
            other = PB.run_system(G, f"M{s:05d}", s, cfg)
            res[lab] = PB.answers_invariance(base, other, back)
            log(f"W{s:05d} {lab}: invariant={res[lab]['invariant']} ({res[lab]['n_diff']} of {res[lab]['n']} candidates differ)")
        (out / f"W{s:05d}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


def cmd_freeze(a) -> int:
    rec = PB.freeze_record(PB.BenchConfig())
    (OUT / "freeze_design.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", nargs="+", default=["0-499"])
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--parts", type=int, default=1)
    r.add_argument("--mode", choices=("full", "screen"), default="full")
    r.add_argument("--mutation", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--stop-file", default=None)
    r.add_argument("--verbose", action="store_true")
    m = sub.add_parser("mutation")
    m.add_argument("--name", required=True)
    m.add_argument("--seeds", nargs="+", default=["0-9"])
    rp = sub.add_parser("report")
    rp.add_argument("--out", default=None)
    h = sub.add_parser("heldout")
    h.add_argument("--upto", type=int, default=1000)
    mm = sub.add_parser("memorization")
    mm.add_argument("--seeds", nargs="+", default=["1"])
    sub.add_parser("freeze")
    a = ap.parse_args(argv)
    if a.cmd == "mutation":
        a = argparse.Namespace(seeds=a.seeds, part=0, parts=1, mode="full", mutation=a.name, out=None, stop_file=None, verbose=False)
        return cmd_run(a)
    return {"run": cmd_run, "report": cmd_report, "heldout": cmd_heldout, "memorization": cmd_memorization, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
