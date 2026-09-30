"""F27 candidate recall on the F19 benchmark's DEVELOPMENT worlds (C75 3A/3B/3I, C70). IMPLEMENTED - NOT VALIDATED.

    python scripts/candidate_recall.py run --seeds 0-19            every development world in the range (held-out seeds are refused)
    python scripts/candidate_recall.py run --seeds 0-19 --null 3   plus 3 NULL worlds (tier 0: no genuine pattern) on development seeds
    python scripts/candidate_recall.py report                      recall before/after by kind x band, trade-off curve -> SUMMARY.md

Per world and per benchmark look (the rolling frame the benchmark's system sees, past data only): the single-feature screen exactly as
pattern_benchmark.run_system runs it (volatility_lab.oriented_scan over scan_universe), then candidate_forms.propose (BLIND: frame and
feature names only) for each prescreen variant, and oriented_scan on the widest proposal kept, so every narrower operating point can be
scored afterwards without re-running (a proposal at (k, t_pre) is a prefix of each family's ranked list). The answer key is used only by
`report`, after generation, to score. Worlds are checkpointed one by one. Results: state/research/candidate_forms/ (never git)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np                                                               # noqa: E402
import pandas as pd                                                              # noqa: E402

from engine.research import candidate_forms as CF                               # noqa: E402
from engine.research import pattern_benchmark as PB                             # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "state" / "research" / "candidate_forms"
VARIANTS = {"all": 1.0, "early": 1.0 / 3.0}      # prescreen on every date / on the first third (before the screen's first test fold)
K_MAX, T_PRE_MIN = 20, 2.0                       # the widest proposal scored by oriented_scan
GRID_K = (1, 2, 4, 8, 12, 20)
GRID_T = (2.0, 2.5, 3.0, 4.0)


def seeds_of(spec: list[str]) -> list[int]:
    out = []
    for s in spec:
        if "-" in s:
            a, b = s.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return list(dict.fromkeys(out))


def provenance(args: dict) -> dict:
    """Code and config stamp WITHOUT git (engine.provenance.stamp shells out to git; builders never run git)."""
    from engine import provenance as PV
    st = PV.code_stamp()
    fc = dataclasses.asdict(CF.FormConfig())
    return {"code_hash": st["code_hash"], "code_mixed": st["code_mixed"], "form_config": fc, "form_config_hash": PV.config_hash(fc),
            "bench_config_hash": PV.config_hash(dataclasses.asdict(PB.BenchConfig())), "variants": VARIANTS, "k_max": K_MAX,
            "t_pre_min": T_PRE_MIN, "args": args, "label": CF.LABEL, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _rows(tab: pd.DataFrame) -> list:
    return [[str(r.feature), *(None if not np.isfinite(v) else round(float(v), 6) for v in (r.auc, r.lo, r.hi))] for r in tab.itertuples()]


def run_world(seed: int, null: bool, log) -> dict:
    """Generation only (blind). The key is stored beside the answers for the scorer; nothing here reads it."""
    from engine.research import volatility_lab as VL
    cfg = PB.BenchConfig(null_world_share=0.99) if null else PB.BenchConfig()
    t0 = time.monotonic()
    w = PB.make_world(seed, cfg)
    F = w.frame
    planted = [c for c in F.columns if c.startswith(PB.PREFIX)]
    dates = pd.DatetimeIndex(F.index.get_level_values(0).unique()).sort_values()
    looks = []
    with PB.registered(planted), CF.session():
        feats = PB.scan_universe(F.columns)
        for li, L in enumerate(cfg.looks()):
            now = dates[L] if L < len(dates) else dates[-1] + pd.Timedelta(days=7)
            lo = dates[max(0, L - cfg.frame_weeks)]
            W = F[(F.index.get_level_values(0) >= lo) & (F.index.get_level_values(0) < now)]
            M = W[pd.to_datetime(W["end"]) < now]
            n_dates = int(M.index.get_level_values(0).nunique())
            lab = VL.LabConfig(min_train_dates=max(8, n_dates // 3), test_step_dates=max(2, n_dates // 8), n_boot=100)
            ts = time.monotonic()
            singles = VL.oriented_scan(M, feats, now, lab)
            rec = {"look": li, "now": str(pd.Timestamp(now).date()), "n_dates": n_dates, "singles": _rows(singles),
                   "singles_s": round(time.monotonic() - ts, 1), "variants": {}}
            for v, frac in VARIANTS.items():
                fc = CF.FormConfig(select_frac=frac)
                prop = CF.propose(M, feats, now, fc)
                wide = prop.at(K_MAX, T_PRE_MIN)
                tc = time.monotonic()
                tab, _ = CF.screen_table(M, feats, now, lab, fc, proposal=dataclasses.replace(prop, proposed=tuple(wide)), singles=False)
                rec["variants"][v] = {"summary": prop.summary(), "ranked": {f: [[s.name, round(s.t, 4)] for s in lst] for f, lst in prop.ranked.items()},
                                      "aliases": len(prop.aliases), "oriented": _rows(tab), "oriented_s": round(time.monotonic() - tc, 1)}
                CF.unregister_all()
            looks.append(rec)
            log(f"W{seed:05d} look {li}: singles {rec['singles_s']}s, " + ", ".join(
                f"{v}: scanned {r['summary']['n_scanned']} propose {r['summary']['seconds']}s oriented {r['oriented_s']}s"
                for v, r in rec["variants"].items()))
    key = {k: w.key[k] for k in ("world_id", "seed", "split", "tier", "eval_era")}
    key["patterns"] = w.key["patterns"]
    return {"seed": seed, "null": null, "world_id": w.key["world_id"] + ("N" if null else ""), "looks": looks, "key": key,
            "seconds": round(time.monotonic() - t0, 1)}


def cmd_run(args) -> None:
    wanted = seeds_of(args.seeds)
    seeds = PB.tuning_seeds([s for s in wanted if not PB.is_heldout(s)])     # held-out seeds are dropped, and refused if one slips in
    if len(seeds) < len(wanted):
        print(f"skipping {len(wanted) - len(seeds)} held-out seeds in the range (never generated)")
    null_seeds = seeds[: args.null]
    (OUT / "worlds").mkdir(parents=True, exist_ok=True)
    (OUT / "provenance.json").write_text(json.dumps(provenance(vars(args)), indent=1, default=str), encoding="utf-8")
    logf = OUT / "run.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    jobs = [(s, False) for s in seeds] + [(s, True) for s in null_seeds]
    for s, null in jobs:
        path = OUT / "worlds" / f"W{s:05d}{'N' if null else ''}.json"
        if path.exists():
            continue
        try:
            res = run_world(s, null, log)
        except Exception:                                   # one broken world must not cost the others; the error is kept
            log(f"W{s:05d} FAILED\n{traceback.format_exc()}")
            continue
        path.write_text(json.dumps(res, default=str), encoding="utf-8")
        log(f"W{s:05d}{'N' if null else ''} done in {res['seconds']}s")


# ================================================================================================================ scoring (after generation)
def _tab(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["feature", "auc", "lo", "hi"]).astype({"auc": float, "lo": float, "hi": float})


def proposal_at(var: dict, k: int, t_pre: float) -> list[str]:
    out = []
    for fam, lst in var["ranked"].items():
        out += [n for n, t in lst if abs(t) >= t_pre][:k]
    return out


def world_stages(res: dict, variant: str, k: int, t_pre: float, cap: int) -> dict:
    """The candidate sets of one world at one operating point, replaying the benchmark's raise rule look by look."""
    before, after, after_unc, proposed = set(), set(), set(), set()
    for rec in res["looks"]:
        S = _tab(rec["singles"])
        before |= set(CF.raise_rule(S, 2.0, cap, before))
        if k == 0:
            continue
        var = rec["variants"][variant]
        prop = set(proposal_at(var, k, t_pre))
        proposed |= prop
        C = _tab(var["oriented"])
        C = C[C["feature"].isin(prop)]
        both = pd.concat([S, C], ignore_index=True)
        after |= set(CF.raise_rule(both, 2.0, cap, after))
        after_unc |= set(CF.raise_rule(both, 2.0, None, after_unc))
    if k == 0:
        after, after_unc = set(before), set()
    return {"proposed": proposed, "raised_before": before, "raised_after": after, "raised_uncapped": after_unc}


def score(results: list[dict], variant: str, k: int, t_pre: float, cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, worlds = [], []
    for res in results:
        st = world_stages(res, variant, k, t_pre, cap)
        key = res["key"]
        if not res["null"]:
            rows += [{**r, "null": False} for r in CF.score_patterns(key, st)]
        comp_after = {n for n in st["raised_after"] if CF.parse(n) is not None}
        fc = CF.false_candidates(key, comp_after)
        worlds.append({"world_id": res["world_id"], "null": res["null"], "tier": key.get("tier"), "proposed": len(st["proposed"]),
                       "raised_before": len(st["raised_before"]), "raised_after": len(st["raised_after"]),
                       "raised_composites": len(comp_after), "false_composites": fc["total"],
                       "raised_uncapped": len(st["raised_uncapped"]),
                       "scanned_per_look": float(np.mean([r["variants"][variant]["summary"]["n_scanned"] for r in res["looks"]])),
                       "propose_s": float(np.sum([r["variants"][variant]["summary"]["seconds"] for r in res["looks"]])),
                       "oriented_s": float(np.sum([r["variants"][variant]["oriented_s"] for r in res["looks"]])),
                       "singles_s": float(np.sum([r["singles_s"] for r in res["looks"]]))})
    return pd.DataFrame(rows), pd.DataFrame(worlds)


def md(t: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if t is None or len(t) == 0:
        return "(none)\n"
    cols = list(t.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in t.iterrows():
        out += "| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values) + " |\n"
    return out


def cmd_report(args) -> None:
    files = sorted((OUT / "worlds").glob("W*.json"))
    results = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    if not results:
        print("no worlds")
        return
    if any(PB.is_heldout(r["seed"]) for r in results):
        raise PB.HeldOutAccess("a held-out world is in the candidate-recall results")
    cap = PB.loop_screen_defaults()[0] * PB.BenchConfig().look_every
    real_worlds = [r for r in results if not r["null"]]
    curve = []
    for variant in VARIANTS:
        for k in (0,) + GRID_K:
            for t_pre in (GRID_T if k else (0.0,)):
                if k == 0 and variant != "all":
                    continue
                R, Wd = score(results, variant, k, t_pre, cap)
                det = R[R["status"] != "UNDETECTABLE_IN_PRINCIPLE"]
                nr = R[R["status"] == "NOT_REPRESENTABLE"]
                real_w = Wd[~Wd["null"]]
                null_w = Wd[Wd["null"]]
                curve.append({"variant": variant if k else "singles", "k": k, "t_pre": t_pre,
                              "recall_true": float(det["raised_after_true"].mean()), "recall_any": float(det["raised_after_any"].mean()),
                              "recall_before": float(det["raised_before_any"].mean()),
                              "recall_not_repr_true": float(nr["raised_after_true"].mean()) if len(nr) else float("nan"),
                              "proposed_true": float(det["proposed_true"].mean()) if k else 0.0,
                              "uncapped_true": float(det["raised_uncapped_true"].mean()) if k else float("nan"),
                              "proposed_per_world": float(real_w["proposed"].mean()), "raised_per_world": float(real_w["raised_after"].mean()),
                              "composites_raised_per_world": float(real_w["raised_composites"].mean()),
                              "false_composites_per_world": float(real_w["false_composites"].mean()),
                              "null_raised_per_world": float(null_w["raised_after"].mean()) if len(null_w) else float("nan"),
                              "null_composites_per_world": float(null_w["raised_composites"].mean()) if len(null_w) else float("nan"),
                              "scanned_per_look": float(real_w["scanned_per_look"].mean()) if k else float("nan")})
    C = pd.DataFrame(curve)
    C.to_csv(OUT / "tradeoff.csv", index=False)
    op = dict(variant=args.variant, k=args.k, t_pre=args.t_pre)
    R, Wd = score(results, op["variant"], op["k"], op["t_pre"], cap)
    R.to_csv(OUT / "recall_rows.csv", index=False)
    Wd.to_csv(OUT / "worlds.csv", index=False)
    stages = ["raised_before", "proposed", "raised_after", "raised_uncapped"]
    kb = CF.recall_table(R, stages)
    by_status = CF.recall_table(R, stages, by=("kind", "status"))
    kinds = CF.recall_table(R, stages, by=("kind",))
    bands = CF.recall_table(R, stages, by=("band",))
    allrec = CF.recall_table(R.assign(all="all"), stages, by=("all",))
    raw_all = CF.recall_table(R.assign(all="all"), stages, by=("all",), detectable_only=False)
    scanned = {}
    for res in real_worlds:
        for rec in res["looks"]:
            for f, n in rec["variants"][op["variant"]]["summary"]["scanned"].items():
                scanned.setdefault(f, []).append(n)
    scanned_mean = {f: int(round(np.mean(v))) for f, v in scanned.items()}
    cost = CF.multiplicity_cost(scanned_mean)
    real_w, null_w = Wd[~Wd["null"]], Wd[Wd["null"]]
    prov = json.loads((OUT / "provenance.json").read_text(encoding="utf-8")) if (OUT / "provenance.json").exists() else {}
    txt = [f"# F27 candidate representation and screen recall - {CF.LABEL}", "",
           f"Development worlds: {len(real_worlds)} (seeds {sorted(r['seed'] for r in real_worlds)}), NULL worlds: {len(null_w)}. "
           f"Held-out seeds: never generated (tuning_seeds refuses them). Operating point: variant={op['variant']}, "
           f"k_per_family={op['k']}, t_pre={op['t_pre']}; raise rule = the benchmark's (oriented_scan t >= 2, cap {cap} per look).",
           f"Code hash {prov.get('code_hash')}, form config {prov.get('form_config_hash')}, bench config {prov.get('bench_config_hash')}.", "",
           "Recall = a real pattern has a raised candidate that is it in its TRUE form (_true) / in any form incl. the single (_any); "
           "patterns an oracle could not detect even in the true form (UNDETECTABLE_IN_PRINCIPLE) are excluded unless stated.", "",
           "## Overall (detectable in their true form)", "", md(allrec),
           "## Overall (all real patterns, undetectable included)", "", md(raw_all),
           "## Per kind", "", md(kinds), "## Per band", "", md(bands), "## Per kind x status", "", md(by_status),
           "## Per kind x band", "", md(kb),
           "## Candidates per world", "", md(Wd.groupby("null")[["proposed", "raised_before", "raised_after", "raised_composites",
                                                                    "false_composites", "raised_uncapped", "scanned_per_look",
                                                                    "propose_s", "oriented_s", "singles_s"]].mean().reset_index()),
           "## Trade-off curve (every operating point; the proposal at (k, t_pre) is a prefix of the stored ranked lists)", "", md(C),
           "## Multiplicity: the |t| bar a single candidate needs (Bonferroni, alpha 0.05, mean tests per look)", "", md(cost),
           "## Runtime", "", f"Per world: singles screen {real_w['singles_s'].mean():.0f}s, form search {real_w['propose_s'].mean():.0f}s, "
           f"screen of the proposed forms {real_w['oriented_s'].mean():.0f}s (sums over looks; per variant).", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", nargs="+", default=["0-19"])
    r.add_argument("--null", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--variant", default="all")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--t_pre", type=float, default=2.5)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
