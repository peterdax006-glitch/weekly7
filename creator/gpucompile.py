"""GPU job compiler (P1.6; MASTER_BLUEPRINT 8.5a; owner 4 Oct: "anything it sets up for the GPU it should train so the GPU can run through
it at max efficiency"). Turns a wishlist into a READY PACKAGE on the PC, before any rental:

  data       existing mixes referenced by sha256 + the planted-data jobs (pod CPU) the thin-mix rule demands; one archive per job
  settings   the fastest setting that kept quality, from creator.gpueff (epochs from the measured 2nd-epoch gain, 4 x 4, grad-ckpt on)
  thin rule  fewer than THIN_STEPS optimizer steps -> the module does not go to the GPU until its data job has run (rows_needed says how many)
  pairing    VRAM-sized: two trainings run at once only when both learned peaks + the eval server fit the card
  evals      base / generalist evals are skipped when cached; the tuned eval runs on the pod beside the next training
  prestage   bases downloaded in order of first use while the previous job trains
  timeline   per job and in total: start, train, merge+GGUF, eval, shipped home; GPU busy %; dollars (rented hours x measured $/h)

Every number carries its basis; anything not measured is named in `assumptions` (nothing is hidden in a constant). Plans only: nothing here
contacts, rents or starts a GPU, and the package is written outside the repository."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from creator import gpueff as GE
from creator import jobcost as JC

VRAM_TOTAL_GB = 31.0                 # same figures as creator.trainmods (5090 minus the CUDA context; eval server with its slots)
EVAL_SERVER_GB = 5.0
PAIR_SLOWDOWN = 1.6                  # ASSUMPTION (unmeasured): two trainings sharing the card each run at 1/1.6 speed (throughput 1.25x)
BOOT_S = 420.0                       # ASSUMPTION: instance start + tunnel + guardian + data upload before the first training
DL_MB_S = 9.5                        # base download rate: the logged 0.4 GB fetch took 42 s (fetch_judge_06b_basegguf)
SHIP_MB_S = 5.0                      # ASSUMPTION: relay transfer home (stalls were seen at ~45 MB)
GEN_S_PER_ROW = 3.2                  # ASSUMPTION: planted-bug row = mutate + run the failing tests; ~54 min for ~1000 rows on the busy PC (4 Oct)
GEN_WORKERS = 8                      # ASSUMPTION: pod CPU workers for the generator
TEARDOWN_S = 60.0
BF16_GB = {"0.6b": 1.2, "1.7b": 3.4, "4b": 8.0}
Q4_GB = {"0.6b": 0.4, "1.7b": 1.1, "4b": 2.5}
MERGE_S_17B, GGUF_S_17B = 261.2, 71.5            # measured, ft_pipeline_17b result.json (merge + Q4 GGUF); other sizes scale with parameters


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _mix(name: str, mixes: Path, hashing: bool = True) -> dict[str, Any]:
    d = mixes / name
    man = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    tr = d / "train.jsonl"
    st = GE.length_stats(tr)
    return {"name": name, "rows": int(man["rows"]["train"]), "tokens_train": float(man["tokens_train"]), "len": st,
            "train_bytes": tr.stat().st_size, "sha256": _sha(tr) if hashing else "", "dev_rows": int(man["rows"]["dev"]),
            "manifest_epochs": man.get("epochs")}


def _merge_len(parts: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Length stats of a union: weighted means for mean/p80/p90, the worst of p99/max (conservative)."""
    n = sum(p["rows"] for p in parts) or 1
    out = {k: round(sum(p["rows"] * p["len"][k] for p in parts) / n, 1) for k in ("mean", "p80", "p90")}
    out.update(p99=max(p["len"]["p99"] for p in parts), max=max(p["len"]["max"] for p in parts), rows=n)
    return out


def family_gain(eff: GE.Eff, family: str) -> Optional[float]:
    g = [x for r in eff.runs if (x := GE.epoch_gain(r)) is not None and GE._family_of(r["name"]) == family]
    return statistics.median(g) if g else None


def mix_epochs(eff: GE.Eff, shares: Mapping[str, float], unmeasured: Mapping[str, int]) -> tuple[int, str]:
    """Epochs for a mixed job: row-share weighted 2nd-epoch gain over the families with a measurement; >= 10% -> 2 epochs. A family without a
    measurement uses its analogue's gain (named in the basis) via `unmeasured` (family -> epochs prior) only when nothing at all is measured."""
    gs = {f: family_gain(eff, f) for f in shares}
    known = {f: g for f, g in gs.items() if g is not None}
    if known:
        tot = sum(shares[f] for f in known)
        gain = sum(shares[f] * g for f, g in known.items()) / tot
        basis = "; ".join(f"{f} {g:.1%} x {shares[f] / tot:.0%}" for f, g in known.items())
        return (2 if gain >= GE.EPOCH_GAIN_MIN else 1), f"weighted measured 2nd-epoch gain {gain:.1%} ({basis})"
    ep = max((unmeasured.get(f, 1) for f in shares), default=1)
    return ep, f"no measured 2nd-epoch gain for {sorted(shares)}: prior {ep}"


def compile_job(w: Mapping[str, Any], eff: GE.Eff, hist: JC.CostModel, mixes: Path, usd_h: float, hashing: bool = True) -> dict[str, Any]:
    """One wishlist entry -> one compiled job (settings, data, predicted seconds). `w`: {id, name, size, families{family: row share},
    mixes[names], extra{name, rows, mean, p80, p99, max, family, gen: bool}, target_rows (optional)}."""
    parts = [_mix(m, mixes, hashing) for m in w.get("mixes", [])]
    ex = w.get("extra")
    rows = sum(p["rows"] for p in parts)
    lens = list(parts)
    if ex:
        target = int(ex.get("rows_target") or ex["rows"])
        lens = parts + [{"rows": target, "len": {k: ex[k] for k in ("mean", "p80", "p90", "p99", "max")}}]
        rows += target
    stats = _merge_len(lens)
    shares = {f: float(s) for f, s in (w.get("families") or {}).items()}
    epochs, ep_basis = mix_epochs(eff, shares, w.get("epoch_prior") or {})
    st = GE.settings(eff, w["size"], next(iter(shares), "other"), rows, sum(p["tokens_train"] for p in parts) + (
        (ex["rows_target"] * ex["mean"]) if ex else 0.0), stats)
    st.update(epochs=epochs, epochs_basis=ep_basis, steps=math.ceil(rows / GE.DEFAULT_BATCH) * epochs)
    steps = st["steps"]
    padded = steps * GE.DEFAULT_BATCH * stats["p80"]
    train_s = eff.train_s(w["size"], padded)
    thin = steps < GE.THIN_STEPS
    min_rows = math.ceil(GE.THIN_STEPS * GE.DEFAULT_BATCH / epochs)
    peak = eff.peak_gb(w["size"], stats["max"])
    big = w["size"]
    p_b = GE.PARAMS_B[big]
    merge_s = (MERGE_S_17B + GGUF_S_17B) * p_b / 1.7
    gen_rows = int(ex["rows_target"] - ex.get("rows_have", 0)) if ex and ex.get("gen") else 0
    ev = hist.predict("roleeval_base")["seconds"] if hist.predict("roleeval_base") else 90.0
    train_bytes = sum(p["train_bytes"] for p in parts) + (ex["rows_target"] * ex["mean"] * 3.6 if ex else 0)
    return {
        "id": w["id"], "name": w["name"], "family": w.get("family", ""), "size": w["size"], "value_usd": float(w.get("value_usd", 0.0)),
        "deps": list(w.get("deps") or []), "deadline": w.get("deadline", ""), "rows": rows, "len": stats, "settings": st,
        "thin_mix": {"steps": steps, "min_steps": GE.THIN_STEPS, "thin": thin, "min_rows_at_these_epochs": min_rows,
                     "rows_needed": max(0, min_rows - sum(p["rows"] for p in parts)) if thin or gen_rows else 0},
        "data": {"mixes": [{k: p[k] for k in ("name", "rows", "tokens_train", "train_bytes", "sha256", "dev_rows")} for p in parts],
                 "gen_rows": gen_rows, "gen_s": gen_rows * GEN_S_PER_ROW / GEN_WORKERS, "archive_mb": round(train_bytes / 2**20 / 4, 1)},
        "predicted": {"train_s": round(train_s, 1), "padded_tokens": round(padded), "merge_gguf_s": round(merge_s, 1),
                      "tuned_eval_s": round(ev, 1), "ship_gb": Q4_GB[big], "ship_s": round(Q4_GB[big] * 1024 / SHIP_MB_S, 1),
                      "base_download_s": round(BF16_GB[big] * 1024 / DL_MB_S, 1)},
        "peak_vram": peak,
        "value_basis": w.get("value_basis", ""), "usd_h": usd_h,
    }


def can_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a["peak_vram"]["gb"] + b["peak_vram"]["gb"] + 2.0 + EVAL_SERVER_GB <= VRAM_TOTAL_GB


def schedule(jobs: Sequence[Mapping[str, Any]], cached_evals: Sequence[str], usd_h: float, boot_s: float = BOOT_S, pair: bool = True) -> dict[str, Any]:
    """Greedy timeline. Lanes: downloads (sequential, in order of first use), pod CPU (data jobs), GPU (training slots, pairs when VRAM allows),
    eval (sequential, beside the next training), ship (home transfers, overlapping). Returns per-job times and the totals."""
    by_id = {j["id"]: j for j in jobs}
    dens = lambda j: j["value_usd"] / max(1.0, j["predicted"]["train_s"])          # noqa: E731
    # downloads: bases in order of first use (value density, data-ready jobs first); the first one is on the critical path
    first = sorted(jobs, key=lambda j: (j["data"]["gen_rows"] > 0, -dens(j)))
    dl_end: dict[str, float] = {}
    t = boot_s * 0.5
    for j in first:
        if j["size"] not in dl_end:
            t += j["predicted"]["base_download_s"]
            dl_end[j["size"]] = t
    # pod CPU lane: data jobs one after another, dependencies and value first
    cpu_t = boot_s
    gen_end: dict[str, float] = {}
    for j in sorted((j for j in jobs if j["data"]["gen_s"] > 0), key=lambda j: (-sum(1 for o in jobs if j["id"] in o["deps"]), -dens(j))):
        cpu_t += j["data"]["gen_s"]
        gen_end[j["id"]] = cpu_t
    gpu_t = max(boot_s, min(dl_end.values(), default=boot_s))
    tl: dict[str, dict[str, Any]] = {}
    busy = 0.0
    slots: list[list[str]] = []
    order: list[str] = []
    todo = {j["id"] for j in jobs}
    start_of = lambda j: max([gpu_t, gen_end.get(j["id"], 0.0), dl_end[j["size"]]])      # noqa: E731
    while todo:
        ready = [by_id[i] for i in todo if all(d not in todo for d in by_id[i]["deps"])]
        t0 = min(start_of(j) for j in ready)
        near = [j for j in ready if start_of(j) <= t0 + 300.0]                        # within 5 min of the earliest start:
        a = max(near, key=lambda j: (sum(1 for o in jobs if j["id"] in o["deps"]), dens(j)))   # unblock dependents first, then value density
        grp = [a]
        if pair:
            mates = [j for j in ready if j["id"] != a["id"] and can_pair(a, j) and start_of(j) <= start_of(a) + 300.0]
            if mates:
                grp.append(min(mates, key=lambda j: start_of(j)))
        start = max(start_of(g) for g in grp)
        if len(grp) == 2:
            s_, l_ = sorted(grp, key=lambda g: g["predicted"]["train_s"])
            ds, dl = s_["predicted"]["train_s"], l_["predicted"]["train_s"]
            ends = {s_["id"]: start + ds * PAIR_SLOWDOWN, l_["id"]: start + dl + (PAIR_SLOWDOWN - 1.0) * ds}
            busy += max(ends.values()) - start
            gpu_t = max(ends.values())
        else:
            ends = {a["id"]: start + a["predicted"]["train_s"]}
            busy += a["predicted"]["train_s"]
            gpu_t = ends[a["id"]]
        for g in grp:
            tl[g["id"]] = {"train_start": round(start, 1), "train_end": round(ends[g["id"]], 1), "paired": len(grp) == 2}
            order.append(g["id"])
            todo.discard(g["id"])
        slots.append([g["id"] for g in grp])
    eval_t = boot_s
    ship_t = 0.0
    for i in order:
        j, p = by_id[i], by_id[i]["predicted"]
        n_cached = sum(1 for k in ("base", "generalist") if f"{j['name']}:{k}" in cached_evals)
        pre = (2 - n_cached) * p["tuned_eval_s"]                     # base + generalist evals not in the cache: once, before the tuned eval
        post = tl[i]["train_end"] + p["merge_gguf_s"]
        e0 = max(eval_t, dl_end[j["size"]]) + pre
        e1 = max(e0, post) + p["tuned_eval_s"] + 6.0
        s1 = max(ship_t, post) + p["ship_s"]
        eval_t, ship_t = e1, s1
        tl[i].update(merged=round(post, 1), pre_evals_s=round(pre, 1), eval_end=round(e1, 1), shipped=round(s1, 1), done=round(max(e1, s1), 1))
    make = max([v["done"] for v in tl.values()] + [gpu_t]) + TEARDOWN_S
    return {"order": order, "slots": slots, "timeline": tl, "makespan_s": round(make, 1), "gpu_train_s": round(busy, 1),
            "gpu_busy_pct": round(100.0 * busy / make, 1), "usd": round(make / 3600.0 * usd_h, 3), "usd_h": usd_h,
            "data_lane_end_s": round(cpu_t, 1), "downloads": {k: round(v, 1) for k, v in dl_end.items()}}


def phase2_wishlist(mixes: Path) -> list[dict[str, Any]]:
    """MASTER_BLUEPRINT 12 Phase 2 as wishlist entries. Row counts of the existing mixes come from their manifests at compile time; the
    planted-bug generator rows (PINPOINT, DEBUG_FIX) are PLANNED data: the thin-mix rule sizes them."""
    pin_have = 1012 + 73                                           # pinpoint_data ds.jsonl + mbfl.jsonl rows on the PC (4 Oct)
    pin = {"name": "pinpoint", "rows": pin_have, "rows_have": pin_have, "rows_target": 6000, "mean": 410.0, "p80": 450.0, "p90": 500.0, "p99": 620.0,
           "max": 760.0, "gen": True}
    dbg = {"name": "debug_fix", "rows": 0, "rows_have": 0, "rows_target": 2000, "mean": 700.0, "p80": 900.0, "p90": 1100.0, "p99": 1500.0,
           "max": 2000.0, "gen": True}
    return [
        {"id": "p2.pinpoint_06b", "name": "pinpoint_06b", "family": "pinpoint", "size": "0.6b", "mixes": [], "extra": pin,
         "families": {"judge": 1.0}, "value_usd": 12.0, "deadline": "",
         "value_basis": "blueprint 2.0: debug is the weakest pipeline step (DEBUG tuned accuracy 0.155); priority-assigned, replace with measured saving"},
        {"id": "p2.checker_06b", "name": "checker_06b", "family": "checker", "size": "0.6b", "mixes": ["calib_17b", "judge_06b"],
         "families": {"calib": 7735 / (7735 + 2192), "judge": 2192 / (7735 + 2192)}, "value_usd": 10.0, "deps": [],
         "value_basis": "CHECKER slot: calib ECE 0.40 -> 0.026 and judge ADOPTED exist on 1.7B; the 0.6B merged slot is ~3x cheaper per call at home"},
        {"id": "p2.coder_17b", "name": "coder_17b", "family": "coder", "size": "1.7b", "mixes": ["coderonly_17b"], "extra": dbg,
         "families": {"code": 1.0}, "value_usd": 10.0, "deps": ["p2.pinpoint_06b"],
         "value_basis": "CODER slot ties the generalist at 25% fewer tokens (coderonly_17b); DEBUG_FIX rows add the weakest step"},
        {"id": "p2.thinker_17b", "name": "thinker_17b", "family": "thinker", "size": "1.7b", "mixes": ["thinker_17b"],
         "families": {"thinker": 1.0}, "epoch_prior": {"thinker": 1}, "value_usd": 6.0,
         "value_basis": "THINKER slot: short forms (5-32 tok) instead of 865-1193 tok reasoning; no 2nd-epoch measurement yet -> 1 epoch (token budget)"},
    ]


PC_TRAIN_TOK_S = {"0.6b": 150.0, "1.7b": 60.0, "4b": 25.0}      # ASSUMPTION (unmeasured; no CPU fine-tune was ever timed): padded tokens/s of LoRA on the 14-thread PC


def place_all(c: Mapping[str, Any], pc_tok_s: Optional[Mapping[str, float]] = None, state_dir: Optional[Path] = None) -> dict[str, Any]:
    """MASTER_BLUEPRINT 7.10 for every compiled job: T_pc (padded tokens / PC rate), T_gpu (the compiled timeline's own seconds for the job),
    placement + wishlist entry, then ONE rental proposal over the whole list. Adds to <state_dir>/wishlist.json when given."""
    rate = pc_tok_s or PC_TRAIN_TOK_S
    usd_h = c["schedule"]["usd_h"]
    tl = c["schedule"]["timeline"]
    decisions, wish = [], {}
    for j in c["jobs"]:
        t_pc = j["predicted"]["padded_tokens"] / rate[j["size"]] + j["data"]["gen_s"] * GEN_WORKERS      # PC: generator on 1 worker
        t_gpu = tl[j["id"]]["done"] - tl[j["id"]]["train_start"] + j["data"]["gen_s"]
        d = JC.place(j["id"], f"ft_{j['size']}", t_pc, t_gpu_s=t_gpu, value_usd=j["value_usd"], deps=j["deps"], deadline=j["deadline"], usd_h=usd_h,
                     setup_s=0.0, recur_per_week=0.25, basis="PC rate is an ASSUMPTION (pc_train_tok_s); GPU seconds from the compiled timeline")
        decisions.append(d)
        if d.get("wishlist_entry"):
            wish[j["id"]] = d["wishlist_entry"]
            if state_dir:
                JC.add_wish(Path(state_dir) / "wishlist_placed.json", d["wishlist_entry"])
    return {"decisions": decisions, "wishlist": wish, "proposal": JC.proposal(wish, usd_h=usd_h, setup_s=JC.DEFAULT_GPU_SETUP_S, plan_gpu_s=c["schedule"]["makespan_s"])}


def cached_evals(jobs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Base/generalist evals already run on the pod (roleeval logs), as '<mix>:<variant>' keys."""
    out: list[str] = []
    for j in jobs:
        n = j["job"]
        if n.startswith("roleeval_") and j["rc"] == 0 and n.rsplit("_", 1)[-1] in ("base", "generalist"):
            out.append(f"{n[len('roleeval_'):].rsplit('_', 1)[0]}:{n.rsplit('_', 1)[-1]}")
    return out


def compile_wishlist(wishlist: Sequence[Mapping[str, Any]], built: Optional[Mapping[str, Any]] = None, mixes: Optional[Path] = None,
                     hashing: bool = True) -> dict[str, Any]:
    built = built or GE.build()
    mixes = Path(mixes or GE.mix_root())
    usd_h = built["usd_h"] or JC.DEFAULT_GPU_USD_H
    eff, hist = built["eff"], JC.CostModel(built["history"])
    jobs = [compile_job(w, eff, hist, mixes, usd_h, hashing) for w in wishlist]
    cache = cached_evals(built["jobs"])
    paired = schedule(jobs, cache, usd_h)
    serial = schedule(jobs, cache, usd_h, pair=False)
    sched = paired if paired["makespan_s"] < serial["makespan_s"] else serial          # pair only when the whole timeline gets shorter
    return {"jobs": jobs, "schedule": sched, "paired_makespan_s": paired["makespan_s"], "unpaired_makespan_s": serial["makespan_s"], "unpaired_usd": serial["usd"],
            "efficiency": {"rate_tok_s": {k: round(v) for k, v in eff.rate.items()}, "load_s": {k: round(v) for k, v in eff.load_s.items()},
                           "epoch_gain": eff.gains(), "validation": eff.validate(built["jobs"])},
            "assumptions": {"pair_slowdown": PAIR_SLOWDOWN, "boot_s": BOOT_S, "ship_mb_s": SHIP_MB_S, "download_mb_s": DL_MB_S,
                            "gen_s_per_row": GEN_S_PER_ROW, "gen_workers": GEN_WORKERS, "usd_h": usd_h,
                            "unmeasured_speedups_not_applied": ["group_by_length", "no grad-ckpt", "flash-attn", "pod-side eval client"]},
            "cached_evals": cache}


def render_timeline(c: Mapping[str, Any]) -> str:
    s = c["schedule"]
    L = ["# Phase 2 package: timeline and cost", "",
         f"Makespan {s['makespan_s'] / 3600:.2f} h, GPU training {s['gpu_train_s'] / 3600:.2f} h, GPU busy {s['gpu_busy_pct']}%, "
         f"cost ${s['usd']:.2f} at ${s['usd_h']}/h (unpaired: {c['unpaired_makespan_s'] / 3600:.2f} h, ${c['unpaired_usd']:.2f}).", "",
         "| job | size | rows | epochs | steps | thin? | train start (min) | train (min) | merged | eval end | shipped home | peak GB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    byid = {j["id"]: j for j in c["jobs"]}
    for i in s["order"]:
        j, t = byid[i], s["timeline"][i]
        L.append(f"| {j['name']} | {j['size']} | {j['rows']} | {j['settings']['epochs']} | {j['settings']['steps']} | "
                 f"{'THIN' if j['thin_mix']['thin'] else 'ok'} | {t['train_start'] / 60:.1f} | {j['predicted']['train_s'] / 60:.1f} | "
                 f"{t['merged'] / 60:.1f} | {t['eval_end'] / 60:.1f} | {t['shipped'] / 60:.1f} | {j['peak_vram']['gb']} |")
    L += ["", "## Data jobs (pod CPU)"]
    for j in c["jobs"]:
        if j["data"]["gen_rows"]:
            L.append(f"- {j['name']}: generate {j['data']['gen_rows']} planted rows, {j['data']['gen_s'] / 60:.0f} min on the pod CPU "
                     f"(thin-mix rule needs >= {j['thin_mix']['min_rows_at_these_epochs']} rows at {j['settings']['epochs']} epoch(s), "
                     f"{j['rows']} planned); run a 2% pilot first")
    L += ["", "## Settings basis"]
    for j in c["jobs"]:
        L.append(f"- {j['name']}: {j['settings']['epochs']} epoch(s) - {j['settings']['epochs_basis']}; max_seq {j['settings']['max_seq']}; "
                 f"VRAM {j['peak_vram']['gb']} GB ({j['peak_vram']['basis']})")
    L += ["", "## Assumptions (unmeasured, named)"] + [f"- {k}: {v}" for k, v in c["assumptions"].items()]
    return "\n".join(L) + "\n"


def write_package(c: Mapping[str, Any], out: Path, wishlist: Sequence[Mapping[str, Any]], loo: Optional[Mapping[str, Any]] = None) -> list[str]:
    out = Path(out)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    (out / "PACKAGE.json").write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
    (out / "TIMELINE.md").write_text(render_timeline(c), encoding="utf-8")
    (out / "wishlist.json").write_text(json.dumps({w["id"]: {k: v for k, v in w.items()} for w in wishlist}, indent=1), encoding="utf-8")
    if loo is not None:
        (out / "jobcost_validation.json").write_text(json.dumps(loo, indent=1), encoding="utf-8")
    for j in c["jobs"]:                                              # queue-format specs (trainmix target_from_spec); not placed in the live queue
        s = j["settings"]
        (out / "specs" / f"{j['name']}.json").write_text(json.dumps({
            "name": j["name"], "size": j["size"], "sources": [m["name"] for m in j["data"]["mixes"]], "epochs": s["epochs"], "batch": s["batch"],
            "accum": s["accum"], "max_seq": s["max_seq"], "packing": s["packing"], "lora_serve": False, "priority": c["schedule"]["order"].index(j["id"]),
            "min_rows": 200, "vram_mib": int(j["peak_vram"]["gb"] * 1024), "why": j["value_basis"], "value": f"${j['value_usd']} (assigned)",
            "thin_mix": j["thin_mix"]["thin"], "gen_rows": j["data"]["gen_rows"]}, indent=1), encoding="utf-8")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main(argv: Optional[Sequence[str]] = None) -> int:
    """python -m creator.gpucompile phase2 [outdir]   (default <runtime>/gpuday/phase2_package)"""
    import sys
    a = list(argv if argv is not None else sys.argv[1:])
    if not a or a[0] != "phase2":
        print(main.__doc__)
        return 2
    out = Path(a[1]) if len(a) > 1 else GE.RUNTIME / "gpuday" / "phase2_package"
    built = GE.build()
    wl = phase2_wishlist(GE.mix_root())
    c = compile_wishlist(wl, built)
    loo = JC.leave_one_out(built["history"])
    files = write_package(c, out, wl, loo)
    pl = place_all(c, state_dir=out)
    (out / "PLACEMENT.json").write_text(json.dumps(pl, indent=1, default=str), encoding="utf-8")
    print(json.dumps(files + ["PLACEMENT.json"], indent=1))
    print(json.dumps({"proposal": pl["proposal"], "placements": {d["job"]: d["placement"] for d in pl["decisions"]}}, indent=1))
    print(render_timeline(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
