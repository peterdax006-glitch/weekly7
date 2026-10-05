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
BF16_GB = {"0.6b": 1.2, "1.7b": 3.4, "4b": 8.0, "30b-a3b": 18.6}     # "30b-a3b": the Q4_K_M GGUF the pod serves (never trained here)
Q4_GB = {"0.6b": 0.4, "1.7b": 1.1, "4b": 2.5, "30b-a3b": 0.0}
# Inference on the pod (llama.cpp servers with parallel slots). ALL ASSUMPTIONS (no server throughput was ever logged); the smoke-first step probes
# each server for 60 s and the time-boxed jobs rescale their row counts to the measured tokens/s.
INFER_TOK_S = {"0.6b": 6000.0, "1.7b": 3500.0, "30b-a3b": 1200.0}     # aggregate decode tok/s over the slots
INFER_LOAD_S = {"0.6b": 20.0, "1.7b": 30.0, "30b-a3b": 150.0}
INFER_GB = {"0.6b": 1.5, "1.7b": 2.9, "30b-a3b": 20.6}              # Q8_0 / Q4_K_M GGUF weights + KV for the slots
VERIFY_S_PER_ATTEMPT = 1.5                                           # ASSUMPTION: run one candidate against its tests on a pod CPU worker
SMOKE_STEPS = 50
SMOKE_OVERHEAD_S = 15.0                                              # adapter re-init + a 64-row dev loss per candidate (ASSUMPTION)
SMOKE_TOL = 0.03                                                     # a candidate may not be worse than the safe setting's dev loss by more than this
SMOKE_MIN_GAIN = 1.03                                                # ...and must be at least this much faster to be worth the switch
PERSISTENT_LOAD_FACTOR = 0.2                                         # ASSUMPTION (expected scenario only): a warm trainer reloads 20% of the cold load time
EVAL_CLIENT_FACTOR = 0.5                                             # ASSUMPTION (expected scenario only): the eval client on the pod halves the eval time
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
    mixes[names], extra{name, rows, mean, p80, p99, max, family, gen: bool} or extras[...], target_rows (optional), min_eff_batch (optional: a small
    curated mix may use a smaller effective batch, 8 or 4, so it still gets THIN_STEPS optimizer steps - never larger than 16)}. An extra with
    `source_job` is produced by another job of the package (a teacher / dialogue job on the GPU): it adds that job to `deps` and needs no pod-CPU
    generation. `kind: infer` / `smoke` wishlist entries go to compile_infer_job / compile_smoke."""
    if w.get("kind") == "infer":
        return compile_infer_job(w, usd_h)
    parts = [_mix(m, mixes, hashing) for m in w.get("mixes", [])]
    exs = list(w.get("extras") or ([w["extra"]] if w.get("extra") else []))
    rows = sum(p["rows"] for p in parts)
    lens = list(parts)
    deps = list(w.get("deps") or [])
    extra_tokens = 0.0
    gen_rows = 0
    for ex in exs:
        target = int(ex.get("rows_target") or ex["rows"])
        lens.append({"rows": target, "len": {k: ex[k] for k in ("mean", "p80", "p90", "p99", "max")}})
        rows += target
        extra_tokens += target * ex["mean"]
        if ex.get("source_job"):
            deps.append(ex["source_job"])
        elif ex.get("gen"):
            # a planted-row generator keeps only `survival` of what it makes after curation (template cap, easy rows): ask for more raw rows
            gen_rows += int(math.ceil((ex["rows_target"] - ex.get("rows_have", 0)) / max(0.05, float(ex.get("survival", 1.0)))))
    deps = sorted(set(deps))
    stats = _merge_len(lens)
    shares = {f: float(s) for f, s in (w.get("families") or {}).items()}
    epochs, ep_basis = mix_epochs(eff, shares, w.get("epoch_prior") or {})
    st = GE.settings(eff, w["size"], next(iter(shares), "other"), rows, sum(p["tokens_train"] for p in parts) + extra_tokens, stats)
    eff_batch = GE.DEFAULT_BATCH
    floor = max(4, int(w.get("min_eff_batch") or GE.DEFAULT_BATCH))
    while eff_batch > floor and math.ceil(rows / eff_batch) * epochs < GE.THIN_STEPS:
        eff_batch //= 2                                            # fewer rows per optimizer step so a small curated mix still gets its steps
    eff_batch = max(eff_batch, 4)
    accum = max(1, eff_batch // st["batch"])
    st.update(epochs=epochs, epochs_basis=ep_basis, accum=accum, eff_batch=st["batch"] * accum, steps=math.ceil(rows / (st["batch"] * accum)) * epochs)
    steps = st["steps"]
    padded = steps * st["batch"] * accum * stats["p80"]
    train_s = eff.train_s(w["size"], padded)
    thin = steps < GE.THIN_STEPS
    min_rows = math.ceil(GE.THIN_STEPS * st["batch"] * accum / epochs)
    peak = eff.peak_gb(w["size"], stats["max"])
    big = w["size"]
    p_b = GE.PARAMS_B[big]
    merge_s = (MERGE_S_17B + GGUF_S_17B) * p_b / 1.7
    ev = hist.predict("roleeval_base")["seconds"] if hist.predict("roleeval_base") else 90.0
    train_bytes = sum(p["train_bytes"] for p in parts) + extra_tokens * 3.6
    gen_s = gen_rows * GEN_S_PER_ROW / GEN_WORKERS
    have = sum(p["rows"] for p in parts)
    st["smoke_candidates"] = GE.candidate_settings(w["size"], peak["gb"], float(peak.get("len") or stats["max"]), stats["max"], base_batch=st["batch"], base_accum=accum)
    return {
        "id": w["id"], "name": w["name"], "kind": "train", "family": w.get("family", ""), "size": w["size"], "value_usd": float(w.get("value_usd", 0.0)),
        "deps": deps, "deadline": w.get("deadline", ""), "rows": rows, "len": stats, "settings": st,
        "thin_mix": {"steps": steps, "min_steps": GE.THIN_STEPS, "thin": thin, "min_rows_at_these_epochs": min_rows,
                     "rows_needed": max(0, min_rows - have) if thin or gen_rows else 0},
        "data": {"mixes": [{k: p[k] for k in ("name", "rows", "tokens_train", "train_bytes", "sha256", "dev_rows")} for p in parts],
                 "gen_rows": gen_rows, "gen_s": gen_s, "archive_mb": round(train_bytes / 2**20 / 4, 1),
                 "fed_by": [ex["source_job"] for ex in exs if ex.get("source_job")]},
        "predicted": {"train_s": round(train_s, 1), "padded_tokens": round(padded), "merge_gguf_s": round(merge_s, 1),
                      "tuned_eval_s": round(ev, 1), "ship_gb": Q4_GB[big], "ship_s": round(Q4_GB[big] * 1024 / SHIP_MB_S, 1),
                      "base_download_s": round(BF16_GB[big] * 1024 / DL_MB_S, 1)},
        "peak_vram": peak,
        "value_basis": w.get("value_basis", ""), "usd_h": usd_h,
        "output": {"merge": "LoRA merged into the bf16 base on the pod (never a Q4 base: that erased adapters)", "gguf": "Q4_K_M merged, shipped home",
                   "also_ship": "LoRA adapter (re-merge at home only from a Q8_0/F16 base)"},
    }


def compile_infer_job(w: Mapping[str, Any], usd_h: float) -> dict[str, Any]:
    """A GPU job that generates tokens instead of training: `servers` [{label, size, tokens}] run in parallel (their VRAM adds up; the card is shared
    at PAIR_SLOWDOWN), `verify_attempts` candidate solutions are tested on the pod CPU workers beside the generation (pipelined). `warm_from`: the
    server of that job is still loaded (no load time). `yields` names the rows it makes and the jobs that train on them."""
    servers = list(w["servers"])
    warm = bool(w.get("warm_from"))
    per = [(0.0 if warm and sv["size"] == w.get("warm_size") else INFER_LOAD_S[sv["size"]]) + sv["tokens"] / INFER_TOK_S[sv["size"]] for sv in servers]
    gen_s = max(per) * (PAIR_SLOWDOWN if len(servers) > 1 else 1.0)
    verify_s = w.get("verify_attempts", 0) * VERIFY_S_PER_ATTEMPT / GEN_WORKERS
    occupancy = max(gen_s, verify_s)
    size = max((sv["size"] for sv in servers), key=lambda z: BF16_GB[z])
    vram = round(sum(INFER_GB[sv["size"]] for sv in servers), 2)
    return {"id": w["id"], "name": w["name"], "kind": "infer", "family": w.get("family", "infer"), "size": size, "value_usd": float(w.get("value_usd", 0.0)),
            "deps": sorted(set(w.get("deps") or [])), "deadline": w.get("deadline", ""), "rows": int(w.get("yield_rows", 0)),
            "len": {"mean": 0.0, "p80": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "rows": 0},
            "settings": {"epochs": 0, "steps": 0, "batch": 0, "accum": 0, "max_seq": 0, "grad_ckpt": False, "packing": False, "group_by_length": False,
                         "epochs_basis": "inference job", "pending": [], "smoke_candidates": []},
            "thin_mix": {"steps": 0, "min_steps": 0, "thin": False, "min_rows_at_these_epochs": 0, "rows_needed": 0},
            "data": {"mixes": [], "gen_rows": 0, "gen_s": 0.0, "archive_mb": 0.0, "fed_by": []},
            "infer": {"servers": [{**sv, "tok_s": INFER_TOK_S[sv["size"]], "load_s": INFER_LOAD_S[sv["size"]]} for sv in servers], "gen_s": round(gen_s, 1),
                      "verify_cpu_s": round(verify_s, 1), "yields": w.get("yields", ""), "yield_rows": int(w.get("yield_rows", 0)),
                      "time_box_s": w.get("time_box_s"), "warm_from": w.get("warm_from", ""), "note": w.get("note", "")},
            "predicted": {"train_s": round(occupancy, 1), "padded_tokens": 0, "merge_gguf_s": 0.0, "tuned_eval_s": 0.0, "ship_gb": 0.0,
                          "ship_s": round(w.get("yield_rows", 0) * 2.0 / 1024 / SHIP_MB_S, 1), "base_download_s": round(BF16_GB[size] * 1024 / DL_MB_S, 1)},
            "peak_vram": {"gb": vram, "basis": "GGUF weights + KV per server (INFER_GB, an estimate; smoke-first measures)"},
            "value_basis": w.get("value_basis", ""), "usd_h": usd_h}


def compile_smoke(size: str, jobs: Sequence[Mapping[str, Any]], eff: GE.Eff, usd_h: float) -> dict[str, Any]:
    """The first GPU job of a size: SMOKE_STEPS steps of every candidate setting on the size's most demanding module (same data, same seed, same effective
    batch); records tok/s, peak VRAM and dev loss per candidate. pick_setting() then chooses the fastest one that does not regress dev loss."""
    mine = [j for j in jobs if j.get("kind") == "train" and j["size"] == size]
    ref = max(mine, key=lambda j: j["len"]["p80"])
    cands = ref["settings"]["smoke_candidates"]
    step_tok = SMOKE_STEPS * ref["settings"]["batch"] * ref["settings"]["accum"] * ref["len"]["p80"]
    secs = eff.load_s.get(size, 60.0) + len(cands) * (step_tok / eff.rate[size] + SMOKE_OVERHEAD_S)
    return {"id": f"p2.smoke_{size.replace('.', '')}", "name": f"smoke_{size}", "kind": "smoke", "family": "smoke", "size": size, "value_usd": 0.0, "deps": [], "deadline": "",
            "rows": 0, "len": ref["len"], "settings": {"epochs": 0, "steps": SMOKE_STEPS * len(cands), "batch": 0, "accum": 0, "max_seq": 0, "grad_ckpt": True,
                                                       "packing": False, "group_by_length": False, "epochs_basis": "smoke", "pending": [], "smoke_candidates": []},
            "thin_mix": {"steps": 0, "min_steps": 0, "thin": False, "min_rows_at_these_epochs": 0, "rows_needed": 0},
            "data": {"mixes": [], "gen_rows": 0, "gen_s": 0.0, "archive_mb": 0.0, "fed_by": []},
            "smoke": {"reference_module": ref["name"], "steps_each": SMOKE_STEPS, "candidates": cands, "tolerance": SMOKE_TOL, "min_gain": SMOKE_MIN_GAIN,
                      "measures": ["tokens_per_s", "peak_vram_gb", "dev_loss_at_step_50", "oom_or_nan"],
                      "rule": "fastest candidate with no OOM/NaN, peak <= SAFE_VRAM_GB, dev loss <= safe x (1 + tolerance), >= min_gain faster than safe; else safe"},
            "predicted": {"train_s": round(secs, 1), "padded_tokens": round(step_tok * len(cands)), "merge_gguf_s": 0.0, "tuned_eval_s": 0.0, "ship_gb": 0.0, "ship_s": 0.0,
                          "base_download_s": round(BF16_GB[size] * 1024 / DL_MB_S, 1)},
            "peak_vram": {"gb": 31.0, "basis": "measurement run: never paired"}, "value_basis": "measures which speed settings are safe and fast before the long runs", "usd_h": usd_h}


def compile_probe(size: str, usd_h: float) -> dict[str, Any]:
    """Smoke-first for an inference server: a 60 s throughput probe, so the time-boxed jobs rescale their row counts to the real tokens/s."""
    j = compile_infer_job({"id": f"p2.probe_{size.replace('.', '').replace('-', '')}", "name": f"probe_{size}", "kind": "infer", "family": "smoke",
                           "servers": [{"label": f"{size} probe", "size": size, "tokens": 60.0 * INFER_TOK_S[size]}], "deps": [],
                           "value_basis": "measures the server's real tokens/s (60 s) so the time-boxed teacher jobs size themselves"}, usd_h)
    j["kind"] = "smoke"
    j["peak_vram"] = {"gb": 31.0, "basis": "measurement run: never paired"}
    return j


def pick_setting(results: Sequence[Mapping[str, Any]], tol: float = SMOKE_TOL, min_gain: float = SMOKE_MIN_GAIN, vram_cap: float = GE.SAFE_VRAM_GB) -> dict[str, Any]:
    """The GPU smoke-first decision. `results`: one dict per tried candidate {name, ok, tok_s, peak_gb, dev_loss}. The 'safe' result is the reference:
    a candidate is eligible only when it ran (ok), fits the card, did not regress dev loss by more than `tol`, and is >= `min_gain` faster; the fastest
    eligible one wins. Without a usable safe result, or without an eligible candidate, the answer is 'safe' (never a guess)."""
    by = {r["name"]: r for r in results}
    safe = by.get("safe")
    if not safe or not safe.get("ok") or not safe.get("tok_s"):
        return {"choice": "safe", "speedup": 1.0, "why": "no usable safe measurement", "rejected": []}
    rejected: list[tuple[str, str]] = []
    best: Optional[Mapping[str, Any]] = None
    for r in results:
        if r["name"] == "safe":
            continue
        if not r.get("ok") or not r.get("tok_s"):
            rejected.append((r["name"], "failed (OOM / NaN / crash)"))
        elif r.get("peak_gb", 0.0) > vram_cap:
            rejected.append((r["name"], f"peak {r['peak_gb']} GB > {vram_cap}"))
        elif r.get("dev_loss") is None or safe.get("dev_loss") is None or r["dev_loss"] > safe["dev_loss"] * (1.0 + tol):
            rejected.append((r["name"], "dev loss regressed beyond the tolerance"))
        elif r["tok_s"] < safe["tok_s"] * min_gain:
            rejected.append((r["name"], "not faster"))
        elif best is None or r["tok_s"] > best["tok_s"]:
            best = r
    if best is None:
        return {"choice": "safe", "speedup": 1.0, "why": "no candidate beat safe within the tolerance", "rejected": rejected}
    return {"choice": best["name"], "speedup": round(best["tok_s"] / safe["tok_s"], 3), "why": "fastest eligible candidate", "rejected": rejected}


def execute_with_safe_retry(run: Any, chosen: Mapping[str, Any], safe: Mapping[str, Any]) -> dict[str, Any]:
    """The first-run safety net: `run(setting) -> {ok, reason}` is the real training. A failure (OOM, NaN, loss spike, hang) with a speed option
    restarts ONCE with the safe setting (automatic safe-mode retry); a failure of the safe setting itself is returned, not retried."""
    first = run(chosen)
    if first.get("ok") or chosen.get("name") == safe.get("name"):
        return {"ok": bool(first.get("ok")), "used": chosen.get("name"), "attempts": [dict(first, setting=chosen.get("name"))], "retried": False}
    second = run(safe)
    return {"ok": bool(second.get("ok")), "used": safe.get("name"), "retried": True,
            "attempts": [dict(first, setting=chosen.get("name")), dict(second, setting=safe.get("name"))]}


def expected_variant(jobs: Sequence[Mapping[str, Any]], eff: GE.Eff) -> list[dict[str, Any]]:
    """The `expected` scenario: every training job at its best smoke candidate's expected speed-up (SPEED_EXPECT, UNVERIFIED) on the token part of the
    time (the load seconds do not shrink), tuned evals through the pod-side client. The `safe` scenario never uses any of this."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for j in jobs:
        k = json.loads(json.dumps(j, default=str))
        if k.get("kind") == "train" and k["settings"]["smoke_candidates"]:
            mult = max(c["expect_speedup"] for c in k["settings"]["smoke_candidates"])
            load = eff.load_s.get(k["size"], 60.0)
            body = k["predicted"]["train_s"] - load
            warm = k["size"] in seen                                  # one trainer process per size: the next job swaps the adapter instead of reloading the base
            seen.add(k["size"])
            k["predicted"]["train_s"] = round((load * PERSISTENT_LOAD_FACTOR if warm else load) + body / mult, 1)
            k["predicted"]["persistent_trainer"] = warm
            k["predicted"]["expected_speedup"] = mult
            k["predicted"]["tuned_eval_s"] = round(k["predicted"]["tuned_eval_s"] * EVAL_CLIENT_FACTOR, 1)
        out.append(k)
    return out


def can_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a["peak_vram"]["gb"] + b["peak_vram"]["gb"] + 2.0 + EVAL_SERVER_GB <= VRAM_TOTAL_GB


def schedule(jobs: Sequence[Mapping[str, Any]], cached_evals: Sequence[str], usd_h: float, boot_s: float = BOOT_S, pair: bool = True) -> dict[str, Any]:
    """Greedy timeline. Lanes: downloads (sequential, in order of first use), pod CPU (data jobs), GPU (training slots, pairs when VRAM allows),
    eval (sequential, beside the next training), ship (home transfers, overlapping). Returns per-job times and the totals."""
    by_id = {j["id"]: j for j in jobs}
    dens = lambda j: j["value_usd"] / max(1.0, j["predicted"]["train_s"])          # noqa: E731
    # downloads: bases in order of first use (value density, data-ready jobs first); the first one is on the critical path
    train_sizes = {j["size"] for j in jobs if j.get("kind", "train") != "infer"}
    first = sorted(jobs, key=lambda j: (j["size"] not in train_sizes, j["data"]["gen_rows"] > 0 and j.get("kind", "train") == "train",
                                        -dens(j) if j.get("kind", "train") == "train" else 0.0, GE.PARAMS_B.get(j["size"], 1.0)))
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
        a = max(near, key=lambda j: (sum(1 for o in jobs if j["id"] in o["deps"]), j["predicted"]["merge_gguf_s"] + j["predicted"]["ship_s"], dens(j)))
        # unblock dependents first; then the job with the longest merge/ship tail (it overlaps the next GPU job; a tail-less job goes last); then value density
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
        if j.get("kind", "train") != "train":                          # smoke / inference jobs: nothing to merge or evaluate; their rows ship home
            s1 = max(ship_t, tl[i]["train_end"]) + p["ship_s"]
            ship_t = s1
            tl[i].update(merged=tl[i]["train_end"], pre_evals_s=0.0, eval_end=tl[i]["train_end"], shipped=round(s1, 1), done=round(max(tl[i]["train_end"], s1), 1))
            continue
        n_cached = sum(1 for k in ("base", "generalist") if f"{j['name']}:{k}" in cached_evals)
        pre = (2 - n_cached) * p["tuned_eval_s"]                     # base + generalist evals not in the cache: once, before the tuned eval
        post = tl[i]["train_end"] + p["merge_gguf_s"]
        e0 = max(eval_t, dl_end[j["size"]]) + pre
        e1 = max(e0, post) + p["tuned_eval_s"] + 6.0
        s1 = max(ship_t, post) + p["ship_s"]
        eval_t, ship_t = e1, s1
        tl[i].update(merged=round(post, 1), pre_evals_s=round(pre, 1), eval_end=round(e1, 1), shipped=round(s1, 1), done=round(max(e1, s1), 1))
    make = max([v["done"] for v in tl.values()] + [gpu_t]) + TEARDOWN_S
    first_start = min((v["train_start"] for v in tl.values()), default=boot_s)
    return {"order": order, "slots": slots, "timeline": tl, "makespan_s": round(make, 1), "gpu_train_s": round(busy, 1),
            "gpu_busy_pct": round(100.0 * busy / make, 1),
            "gpu_busy_after_first_start_pct": round(100.0 * busy / max(1.0, make - first_start - TEARDOWN_S), 1),
            "gpu_idle_gaps_s": round(sum(max(0.0, tl[b]["train_start"] - tl[a]["train_end"]) for a, b in zip(order, order[1:])
                                         if not tl[a]["paired"] and not tl[b]["paired"]), 1),
            "usd": round(make / 3600.0 * usd_h, 3), "usd_h": usd_h,
            "data_lane_end_s": round(cpu_t, 1), "downloads": {k: round(v, 1) for k, v in dl_end.items()}}


def rows_in_box(box_s: float, size: str, tokens_per_row: float, load_s: Optional[float] = None) -> int:
    """How many rows a time-boxed generation job yields at the (assumed, later measured) server throughput."""
    ld = INFER_LOAD_S[size] if load_s is None else load_s
    return max(0, int((box_s - ld) * INFER_TOK_S[size] / tokens_per_row))


TEACHER_BOX_S = 1500.0               # time box of the teacher solving job (rows scale with the measured tokens/s)
TEACHER_ATTEMPTS = 2.0               # ASSUMPTION: mean attempts per verified row (60% pass x up to 3 samples)
TEACHER_OUT_TOK = 650.0              # ASSUMPTION: mean tokens of one teacher attempt
VOICE_ROWS = 3600                    # 225 optimizer steps at an effective batch of 16 (>= THIN_STEPS)
VOICE_TOK = 250.0                    # mean tokens of one model-written persona dialogue
DEBUG_MW_ROWS = 1214                 # curated debug_fix rows whose template REASONING line the teacher rewrites in its own words
DEBUG_MW_TOK = 150.0                 # reasoning line out + the prefill share of the prompt, per row


def _man(mixes: Path, name: str) -> dict[str, Any]:
    try:
        return dict(json.loads((Path(mixes) / name / "MANIFEST.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


from creator.pinned_extras import pinned  # noqa: E402


@pinned
def phase2_wishlist(mixes: Path) -> list[dict[str, Any]]:
    """MASTER_BLUEPRINT 12 Phase 2 as wishlist entries, on the CURATED mixes of creator.gpucurate (data that exists is used as it is; the planted-bug
    generator only tops PINPOINT up to the rows its thin-mix rule needs). Plus the owner's 5 Oct additions: failure mining, the teacher (30B-A3B) solving
    what Nupen fails, model-written DEBUG_FIX / persona data, voice persona training, and the big 203-task suite eval on four parallel servers."""
    pm = _man(mixes, "pinpoint.c")
    cur = (pm.get("curation") or {})
    survival = (cur.get("rows_after", 1) / cur["rows_before"]) if cur.get("rows_before") else 0.31
    pin_rows = int(pm.get("rows", {}).get("train", 0)) or 1000
    pin_floor = 1600                                               # 200 steps at an effective batch of 8
    pin_top = {"name": "pinpoint_topup", "rows": 0, "rows_have": 0, "rows_target": max(0, pin_floor - pin_rows), "mean": 700.0, "p80": 840.0, "p90": 860.0,
               "p99": 930.0, "max": 1000.0, "gen": True, "survival": round(survival, 3)}
    teacher_rows = rows_in_box(TEACHER_BOX_S, "30b-a3b", TEACHER_ATTEMPTS * TEACHER_OUT_TOK)
    teacher = {"name": "teacher_verified", "rows": 0, "rows_target": teacher_rows, "mean": 800.0, "p80": 950.0, "p90": 1100.0, "p99": 1500.0, "max": 1800.0,
               "source_job": "p2.teacher_moe"}
    voice = {"name": "voice_persona", "rows": 0, "rows_target": VOICE_ROWS, "mean": 380.0, "p80": 450.0, "p90": 520.0, "p99": 700.0, "max": 900.0,
             "source_job": "p2.dialogue_moe"}
    n_code = int(_man(mixes, "code_verified.c").get("rows", {}).get("train", 1400))
    n_dbg = int(_man(mixes, "debug_fix.c").get("rows", {}).get("train", DEBUG_MW_ROWS))
    mine_tok = n_code * 400.0 + n_dbg * 200.0                      # base 1.7B, one attempt per train-side task
    suite_small = 150 * 500.0 + 53 * 3000.0                         # fn tasks + app tasks, one attempt each
    suite_moe = 203 * 2500.0                                        # the MoE thinks longer
    cal = _man(mixes, "calib_17b.c").get("rows", {}).get("train", 7638)
    jud = _man(mixes, "judge_06b.c").get("rows", {}).get("train", 2174)
    return [
        {"id": "p2.pinpoint_06b", "name": "pinpoint_06b", "family": "pinpoint", "size": "0.6b", "mixes": ["pinpoint.c"], "extras": [pin_top] if pin_top["rows_target"] else [],
         "families": {"judge": 1.0}, "min_eff_batch": 8, "value_usd": 12.0, "deadline": "",
         "value_basis": "blueprint 2.0: debug is the weakest pipeline step (DEBUG tuned accuracy 0.155); priority-assigned, replace with measured saving"},
        {"id": "p2.checker_06b", "name": "checker_06b", "family": "checker", "size": "0.6b", "mixes": ["calib_17b.c", "judge_06b.c"],
         "families": {"calib": cal / (cal + jud), "judge": jud / (cal + jud)}, "value_usd": 10.0, "deps": [],
         "value_basis": "CHECKER slot: calib ECE 0.40 -> 0.026 and judge ADOPTED exist on 1.7B; the 0.6B merged slot is ~3x cheaper per call at home"},
        {"id": "p2.thinker_17b", "name": "thinker_17b", "family": "thinker", "size": "1.7b", "mixes": ["thinker_17b.c"],
         "families": {"thinker": 1.0}, "epoch_prior": {"thinker": 1}, "value_usd": 6.0,
         "value_basis": "THINKER slot: short forms (5-32 tok) instead of 865-1193 tok reasoning; no 2nd-epoch measurement yet -> 1 epoch (token budget)"},
        {"id": "p2.coder_17b", "name": "coder_17b", "family": "coder", "size": "1.7b", "mixes": ["coderonly_17b.c", "debug_fix.c", "code_verified.c"], "extras": [teacher],
         "families": {"code": 1.0}, "value_usd": 10.0, "deps": ["p2.pinpoint_06b", "p2.dialogue_moe"],
         "value_basis": "CODER slot ties the generalist at 25% fewer tokens (coderonly_17b); DEBUG_FIX (model-written reasoning) + teacher-verified rows add the weakest step"},
        {"id": "p2.voice_persona_17b", "name": "voice_persona_17b", "family": "voice", "size": "1.7b", "mixes": [], "extras": [voice], "families": {"voice": 1.0},
         "epoch_prior": {"voice": 2}, "value_usd": 4.0,
         "value_basis": "owner 5 Oct: Jarvis-style persona for the talk slot, from original model-written dialogues (the iPhone voice is the fallback)"},
        {"id": "p2.mine_fail_17b", "name": "mine_fail_17b", "kind": "infer", "family": "infer", "servers": [{"label": "base 1.7B", "size": "1.7b", "tokens": mine_tok}],
         "verify_attempts": n_code + n_dbg, "yields": "the train-side tasks the base 1.7B fails (the teacher's worklist)", "yield_rows": int((n_code + n_dbg) * 0.6),
         "value_usd": 5.0, "note": "tasks from phase2_data and the suite TRAIN-side sources only; eval_suite200 and the 32-task suite are excluded (EXCLUDE.json)",
         "value_basis": "teacher time is spent only on what Nupen fails"},
        {"id": "p2.teacher_moe", "name": "teacher_moe", "kind": "infer", "family": "infer", "servers": [{"label": "MoE 30B-A3B teacher", "size": "30b-a3b",
                                                                                                        "tokens": teacher_rows * TEACHER_ATTEMPTS * TEACHER_OUT_TOK}],
         "verify_attempts": int(teacher_rows * TEACHER_ATTEMPTS), "yields": "verified teacher solutions -> coder_17b training rows", "yield_rows": teacher_rows,
         "time_box_s": TEACHER_BOX_S, "deps": ["p2.mine_fail_17b"], "value_usd": 10.0, "note": "27B dense is slower per token; the A3B MoE does the same job in less time",
         "value_basis": "teacher distillation on the failures only (blueprint 8.4); every row re-run against its tests before it is kept"},
        {"id": "p2.dialogue_moe", "name": "dialogue_moe", "kind": "infer", "family": "infer", "warm_from": "p2.teacher_moe", "warm_size": "30b-a3b",
         "servers": [{"label": "MoE 30B-A3B writer", "size": "30b-a3b", "tokens": DEBUG_MW_ROWS * DEBUG_MW_TOK + VOICE_ROWS * VOICE_TOK}],
         "yields": "model-written DEBUG_FIX reasoning + persona dialogues -> coder_17b and voice_persona_17b", "yield_rows": DEBUG_MW_ROWS + VOICE_ROWS,
         "deps": ["p2.teacher_moe"], "value_usd": 6.0, "value_basis": "mass model-written data where the template text is the weak part"},
        {"id": "p2.suite_eval", "name": "suite_eval", "kind": "infer", "family": "infer",
         "servers": [{"label": "base 1.7B", "size": "1.7b", "tokens": suite_small}, {"label": "trained CODER 1.7B", "size": "1.7b", "tokens": suite_small},
                     {"label": "0.6B", "size": "0.6b", "tokens": suite_small}, {"label": "MoE 30B-A3B", "size": "30b-a3b", "tokens": suite_moe}],
         "verify_attempts": 203 * 4, "yields": "pass rate per model on the 203-task suite (eval_suite200) = the north-star measurement", "yield_rows": 0,
         "deps": ["p2.coder_17b"], "value_usd": 8.0, "note": "four servers in parallel on one card (GGUF Q8 / Q4); the harness runs on the pod CPU workers",
         "value_basis": "measures cost per solved task of the trained CODER against the MoE and the base models; nothing from this suite is ever trained on"},
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
        if j.get("kind", "train") != "train":
            continue
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


def add_measurement_jobs(jobs: list[dict[str, Any]], eff: GE.Eff, usd_h: float) -> list[dict[str, Any]]:
    """GPU smoke-first (owner 5 Oct): the first GPU job of every training size measures the speed options on 50 steps each (compile_smoke) and
    pick_setting() chooses; every training job of that size waits for it. Every inference size gets a 60 s throughput probe the same way."""
    out = [dict(j) for j in jobs]
    new: list[dict[str, Any]] = []
    for size in sorted({j["size"] for j in out if j.get("kind", "train") == "train"}, key=lambda z: GE.PARAMS_B[z]):
        sm = compile_smoke(size, out, eff, usd_h)
        new.append(sm)
        for j in out:
            if j.get("kind", "train") == "train" and j["size"] == size:
                j["deps"] = sorted(set(j["deps"]) | {sm["id"]})
    for size in sorted({sv["size"] for j in out if j.get("kind") == "infer" for sv in j["infer"]["servers"] if sv["size"] in ("30b-a3b",)}):
        pr = compile_probe(size, usd_h)
        new.append(pr)
        for j in out:
            if j.get("kind") == "infer" and any(sv["size"] == size for sv in j["infer"]["servers"]):
                j["deps"] = sorted(set(j["deps"]) | {pr["id"]})
    return new + out


def compile_wishlist(wishlist: Sequence[Mapping[str, Any]], built: Optional[Mapping[str, Any]] = None, mixes: Optional[Path] = None,
                     hashing: bool = True, smoke_first: bool = False) -> dict[str, Any]:
    """smoke_first: add the measurement jobs (add_measurement_jobs) and also schedule the `expected` scenario (speed options at their expected
    gain). The primary `schedule` is always the SAFE scenario: only settings proven on the GPU on 3-4 Oct, nothing assumed faster."""
    built = built or GE.build()
    mixes = Path(mixes or GE.mix_root())
    usd_h = built["usd_h"] or JC.DEFAULT_GPU_USD_H
    eff, hist = built["eff"], JC.CostModel(built["history"])
    jobs = [compile_job(w, eff, hist, mixes, usd_h, hashing) for w in wishlist]
    if smoke_first:
        jobs = add_measurement_jobs(jobs, eff, usd_h)
    cache = cached_evals(built["jobs"])
    paired = schedule(jobs, cache, usd_h)
    serial = schedule(jobs, cache, usd_h, pair=False)
    sched = paired if paired["makespan_s"] < serial["makespan_s"] else serial          # pair only when the whole timeline gets shorter
    out = {"jobs": jobs, "schedule": sched, "paired_makespan_s": paired["makespan_s"], "unpaired_makespan_s": serial["makespan_s"], "unpaired_usd": serial["usd"],
           "efficiency": {"rate_tok_s": {k: round(v) for k, v in eff.rate.items()}, "load_s": {k: round(v) for k, v in eff.load_s.items()},
                          "epoch_gain": eff.gains(), "validation": eff.validate(built["jobs"])},
           "assumptions": {"pair_slowdown": PAIR_SLOWDOWN, "boot_s": BOOT_S, "ship_mb_s": SHIP_MB_S, "download_mb_s": DL_MB_S,
                           "gen_s_per_row": GEN_S_PER_ROW, "gen_workers": GEN_WORKERS, "usd_h": usd_h,
                           "unmeasured_speedups_not_applied": ["group_by_length", "no grad-ckpt", "micro-batch 8", "packing", "flash-attn", "pod-side eval client"],
                           "infer_tok_s": INFER_TOK_S, "infer_load_s": INFER_LOAD_S, "infer_gb": INFER_GB, "verify_s_per_attempt": VERIFY_S_PER_ATTEMPT},
           "cached_evals": cache}
    if smoke_first:
        exp_jobs = expected_variant(jobs, eff)
        e_paired, e_serial = schedule(exp_jobs, cache, usd_h), schedule(exp_jobs, cache, usd_h, pair=False)
        e = e_paired if e_paired["makespan_s"] < e_serial["makespan_s"] else e_serial
        out["expected"] = {"schedule": e, "jobs": {j["id"]: {"train_s": j["predicted"]["train_s"], "speedup": j["predicted"].get("expected_speedup", 1.0)} for j in exp_jobs},
                           "basis": "SPEED_EXPECT multipliers (SPEED_PLAN B) on the token part of each training job; UNVERIFIED until the smoke-first step measures them"}
        out["assumptions"]["expected_scenario_applies"] = sorted(GE.SPEED_EXPECT) + ["pod-side eval client x%.1f" % EVAL_CLIENT_FACTOR, "persistent trainer: later jobs of a size load x%.1f" % PERSISTENT_LOAD_FACTOR]
    return out


def _fmt_min(x: float) -> str:
    return f"{x / 60:.1f}"


def render_timeline(c: Mapping[str, Any], before: Optional[Mapping[str, Any]] = None) -> str:
    s = c["schedule"]
    L = ["# Phase 2 package: timeline and cost (recompiled 5 Oct with the 10x data + settings work)", "",
         f"SAFE scenario (only settings proven on the GPU): makespan {s['makespan_s'] / 3600:.2f} h, GPU work {s['gpu_train_s'] / 3600:.2f} h, "
         f"GPU busy {s['gpu_busy_pct']}% of the rental ({s.get('gpu_busy_after_first_start_pct', '?')}% from the first job to the end), "
         f"cost ${s['usd']:.2f} at ${s['usd_h']}/h (unpaired: {c['unpaired_makespan_s'] / 3600:.2f} h, ${c['unpaired_usd']:.2f})."]
    if c.get("expected"):
        e = c["expected"]["schedule"]
        L.append(f"EXPECTED scenario (smoke-first picks the speed options; multipliers unverified): makespan {e['makespan_s'] / 3600:.2f} h, "
                 f"GPU busy {e['gpu_busy_pct']}% ({e.get('gpu_busy_after_first_start_pct', '?')}% from the first job), cost ${e['usd']:.2f}.")
    L += ["", "| job | size | kind | rows | epochs | steps | thin? | start (min) | GPU (min) | expected GPU (min) | merged | eval end | shipped home | peak GB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    byid = {j["id"]: j for j in c["jobs"]}
    ex_jobs = (c.get("expected") or {}).get("jobs", {})
    for i in s["order"]:
        j, t = byid[i], s["timeline"][i]
        L.append(f"| {j['name']} | {j['size']} | {j.get('kind', 'train')} | {j['rows']} | {j['settings']['epochs']} | {j['settings']['steps']} | "
                 f"{'THIN' if j['thin_mix']['thin'] else 'ok'} | {_fmt_min(t['train_start'])} | {_fmt_min(j['predicted']['train_s'])} | "
                 f"{_fmt_min(ex_jobs[i]['train_s']) if i in ex_jobs else '-'} | {_fmt_min(t['merged'])} | {_fmt_min(t['eval_end'])} | {_fmt_min(t['shipped'])} | {j['peak_vram']['gb']} |")
    if before:
        L += ["", "## Before (the 4 Oct package) -> after, training minutes per module", "", "| module | before (min) | after SAFE (min) | after EXPECTED (min) |", "|---|---|---|---|"]
        bj = {j["name"]: j for j in before["jobs"]}
        for j in c["jobs"]:
            if j.get("kind", "train") == "train" and j["name"] in bj:
                L.append(f"| {j['name']} | {_fmt_min(bj[j['name']]['predicted']['train_s'])} | {_fmt_min(j['predicted']['train_s'])} | "
                         f"{_fmt_min(ex_jobs[j['id']]['train_s']) if j['id'] in ex_jobs else '-'} |")
        L.append(f"| TOTAL rental | {before['schedule']['makespan_s'] / 60:.1f} (${before['schedule']['usd']:.2f}) | {s['makespan_s'] / 60:.1f} (${s['usd']:.2f}) | "
                 f"{c['expected']['schedule']['makespan_s'] / 60:.1f} (${c['expected']['schedule']['usd']:.2f}) |" if c.get("expected") else "")
    L += ["", "## GPU jobs that make data or measure (not trainings)"]
    for j in c["jobs"]:
        if j.get("kind") in ("infer", "smoke"):
            inf = j.get("infer")
            if inf:
                L.append(f"- {j['name']} ({j.get('kind')}): servers {', '.join(sv['label'] + ' ' + sv['size'] for sv in inf['servers'])}; GPU {_fmt_min(j['predicted']['train_s'])} min; "
                         f"pod CPU verify {_fmt_min(inf['verify_cpu_s'])} min (overlapped); yields {inf['yields'] or 'a measurement'}"
                         f"{' (time box %s min, ~%d rows)' % (_fmt_min(inf['time_box_s']), inf['yield_rows']) if inf.get('time_box_s') else ''}")
            else:
                L.append(f"- {j['name']} (smoke-first): {len(j['smoke']['candidates'])} candidates x {j['smoke']['steps_each']} steps on {j['smoke']['reference_module']}; "
                         f"GPU {_fmt_min(j['predicted']['train_s'])} min; picks the fastest with dev loss within {j['smoke']['tolerance']:.0%} of safe")
    L += ["", "## Data jobs (pod CPU)"]
    for j in c["jobs"]:
        if j["data"]["gen_rows"]:
            L.append(f"- {j['name']}: generate {j['data']['gen_rows']} raw planted rows (curation keeps ~a third), {j['data']['gen_s'] / 60:.0f} min on the pod CPU "
                     f"(thin-mix rule needs >= {j['thin_mix']['min_rows_at_these_epochs']} rows at {j['settings']['epochs']} epoch(s) x effective batch "
                     f"{j['settings'].get('eff_batch', 16)}); run a 2% pilot first")
        if j["data"].get("fed_by"):
            L.append(f"- {j['name']}: its extra rows come from {', '.join(j['data']['fed_by'])} (a GPU job; the training waits for it)")
    L += ["", "## Settings basis"]
    for j in c["jobs"]:
        if j.get("kind", "train") != "train":
            continue
        L.append(f"- {j['name']}: {j['settings']['epochs']} epoch(s) - {j['settings']['epochs_basis']}; batch {j['settings']['batch']} x {j['settings']['accum']}; "
                 f"max_seq {j['settings']['max_seq']} (from the curated p99 {j['len']['p99']:.0f}); VRAM {j['peak_vram']['gb']} GB ({j['peak_vram']['basis']})")
    L += ["", "## Assumptions (unmeasured, named)"] + [f"- {k}: {v}" for k, v in c["assumptions"].items()]
    return "\n".join(L) + "\n"


from creator.pinned_extras import pinned_written  # noqa: E402


@pinned_written
def write_package(c: Mapping[str, Any], out: Path, wishlist: Sequence[Mapping[str, Any]], loo: Optional[Mapping[str, Any]] = None,
                  before: Optional[Mapping[str, Any]] = None) -> list[str]:
    out = Path(out)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    (out / "PACKAGE.json").write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
    (out / "TIMELINE.md").write_text(render_timeline(c, before), encoding="utf-8")
    (out / "wishlist.json").write_text(json.dumps({w["id"]: {k: v for k, v in w.items()} for w in wishlist}, indent=1), encoding="utf-8")
    if loo is not None:
        (out / "jobcost_validation.json").write_text(json.dumps(loo, indent=1), encoding="utf-8")
    order = c["schedule"]["order"]
    for j in c["jobs"]:                                              # queue-format specs (trainmix target_from_spec); not placed in the live queue
        s = j["settings"]
        if j.get("kind", "train") == "train":
            spec = {"name": j["name"], "size": j["size"], "sources": [m["name"] for m in j["data"]["mixes"]], "epochs": s["epochs"], "batch": s["batch"],
                    "accum": s["accum"], "max_seq": s["max_seq"], "packing": s["packing"], "group_by_length": s["group_by_length"], "grad_ckpt": s["grad_ckpt"],
                    "lora_serve": False, "priority": order.index(j["id"]), "min_rows": 200, "vram_mib": int(j["peak_vram"]["gb"] * 1024),
                    "why": j["value_basis"], "value": f"${j['value_usd']} (assigned)", "thin_mix": j["thin_mix"]["thin"], "gen_rows": j["data"]["gen_rows"],
                    "smoke_first": {"candidates": s["smoke_candidates"], "choose": "creator.gpucompile.pick_setting", "on_failure": "execute_with_safe_retry: one restart with 'safe'"},
                    "merge": j["output"], "eval": {"cache_base_generalist": True, "pod_side_client": True, "dev_rows": "package mix dev.jsonl"},
                    "reserve_topup": "mix reserve.jsonl: appended by a follow-up job only if the held-out result misses the adopt rule", "deps": j["deps"]}
        else:
            spec = {"name": j["name"], "kind": j.get("kind"), "deps": j["deps"], "infer": j.get("infer"), "smoke": j.get("smoke"), "priority": order.index(j["id"]),
                    "vram_mib": int(j["peak_vram"]["gb"] * 1024), "why": j["value_basis"]}
        (out / "specs" / f"{j['name']}.json").write_text(json.dumps(spec, indent=1, default=str), encoding="utf-8")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and "mixes" not in p.parts and "prev" not in p.parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """python -m creator.gpucompile phase2 [outdir]   (default <runtime>/gpuday/phase2_package)
    Curates the Phase 2 data (creator.gpucurate) into <outdir>/mixes, compiles the wishlist with the smoke-first step and writes the package.
    The previous package is kept under <outdir>/prev (first run only) so the before/after table can be rendered."""
    import shutil
    import sys
    from creator import gpucurate as CU
    a = list(argv if argv is not None else sys.argv[1:])
    if not a or a[0] != "phase2":
        print(main.__doc__)
        return 2
    out = Path(a[1]) if len(a) > 1 else GE.RUNTIME / "gpuday" / "phase2_package"
    prev = out / "prev"
    if (out / "PACKAGE.json").exists() and not prev.exists():
        prev.mkdir(parents=True)
        for f in ("PACKAGE.json", "TIMELINE.md", "wishlist.json"):
            shutil.copy2(out / f, prev / f)
    before = json.loads((prev / "PACKAGE.json").read_text(encoding="utf-8")) if (prev / "PACKAGE.json").exists() else None
    built = GE.build()
    mixes = out / "mixes"
    cur = CU.build_phase2_mixes(mixes, GE.RUNTIME / "gpuday")
    wl = phase2_wishlist(mixes)
    c = compile_wishlist(wl, built, mixes, smoke_first=True)
    c["curation"] = cur
    loo = JC.leave_one_out(built["history"])
    files = write_package(c, out, wl, loo, before)
    (out / "CURATION.json").write_text(json.dumps(cur, indent=1), encoding="utf-8")
    pl = place_all(c, state_dir=out)
    (out / "PLACEMENT.json").write_text(json.dumps(pl, indent=1, default=str), encoding="utf-8")
    print(json.dumps(files + ["PLACEMENT.json", "CURATION.json"], indent=1))
    print(render_timeline(c, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
