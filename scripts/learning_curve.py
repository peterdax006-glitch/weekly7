"""Repeated-run learning curve on real revealed windows (canon C57, with C55 disguise and C58 time gating).

usage: learning_curve.py --tag NAME [--windows 6] [--K 20] [--seed 1] [--identity-windows 2] [--min-ram 2.5]

For each chosen revealed window W (mixed eras, no THIN universe): play W K times in sequence, every run under a FRESH disguise
(new order-preserving codes and a new date shift), the learner state carried forward (the long-term memory bank is never
wiped) and released to each run only as evidence whose outcome had matured by that simulated moment (TimeGate, verified after
every run). Then three other revealed years, then W three more times (do the gains survive other years?).
Controls per window: reset-state chain (no learning, K runs: disguise noise); on the first --identity-windows also the
identity-recall chain under disguise (must be flat) and undisguised (must be able to rise).
Every finished run is checkpointed (state + records), so a kill costs one run; rerunning with the same tag resumes, and a
checkpoint written by different code is discarded, never mixed. Output: state/research/learning_delta/<tag>/
{summary.json, report.md, curves.csv, ckpt/}. Only revealed windows are read.
Launch detached (PowerShell): Start-Process .venv/Scripts/python.exe -ArgumentList '-u','scripts/learning_curve.py','--tag','lc1'"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import config as K, provenance
from engine import learning_delta as L
from engine.improve import log_experiment

OUT = K.STATE / "research" / "learning_delta"
LIVE = K.STATE / "livesim"
THIN_NAMES = 500               # same rule as scripts/livesim_loop2.py: fewer names than this is a THIN (survivor) universe
N_OTHER, N_POST = 3, 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def n_names(entry):
    d = Path(entry["dir"])
    f = d / "closes_v2.parquet" if (d / "closes_v2.parquet").exists() else d / "closes.parquet"
    return len(pq.ParquetFile(f).schema.names) - 1          # minus the date index column


def choose_windows(pool, n, seed):
    """Non-THIN revealed windows spread over the calendar: evenly spaced by real start among the eligible ones."""
    ok = sorted([p for p in pool if n_names(p) >= THIN_NAMES and list(Path(p["dir"]).glob("*snap_*.parquet"))], key=lambda p: p["real_start"])
    if len(ok) <= n:
        return ok, ok
    idx = np.unique(np.linspace(0, len(ok) - 1, n).round().astype(int))
    return [ok[i] for i in idx], ok


def others_for(W, eligible, seed, k=N_OTHER):
    """k other years, preferring years that ENDED before W began (their evidence has matured), otherwise any other year."""
    rng = np.random.default_rng(L.derive_seed(seed, W["id"], "others"))
    earlier = [p for p in eligible if p["id"] != W["id"] and p["real_end"] < W["real_start"]]
    rest = [p for p in eligible if p["id"] != W["id"] and p not in earlier]
    take = [earlier[i] for i in rng.permutation(len(earlier))[:k]]
    take += [rest[i] for i in rng.permutation(len(rest))[:k - len(take)]]
    return take


def run_arm(name, entry, steps_spec, player, learner, s0_fn, out, code, seed, arm, reset=False, disguise=True, min_ram=2.5, trust_done=False):
    """One chain with per-run checkpoints. steps_spec = [(entry, tag)]; windows are loaded once per distinct id."""
    ck = out / "ckpt" / f"{name}.pkl"
    ck.parent.mkdir(parents=True, exist_ok=True)
    done = out / "ckpt" / f"{name}.done.json"
    if done.exists() and (trust_done or json.loads(done.read_text()).get("code") == code.get("code_hash")):
        return json.loads(done.read_text())
    wins = {}
    for e, _ in steps_spec:
        if e["id"] not in wins:
            wins[e["id"]] = L.load_window(e)
    W = wins[entry["id"]]
    steps = [(wins[e["id"]], t) for e, t in steps_spec]
    resume = None
    if ck.exists():
        c = pickle.loads(ck.read_bytes())
        if c.get("code") == code.get("code_hash"):
            resume = c["resume"]
            for w in wins.values():                                         # every id's shift history is part of the resume point
                w.used_shifts[:] = c["all_shifts"].get(w.id, [])
            log(f"{name}: resuming at run {resume['i'] + 1}/{len(steps)}")

    def on_run(i, rec, state, shifts):
        L.wait_for_ram(min_ram, log_fn=log)
        blob = {"code": code.get("code_hash"), "resume": {"i": i, "state": state, "recs": pickle.loads(pickle.dumps(on_run.recs + [rec])),
                                                          "shifts": shifts}, "all_shifts": {k: list(w.used_shifts) for k, w in wins.items()}}
        on_run.recs = on_run.recs + [rec]
        tmp = ck.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(blob))
        os.replace(tmp, ck)
    on_run.recs = list(resume["recs"]) if resume else []
    res = L.play_curve(W, player, learner, s0_fn(), steps, seed, arm, reset=reset, disguise=disguise, log_fn=log, resume=resume, on_run=on_run)
    out_rec = {"code": code.get("code_hash"), "recs": res["recs"], "tests": res["tests"], "final_episodes": res["state"].n_episodes()}
    done.write_text(json.dumps(out_rec, default=str))
    return out_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--identity-windows", type=int, default=2)
    ap.add_argument("--identity-K", type=int, default=8)
    ap.add_argument("--min-ram", type=float, default=2.5)
    ap.add_argument("--finalize", action="store_true", help="rebuild summary/report from finished chains without rerunning (they keep the code hash that played them)")
    a = ap.parse_args()
    out = OUT / a.tag
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "pid.txt").write_text(str(os.getpid()))
    if not L.wait_for_ram(a.min_ram, log_fn=log):
        (out / "ABORTED.txt").write_text("RAM")
        log("free RAM stayed under the floor for 20 minutes: not running")
        return 2
    code = provenance.code_stamp()
    log(f"learning curve '{a.tag}' K={a.K} seed {a.seed}, code {code.get('code_hash')}")
    sc = L.curve_selfcheck(a.seed, log_fn=log)
    (out / "selfcheck.json").write_text(json.dumps(sc, indent=1, default=str))
    pool = L.revealed_pool()
    chosen, eligible = choose_windows(pool, a.windows, a.seed)
    log(f"{len(pool)} revealed, {len(eligible)} non-THIN; chosen: {[(c['id'], c['real_start'].year) for c in chosen]}")
    st = json.loads((LIVE / "loop2.json").read_text())
    bank = pd.read_parquet(LIVE / "memory_bank.parquet") if (LIVE / "memory_bank.parquet").exists() else None
    player, learner = L.ReplayPlayer(), L.MemoryBankLearner()
    results = {}
    for wi, e in enumerate(chosen):
        s0_fn = lambda e=e: L.LearnedState(dict(st["cfg"]), dict(st["meta"]), L.causal_bank(bank, e["real_start"]), lineage=[f"S0 basis v{st['version']}"])
        others = others_for(e, eligible, a.seed)
        steps = [(e, "main")] * a.K + [(o, "other") for o in others] + [(e, "post")] * N_POST
        log(f"=== window {e['id']} ({e['real_start'].year}), {len(steps)} runs, other years {[o['id'] for o in others]}")
        r = {"entry": e, "others": [o["id"] for o in others]}
        r["main"] = run_arm(f"{e['id']}_main", e, steps, player, learner, s0_fn, out, code, a.seed, "main", min_ram=a.min_ram, trust_done=a.finalize)
        r["reset"] = run_arm(f"{e['id']}_reset", e, [(e, "main")] * a.K, player, learner, s0_fn, out, code, a.seed, "reset", reset=True, min_ram=a.min_ram, trust_done=a.finalize)
        if wi < a.identity_windows:
            zero = lambda e=e: L.LearnedState(dict(st["cfg"]), dict(st["meta"]), None)
            ik = [(e, "main")] * a.identity_K
            r["ident"] = run_arm(f"{e['id']}_ident", e, ik, L.IdentityRecallPlayer(), L.IdentityRecallLearner(), zero, out, code, a.seed, "ident", min_ram=a.min_ram, trust_done=a.finalize)
            r["ident_raw"] = run_arm(f"{e['id']}_identraw", e, ik, L.IdentityRecallPlayer(), L.IdentityRecallLearner(), zero, out, code, a.seed,
                                     "identraw", disguise=False, min_ram=a.min_ram, trust_done=a.finalize)
        results[e["id"]] = r
        log(f"window {e['id']} done: mean_week run1 {r['main']['recs'][0]['mean_week']:+.4f} -> run{a.K} {r['main']['recs'][a.K - 1]['mean_week']:+.4f}; "
            f"reset {r['reset']['recs'][0]['mean_week']:+.4f} -> {r['reset']['recs'][-1]['mean_week']:+.4f}")
    family = int(sum(r["main"]["tests"] for r in results.values()))
    npm = L.perm_budget(family)
    log(f"family of candidate patterns tried across all chains: {family}; bar {L.alpha_bar(family):.2g}; {npm} permutations")
    windows, rows = [], []
    for wid, r in results.items():
        e = r["entry"]
        mrecs = r["main"]["recs"]
        cur = L.curve_stats(mrecs, "main", a.seed, family)
        ps = lambda arm: L.curve_stats(r[arm]["recs"], "main", a.seed) if arm in r else None
        reset_c, ident_c, raw_c = ps("reset"), ps("ident"), ps("ident_raw")
        pers = L.persistence(mrecs)
        v = L.curve_verdict(cur[L.PRIMARY], reset_c[L.PRIMARY], ident_c[L.PRIMARY] if ident_c else None, pers, n_tests=family)
        if raw_c is not None and not L.curve_verdict(cur[L.PRIMARY], reset_c[L.PRIMARY], raw_c[L.PRIMARY])["label"] == "IDENTITY_LEAK":
            v["why"] += "; NOTE: the undisguised identity control did not rise, so this window cannot vouch that its control could"
        windows.append({"window": wid, "era": L.BG.era_of(e["real_start"]), "year": int(e["real_start"].year), "others": r["others"],
                        "curve": cur, "reset_curve": reset_c, "identity_curve": ident_c, "identity_raw_curve": raw_c, "persistence": pers,
                        "n_tests": r["main"]["tests"], "verdict": v})
        for arm in ("main", "reset", "ident", "ident_raw"):
            for rec in r.get(arm, {}).get("recs", []):
                rows.append({"window": wid, "chain": arm, **{k: rec[k] for k in ["i", "tag", *L.METRICS, "n_weeks", "episodes", "seconds"]}})
    pd.DataFrame(rows).to_csv(out / "curves.csv", index=False)
    summary = {"tag": a.tag, "seed": a.seed, "K": a.K, "selfcheck": sc, "n_tests_family": family, "alpha_bar": L.alpha_bar(family),
               "provenance": provenance.stamp({"cfg": st["cfg"], "meta": st["meta"]}, seed=a.seed), "windows": windows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text(L.render_curve_report(summary), encoding="utf-8")
    good = sum(w["verdict"]["label"] == "SAME_YEAR_LEARNING" for w in windows)
    log_experiment({"event": "learning_curve", "tag": a.tag, "K": a.K, "n_windows": len(windows),
                    "metrics": {"windows_rising": good, "n_tests_family": family,
                                "mean_last5_minus_first5": float(np.nanmean([w["curve"][L.PRIMARY]["diff"] for w in windows]))},
                    "gates": {"selfcheck_valid": sc["valid"], "c58_time_gate": "verified after every run"},
                    "window_ids": list(results), "outcome": "continue_testing",
                    "reason": "; ".join(f"{w['window']}:{w['verdict']['label']}" for w in windows)},
                   cfg={"cfg": st["cfg"], "meta": st["meta"]}, seed=a.seed)
    log(f"done: {good} of {len(windows)} windows show a same-year learning curve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
