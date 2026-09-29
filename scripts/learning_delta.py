"""Real-cache runner for the learning-delta harness (canon C54/C55; Bible Phases 11, 21-23, 45).

usage: learning_delta.py --tag NAME [--pairs 12] [--memo-pairs 4] [--seed 1] [--basis] [--min-ram 2.5]

Plays revealed windows twice (second time under a fresh disguise, after the system's own learning), runs the no-learning,
transfer and memoriser controls, and writes state/research/learning_delta/<tag>/{summary.json,report.md,pairs/*.json}.
Each finished pair is checkpointed, so a kill costs at most one pair; rerunning with the same tag resumes (a pair written
by different code is redone, never mixed). Launch detached:
  Start-Process .venv/Scripts/python.exe -ArgumentList '-u','scripts/learning_delta.py','--tag','ld1' -WindowStyle Hidden
Only revealed windows are ever read (engine.learning_delta.revealed_pool); state/livesim seals are not touched."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from engine import config as K, provenance
from engine import learning_delta as L
from engine.improve import log_experiment

OUT = K.STATE / "research" / "learning_delta"
LIVE = K.STATE / "livesim"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_state0():
    st = json.loads((LIVE / "loop2.json").read_text())
    bank = pd.read_parquet(LIVE / "memory_bank.parquet") if (LIVE / "memory_bank.parquet").exists() else None
    return st, bank


def state_for(W, st, bank):
    """S0 for window W: the current basis and the part of the long-term bank that ended before W began (C34)."""
    return L.LearnedState(dict(st["cfg"]), dict(st["meta"]), L.causal_bank(bank, W.real_start),
                          lineage=[f"S0 basis v{st['version']}"])


def basis_train_fn(pool, seed, n_extra=4):
    """The system's own basis search (livesim_loop2.train_basis, firewall included) on the window just played plus a few
    other revealed windows that are neither the transfer window nor the played one. Adopts only what the firewall adopts."""
    import scripts.livesim_loop2 as LP2                         # noqa: E402 - heavy import kept lazy

    def fn(run, ctx, state):
        rng = np.random.default_rng(L.derive_seed(seed, ctx.window_id, "basis"))
        others = [p for p in pool if p["id"] != ctx.window_id and p["real_end"] < ctx.real_start]
        pick = [others[i] for i in rng.permutation(len(others))[:n_extra]]
        wins = [LP2.load_window(p["dir"]) if list(Path(p["dir"]).glob("wsnap_*")) else None for p in pick]
        wins = [w for w in wins if w is not None]
        if len(wins) < 2:
            return None
        res = LP2.train_basis(wins, state.cfg, state.meta, seed=L.derive_seed(seed, ctx.window_id) % 100000, as_of=ctx.real_end)
        return (res.cfg, res.meta, res.reason) if res.adopted else None
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--memo-pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--basis", action="store_true", help="also let the system's basis search learn from Run 1")
    ap.add_argument("--min-ram", type=float, default=2.5)
    a = ap.parse_args()
    out = OUT / a.tag
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "pid.txt").write_text(str(os.getpid()))
    if not L.wait_for_ram(a.min_ram, log_fn=log):
        log(f"free RAM stayed under {a.min_ram} GB for 20 minutes: not running")
        (out / "ABORTED.txt").write_text("RAM")
        return 2
    code = provenance.code_stamp()
    log(f"learning delta '{a.tag}' seed {a.seed}, code {code.get('code_hash')}")
    sc = L.harness_selfcheck(a.seed, log_fn=log)
    (out / "selfcheck.json").write_text(json.dumps(sc, indent=1, default=str))
    if not sc["valid"]:
        log("HARNESS SELF-CHECK FAILED: the harness cannot see what it claims to; real windows will not be judged")
    pool = L.revealed_pool()
    log(f"{len(pool)} revealed windows in the pool")
    pairs = L.choose_pairs(pool, a.pairs, a.seed)
    st, bank = read_state0()
    learners = [L.MemoryBankLearner()] + ([L.BasisLearner(basis_train_fn(pool, a.seed))] if a.basis else [])
    learner = L.ChainLearner(learners) if len(learners) > 1 else learners[0]
    player = L.ReplayPlayer()

    def play_pairs(prefix, plr, lrn, plist):
        recs = []
        for we, be in plist:
            f = out / "pairs" / f"{prefix}__{we['id']}__{be['id']}.json"
            if f.exists():
                r = json.loads(f.read_text())
                if r.get("_code") == code.get("code_hash"):
                    recs.append(r)
                    continue
            if not L.wait_for_ram(a.min_ram, log_fn=log):
                log("RAM never freed up; stopping with what is checkpointed")
                return recs
            W, B = L.load_window(we), L.load_window(be)
            r = L.run_pair(W, B, plr, lrn, state_for(W, st, bank), L.derive_seed(a.seed, prefix), log_fn=log)
            r["_code"] = code.get("code_hash")
            f.write_text(json.dumps(r, default=str))
            recs.append(r)
            del W, B
        return recs

    recs = play_pairs("sys", player, learner, pairs)
    mrecs = play_pairs("memo", L.MemorisingReplayPlayer(), L.MemoriserLearner(), pairs[:a.memo_pairs])
    agg = L.aggregate(recs, a.seed)
    magg = L.aggregate(mrecs, a.seed) if mrecs else None
    n_audits = sum(len(r["blindness"]) for r in recs + mrecs)
    n_failed = sum(1 for r in recs + mrecs for b in r["blindness"] if not b["passed"])
    memo_ok = None
    if magg is not None:
        mm = magg["metrics"][L.PRIMARY]
        memo_ok = bool(mm["same"]["mean"] > 3 * max(abs(mm["transfer"]["mean"]), 1e-4) and mm["same"]["mean"] > 0)
    if not sc["valid"] or memo_ok is False:
        agg["verdict"] = {"label": "VOID", "why": "harness self-check or the real-window memoriser control failed; verdict withheld",
                          "notes": []}
    summary = {"tag": a.tag, "seed": a.seed, "learner": getattr(learner, "name", "?"), "basis_version": st["version"],
               "provenance": provenance.stamp({"cfg": st["cfg"], "meta": st["meta"]}, seed=a.seed),
               "selfcheck": sc, "aggregate": agg, "memoriser_control": {"aggregate": magg, "sees_memorisation": memo_ok} if magg else None,
               "blindness": {"n_audits": n_audits, "n_failed": n_failed}, "caveats": L.CAVEATS,
               "pairs": [{"window": w["id"], "transfer": b["id"]} for w, b in pairs]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text(L.render_report(summary), encoding="utf-8")
    m = agg["metrics"][L.PRIMARY]
    log_experiment({"event": "learning_delta", "tag": a.tag, "n_pairs": len(recs), "verdict": agg["verdict"]["label"],
                    "metrics": {"same_year_delta_mean_week": m["same"]["mean"], "learning_effect": m["effect"]["mean"],
                                "transfer": m["transfer"]["mean"], "noise": m["noise"]["mean"]},
                    "gates": {"selfcheck_valid": sc["valid"], "memoriser_seen_on_real": memo_ok, "blindness_failures": n_failed},
                    "window_ids": [w["id"] for w, _ in pairs], "outcome": "continue_testing", "reason": agg["verdict"]["why"]},
                   cfg={"cfg": st["cfg"], "meta": st["meta"]}, seed=a.seed)
    log(f"done: verdict {agg['verdict']['label']} - {agg['verdict']['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
