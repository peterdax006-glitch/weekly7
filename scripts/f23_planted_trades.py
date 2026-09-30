"""F23 (C75 Phase 1E, C68, C66 RF28): does the default research loop now TRADE on the planted world for the right reason, and does
the C68 chain (expectations -> outcomes -> errors -> what_changed -> research questions) run on those positions?

Runs the real loop (engine.research.loop.step) at default settings (the same LoopConfig as tests/test_regate_sequential.run_loop:
checkpoint per cycle, free_gb pinned, TwoStageConfig(gate_min_weeks=26)) for at most --cycles cycles on planted world --seed, either
the default world (with the F23 run mechanism) or the null world (feeds.NULL_PLANT: no volatility state, hence no runs). Per cycle it
records every stage's status and the two-stage decision (positions, abstention reasons), then writes the stage table:

    python scripts/f23_planted_trades.py --world default --seed 0 --cycles 40
    python scripts/f23_planted_trades.py --world null --seed 0 --cycles 40

Results: state/research/f23_planted_trades/<world>_s<seed>.json (+ run folder <world>_s<seed>/), with provenance."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.research import error_loop as C68                                     # noqa: E402,F401 - registers the C68 stages
from engine.research import feeds as FD                                           # noqa: E402
from engine.research import loop as LP                                            # noqa: E402
from engine.research import two_stage as TS                                       # noqa: E402

OUT = ROOT / "state" / "research" / "f23_planted_trades"
CHAIN = ("c68.expectations", "c68.outcomes_errors", "c68.what_changed", "c68.error_research", "c68.research_depth",
         "questions.generate", "learn.decision_bridge", "evaluate.two_stage", "c68.selection_policy")


def _provenance(args: dict) -> dict:
    try:
        from engine import provenance
        return provenance.stamp(args, int(args.get("seed", 0)))
    except Exception as e:                                                      # noqa: BLE001 - recorded, never silent
        return {"error": f"{type(e).__name__}: {e}"}


def stage_table(cycles: list[dict], warm: int) -> dict:
    """Per stage: status counts over all cycles and over the cycles after the first `warm` (the warm-up the chain needs: a position
    opened at cycle k resolves one to two cycles later, and its error is investigated the cycle after)."""
    allc, post = defaultdict(Counter), defaultdict(Counter)
    for c in cycles:
        for s, st in c["stages"].items():
            allc[s][st] += 1
            if c["cycle"] >= warm:
                post[s][st] += 1
    return {s: {"all": dict(allc[s]), "after_warmup": dict(post[s])} for s in sorted(allc)}


def active_runs(truth: dict, sessions, now) -> dict[str, int]:
    """name -> sign of the planted run in force at the decision close before `now` (research-side scoring only: the loop never sees
    this). A position taken on a name with sign +1 was taken for the planted reason."""
    import pandas as pd
    pos = {d: i for i, d in enumerate(sessions.strftime("%Y-%m-%d"))}
    t = int(sessions.searchsorted(pd.Timestamp(now))) - 1
    return {tk: int(s) for d, tk, s, k in truth.get("trend_runs") or () if pos[d] <= t < pos[d] + k}


def right_reason(cycles: list[dict]) -> dict:
    """Share of positions (and of predicted movers) that sat in a planted up-run at the decision close."""
    P = [p for c in cycles for p in c["positions"]]
    M = [m for c in cycles for m in c.get("movers", ())]
    return {"positions": len(P), "in_up_run": sum(p["run"] > 0 for p in P), "in_down_run": sum(p["run"] < 0 for p in P),
            "movers": len(M), "movers_in_run": sum(m["run"] != 0 for m in M),
            "up_run_movers_by_reason": dict(Counter(m["reason"] for m in M if m["run"] > 0))}


def run(world: str, seed: int, cycles: int, fresh: bool = True, warm: int = 6) -> dict:
    plant = {"seed": seed, **(FD.NULL_PLANT if world == "null" else {})}
    feed = LP.world_feed("planted", plant=plant)
    truth = feed.source.world.truth
    run_id = f"f23_{world}_s{seed}"
    root = OUT / f"{world}_s{seed}"
    root.mkdir(parents=True, exist_ok=True)
    cfg = LP.LoopConfig(run_id=run_id, seed=seed, checkpoint="cycle", free_gb=12.0, two_stage=TS.TwoStageConfig(gate_min_weeks=26))
    state, rt, info = LP.open_loop(feed, root, cfg, sweeps=feed.sweeps(), fresh=fresh)
    rec = {"world": world, "seed": seed, "code_hash": rt.code_hash, "resume": info.get("action"), "cycles": [],
           "truth": {"trend_share": truth.get("trend_share"), "n_runs": len(truth.get("trend_runs") or ()),
                     "genuine_direction": truth.get("genuine_direction")},
           "provenance": _provenance({"world": world, "seed": seed, "cycles": cycles, "run_id": run_id})}
    out = OUT / f"{world}_s{seed}.json"
    t0 = time.monotonic()
    for _ in range(cycles):
        rep = LP.step(state, rt)
        if rep is None:
            break
        dec = state.decisions[-1] if state.decisions else None
        pos = dec.positions if dec is not None else None
        live = active_runs(truth, feed.source.world.sessions, rep["now"])
        movers = [] if dec is None else [
            {"ticker": str(ix[-1]), "reason": str(r["reason"]), "p_up": float(r["p_up"]), "gain_pred": float(r.get("gain_pred", float("nan"))),
             "run": live.get(str(ix[-1]), 0)} for ix, r in dec.table[dec.table["mover"].astype(bool)].iterrows()]
        rec["cycles"].append({
            "cycle": rep["cycle"], "now": rep["now"],
            "stages": {s["stage"]: s["status"] for s in rep["stages"]},
            "reasons": {s["stage"]: s["reason"][:160] for s in rep["stages"] if s["stage"] in CHAIN},
            "positions": [] if pos is None else [{"ticker": str(ix[-1]), "side": int(r["side"]), "p_up": float(r["p_up"]),
                                                  "gain_pred": float(r.get("gain_pred", float("nan"))), "run": live.get(str(ix[-1]), 0)}
                                                 for ix, r in pos.iterrows()],
            "movers": movers,
            "abstentions": {} if dec is None else {str(k): int(v) for k, v in dec.reasons().items()},
            "knowledge": sorted(k["feature"] for k in state.knowledge.values())})
        rec["seconds"] = round(time.monotonic() - t0, 1)
        rec["stage_table"] = stage_table(rec["cycles"], warm)
        rec["n_positions"] = sum(len(c["positions"]) for c in rec["cycles"])
        rec["cycles_with_positions"] = sum(1 for c in rec["cycles"] if c["positions"])
        rec["right_reason"] = right_reason(rec["cycles"])
        out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        print(f"[f23] {world} s{seed} cycle {rep['cycle']} {rep['now']} positions {len(rec['cycles'][-1]['positions'])} "
              f"{ {s: rec['cycles'][-1]['stages'].get(s) for s in CHAIN[:5]} }", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", choices=("default", "null"), default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=40)
    ap.add_argument("--resume", action="store_true", help="continue from the run folder's checkpoint instead of a fresh start")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rec = run(a.world, a.seed, a.cycles, fresh=not a.resume)
    print(json.dumps({"n_positions": rec.get("n_positions"), "cycles_with_positions": rec.get("cycles_with_positions"),
                      "right_reason": rec.get("right_reason"),
                      "chain": {s: rec.get("stage_table", {}).get(s) for s in CHAIN}}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
