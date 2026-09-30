"""F11 runner (C69 ledger W-06; C68 PZ20, PC16): does a genuine, persistent prediction error get FIXED by the real research loop - a fix
proposed, validated out of sample and PROMOTED by the real quality gate on its own measured evidence - while the null twin's fix never
is, and is a promoted fix that later degrades rolled back by the real monitor? Planted worlds only (C63).

Worlds (engine.research.error_loop.plant_world): 'planted' = one stock type (sector SEC1) gives back part of its 20-session run-up
every session, so a pooled r20 slope over-predicts its strong names in every year; 'null' = the same seed and draws without it;
'flip' = planted until `flip_at` of the sample, after which that sector's give-back stops (the promoted slope is now wrong;
a stronger reversal, an ACCELERATING momentum, compounds into exploding prices in this world and was rejected as unrealistic).

    .venv/Scripts/python scripts/c68_fix_promote.py --kinds planted,null --seeds 6 --workers 3
    .venv/Scripts/python scripts/c68_fix_promote.py --kinds flip --seeds 1

Results: state/research/c68_fix_promote/<run>/rows.csv, rates.json, validations.jsonl, provenance.json, summary.md."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

OUT = ROOT / "state" / "research" / "c68_fix_promote"
# The planted effect is sized so the fix's adjustment stays below the firewall's implausible-IC cap (0.15 rank IC against the
# incumbent's error) while its gain is material (the complexity gate's +-0.0005 equivalence margin): a realistic-size error.
PLANT = {"n_days": 1150, "n_names": 60, "regime_switch": False, "n_sectors": 3, "sigma": 0.024, "weak_revert": 0.06}
FLIP = {"n_days": 1400, "flip_at": 0.8, "flip_revert": 0.0}         # the give-back stops: the promoted slope is now wrong
CADENCE = 4


def world_for(kind: str, seed: int, plant: dict | None = None):
    from engine.research import error_loop as EL
    p = {**PLANT, **(plant or {})}
    kw = dict(n_days=int(p["n_days"]), n_names=int(p["n_names"]), seed=int(seed), regime_switch=bool(p["regime_switch"]), n_sectors=int(p["n_sectors"]),
              base_sigma=(float(p["sigma"]), 1.75 * float(p["sigma"])), weak_revert=float(p["weak_revert"]))
    if kind == "planted":
        return EL.plant_world(EL.C68Plant(weak_sector=1, **kw))
    if kind == "null":
        return EL.plant_world(EL.C68Plant(weak_sector=-1, **kw))
    if kind == "flip":
        f = {**FLIP, **(plant or {})}
        kw["n_days"] = int(f["n_days"])
        return EL.plant_world(EL.C68Plant(weak_sector=1, weak_until=float(f["flip_at"]), weak_revert_after=float(f["flip_revert"]), **kw))
    raise ValueError(f"unknown world kind {kind!r}")


def loop_config(run_id: str):
    from engine.research import loop as LP
    from engine.research import two_stage as TS
    keep = ("observe.panel", "evaluate.two_stage")
    # the C68 stages that do not feed self-correction are switched off to keep a multi-year run cheap; the path under test
    # (policy -> expectations -> outcomes -> validate/promote -> monitor/audit) runs exactly as in production
    off = ("c68.market_regime", "c68.pattern_change", "c68.what_changed", "c68.error_research", "c68.research_depth")
    return LP.LoopConfig(run_id=run_id, free_gb=12.0, code_hash="f13-c68-regime", checkpoint="off", cadence={"c68.validate_promote": CADENCE},
                         disabled=tuple(n for n in LP.BUILTIN_STAGES if n not in keep and n != "report.cycle") + off,
                         two_stage=TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))


def run_one(kind: str, seed: int, root, plant: dict | None = None) -> dict:
    """One real-loop run to the end of the world. Returns the row the rate tables are made of; the per-validation records go to
    `root`/validations.jsonl."""
    warnings.filterwarnings("ignore")
    from engine.research import error_loop as EL
    from engine.research import feeds as FD
    from engine.research import loop as LP
    EL.register()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    world = world_for(kind, seed, plant)
    state, rt, _ = LP.open_loop(FD.WorldFeed(FD.InMemorySource(world), FD.FeedConfig(warm_weeks=50)), root / "loop", loop_config(f"f11{kind}{seed}"),
                                clock=lambda: 1_700_000_000.0)
    t0, cycles, bad = time.time(), 0, []
    while (rep := LP.step(state, rt)) is not None:
        cycles += 1
        bad += [f"{rep['cycle']}|{s['stage']}|{s['status']}" for s in rep["stages"] if s["stage"].startswith("c68.") and s["status"] in ("FAILED", "REFUSED_LEAK")]
    st = state.modules["c68"]
    vals = st.corrections
    with open(root / "validations.jsonl", "w", encoding="utf-8") as fh:
        for c in vals:
            fh.write(json.dumps({"kind": kind, "seed": seed, **c}, default=str) + "\n")
    EL.correction_frame(st, "2100-01-01").to_pickle(root / "frame.pkl")      # every matured forecast: gate decisions replay offline
    promo = [c for c in vals if "sector_slope" in c.get("promoted", [])]
    others = sorted({p for c in vals for p in c.get("promoted", []) if p != "sector_slope"})
    last = vals[-1] if vals else {}
    groups = [c.get("detail", {}).get("sector_slope", {}).get("group") for c in vals]
    mon = [m for m in st.monitoring if "t" in m]
    flip = world.truth.get("weak_until")
    rb = next((m for m in mon if m.get("rolled_back")), None)
    first_promo = promo[0]["now"] if promo else None
    at_promo = (promo[0].get("regime") or {}).get("sector_slope", {}) if promo else {}
    declared = next((m.get("change_declared") for m in mon if m.get("change_declared")), None)
    return {"kind": kind, "seed": seed, "world": "null" if kind == "null" else "planted", "cycles": cycles, "seconds": round(time.time() - t0, 1),
            "validations": len(vals), "promoted_sector_slope": int(bool(promo)), "first_promote": promo[0]["now"] if promo else None,
            "promote_count": len(promo), "promoted_other": ",".join(others), "applied": int(st.counters.get("fixes_applied", 0)),
            "rolled_back": int(st.counters.get("rolled_back", 0)), "retired": json.dumps(dict(EL._retired(st))),
            "final_production": st.production.get("name"), "last_verdict": (last.get("verdicts") or {}).get("sector_slope"),
            "last_blocking": ",".join((last.get("blocking") or {}).get("sector_slope", [])),
            "last_effect": (last.get("effects") or {}).get("sector_slope", (None, None, None))[0],
            "last_t": (last.get("effects") or {}).get("sector_slope", (None, None, None))[1],
            "picked_weak_sector_share": float(np.mean([g == "SEC1" for g in groups if g])) if any(groups) else 0.0,
            "monitor_min_t": min((m["t"] for m in mon), default=None), "stage_failures": ";".join(bad[:5]),
            "weak_sector": world.truth.get("weak_sector"), "flip_date": flip,
            # F13: a promotion after the flip is a promotion on evidence that predates the change (the fix is wrong from the flip on)
            "post_flip_promotions": sum(1 for c in promo if flip and str(c["now"]) > str(flip)),
            "promotion_regime": at_promo.get("verdict"), "promotion_stale_share": at_promo.get("stale_share"),
            "held_by_regime": int(st.counters.get("promotion_held_by_regime", 0)),
            "held_dates": ",".join(str(c["now"]) for c in vals if "sector_slope" in (c.get("held_by_regime") or [])),
            "rollback_at": rb["now"] if rb else None, "rollback_why": rb.get("why") if rb else None,
            "change_declared": declared,
            "rollback_delay_weeks": round((pd.Timestamp(rb["now"]) - pd.Timestamp(flip)).days / 7, 1)
            if rb and flip and first_promo and str(first_promo) <= str(flip) else None}


def _job(args):
    kind, seed, root, plant = args
    return run_one(kind, seed, root, plant)


def rates(tab: pd.DataFrame) -> dict:
    out = {}
    for k, g in tab.groupby("kind"):
        out[k] = {"n": int(len(g)), "promote_rate": float(g["promoted_sector_slope"].mean()),
                  "promoted_seeds": [int(s) for s in g[g.promoted_sector_slope == 1]["seed"]],
                  "any_other_fix_promoted": int((g["promoted_other"].fillna("") != "").sum()),
                  "rollback_rate": float(g["rolled_back"].clip(upper=1).mean()),
                  "first_promote": sorted(x for x in g["first_promote"].dropna()), "last_verdicts": g["last_verdict"].value_counts().to_dict(),
                  "stage_failures": int((g["stage_failures"].fillna("") != "").sum())}
        if "post_flip_promotions" in g:
            out[k] |= {"post_flip_promotions": int(g["post_flip_promotions"].sum()), "held_by_regime": int(g["held_by_regime"].sum()),
                       "rollback_delay_weeks": sorted(float(x) for x in g["rollback_delay_weeks"].dropna()),
                       "rollback_why": g["rollback_why"].dropna().value_counts().to_dict()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kinds", default="planted,null")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--run", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--plant", default="{}", help="JSON overrides of PLANT / FLIP")
    ap.add_argument("--delay-sims", type=int, default=0, help="also simulate the rollback delay distribution (self_correct.rollback_delay_curve)")
    a = ap.parse_args(argv)
    import psutil
    free = psutil.virtual_memory().available / 1e9
    if free < 2.5:
        print(f"only {free:.1f} GB free (< 2.5): not launching (CONTEXT rule 10)")
        return 2
    plant = json.loads(a.plant)
    out = OUT / a.run
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s, str(out / f"{k}_{s}"), plant) for k in a.kinds.split(",") for s in range(a.seed0, a.seed0 + a.seeds)]
    from engine.provenance import stamp
    (out / "provenance.json").write_text(json.dumps({**stamp({"plant": {**PLANT, **plant}, "flip": FLIP, "cadence": CADENCE, "jobs": len(jobs)}),
                                                     "label": "IMPLEMENTED - NOT VALIDATED (planted worlds)"}, default=str, indent=1), encoding="utf-8")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            print(json.dumps(r, default=str), flush=True)
            pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
    tab = pd.DataFrame(rows).sort_values(["kind", "seed"])
    tab.to_csv(out / "rows.csv", index=False)
    rt = rates(tab)
    (out / "rates.json").write_text(json.dumps(rt, indent=1, default=str), encoding="utf-8")
    if a.delay_sims:
        from engine.research import self_correct as SCX
        rt["rollback_delay_curve_snr0.8"] = [dataclasses.asdict(x) for x in SCX.rollback_delay_curve(n_sim=a.delay_sims, seed=0)]
        (out / "rates.json").write_text(json.dumps(rt, indent=1, default=str), encoding="utf-8")
    cols = ["kind", "seed", "promoted_sector_slope", "first_promote", "promoted_other", "rolled_back", "final_production", "last_verdict",
            "last_blocking", "last_effect", "last_t", "picked_weak_sector_share", "post_flip_promotions", "promotion_regime", "held_by_regime",
            "rollback_at", "rollback_why", "rollback_delay_weeks"]
    lines = ["# C68 fix promotion (F11)", "", "IMPLEMENTED - NOT VALIDATED. Planted worlds only.", "", "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c)) for c in cols) + " |" for _, r in tab.iterrows()]
    lines += ["", "```", json.dumps(rt, indent=1, default=str), "```"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(rt, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
