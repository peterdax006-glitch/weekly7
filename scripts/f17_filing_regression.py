"""F17 (C69 sections 12-14, 27, 28, 31): why the full research loop filed less after F14 + F16 - re-gate the RECORDED looks with each
change toggled.

Every GATE look recorded by the final-code loops (git cac30e76, the 'before' runs) and by the F14+F16 loops (the 'after' runs, current
state/research/regate_sequential/loop_default_s*.json) is re-gated on the planted world at the same date, feature and look number, in time
order with a persistent replication ledger per finding (as the loop does), under five configurations:

    current       F16 rolling 156-week frame, F14 year-end train cut, F14 failure floor (episodes OR exposure)
    no_f14_cut    EvidenceConfig(align_years=False): the pre-F14 fractional train cut
    no_f14_floor  max_failure_rate ~ 0: the exposure route can never be met (the pre-F14 5-episode floor)
    no_f16_frame  FeedConfig(frame_weeks=0): the pre-F16 calendar-year frame
    pre_f14       all three off (the gate the 'before' runs used)

The loop's own n_tests_searched (len(state.screened) at that cycle) is not recorded per look; 40 is used (the runs end with 25-46), and
has_falsifier=True (every genuine feature had a science item by the end). Fidelity is checked by comparing the 'current' verdicts with
the after-runs' recorded verdicts and 'pre_f14' with the before-runs'.

    python scripts/f17_filing_regression.py --seeds 0 1 2
Results: state/research/regate_sequential/f17/regate_toggles_s<seed>.json (with provenance)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.research import evidence as EV                                    # noqa: E402
from engine.research import feeds as FD                                       # noqa: E402
from engine.research import loop as LP                                        # noqa: E402
from engine.research import quality_gate as QG                                # noqa: E402
from engine.research import replication as RP                                 # noqa: E402

OUT = ROOT / "state" / "research" / "regate_sequential"
BEFORE_COMMIT = "cac30e76"
CONFIGS = {
    "current": (156, EV.EvidenceConfig()),
    "no_f14_cut": (156, EV.EvidenceConfig(align_years=False)),
    "no_f14_floor": (156, EV.EvidenceConfig(max_failure_rate=1e-9)),
    "no_f16_frame": (0, EV.EvidenceConfig()),
    "pre_f14": (0, EV.EvidenceConfig(align_years=False, max_failure_rate=1e-9)),
}


def recorded_looks(seed: int) -> list[dict]:
    """Every recorded look of both runs: (run, gate id, feature, look, date, cycle, recorded verdict, recorded blocking)."""
    rel = f"state/research/regate_sequential/loop_default_s{seed}.json"
    before = json.loads(subprocess.run(["git", "show", f"{BEFORE_COMMIT}:{rel}"], capture_output=True, text=True, cwd=ROOT).stdout)
    after = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    rows = []
    for run, rec in (("before", before), ("after", after)):
        for gid, g in (rec.get("gates") or {}).items():
            for h in g.get("history", ()):
                rows.append({"run": run, "gate": gid, "feature": g["feature"], "look": int(h["look"]), "at": h["at"], "cycle": h["cycle"],
                             "rec_verdict": h["verdict"], "rec_blocking": sorted(h["blocking"]), "rec_n_test": h.get("n_test")})
    return rows


def regate(seed: int, looks: list[dict], name: str, n_tests: int = 40) -> list[dict]:
    frame_weeks, ec = CONFIGS[name]
    feed = FD.planted_feed(FD.PlantConfig(seed=seed), dataclasses.replace(FD.FeedConfig(), frame_weeks=frame_weeks))
    plan, store, leds, out = LP.REGATE_PLAN, QG.QuarantineStore(), {}, []
    for now in sorted({r["at"] for r in looks}):
        F = feed.store.upto(now)
        M = F[pd.to_datetime(F["end"]) < pd.Timestamp(now)]
        for r in sorted((r for r in looks if r["at"] == now), key=lambda r: (r["run"], r["gate"])):
            sid = f"{r['run']}:{r['gate']}"
            spec = EV.FindingSpec(sid, r["feature"], 1.0, "VOLATILITY", n_tests_searched=n_tests, has_falsifier=True, seed=seed)
            led = leds.setdefault(sid, RP.ReplicationLedger())
            b = EV.assemble(M, spec, now, code_hash="f17", data_hash=f"planted{seed}", created_real="2026-09-30T00:00:00+00:00",
                            cfg=ec, ledger=led, look=r["look"], plan=plan)
            rep = EV.gate([b], now, "f17", store=store, looks={sid: r["look"]}, plan=plan, cfg=ec)
            ff = EV.failure_floor(b, plan.quality_policy(r["look"], "f17"), plan.alpha_at(r["look"]), ec)
            out.append({**r, "config": name, "verdict": EV.verdicts(rep)[sid], "blocking": sorted(EV.blocking(rep, sid)),
                        "n_test": b.parts.get("n_test"), "n_train": b.parts.get("n_train"), "train_end": b.parts.get("train_end"),
                        "first_test": b.parts.get("first_test"), "train_moved": b.parts.get("train_moved_to_year_end"),
                        "frame_first": str(pd.to_datetime(M.index.get_level_values(0)).min().date()) if len(M) else None,
                        "effect_test": b.parts.get("effect_test"), "failure_episodes": ff.episodes, "failure_route": ff.route,
                        "failure_rate_upper": ff.rate_upper, "missing": dict(b.missing)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args(argv)
    (OUT / "f17").mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        t0 = time.monotonic()
        looks = recorded_looks(s)
        rows = []
        for name in a.configs:
            rows += regate(s, looks, name)
            print(f"seed {s} {name}: {len(looks)} looks, {time.monotonic() - t0:.0f}s", flush=True)
        try:
            from engine import provenance
            prov = provenance.stamp({"seed": s, "configs": a.configs, "before": BEFORE_COMMIT}, s)
        except Exception as e:                                                  # noqa: BLE001 - recorded, never silent
            prov = {"error": f"{type(e).__name__}: {e}"}
        (OUT / "f17" / f"regate_toggles_s{s}.json").write_text(
            json.dumps({"seed": s, "rows": rows, "seconds": round(time.monotonic() - t0, 1), "provenance": prov}, indent=1, default=str),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
