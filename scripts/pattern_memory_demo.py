"""Demo / experiment for engine.pattern_memory (canon C55-C60): the same year re-run many times keeps improving on its
PATTERNS (never on stock identities), without ever seeing evidence from a date that had not yet occurred.

Synthetic world with known ground truth (planted): universal patterns, era-specific patterns, patterns that break in the
target year (some mid-year), and noise candidates. Each simulated run explores a random subset of the candidate space and
writes what it saw into the timeline memory; after chosen run counts we read the memory at three moments INSIDE the target
year and score it against the truth as of that moment. Results go to state/research/pattern_memory/.

Usage: python scripts/pattern_memory_demo.py [--runs 120] [--seed 7] [--candidates 120] [--out DIR]"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import config as K                                  # noqa: E402
from engine import pattern_memory as pm                         # noqa: E402

Y0, YT = 2004, 2016                                              # history start, target (re-run) year
CHECK_DAYS = [(3, 1), (7, 1), (11, 1)]                           # moments inside the target year
ERAS = {"2004-2008": (2004, 2008), "2009-2012": (2009, 2012), "2013-2016": (2013, 2016)}


def build_world(rng, n_cand):
    """Return {key: {'kind', 'monthly_t'}} where monthly_t is a length-(12*years) array of per-month t-statistics that the
    market 'generated' - the data every run sees identically - plus a truth function active(key, date)."""
    months = pd.date_range(f"{Y0}-01-31", f"{YT}-12-31", freq="ME")
    world, truth = {}, {}
    n_u = n_e = n_b = max(n_cand // 20, 2)
    for i in range(n_cand):
        key = f"feat{i:03d}_q{rng.integers(1, 6)}"
        sign = rng.choice([-1.0, 1.0])
        t = rng.normal(0, 1.0, len(months))
        if i < n_u:
            kind, on = "universal", np.ones(len(months), bool)
            t += sign * 1.3
        elif i < n_u + n_e:
            kind = "era"
            y = int(rng.integers(Y0, YT - 5))
            on = (months.year >= y) & (months.year <= y + 3)
            t += sign * 1.6 * on
        elif i < n_u + n_e + n_b:
            kind = "breaker"
            brk = pd.Timestamp(YT, int(rng.integers(3, 9)), 1)              # breaks inside the target year
            on = months < brk
            t += sign * 1.3 * on - sign * 1.8 * (~on) * (months.year >= YT)
        else:
            kind, on = "noise", np.zeros(len(months), bool)
        world[key] = {"kind": kind, "t": t, "on": on, "sign": sign}
    return months, world


def truth_active(world_entry, months, now):
    """Ground truth as of `now`: is the pattern one a trader should be using?"""
    m = months[months < now]
    on = world_entry["on"][: len(m)]
    k = world_entry["kind"]
    if k == "noise" or len(m) == 0:
        return False
    if k == "universal":
        return True
    return bool(on[-3:].all())                 # era / breaker: only while its effect is still on


def observe(mem, run_id, keys, months, world, run_now):
    obs = []
    for key in keys:
        w = world[key]
        for j, d in enumerate(months):
            if pm.add_sessions(d, 5) > run_now:
                continue
            obs.append({"key": key, "obs_date": d, "effect": 0.002 * np.sign(w["t"][j]), "n": 20, "t": float(w["t"][j]),
                        "ctx": {"m_vix": float(np.sin(j / 9.0)), "m_breadth": float(np.cos(j / 13.0))}})
    return mem.add_observations(run_id, run_now, obs, n_tried=len(keys))


def score(mem, months, world, day):
    v = mem.view(day, {"m_vix": float(np.sin(len(months[months < day]) / 9.0)), "m_breadth": 0.0})
    act = {k for k, w in v.weights.items() if w.weight > 0}
    should = {k for k, w in world.items() if truth_active(w, months, day)}
    tp, fp, fn = len(act & should), len(act - should), len(should - act)
    noise_mass = sum(w.weight for k, w in v.weights.items() if world[k]["kind"] == "noise")
    return {"tp": tp, "fp": fp, "fn": fn, "recall": tp / max(len(should), 1), "precision": tp / max(len(act), 1),
            "noise_active": sum(world[k]["kind"] == "noise" for k in act), "noise_weight": float(noise_mass),
            "seen": len(v.weights), "should": len(should)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--candidates", type=int, default=120)
    ap.add_argument("--explore", type=float, default=0.06, help="share of the candidate space each run explores")
    ap.add_argument("--out", default=os.path.join(K.STATE, "research", "pattern_memory"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    months, world = build_world(rng, a.candidates)
    keys = sorted(world)
    run_now = pd.Timestamp(YT + 1, 1, 20)
    checkpoints = sorted({1, 3, 10, 30, a.runs // 2, a.runs})
    t0 = time.time()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        mem = pm.PatternMemory(tmp)
        for r in range(1, a.runs + 1):
            sub = list(rng.choice(keys, size=max(int(len(keys) * a.explore), 1), replace=False))
            observe(mem, f"run{r:04d}", sub, months, world, run_now)
            if r in checkpoints:
                for (m, d) in CHECK_DAYS:
                    day = pd.Timestamp(YT, m, d)
                    rows.append({"runs": r, "day": f"{YT}-{m:02d}-{d:02d}", **score(mem, months, world, day),
                                 "cum_tries": mem.cumulative_tries()["total_tries"], "keys_known": len(mem.keys())})
                print(f"run {r:4d}: known={len(mem.keys()):4d} tries={mem.cumulative_tries()['total_tries']:6d} "
                      + " ".join(f"rec={x['recall']:.2f}/prec={x['precision']:.2f}" for x in rows[-3:]), flush=True)
        df = pd.DataFrame(rows)
        # intra-year fade of the breakers: weight through the target year
        fade = []
        an = pm.Analytics(mem)
        for k, w in world.items():
            if w["kind"] == "breaker" and k in mem.keys():
                ws = [mem.view(pd.Timestamp(YT, m, 15)).weights.get(k) for m in range(2, 13)]
                fade.append({"key": k, "weights": [None if x is None else round(x.weight, 3) for x in ws]})
        audit = pm.audit_prefix_invariance(mem, [f"{YT}-03-01", f"{YT}-07-01", f"{YT}-11-01", f"{YT - 3}-06-15"],
                                           {"m_vix": 0.1, "m_breadth": 0.0})
        md = an.markdown(pd.Timestamp(YT, 12, 31), {"m_vix": 0.0, "m_breadth": 0.0}, ERAS)
        tries = mem.cumulative_tries()
        chain = mem.verify()
    out = {"seed": a.seed, "runs": a.runs, "candidates": a.candidates, "explore": a.explore, "target_year": YT,
           "seconds": round(time.time() - t0, 1), "prefix_invariance_violations": audit, "chain_ok": chain["ok"],
           "cumulative_tries": {k: tries[k] for k in ("total_tries", "distinct_keys", "runs", "expected_best_null_t")},
           "table": rows, "breaker_fade": fade}
    try:
        from engine.provenance import stamp
        out["provenance"] = stamp({"runs": a.runs, "candidates": a.candidates, "explore": a.explore}, a.seed)
    except Exception as e:                                       # provenance is a bonus, not a dependency of the demo
        out["provenance"] = {"unavailable": str(e)}
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    with open(os.path.join(a.out, "report.md"), "w") as fh:
        fh.write(md + "\n## Score against planted truth\n\n" + df.to_string(index=False) + "\n")
    print("prefix-invariance violations:", len(audit), "| chain ok:", chain["ok"], "| wrote", a.out)
    return 0 if not audit and chain["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
