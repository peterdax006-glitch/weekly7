"""A13 measurement: over many seeds, how often does the miner recover each planted effect, how many false patterns
does it admit, and is P(real) calibrated (patterns called ~80% real should be real ~80% of the time)?"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K
from engine.patterns import PatternMiner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_planted_patterns import make

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
REAL = {"f0 q4": "strong +1.2%", "f1 q4": "weak +0.4%", "f2 q4": "negative -0.8%", "f3 q4": "regime-only +1.0%"}
rows, calib = [], []
for seed in range(SEEDS):
    X, y = make(seed=100 + seed)
    M = PatternMiner({"max_pairs": 400, "max_unless": 80, "null_reps": 2, "min_n": 200, "half_life_years": 50}).fit(
        X, y, now=X.index.get_level_values(0).max())
    P = M.patterns
    act = P[P["status"].isin(["active", "rescoped"])]
    found = {k: bool((act["key_named"] == k).any()) for k in REAL}
    # a pattern is "truly real" if it involves a planted feature/quintile in its conditions
    def truly(n):
        return any(tok in n for tok in ("f0 q4", "f1 q4", "f2 q4", "f3 q4"))
    false_active = [n for n in act["key_named"] if not truly(n)]
    rows.append({"seed": seed, **found, "false_active": len(false_active), "active": len(act)})
    for n, pr in zip(P["key_named"], P["p_real"]):
        calib.append((float(pr), truly(n)))
    print(seed, found, "false active:", len(false_active), "of", len(act), flush=True)
R = pd.DataFrame(rows)
C = pd.DataFrame(calib, columns=["p_real", "truly"])
C["bin"] = pd.cut(C["p_real"], [-0.01, 0.2, 0.5, 0.8, 0.9, 1.0])
cal = C.groupby("bin", observed=True)["truly"].agg(["mean", "size"])
out = {"seeds": SEEDS, "recovery": {REAL[k]: float(R[k].mean()) for k in REAL},
       "false_active_per_run": float(R["false_active"].mean()), "active_per_run": float(R["active"].mean()),
       "false_share_of_active": float(R["false_active"].sum() / max(R["active"].sum(), 1)),
       "calibration": {str(b): {"share_truly_real": float(r["mean"]), "n": int(r["size"])} for b, r in cal.iterrows()}}
print(json.dumps(out, indent=1))
(K.STATE / "research" / "algorithm").mkdir(parents=True, exist_ok=True)
(K.STATE / "research" / "algorithm" / "planted_calibration.json").write_text(json.dumps(out, indent=1))
from engine.improve import log_experiment
log_experiment({"event": "planted_calibration", **out}, seed=100)
