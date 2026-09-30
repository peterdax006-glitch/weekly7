"""F25 recall on the existing planted patterns (engine.planted scenarios, planted_calibration MINER), before vs after,
same seeds, one process. usage: python f25_recall.py <n_seeds> [scenario,...]"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\Peter\weekly7"); sys.path.insert(0, str(ROOT))
import engine  # noqa
from engine import planted as PL
from engine.patterns import PatternMiner as After
spec = importlib.util.spec_from_file_location("engine._f25_before", Path(__file__).with_name("f25_patterns_before.py"))
mod = importlib.util.module_from_spec(spec); mod.__package__ = "engine"; spec.loader.exec_module(mod)
Before = mod.PatternMiner
MINER = {"max_pairs": 400, "max_unless": 80, "null_reps": 2, "min_n": 200, "half_life_years": 50}
n = int(sys.argv[1])
names = sys.argv[2].split(",") if len(sys.argv) > 2 else [s.name for s in PL.scenarios()]
res = {}
t0 = time.time()
for sc in PL.scenarios():
    if sc.name not in names:
        continue
    for seed in range(100, 100 + n):
        X, y, truth = PL.generate(sc, seed=seed)
        now = X.index.get_level_values(0).max()
        for tag, M in (("before", Before), ("after", After)):
            r = PL.score_run(M(MINER).fit(X, y, now).patterns, sc, X=X, truth=truth)
            d = res.setdefault(sc.name, {}).setdefault(tag, {"runs": 0, "false": 0, "false_true": 0, "plants": {}})
            d["runs"] += 1; d["false"] += r["n_false_active"]; d["false_true"] += r.get("n_false_active_true", 0)
            for pn, p in r["per_plant"].items():
                q = d["plants"].setdefault(pn, {"should": p["should_admit"], "admitted": 0, "via_child": 0})
                q["admitted"] += int(p["admitted"]); q["via_child"] += int(p["admitted_via_child"])
        print(sc.name, seed, f"{time.time() - t0:.0f}s", flush=True)
print(json.dumps(res, indent=1))
Path(__file__).with_name("f25_recall.json").write_text(json.dumps(res, indent=1))
