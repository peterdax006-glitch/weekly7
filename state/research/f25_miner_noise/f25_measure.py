"""F25: false admissions of PatternMiner on pure noise (test panel generator), before vs after.
usage: python f25_measure.py <before|after> <n_panels> [seed0] [extra-json-params]"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Peter\weekly7")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import engine  # noqa
from test_patterns_integration import panel, FAST

which, n = sys.argv[1], int(sys.argv[2])
seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
if which == "before":
    spec = importlib.util.spec_from_file_location("engine._f25_before", Path(__file__).with_name("f25_patterns_before.py"))
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "engine"; spec.loader.exec_module(mod)
    PM = mod.PatternMiner
else:
    from engine.patterns import PatternMiner as PM

rows, per = [], []
t0 = time.time()
for s in range(seed0, seed0 + n):
    X, y = panel(weeks=200, stocks=80, seed=s, regime_effect=0.0)
    m = PM({**FAST, "null_reps": 2, "null_search": "full", "p_method": "bh", **extra}).fit(X, y, X.index.get_level_values(0).max())
    P = m.patterns
    A = P[P["status"].isin(["active", "rescoped"])] if len(P) else P
    per.append(len(A))
    for r in A.itertuples():
        rows.append({"seed": s, "key": r.key_named, "origin": r.origin, "status": r.status, "t_disc": r.t_disc, "t_conf": r.t_conf,
                     "p": r.p_coincidence, "q": r.q_value, "ph": r.p_hallucinated, "cf": r.conf_factor, "p_real": r.p_real,
                     "tested": m.report.get("tested"), "fdr_pass": m.report.get("fdr_pass"), "fdr_rows_pass": bool(r.fdr_pass)})
    # diagnostic counts along the admission chain
    if len(P):
        conf = P["p_real"] >= 0.8
        rows.append({"seed": s, "key": "__chain__", "tested": len(P), "q<=.2": int((P["q_value"] <= 0.2).sum()),
                     "ph<=.2": int((P["p_hallucinated"] <= 0.2).sum()), "confirmed": int(conf.sum()),
                     "no_gain": int((P["status"] == "no_gain").sum()), "dup": int((P["status"] == "duplicate").sum()),
                     "disc": int((P["status"] == "discarded").sum())})
per = np.array(per)
k = int((per > 0).sum())
from scipy.stats import beta
lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
se = per.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
out = {"which": which, "n": n, "seed0": seed0, "extra": extra, "mean_false_per_panel": per.mean(),
       "mean_ci95": [per.mean() - 1.96 * se, per.mean() + 1.96 * se], "panels_with_any": k, "p_any": k / n,
       "p_any_ci95_clopper": [lo, hi], "hist": np.bincount(per).tolist(), "secs": time.time() - t0}
print(json.dumps(out, default=float))
tag = f"{which}_{seed0}_{n}_{abs(hash(json.dumps(extra, sort_keys=True))) % 10**6}"
pd.DataFrame(rows).to_csv(Path(__file__).with_name(f"f25_rows_{tag}.csv"), index=False)
Path(__file__).with_name(f"f25_sum_{tag}.json").write_text(json.dumps(out, default=float))
