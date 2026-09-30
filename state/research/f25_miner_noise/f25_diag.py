"""F25 diagnosis: on pure noise, are discovery p-values uniform per candidate origin? Is confirmation uniform?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"C:\Users\Peter\weekly7"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_patterns_integration import panel, FAST
from engine.patterns import PatternMiner
from engine import pattern_stats as S
fr = []
nullq = []
for s in range(2000, 2000 + int(sys.argv[1])):
    X, y = panel(weeks=200, stocks=80, seed=s, regime_effect=0.0)
    m = PatternMiner({**FAST, "null_reps": 2}).fit(X, y, X.index.get_level_values(0).max())
    P = m.patterns.copy(); P["seed"] = s
    fr.append(P[["seed", "origin", "t_disc", "t_conf", "m_disc", "m_conf", "p_coincidence", "p_hallucinated"]])
    nullq.append((m.null_summary["null_t_95pct"], m.null_summary["real_t_95pct"]))
D = pd.concat(fr)
D["pconf1"] = 1 - S.norm_cdf(np.abs(D["t_conf"])) * (np.sign(D["m_conf"]) == np.sign(D["m_disc"])) - 0.5 * (np.sign(D["m_conf"]) != np.sign(D["m_disc"]))
D["conf_same_sign"] = np.sign(D["m_conf"]) == np.sign(D["m_disc"])
g = D.groupby("origin")
print(pd.DataFrame({"n": g.size(), "disc_p<.05": g["p_coincidence"].apply(lambda p: (p < .05).mean()),
                    "disc_p<.01": g["p_coincidence"].apply(lambda p: (p < .01).mean()),
                    "conf_same_sign": g["conf_same_sign"].mean(),
                    "|t_conf|>1.96": g["t_conf"].apply(lambda t: (t.abs() > 1.96).mean()),
                    "ph==0": g["p_hallucinated"].apply(lambda p: (p == 0).mean())}).to_string())
top = D[D["p_coincidence"] < 0.01]
print("disc p<.01 rows:", len(top), "conf same sign", top["conf_same_sign"].mean(), "|t_conf|>1.96", (top["t_conf"].abs() > 1.96).mean())
print("null_t_95 / real_t_95 mean:", np.mean(nullq, axis=0))
