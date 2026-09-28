"""Canon C14: what moves the weekly average, how much, and how consistently - measured on every archived
hidden year with the re-tester (which reproduces every live-clock run exactly).

One-at-a-time sweeps around the current champion: every portfolio knob across its range, and every evidence
indicator's weight at x0 / x0.5 / x2 / reversed. For each variant, per hidden year: change in average week,
+7% weeks and worst drawdown vs the champion. Summaries: mean effect, t-stat across years, share of years
improved (consistency), and a class (significant / slight / negligible; consistent / inconsistent).
Writes state/research/sensitivity.json and docs/sensitivity.json."""
import glob, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, policy

src = open("scripts/livesim_cycle.py", encoding="utf-8").read()
ns = {"__file__": "scripts/livesim_cycle.py"}
_a = sys.argv; sys.argv = ["x"]
exec(compile(src[:src.index("def worker(")], "lc", "exec"), ns)
sys.argv = _a
replay_variant = ns["replay_variant"]

st = json.loads((K.STATE / "livesim" / "cycles.json").read_text())
champion = {**st["config"], "ew": policy.default_evidence_weights()}     # corrected system: evidence switched on
buggy = dict(st["config"])                                                 # as the blind runs traded (evidence stuck at 1.0)
years = []
for c in st["cycles"]:
    a = K.STATE / "livesim" / c["run_id"]
    snaps = {f.stem[5:]: pd.read_parquet(f) for f in sorted(a.glob("snap_*.parquet"))}
    if not snaps or "e_dist_52wh" not in next(iter(snaps.values())):
        continue
    sc = pd.read_parquet(a / "sic.parquet")
    years.append((c["run_id"], c.get("revealed_year"), snaps, pd.read_parquet(a / "closes.parquet"),
                  json.loads((a / "meta.json").read_text())["cost_bps"],
                  {t: policy.sic_division(x) for t, x in zip(sc["ticker"], sc["sic"])}))
print(f"{len(years)} hidden years, champion {champion}", flush=True)


def run(cfg):
    return [replay_variant(cfg, s, c, b, d) for _, _, s, c, b, d in years]


t0 = time.time()
base = run(champion)
KNOBS = {"k": [2, 3, 4, 6, 8, 12, 16], "exit_q": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], "rebalance_weeks": [1, 2, 3, 4],
         "brake": [None, 0.03, 0.05, 0.08, 0.12, 0.2], "max_per_sector": [None, 1, 2, 3], "w_model": [0.0, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0],
         "pick": ["top", "hivol"], "pool_q": [0.8, 0.9, 0.95, 0.98, 0.99], "liq_q": [0.0, 0.2, 0.3, 0.5, 0.7, 0.85],
         "vol_filter": [True, False], "stress_thr": [None, 0.9, 0.95, 1.0, 1.05, 1.1], "stress_k": [2, 3, 4, 6],
         "trend_filter": [None, -0.1, -0.05, 0.0], "trend_gross": [0.0, 0.25, 0.5, 0.75]}
EW0 = policy.default_evidence_weights()


def summarise(rs, label, group, value):
    d_mw = np.array([r["mean_week"] - b["mean_week"] for r, b in zip(rs, base)])
    d_7 = np.array([r["weeks_ge_7"] - b["weeks_ge_7"] for r, b in zip(rs, base)])
    d_dd = np.array([r["max_dd"] - b["max_dd"] for r, b in zip(rs, base)])
    n = len(d_mw)
    mean, sd = float(d_mw.mean()), float(d_mw.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    same = float(np.mean(np.sign(d_mw) == np.sign(mean))) if mean != 0 else 0.0
    size = "significant" if abs(mean) >= 0.0005 and abs(t) >= 2 else "slight" if abs(t) >= 1 or abs(mean) >= 0.0002 else "negligible"
    cons = "consistent" if same >= 0.65 else "inconsistent"
    return {"group": group, "knob": label, "value": value, "d_mean_week": mean, "t": float(t), "share_years_same_way": same,
            "d_weeks_ge7_per_year": float(d_7.mean()), "d_worst_dd": float(d_dd.mean()), "size": size, "consistency": cons,
            "abs_mean_week": float(np.mean([r["mean_week"] for r in rs])), "per_year": [round(x, 5) for x in d_mw]}


rows = [summarise(run(buggy), "evidence score (bug fix)", "fix", "off, as traded")]
print(f"  bug-fix effect measured ({time.time() - t0:.0f}s)", flush=True)
for k, vals in KNOBS.items():
    for v in vals:
        if champion.get(k) == v:
            continue
        cfg = {**champion, k: v}
        rows.append(summarise(run(cfg), k, "knob", v))
    print(f"  {k} done ({time.time() - t0:.0f}s)", flush=True)
for f, w in EW0.items():
    for mult, lab in ((0.0, "off"), (0.5, "x0.5"), (2.0, "x2"), (-1.0, "reversed")):
        ew = dict(EW0); ew[f] = w * mult
        rows.append(summarise(run({**champion, "ew": ew}), f, "indicator", lab))
    print(f"  indicator {f} done ({time.time() - t0:.0f}s)", flush=True)

# per knob: best value, the dose-response range, and the recommended step for the adaptive system
knob_summary = []
for (g, k), grp in pd.DataFrame(rows).groupby(["group", "knob"]):
    best = grp.loc[grp["d_mean_week"].idxmax()]
    span = float(grp["d_mean_week"].max() - grp["d_mean_week"].min())
    knob_summary.append({"group": g, "knob": k, "range_of_effect": span, "best_value": best["value"],
                         "best_gain": float(best["d_mean_week"]), "best_t": float(best["t"]),
                         "best_consistency": float(best["share_years_same_way"]),
                         "class": ("significant" if (grp["size"] == "significant").any() else
                                   "slight" if (grp["size"] == "slight").any() else "negligible")})
knob_summary.sort(key=lambda r: -r["range_of_effect"])
out = {"hidden_years": len(years), "years": [y for _, y, *_ in years], "champion": champion,
       "champion_mean_week": float(np.mean([b["mean_week"] for b in base])),
       "variants": rows, "by_knob": knob_summary, "runtime_s": round(time.time() - t0)}
for p in (K.STATE / "research" / "sensitivity.json", K.SITE / "sensitivity.json"):
    p.write_text(json.dumps(out, indent=1, default=lambda x: None if x != x else x))
print(f"done in {time.time() - t0:.0f}s")
for r in knob_summary[:12]:
    print(f"  {r['group']:9s} {r['knob']:14s} range {r['range_of_effect']*100:+.3f}%/wk  best {r['best_value']} "
          f"({r['best_gain']*100:+.3f}%/wk, t={r['best_t']:.1f}, {r['best_consistency']:.0%} of years)  [{r['class']}]")
