"""Score every simulated configuration on hundreds of random 4-week months.
TUNE = odd years, LOCKED = even years. Picks the best on TUNE, then reports it on LOCKED
against the champion, with a paired bootstrap and a multiple-testing (Bonferroni) haircut."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from statistics import NormalDist
from engine import config as K, data
from engine.improve import log_experiment

import os
SUFFIX = os.environ.get("W7_SUFFIX", "")
DIR = K.STATE / "research" / f"tuning{SUFFIX}"
PRIOR_TRIALS = int(os.environ.get("W7_PRIOR_TRIALS", "0"))      # configs tried in earlier rounds (Bonferroni)
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = np.random.default_rng(7)
spy = data.load("market")["Close"]["SPY"]

eqs = {}
for f in DIR.glob("eq_*.parquet"):
    e = pd.read_parquet(f).iloc[:, 0]
    eqs[f.stem[3:]] = e
names = sorted(eqs)
days = eqs[names[0]].index
mondays = [d for i, d in enumerate(days[1:], 1) if d.weekday() < days[i - 1].weekday() or (d - days[i - 1]).days > 2]

def episodes(years):
    starts = [d for d in mondays if d.year in years]
    idx = rng.choice(len(starts), size=N_EPISODES, replace=True)
    return [starts[i] for i in idx]

def month_stats(e, starts):
    rows = []
    for s in starts:
        i = e.index.get_loc(s) - 1                  # buy at the prior close
        if i < 0 or i + 20 >= len(e):
            continue
        seg = e.iloc[i:i + 21] / e.iloc[i] * 1000
        wk = seg.iloc[[0, 5, 10, 15, 20]].pct_change().dropna()
        sp = spy.reindex(seg.index).ffill()
        rows.append({"month": seg.iloc[-1] / 1000 - 1, "spy": sp.iloc[-1] / sp.iloc[0] - 1,
                     "weeks_ge7": int((wk >= 0.07).sum()), "worst_week": wk.min(), "best_week": wk.max()})
    return pd.DataFrame(rows)

years = sorted(set(days.year))
if os.environ.get("W7_LOCK_YEARS"):            # e.g. "2008-2016": an era no earlier round has seen
    a, b = map(int, os.environ["W7_LOCK_YEARS"].split("-"))
    LOCK = [y for y in years if a <= y <= b]
    TUNE = [y for y in years if y not in LOCK]
else:
    TUNE = [y for y in years if y % 2 == 1]
    LOCK = [y for y in years if y % 2 == 0]
st_tune, st_lock = episodes(TUNE), episodes(LOCK)
res = {}
for n in names:
    a, b = month_stats(eqs[n], st_tune), month_stats(eqs[n], st_lock)
    def summ(d):
        return {"mean_month": d["month"].mean(), "median_month": d["month"].median(),
                "p5_month": d["month"].quantile(0.05), "beat_spy": (d["month"] > d["spy"]).mean(),
                "weeks_ge7_per_month": d["weeks_ge7"].mean(), "any_7pct_week": (d["weeks_ge7"] > 0).mean(),
                "worst_week_avg": d["worst_week"].mean(), "all_4_weeks_ge7": (d["weeks_ge7"] == 4).mean()}
    def dd(years):
        worst = 0.0
        for y in years:
            e = eqs[n][eqs[n].index.year == y]
            if len(e) > 5:
                worst = min(worst, float((e / e.cummax() - 1).min()))
        return worst
    ta, tb = summ(a), summ(b)
    ta.update({"spy_mean_month": a["spy"].mean(), "max_dd_in_year": dd(TUNE)})
    tb.update({"spy_mean_month": b["spy"].mean(), "max_dd_in_year": dd(LOCK)})
    res[n] = {"tune": ta, "lock": tb, "_lock_weeks7": b["weeks_ge7"].values.astype(float)}

def objective(s):
    """Canon C8: primary = expected +7% weeks per month. Hard guards (round-1 lesson: single-month
    guards let a -73% drawdown through): mean month >= S&P's mean month on the same episodes, and the
    worst within-year drawdown on these years no worse than -50%."""
    if s["mean_month"] < s["spy_mean_month"] or s["max_dd_in_year"] < -0.50:
        return -1.0 + s["weeks_ge7_per_month"] / 100          # fails the guards: ranked below every passer
    return s["weeks_ge7_per_month"]

# cheating detector: the clairvoyant control must be far ahead; if any real config is close, we leak
cheat = [n for n in names if n.startswith("CHEAT")]
if cheat:
    c_obj = objective(res[cheat[0]]["tune"])
    real_best = max(objective(res[n]["tune"]) for n in names if not n.startswith("CHEAT"))
    leak_ok = real_best < 0.5 * c_obj
    print(f"cheat control: clairvoyant {c_obj:.3f} +7%-weeks/month vs best real {real_best:.3f} -> "
          f"{'NO LEAK' if leak_ok else 'POSSIBLE LEAK - results invalid'}")
    names = [n for n in names if not n.startswith("CHEAT")]
champ = os.environ.get("W7_CHAMPION", "CHAMPION_v1.2")
ranked = sorted(names, key=lambda n: -objective(res[n]["tune"]))
best = ranked[0] if ranked[0] != champ else ranked[1]
diff = res[best]["_lock_weeks7"] - res[champ]["_lock_weeks7"]     # same episodes: paired
boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
z = diff.mean() / (np.std(boots) + 1e-12)
z_need = NormalDist().inv_cdf(1 - 0.05 / (len(names) + PRIOR_TRIALS))
verdict = "PROMOTE" if (z >= z_need and objective(res[best]["lock"]) > objective(res[champ]["lock"])) else "KEEP CHAMPION"
summary = {"configs": len(names), "episodes_per_split": N_EPISODES, "tune_years": TUNE, "locked_years": LOCK,
           "champion": {k: res[champ][k] for k in ("tune", "lock")},
           "best_on_tune": best, "best": {k: res[best][k] for k in ("tune", "lock")},
           "locked_gain_weeks7_per_month": float(diff.mean()), "z": float(z), "z_needed": float(z_need), "verdict": verdict,
           "top10_tune": [(n, round(objective(res[n]["tune"]), 4), round(objective(res[n]["lock"]), 4)) for n in ranked[:10]]}
(K.STATE / "research" / f"tuning_summary{SUFFIX}.json").write_text(json.dumps(summary, indent=1, default=float))
log_experiment({"event": "tuning_lab", **{k: v for k, v in summary.items() if k != "top10_tune"}})
print(json.dumps(summary, indent=1, default=lambda x: round(float(x), 4)))
