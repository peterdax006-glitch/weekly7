"""Canon C10 loop: random years from the last 50 (1976-2025), each replayed as if live, adjusting the
system as it goes, until the fresh-year average reaches +7% a week or every year has been played.

Honesty rules (canon C8):
  - each year is played ONCE with the configuration that existed before it was drawn -> 'fresh' result;
    only fresh results count toward the 7%-a-week goal;
  - adjustments are chosen on years already played, never on the next year;
  - an adjustment must also keep up with the S&P and never lose more than 50% within a year on seen years."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, replay
from engine.improve import log_experiment

STATE = K.STATE / "replays" / "loop.json"
TARGET = 0.07
ADJUST_EVERY = 3
N_CANDIDATES = 24
SPACE = {"k": [2, 3, 4, 5, 6], "exit_q": [0.6, 0.7, 0.8, 0.9], "brake": [None, 0.05, 0.08, 0.12],
         "max_per_sector": [None, 2], "w_model": [0.3, 0.5, 0.7], "pick": ["top", "hivol"],
         "pool_q": [0.9, 0.95, 0.98], "liq_q": [0.3, 0.5, 0.7], "vol_filter": [True, False]}

st = json.loads(STATE.read_text()) if STATE.exists() else None
rng = np.random.default_rng()
if st is None:
    order = [int(y) for y in rng.permutation(np.arange(1976, 2026))]
    st = {"order": order, "played": [], "config": dict(replay.CHAMPION), "version": 1, "adjustments": []}


def save():
    STATE.write_text(json.dumps(st, indent=1, default=float))


def objective(cfg, years, cap=15):
    years = years[-cap:]                   # the most recent seen years keep each adjustment affordable
    rows = [replay.simulate(y, cfg) for y in years]
    mw = np.mean([r["mean_week"] for r in rows])
    excess = np.mean([r["mean_week"] - r["spy_mean_week"] for r in rows])
    worst_dd = min(r["max_drawdown"] for r in rows)
    ok = excess >= 0 and worst_dd >= -0.50
    return (mw if ok else -1 + mw), {"mean_week": mw, "excess_vs_spy": excess, "worst_dd": worst_dd, "passes": ok}


t0 = time.time()
while len(st["played"]) < len(st["order"]):
    Y = st["order"][len(st["played"])]
    try:
        replay.prepare_year(Y)
    except Exception as e:                      # not enough data for this year (e.g. too few listed survivors)
        st["played"].append({"year": Y, "skipped": str(e)[:200]}); save()
        print(f"{Y}: skipped ({e})", flush=True); continue
    r = replay.simulate(Y, st["config"])
    r.update({"fresh": True, "config_version": st["version"]})
    st["played"].append(r); save()
    fresh = [p for p in st["played"] if p.get("fresh")]
    avg = np.mean([p["mean_week"] for p in fresh])
    print(f"[{time.time() - t0:6.0f}s] {Y} (v{st['version']}): {r['year_return']:+.1%} for the year "
          f"(S&P {r['spy_year_return']:+.1%}), avg week {r['mean_week']:+.2%}, {r['weeks_ge_7']} weeks >= +7%, "
          f"max DD {r['max_drawdown']:.0%} | fresh-year average so far {avg:+.2%}/week over {len(fresh)} years", flush=True)
    log_experiment({"event": "loop_fresh_year", **{k: v for k, v in r.items() if k not in ("weekly", "trades")}})
    if avg >= TARGET and len(fresh) >= 10:
        print("TARGET REACHED on fresh years", flush=True); break
    if len(fresh) % ADJUST_EVERY == 0:
        seen = [p["year"] for p in fresh]
        cur_obj, cur_info = objective(st["config"], seen)
        best = (cur_obj, st["config"], cur_info)
        for _ in range(N_CANDIDATES):
            c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
            o, info = objective(c, seen)
            if o > best[0]:
                best = (o, c, info)
        if best[1] is not st["config"] and best[0] > cur_obj + 0.0005:
            st["version"] += 1
            st["adjustments"].append({"after_years": seen, "from": st["config"], "to": best[1], "old": cur_info,
                                      "new": best[2], "version": st["version"]})
            st["config"] = best[1]
            print(f"   ADJUSTED -> v{st['version']}: {best[1]}  seen-years avg week {best[2]['mean_week']:+.2%} "
                  f"(was {cur_info['mean_week']:+.2%}), worst DD {best[2]['worst_dd']:.0%}", flush=True)
            log_experiment({"event": "loop_adjustment", "version": st["version"], "config": best[1], **best[2]})
        else:
            print(f"   no adjustment beat v{st['version']} on the {len(seen)} seen years "
                  f"(avg week {cur_info['mean_week']:+.2%})", flush=True)
        save()

fresh = [p for p in st["played"] if p.get("fresh")]
summary = {"years_played": len(fresh), "fresh_avg_week": float(np.mean([p["mean_week"] for p in fresh])),
           "fresh_avg_spy_week": float(np.mean([p["spy_mean_week"] for p in fresh])),
           "years_beating_spy": int(sum(p["year_return"] > p["spy_year_return"] for p in fresh)),
           "weeks_ge_7_total": int(sum(p["weeks_ge_7"] for p in fresh)), "weeks_total": int(sum(p["weeks"] for p in fresh)),
           "target_reached": bool(np.mean([p["mean_week"] for p in fresh]) >= TARGET), "final_config": st["config"],
           "versions": st["version"]}
st["summary"] = summary; save()
print(json.dumps(summary, indent=1, default=float))
