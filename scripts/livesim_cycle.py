"""Canon C11 cycle: run a sealed hidden year on the live clock -> examine blind -> adjust -> reveal.
Repeats with a NEW hidden year each cycle until a run averages +7% a week (or max cycles).

usage: livesim_cycle.py [max_cycles]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, livesim, policy
from engine.improve import log_experiment

DIR = livesim.DIR
STATE = DIR / "cycles.json"
MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TARGET = 0.07
SPACE = {"k": [2, 3, 4, 5, 6, 8], "exit_q": [0.6, 0.7, 0.8, 0.9], "brake": [None, 0.05, 0.08, 0.12],
         "max_per_sector": [None, 2], "w_model": [0.3, 0.5, 0.7], "pick": ["top", "hivol"],
         "pool_q": [0.9, 0.95, 0.98], "liq_q": [0.3, 0.5, 0.7], "vol_filter": [True, False]}
START_CFG = {"k": 4, "exit_q": 0.8, "brake": 0.08, "max_per_sector": 2, "w_model": 0.5, "pick": "top",
             "pool_q": 0.95, "liq_q": 0.5, "vol_filter": True}
st = json.loads(STATE.read_text()) if STATE.exists() else {"cycles": [], "config": START_CFG, "version": 1}
save = lambda: STATE.write_text(json.dumps(st, indent=1, default=float))


def replay_variant(cfg, snaps, closes, cost_bps, divs):
    """Re-trade a finished hidden year from its decision snapshots under another config (no new information)."""
    sessions = closes.index
    dec = {pd.Timestamp(k): v for k, v in snaps.items()}
    cash, pos, week_start, capped, weeks, eq = K.START_CASH, {}, K.START_CASH, False, [], []
    for i, d in enumerate(sessions):
        px = closes.loc[d]
        val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
        week_end = i + 1 >= len(sessions) or sessions[i + 1].isocalendar().week != d.isocalendar().week
        target = None
        if d in dec:
            p = dec[d]
            s = policy.score(p, cfg["w_model"])
            ok = ~((p["ev_red_flag"] > 0) | ((p["ev_offering"] > 0) & (p["log_dv"].rank(pct=True) < 0.5)))
            if cfg["vol_filter"]:
                ok &= ~((p["vol20"].rank(pct=True) > 0.9) | (p["max20"].rank(pct=True) > 0.9))
            ok &= p["log_dv"].rank(pct=True) >= cfg["liq_q"]
            target = policy.topk_targets(s[ok], list(pos), cfg["k"], cfg["exit_q"], divs if cfg["max_per_sector"] else None,
                                         cfg["max_per_sector"], pick=cfg["pick"], vol=p["vol20"], pool_q=cfg["pool_q"])
        elif not capped and cfg["brake"] and val / week_start - 1 <= -cfg["brake"]:
            target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * policy.TOPK["brake_exposure"]
            capped = True
        if target is not None:
            for t in set(pos) | set(target.index):
                pr = px.get(t, np.nan)
                if not np.isfinite(pr):
                    continue
                dv = target.get(t, 0.0) * 0.985 * val - pos.get(t, 0.0) * pr
                if abs(dv) >= 1:
                    cash -= dv + abs(dv) * cost_bps / 1e4
                    pos[t] = pos.get(t, 0.0) + dv / pr
            val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
        eq.append(val)
        if week_end:
            weeks.append(val / week_start - 1)
            week_start, capped = val, False
    e = pd.Series(eq)
    return {"mean_week": float(np.mean(weeks)), "weeks_ge_7": int(sum(w >= 0.07 for w in weeks)),
            "year_return": float(e.iloc[-1] / K.START_CASH - 1), "max_dd": float((e / e.cummax() - 1).min())}


def examine(feed, trader):
    """Blind diagnosis: what held the returns back (no year, disguised names)."""
    w = pd.Series([x["ret"] for x in trader.weeks])
    days = pd.DataFrame(trader.days).set_index("session")
    closes = feed._stocks["Close"].loc[feed.first_live:]          # post-run analysis of the finished year only
    spy = feed._market["Close"]["SPY"].loc[feed.first_live - pd.Timedelta(days=7):]
    spy_w = spy.resample("W-FRI").last().pct_change().dropna()
    # holding-period attribution: each week's names and what they returned
    contrib = []
    for a, b in zip([None] + trader.weeks[:-1], trader.weeks):
        names = b["holdings"]
        start = a["week_end"] if a else str(feed.first_live.date())
        seg = closes.loc[start:b["week_end"], names] if names else None
        if seg is not None and len(seg) > 1:
            r = seg.iloc[-1] / seg.iloc[0] - 1
            for n, v in r.items():
                contrib.append({"week_end": b["week_end"], "code": n, "ret": float(v)})
    C = pd.DataFrame(contrib)
    costs = sum(abs(t["dollars"]) for t in trader.broker.log) * feed.cost_bps / 1e4
    turnover = sum(abs(t["dollars"]) for t in trader.broker.log) / K.START_CASH
    # opportunity check: how did the top-ranked names the system SKIPPED do (eligibility filters, k)?
    missed = []
    for sdate, snap in trader.snaps.items():
        nxt = [x for x in trader.weeks if x["week_end"] > sdate][:1]
        if not nxt:
            continue
        seg = closes.loc[sdate:nxt[0]["week_end"]]
        top = snap["score"].nlargest(20).index
        r = (seg.iloc[-1] / seg.iloc[0] - 1).reindex(top).dropna()
        held = [t for t in trader.picks if t["session"] == sdate]
        held = set(held[0]["names"]) if held else set()
        missed.append({"held": float(r[r.index.isin(held)].mean()) if held else np.nan,
                       "top20": float(r.mean()), "top20_hivol": float(r.reindex(snap.loc[top, "vol20"].nlargest(5).index).mean())})
    M = pd.DataFrame(missed)
    return {
        "mean_week": float(w.mean()), "median_week": float(w.median()), "weeks": len(w),
        "weeks_ge_7": int((w >= 0.07).sum()), "weeks_le_m7": int((w <= -0.07).sum()),
        "year_return": float(days["equity"].iloc[-1] / K.START_CASH - 1),
        "market_year_return": float((1 + spy_w).prod() - 1), "market_mean_week": float(spy_w.mean()),
        "max_dd": float((days["equity"] / days["equity"].cummax() - 1).min()),
        "brakes": int(sum(x["brake"] for x in trader.weeks)), "costs_dollars": float(costs), "turnover_x": float(turnover),
        "worst_positions": C.nsmallest(5, "ret").to_dict("records") if len(C) else [],
        "best_positions": C.nlargest(5, "ret").to_dict("records") if len(C) else [],
        "share_of_positions_losing": float((C["ret"] < 0).mean()) if len(C) else None,
        "held_vs_top20_vs_hivol_week": M.mean().to_dict() if len(M) else {},
    }


def worker(run_id, cfg):
    """One sealed hidden year: live-clock run, blind diagnosis, archive. Never reveals the year."""
    feed, trader, sealed, wall = livesim.run(cfg, run_id, log=lambda *a: print(f"[{run_id}]", *a, flush=True))
    sessions = len(trader.days)
    diag = examine(feed, trader)
    arch = DIR / run_id
    arch.mkdir(exist_ok=True)
    feed._stocks["Close"].loc[feed.first_live:].to_parquet(arch / "closes.parquet")
    for k, v in trader.snaps.items():
        v.to_parquet(arch / f"snap_{k}.parquet")
    (arch / "meta.json").write_text(json.dumps({"cost_bps": feed.cost_bps}))
    feed.sic.to_parquet(arch / "sic.parquet")
    (arch / "result.json").write_text(json.dumps({"run_id": run_id, "config": cfg, "diagnosis": diag, "clock_seconds": wall,
                                                  "sessions": sessions, "ms_per_day": 1000 * wall / sessions}, default=float))


if len(sys.argv) > 2 and sys.argv[1] == "--worker":
    worker(sys.argv[2], json.loads(sys.argv[3]))
    sys.exit(0)

import subprocess
PAR = 3
rnd = len(st["cycles"]) // PAR + 1
while len(st["cycles"]) < MAX:
    ids = [f"r{rnd:02d}{x}" for x in "abc"[:PAR]]
    print(f"\n=== round {rnd}: {PAR} sealed hidden years in parallel, config v{st['version']} ===", flush=True)
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, "-u", __file__, "--worker", r, json.dumps(st["config"])]) for r in ids]
    codes = [p.wait() for p in procs]
    print(f"  round wall time {time.perf_counter() - t0:.0f}s", flush=True)
    done = [r for r, c in zip(ids, codes) if c == 0 and (DIR / r / "result.json").exists()]
    for r in done:
        res = json.loads((DIR / r / "result.json").read_text())
        d = res["diagnosis"]
        print(f"  [{r}] avg week {d['mean_week']:+.2%} | {d['weeks_ge_7']} weeks >= +7% | year {d['year_return']:+.1%} "
              f"vs market {d['market_year_return']:+.1%} | max DD {d['max_dd']:.0%} | costs ${d['costs_dollars']:.0f} "
              f"| clock {res['ms_per_day']:.0f} ms/trading day", flush=True)
        print(f"       held vs top-20 vs top-20-high-vol (avg week): {json.dumps({k: round(v, 4) for k, v in d['held_vs_top20_vs_hivol_week'].items()})}"
              f" | losing positions {d['share_of_positions_losing']:.0%}", flush=True)
        st["cycles"].append({**res, "config_version": st["version"]})
    # ---- adjust across EVERY hidden year played so far ----
    years = []
    for c in st["cycles"]:
        a = DIR / c["run_id"]
        snaps = {f.stem[5:]: pd.read_parquet(f) for f in sorted(a.glob("snap_*.parquet"))}
        sc = pd.read_parquet(a / "sic.parquet")
        years.append((snaps, pd.read_parquet(a / "closes.parquet"), json.loads((a / "meta.json").read_text())["cost_bps"],
                      {t: policy.sic_division(x) for t, x in zip(sc["ticker"], sc["sic"])}))

    def score_cfg(cfg):
        rs = [replay_variant(cfg, s_, c_, b_, dv_) for s_, c_, b_, dv_ in years]
        mw = float(np.mean([r["mean_week"] for r in rs]))
        worst = min(r["max_dd"] for r in rs)
        return (mw if worst >= -0.5 else -1 + mw), mw, worst

    t1 = time.perf_counter()
    base = score_cfg(st["config"])
    rng = np.random.default_rng()
    best = (base[0], st["config"], base)
    for _ in range(60):
        c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        sc_ = score_cfg(c)
        if sc_[0] > best[0]:
            best = (sc_[0], c, sc_)
    adopted = best[1] is not st["config"] and best[0] > base[0] + 0.0005
    print(f"  adjustment search: 60 candidates x {len(years)} hidden years in {time.perf_counter() - t1:.0f}s", flush=True)
    if adopted:
        st["version"] += 1
        st["config"] = best[1]
        print(f"  ADJUSTED -> v{st['version']}: {best[1]}  (avg week over {len(years)} hidden years "
              f"{base[1]:+.2%} -> {best[2][1]:+.2%}, worst DD {best[2][2]:.0%})", flush=True)
    else:
        print(f"  kept v{st['version']}: nothing beat it across {len(years)} hidden years ({base[1]:+.2%}/week)", flush=True)
    # reveal only now, after the adjustment is locked in
    for c in st["cycles"][-len(done):]:
        c["revealed_year"] = livesim.SealedYear(c["run_id"]).reveal()
        log_experiment({"event": "livesim_cycle", "run_id": c["run_id"], "year": c["revealed_year"],
                        "mean_week": c["diagnosis"]["mean_week"], "weeks_ge_7": c["diagnosis"]["weeks_ge_7"],
                        "year_return": c["diagnosis"]["year_return"], "market": c["diagnosis"]["market_year_return"],
                        "version": c["config_version"]})
    print("  revealed:", {c["run_id"]: c["revealed_year"] for c in st["cycles"][-len(done):]}, flush=True)
    st.setdefault("rounds", []).append({"round": rnd, "adopted": adopted, "config_after": st["config"],
                                        "avg_week_all_years_before": base[1], "avg_week_all_years_after": best[2][1]})
    save()
    fresh = [c["diagnosis"]["mean_week"] for c in st["cycles"]]
    print(f"  running average over all {len(fresh)} hidden years played: {np.mean(fresh):+.2%}/week (target +7.00%)", flush=True)
    if any(c["diagnosis"]["mean_week"] >= TARGET for c in st["cycles"][-len(done):]):
        print("TARGET REACHED: a hidden year averaged +7% a week", flush=True)
        break
    rnd += 1
