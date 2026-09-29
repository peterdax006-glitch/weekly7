"""Outer loop v2 (canon C11, C15-C21): blind 12-month windows played by the SELF-ADJUSTING system; between
rounds the loop trains the training basis (starting defaults + adaptation meta-parameters). No manual
changes mid-test (C16). Anti-cheat gates every round (C18). Volatility-first while below 1%/week (C21).

usage: livesim_loop2.py [max_windows]"""
import json, sys, time, subprocess, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, livesim, policy, adaptive as A
from engine.improve import log_experiment

DIR = livesim.DIR
STATE = DIR / "loop2.json"
MAXW = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
PAR, SCREEN_N, N_CAND = 3, 10, 24

# C22: the owner wants +/-200% years -> weekly swings near 15%. Aggressive settings dominate the search.
VOL_TARGET = 0.15
CFG_SPACE = {"k": [1, 1, 2, 2, 3, 4], "exit_q": [0.5, 0.7, 0.8, 0.9], "rebalance_weeks": [1, 1, 2], "brake": [None, None, 0.15],
             "max_per_sector": [None, 2], "w_model": [0.85, 1.0, 1.0], "pick": ["hivol", "hivol", "top"], "pool_q": [0.3, 0.5, 0.7, 0.9, 0.95],
             "liq_q": [0.0, 0.0, 0.2], "vol_filter": [False, False, True], "stress_thr": [None, 1.0, 1.05], "stress_k": [2, 3, 4],
             "trend_filter": [None, -0.05], "trend_gross": [0.0, 0.5],
             "w_move": [0.0, 0.3, 0.5, 0.7], "w_mom": [0.0, 0.2, 0.4]}
META_SPACE = {"half_life": [3, 6, 12], "prior_weeks": [4, 8, 16], "switch_z": [1.5, 2.0, 3.0], "min_weeks": [3, 6],
              "cooldown": [2, 4], "revert_drop": [0.02, 0.04, 0.08], "ic_beta": [0.0, 0.5, 1.0, 2.0],
              "det_max": [0.0, 0.25, 0.5], "det_min_weeks": [4, 8],
              # C34 memory factors: fade speed, market-similarity width, weight of earlier windows, shrinkage, shock
              "mem_half_life": [4, 8, 16, 32], "mem_bandwidth": [0.75, 1.5, 3.0], "mem_prior_scale": [0.0, 0.1, 0.3, 0.6],
              "mem_shrink": [2, 6, 12], "mem_shock_k": [1.5, 2.5, 4.0], "mem_shock_cut": [0.1, 0.25, 0.5]}

st = json.loads(STATE.read_text()) if STATE.exists() else {
    "windows": [], "version": 1,
    # starting defaults from the sensitivity study: model only (+0.32%/wk, t=3.8), lower liquidity floor
    "cfg": {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
            "pick": "hivol", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
            "trend_filter": None, "trend_gross": 0.0},
    "meta": dict(A.META_DEFAULT), "phase": "volatility"}
save = lambda: STATE.write_text(json.dumps(st, indent=1, default=str))


def load_window(a):
    a = Path(a)
    ws = {f.stem[6:]: pd.read_parquet(f) for f in sorted(a.glob("wsnap_*.parquet"))}
    closes = pd.read_parquet(a / ("closes_v2.parquet" if (a / "closes_v2.parquet").exists() else "closes.parquet"))
    opens = pd.read_parquet(a / "opens_v2.parquet") if (a / "opens_v2.parquet").exists() else None
    ltm = pd.read_parquet(a / "ltm.parquet") if (a / "ltm.parquet").exists() else None
    sc = pd.read_parquet(a / "sic.parquet")
    return {"id": a.name, "snaps": ws, "closes": closes, "opens": opens, "ltm": ltm, "bps": json.loads((a / "meta.json").read_text())["cost_bps"],
            "divs": {t: policy.sic_division(x) for t, x in zip(sc["ticker"], sc["sic"])}}


def run_window(w, cfg, meta):
    S = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=meta, opens=w["opens"],
                 long_term=w["ltm"])
    r = S.result()
    wk = np.array(S.weeks)
    r["sd_week"] = float(wk.std()) if len(wk) > 1 else 0.0
    r["win_weeks"] = float((wk > 0).mean()) if len(wk) else 0.5
    band = (np.abs(wk) >= BAND[0]) & (np.abs(wk) <= BAND[1])
    r["in_band"] = float(band.mean()) if len(wk) else 0.0                 # tier 1: share of ~7% weeks
    r["over_band"] = float((np.abs(wk) > BAND[1]).mean()) if len(wk) else 0.0
    r["worst5"] = float(np.quantile(wk, 0.05)) if len(wk) > 5 else 0.0   # tier 2: tail risk
    r["pos_in_band"] = float((wk[band] > 0).mean()) if band.any() else 0.0  # tier 3
    return r


BAND = (0.05, 0.10)          # "about 7%": below 5% is too low, above 10% too risky (C38)


def tiered(rows):
    """C39: tier 1 dominates until most weeks are ~7% (majority); then risk; then the positive share."""
    t1 = float(np.mean([r["in_band"] for r in rows]))
    over = float(np.mean([r["over_band"] for r in rows]))
    risk = float(np.mean([r["worst5"] for r in rows])) + float(min(r["max_dd"] for r in rows)) / 4
    t3 = float(np.mean([r["pos_in_band"] for r in rows]))
    reached = t1 >= 0.5
    return (100 * min(t1, 0.5) - 50 * over + (10 * (risk + 0.3) + t3 if reached else 0.0)), t1, risk, t3


def objective(rows, phase):
    """Kept for the old call sites; now defers to the tiered objective."""
    mw = float(np.mean([r["mean_week"] for r in rows]))
    sd = float(np.mean([r["sd_week"] for r in rows]))
    worst = min(r["max_dd"] for r in rows)
    if worst < -0.99:
        return -9.0, mw, sd, worst
    return tiered(rows)[0], mw, sd, worst


def _old_objective(rows, phase):
    mw = float(np.mean([r["mean_week"] for r in rows]))
    sd = float(np.mean([r["sd_week"] for r in rows]))
    worst = min(r["max_dd"] for r in rows)
    if worst < -0.99:                                       # only floor left (C22): not a total wipe-out
        return -9.0, mw, sd, worst
    # C31: first the weekly average toward +7%, second the accuracy (share of winning weeks)
    acc = float(np.mean([r.get("win_weeks", 0.5) for r in rows]))
    return mw + 0.004 * (acc - 0.5), mw, sd, worst


def worker(run_id, cfg, meta):
    stale = DIR / run_id / "result2.json"
    if stale.exists():
        stale.unlink()                                       # never let an older run's result stand in for this one
    feed, trader, sealed, wall = livesim.run(cfg, run_id, log=lambda *a: print(f"[{run_id}]", *a, flush=True),
                                             adaptive=True, meta=meta)
    a = DIR / run_id
    a.mkdir(exist_ok=True)
    feed._stocks["Close"].loc[feed.first_live:].to_parquet(a / "closes_v2.parquet")
    feed._stocks["Open"].loc[feed.first_live:].to_parquet(a / "opens_v2.parquet")
    if trader.long_term is not None:
        trader.long_term.to_parquet(a / "ltm.parquet")
    ep = trader.session.adapter.mem.export()
    if len(ep):                                           # add this window's lessons to the long-term bank
        ep["real_end"] = feed.real_end()
        ep["window"] = run_id
        bank = DIR / "memory_bank.parquet"
        import os
        lock = str(bank) + ".lock"
        for _ in range(600):                          # three workers finish together: one writer at a time
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL); os.close(fd); break
            except FileExistsError:
                time.sleep(0.5)
        try:
            (pd.concat([pd.read_parquet(bank), ep]) if bank.exists() else ep).to_parquet(bank)
        finally:
            os.remove(lock)
    for k, v in trader.snaps.items():
        v.to_parquet(a / f"wsnap_{k}.parquet")
    for k, v in trader.warm_snaps.items():
        v.to_parquet(a / f"warm_{k}.parquet")
    (a / "meta.json").write_text(json.dumps({"cost_bps": feed.cost_bps}))
    feed.sic.to_parquet(a / "sic.parquet")
    r = trader.session.result()
    wk = np.array(trader.session.weeks)
    r["sd_week"] = float(wk.std()) if len(wk) > 1 else 0.0
    band = (np.abs(wk) >= BAND[0]) & (np.abs(wk) <= BAND[1])
    r["in_band"] = float(band.mean()) if len(wk) else 0.0
    r["pos_in_band"] = float((wk[band] > 0).mean()) if band.any() else 0.0
    r["weekly_returns"] = [float(x) for x in wk]
    from engine import provenance
    r["provenance"] = provenance.stamp({"cfg": cfg, "meta": meta}, seed=run_id)
    r.update({"run_id": run_id, "prior_cfg": cfg, "meta": meta, "preseason": trader.preseason, "clock_s": wall,
              "ms_per_day": 1000 * wall / max(1, len(trader.session.days)), "used_cfg": trader.cfg})
    (a / "result2.json").write_text(json.dumps(r, default=str))


if len(sys.argv) > 2 and sys.argv[1] == "--worker":
    worker(sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4]))
    sys.exit(0)

rnd = len(st["windows"]) // PAR + 1
while len(st["windows"]) < MAXW:
    ids = [f"w{rnd:02d}{x}" for x in "abc"[:PAR]]
    for r in ids:
        livesim.SealedYear(r)                                # sealed one at a time: no duplicate draws
    print(f"\n=== round {rnd} ({st['phase']} phase): {PAR} sealed 12-month windows, basis v{st['version']} ===", flush=True)
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, "-u", __file__, "--worker", r, json.dumps(st["cfg"]), json.dumps(st["meta"])]) for r in ids]
    codes = [p.wait() for p in procs]
    done = [r for r, c in zip(ids, codes) if c == 0 and (DIR / r / "result2.json").exists()]
    for r, c in zip(ids, codes):
        if c != 0:
            print(f"  [{r}] worker FAILED (exit {c}) - window excluded, see trace above", flush=True)
    if not done:
        print("  no window finished - stopping", flush=True); break
    # ---- anti-cheat gates (C18) ----
    gate_ok = True
    from engine import provenance
    if provenance.stale(provenance.code_stamp()):
        # the loop itself runs old code: its replays would judge the workers with the wrong rules
        save(); print("  engine code changed since the loop started - restart the loop (no gate judged on mixed code)", flush=True); break
    for r in list(done):
        for attempt in range(2):                             # a worker that outlived a code edit is rerun, not judged
            res = json.loads((DIR / r / "result2.json").read_text())
            pv = res.get("provenance", {})
            if not provenance.stale(pv):
                break
            print(f"  [{r}] STALE CODE (code {pv.get('code_hash')}, edited mid-run: {pv.get('code_mixed')}, now "
                  f"{provenance.code_hash(pv.get('code_files'))}) - rerunning the window under current code", flush=True)
            subprocess.run([sys.executable, "-u", __file__, "--worker", r, json.dumps(res["prior_cfg"]), json.dumps(res["meta"])])
        else:
            print(f"  [{r}] still stale after reruns (code is changing) - excluded", flush=True)
            done.remove(r); continue
        w = load_window(DIR / r)
        re = run_window(w, res["used_cfg"], res["meta"])
        rep = abs(re["year_return"] - res["year_return"]) < 0.005
        S1 = A.replay(res["used_cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=res["meta"],
                      opens=w["opens"], long_term=w["ltm"])
        cut = w["closes"].index[len(w["closes"]) // 2]
        S2 = A.replay(res["used_cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=res["meta"],
                      scramble_after=cut, seed=len(r), opens=w["opens"], long_term=w["ltm"])
        before = lambda S: [d for d in S.decisions if pd.Timestamp(d[0]) <= cut]
        scram = before(S1) == before(S2)
        wk_ = np.array(res.get("weekly_returns", [])) if res.get("weekly_returns") else None
        print(f"  [{r}] ~7% weeks (5-10% moves) {res.get('in_band', float('nan')):.0%} | avg week {res['mean_week']:+.2%} | swing (sd) {res['sd_week']:.2%} | {res['weeks_ge_7']} weeks >= +7% | "
              f"window {res['year_return']:+.1%} | max DD {res['max_dd']:.0%} | {len(res['adaptations'])} self-adjustments | "
              f"missed winners studied {sum(m['winners'] for m in res['missed_winners'])} | clock {res['ms_per_day']:.0f} ms/day", flush=True)
        from engine import fill_audit
        fill_ok, fill_msg = fill_audit.gate(S1, w["opens"], w["closes"])       # C33: next-open fills, proven per fill
        print(f"       gates: re-tester {'OK' if rep else 'MISMATCH'} | future-scramble {'OK' if scram else 'LEAK'} | {fill_msg}", flush=True)
        gate_ok &= rep and scram and fill_ok
        st["windows"].append({**{k: v for k, v in res.items() if k not in ("missed_winners",)},
                              "missed_summary": {"winners": sum(m["winners"] for m in res["missed_winners"]),
                                                 "caught": sum(m["caught"] for m in res["missed_winners"]),
                                                 "detector_final_weight": (res["missed_winners"][-1]["detector_weight"]
                                                                           if res["missed_winners"] else 0)},
                              "basis_version": st["version"], "phase": st["phase"]})
    if not gate_ok:
        save(); print("  ANTI-CHEAT GATE FAILED - stopping before any retraining", flush=True); break
    # ---- train the training basis on every window with weekly snapshots ----
    wins = [load_window(a) for a in sorted(glob.glob(str(DIR / "*"))) if Path(a).is_dir() and list(Path(a).glob("wsnap_*"))]
    rng = np.random.default_rng()
    screen = [wins[i] for i in rng.choice(len(wins), size=min(SCREEN_N, len(wins)), replace=False)]
    t1 = time.perf_counter()
    base = objective([run_window(w, st["cfg"], st["meta"]) for w in screen], st["phase"])
    cands = []
    for _ in range(N_CAND):
        c = {k: v[rng.integers(len(v))] for k, v in CFG_SPACE.items()}
        m = {**st["meta"], **{k: v[rng.integers(len(v))] for k, v in META_SPACE.items()}}
        cands.append((objective([run_window(w, c, m) for w in screen], st["phase"]), c, m))
    cands.sort(key=lambda x: -x[0][0])
    full_base = objective([run_window(w, st["cfg"], st["meta"]) for w in wins], st["phase"])
    best = (full_base, st["cfg"], st["meta"])
    for o, c, m in cands[:3]:                                  # confirm the screen's top 3 on every window
        fo = objective([run_window(w, c, m) for w in wins], st["phase"])
        if fo[0] > best[0][0]:
            best = (fo, c, m)
    adopted = best[1] is not st["cfg"]
    print(f"  training basis: {N_CAND} candidates screened on {len(screen)} windows, top 3 confirmed on all {len(wins)} "
          f"({time.perf_counter() - t1:.0f}s)", flush=True)
    if adopted:
        st["version"] += 1
        st["cfg"], st["meta"] = best[1], best[2]
        print(f"  NEW BASIS v{st['version']}: avg week {full_base[1]:+.2%} -> {best[0][1]:+.2%}, swing {full_base[2]:.2%} -> "
              f"{best[0][2]:.2%}, worst DD {best[0][3]:.0%}\n     defaults {best[1]}\n     adaptation {best[2]}", flush=True)
    else:
        print(f"  kept basis v{st['version']} (avg week {full_base[1]:+.2%}, swing {full_base[2]:.2%})", flush=True)
    fresh = [w["mean_week"] for w in st["windows"]]
    if st["phase"] == "volatility" and np.mean(fresh[-6:]) >= 0.01:
        st["phase"] = "direction"
    for w in st["windows"][-len(done):]:
        w["revealed"] = livesim.SealedYear(w["run_id"]).reveal()
    print("  revealed:", {w["run_id"]: w["revealed"] for w in st["windows"][-len(done):]}, flush=True)
    print(f"  running average over {len(fresh)} fresh windows: {np.mean(fresh):+.2%}/week (target +7.00%)", flush=True)
    log_experiment({"event": "loop2_round", "round": rnd, "basis": st["version"], "phase": st["phase"],
                    "fresh_avg": float(np.mean(fresh)), "windows": [w["run_id"] for w in st["windows"][-len(done):]]})
    save()
    if any(w["mean_week"] >= 0.07 for w in st["windows"][-len(done):]):
        print("TARGET REACHED", flush=True); break
    rnd += 1
