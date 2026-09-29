"""Outer loop v2 (canon C11, C15-C21): blind 12-month windows played by the SELF-ADJUSTING system; between
rounds the loop trains the training basis (starting defaults + adaptation meta-parameters). No manual
changes mid-test (C16). Anti-cheat gates every round (C18). Volatility-first while below 1%/week (C21).

usage: livesim_loop2.py [max_windows]"""
import json, sys, time, subprocess, glob, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, livesim, policy, adaptive as A, objective as O, basis_search as B
from engine import blind_gates, health, provenance
from engine.improve import log_experiment

DIR = livesim.DIR
SRC_FILE = Path(__file__).resolve()
STATE = DIR / "loop2.json"
MAXW = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
PAR, SCREEN_N, N_CAND = 3, 10, 24
MODEL_SEED = 7                    # random_state of every model the trader fits; recorded as the worker seed
WORKER_TIMEOUT_S, WORKER_MEM_MB, WORKER_HEARTBEAT_S, BEAT_EVERY_S = 3 * 3600, 6000, 900, 30
WORKER_FILES = ("result2.json", "blind_audit2.json", "health2.jsonl")

# C22: the owner wants +/-200% years -> weekly swings near 15%. Aggressive settings dominate the search.
VOL_TARGET = 0.15
CFG_SPACE = {"k": [1, 1, 2, 2, 3, 4], "exit_q": [0.5, 0.7, 0.8, 0.9], "rebalance_weeks": [1, 1, 2], "brake": [None, None, 0.15],
             "max_per_sector": [None, 2], "w_model": [0.85, 1.0, 1.0], "pick": ["hivol", "hivol", "top"], "pool_q": [0.3, 0.5, 0.7, 0.9, 0.95],
             "liq_q": [0.0, 0.0, 0.2], "vol_filter": [False, False, True], "stress_thr": [None, 1.0, 1.05], "stress_k": [2, 3, 4],
             "trend_filter": [None, -0.05], "trend_gross": [0.0, 0.5],
             "w_move": [0.0, 0.3, 0.5, 0.7], "w_mom": [0.0, 0.2, 0.4]}
META_SPACE = B.META_SPACE          # single source (engine/basis_search.py); tests pin it to the Bible Phase 19 list

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
    daily_dd = r["max_dd"]
    wk = np.array(S.weeks)
    r.update(O.week_row(wk, BAND) if len(wk) else O.week_row([]))        # tiers 1-3 statistics, one definition (engine/objective)
    r["win_weeks"] = float((wk > 0).mean()) if len(wk) else 0.5
    r["max_dd"] = daily_dd                                                # tier 2 reads the daily-path drawdown
    r["weekly_returns"] = [float(x) for x in wk]                          # the distribution, not just the mean (Phase 46)
    return r


BAND = O.BAND                # "about 7%": below 5% is too low, above 10% too risky (C38)


def tiered(rows):
    """C39 via the tested library: (score, tier1, risk, tier3). The score is an integer-packed lexicographic key, so a
    lower-tier gain can never outweigh a higher-tier loss. Old scalar: engine.objective.tiered_legacy."""
    s = O.evaluate(rows)
    return s.scalar, s.t1, s.risk, s.t3


def objective(rows, phase):
    """Kept for the old call sites: (score, mean week, swing, worst drawdown)."""
    rows = list(rows)
    mw = float(np.mean([r["mean_week"] for r in rows])) if rows else 0.0
    sd = float(np.mean([r["sd_week"] for r in rows])) if rows else 0.0
    worst = min((r["max_dd"] for r in rows), default=0.0)
    return tiered(rows)[0], mw, sd, worst


def archive_dirs(root):
    """Window directories the basis trains on: those with weekly snapshots. Directories starting with '_' are backups
    (e.g. _w01c_original duplicates w01c) and would double-count a window, so they are skipped. Sorted by name."""
    root = Path(root)
    return [a for a in sorted(root.glob("*")) if a.is_dir() and not a.name.startswith("_") and list(a.glob("wsnap_*"))]


def as_search_window(w):
    """Tag a loaded window for basis_search: an id and an `end` used only to forbid windows after an as_of."""
    return {**w, "end": w["closes"].index[-1]}


def train_basis(wins, cfg, meta, seed, evaluate=None, as_of=None, config=None):
    """Phase 19 outer step: 24 random configs screened on 10 random archived windows, top 3 (+ incumbent) confirmed on all,
    adopted only through the firewall + held-out bootstrap (engine.basis_search). `evaluate` defaults to run_window."""
    ev = evaluate or run_window
    sc = config or B.SearchConfig(n_start=N_CAND, n_screen=SCREEN_N, seed=seed)
    return B.BasisSearch(ev, [w if "end" in w else as_search_window(w) for w in wins], cfg, meta, sc,
                         cfg_space=CFG_SPACE, meta_space=META_SPACE).run(as_of=as_of)


def _old_objective(rows, phase):
    mw = float(np.mean([r["mean_week"] for r in rows]))
    sd = float(np.mean([r["sd_week"] for r in rows]))
    worst = min(r["max_dd"] for r in rows)
    if worst < -0.99:                                       # only floor left (C22): not a total wipe-out
        return -9.0, mw, sd, worst
    # C31: first the weekly average toward +7%, second the accuracy (share of winning weeks)
    acc = float(np.mean([r.get("win_weeks", 0.5) for r in rows]))
    return mw + 0.004 * (acc - 0.5), mw, sd, worst


def expected_config(cfg, meta):
    """What a worker must report as its config (JSON round-trip, so tuples/None compare the way they are logged)."""
    return json.loads(json.dumps({"cfg": cfg, "meta": meta}, default=str))


def reset_worker_files(run_dir):
    """A new run never inherits an older run's result, audit or health log."""
    for f in WORKER_FILES:
        (Path(run_dir) / f).unlink(missing_ok=True)


def _heartbeat(wl, stop):
    while not stop.wait(BEAT_EVERY_S):
        wl.beat()


def worker(run_id, cfg, meta):
    """One sealed window on the live clock. Reports through health.WorkerLog (start, config, window, seed, heartbeat every
    30 s, complete). The Phase 21/22 audit of the run must pass BEFORE anything is archived: on failure the worker
    records `invalid` and writes no result, so its window is excluded (and never enters the long-term memory bank)."""
    a = DIR / run_id
    a.mkdir(exist_ok=True)
    for f in ("result2.json", "blind_audit2.json"):
        (a / f).unlink(missing_ok=True)                      # never let an older run's result stand in for this one
    wl = health.WorkerLog(a / "health2.jsonl", run_id)
    wl.begin(expected_config(cfg, meta), run_id, MODEL_SEED, code=provenance.code_stamp())
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(wl, stop), daemon=True).start()
    try:
        feed, trader, sealed, wall = livesim.run(cfg, run_id, log=lambda *x: print(f"[{run_id}]", *x, flush=True),
                                                 adaptive=True, meta=meta)
        findings = feed.audit()
        blind_gates.save_report(findings, a / "blind_audit2.json")
        bad = [f for f in findings if f.severity == "fail"]
        if bad:
            wl.emit("invalid", why="blind gates failed: " + "; ".join(str(f) for f in bad[:3]))
            print(f"[{run_id}] BLIND GATES FAILED - window excluded: {bad[0]}", flush=True)
            return
        _archive(run_id, cfg, meta, feed, trader, wall, a, wl)
    finally:
        stop.set()


def _archive(run_id, cfg, meta, feed, trader, wall, a, wl):
    problems = health.validate_result({"window": run_id, "seed": MODEL_SEED,
                                       "weekly_returns": [float(x) for x in trader.session.weeks]}, run_id, MODEL_SEED)
    if problems:                                             # before ANY archive write: an invalid window must not feed the memory bank
        wl.emit("invalid", why="; ".join(problems))
        print(f"[{run_id}] result failed validation - window excluded: {problems}", flush=True)
        return
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
    r["provenance"] = provenance.stamp({"cfg": cfg, "meta": meta}, seed=run_id)
    r.update({"window": run_id, "seed": MODEL_SEED, "run_id": run_id, "prior_cfg": cfg, "meta": meta, "preseason": trader.preseason, "clock_s": wall,
              "ms_per_day": 1000 * wall / max(1, len(trader.session.days)), "used_cfg": trader.cfg})
    (a / "result2.json").write_text(json.dumps(r, default=str))
    wl.done(a / "result2.json", code=provenance.code_stamp())


def run_workers(ids, cfg, meta, supervise=health.supervise):
    """One supervised child per window (timeout, memory ceiling, heartbeat), all in parallel; worker stdout is inherited."""
    out = {}

    def launch(r):
        out[r] = supervise([sys.executable, "-u", str(SRC_FILE), "--worker", r, json.dumps(cfg), json.dumps(meta)],
                           DIR / r / "health2.jsonl", r, WORKER_TIMEOUT_S, WORKER_MEM_MB, WORKER_HEARTBEAT_S, stdout="inherit")
    threads = [threading.Thread(target=launch, args=(r,)) for r in ids]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return out


def classify_round(ids, cfg, meta, rnd=None, root=None, stale_check=None):
    """Phase 24: classify every worker; anything not OK is EXCLUDED (never replaced, reused or averaged)."""
    root = Path(root) if root else DIR
    stale_check = stale_check or (lambda row: provenance.stale({k: row.get(k) for k in ("code_hash", "code_files", "code_mixed")}))
    want = expected_config(cfg, meta)
    workers = {}
    for r in ids:
        rp = root / r / "result2.json"
        try:
            res = json.loads(rp.read_text()) if rp.exists() else None
        except ValueError:
            res = None                                       # torn result file: classify() reports it invalid
        workers[r] = {"log": health.read_log(root / r / "health2.jsonl"), "config": want, "window": r, "seed": MODEL_SEED,
                      "result": res}
    report = health.exclusion_report(workers, stale_check=stale_check)
    if rnd is not None:
        health.write_report(report, root / f"round_{rnd:02d}_loop2_health.json")
    return report


def reveal_round(run_ids, adjustments_locked, sealed=None):
    """Reveal the true periods only through a RevealGate opened AFTER the basis decision (Phase 21). If adjustments are
    not locked the gate refuses (PermissionError) and nothing is revealed."""
    sealed = sealed or livesim.SealedYear
    gate = blind_gates.RevealGate(window_label="")
    if adjustments_locked:
        gate.lock_adjustments()
        gate.predictions_recorded = gate.trades_completed = gate.learning_finalized = True
    return {r: sealed(r).reveal(gate) for r in run_ids}


def main():
    rnd = len(st["windows"]) // PAR + 1
    while len(st["windows"]) < MAXW:
        ids = [f"w{rnd:02d}{x}" for x in "abc"[:PAR]]
        for r in ids:
            livesim.SealedYear(r)                                # sealed one at a time: no duplicate draws
        print(f"\n=== round {rnd} ({st['phase']} phase): {PAR} sealed 12-month windows, basis v{st['version']} ===", flush=True)
        t0 = time.perf_counter()
        for r in ids:
            (DIR / r).mkdir(exist_ok=True)
            reset_worker_files(DIR / r)
        run_workers(ids, st["cfg"], st["meta"])
        print(f"  round wall time {time.perf_counter() - t0:.0f}s", flush=True)
        report = classify_round(ids, st["cfg"], st["meta"], rnd)
        for attempt in range(2):                                 # a worker that outlived a code edit is rerun, not judged
            stale_ids = [e["worker"] for e in report["excluded"] if e["status"] == health.STALE]
            if not stale_ids:
                break
            for r in stale_ids:
                print(f"  [{r}] STALE CODE - rerunning the window under current code", flush=True)
                reset_worker_files(DIR / r)
            run_workers(stale_ids, st["cfg"], st["meta"])
            report = classify_round(ids, st["cfg"], st["meta"], rnd)
        for e in report["excluded"]:
            print(f"  [{e['worker']}] EXCLUDED {e['status']}: {e['why']}", flush=True)
        done = [r for r in ids if r in report["included"]]     # only included workers are ever averaged
        assert not health.check_no_silent_replacement(report, done)
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
        wins = [load_window(a) for a in archive_dirs(DIR)]
        t1 = time.perf_counter()
        res = train_basis(wins, st["cfg"], st["meta"], seed=1000 * st["version"] + rnd)
        inc, win_ = res.incumbent.confirm, res.winner.confirm
        print(f"  training basis: {res.n_evals} replays; {N_CAND} candidates screened on {res.n_screen} of {res.n_windows} "
              f"windows, top 3 confirmed on all ({time.perf_counter() - t1:.0f}s)", flush=True)
        if res.adopted:
            st["version"] += 1
            st["cfg"], st["meta"] = res.cfg, res.meta
            print(f"  NEW BASIS v{st['version']}: weeks in band {inc.t1:.0%} -> {win_.t1:.0%}, risk {inc.risk:+.3f} -> {win_.risk:+.3f} "
                  f"({res.reason})", flush=True)
            print(f"     defaults {res.cfg}", flush=True)
            print(f"     adaptation {res.meta}", flush=True)
        else:
            print(f"  kept basis v{st['version']} (weeks in band {inc.t1:.0%}, risk {inc.risk:+.3f}): {res.reason}", flush=True)
        fresh = [w["mean_week"] for w in st["windows"]]
        if st["phase"] == "volatility" and np.mean(fresh[-6:]) >= 0.01:
            st["phase"] = "direction"
        # the basis decision above is final: only now may the true periods be revealed (RevealGate, Phase 21)
        for w, y in zip(st["windows"][-len(done):], reveal_round([w["run_id"] for w in st["windows"][-len(done):]], True).values()):
            w["revealed"] = y
        print("  revealed:", {w["run_id"]: w["revealed"] for w in st["windows"][-len(done):]}, flush=True)
        print(f"  running average over {len(fresh)} fresh windows: {np.mean(fresh):+.2%}/week (target +7.00%)", flush=True)
        log_experiment({**res.to_record(), "event": "loop2_basis_search", "round": rnd})
        log_experiment({"event": "loop2_round", "round": rnd, "basis": st["version"], "phase": st["phase"],
                        "fresh_avg": float(np.mean(fresh)), "windows": [w["run_id"] for w in st["windows"][-len(done):]]})
        save()
        if any(w["mean_week"] >= 0.07 for w in st["windows"][-len(done):]):
            print("TARGET REACHED", flush=True); break
        rnd += 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4]))
        sys.exit(0)
    main()
