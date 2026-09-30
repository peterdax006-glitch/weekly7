"""Outer loop v2 (canon C11, C15-C21): blind 12-month windows played by the SELF-ADJUSTING system; between
rounds the loop trains the training basis (starting defaults + adaptation meta-parameters). No manual
changes mid-test (C16). Anti-cheat gates every round (C18). Volatility-first while below 1%/week (C21).

usage: livesim_loop2.py [max_windows] [--learner legit|off]

--learner legit (canon C64, default off until Stage 3 validation): every worker also runs the LegitimateLearner as a SHADOW decider
behind engine.learning.test_path - the trader sees only the curator's TraderDay plus the hardened feed, what it learns is filed under
the real year on the trusted side. The traded path (adaptive.Session) is unchanged, so the re-tester, future-scramble, fill audit,
RevealGate, per-round checkpoint and basis lineage all judge exactly what they judged before. The learner's report is stored under
`legit` in result2.json; its own gates (trader path clean, no date-like release, intact store chains) exclude a window on failure."""
import json, sys, time, subprocess, glob, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, livesim, policy, adaptive as A, objective as O, basis_search as B
from engine import blind_gates, health, provenance
from engine import leak_audit as LA
from engine.improve import log_experiment

DIR = livesim.DIR
SRC_FILE = Path(__file__).resolve()
STATE = DIR / "loop2.json"
MAXW = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60


def learner_flag(argv):
    """The --learner mode named on the command line: 'off' (default) or 'legit'. Anything else is refused, never guessed."""
    if "--learner" not in argv:
        return "off"
    i = argv.index("--learner")
    mode = argv[i + 1] if i + 1 < len(argv) else ""
    if mode not in ("off", "legit"):
        raise SystemExit(f"--learner must be 'legit' or 'off', not {mode!r}")
    return mode


LEARNER = learner_flag(sys.argv)
LEARNER_ARGS = ["--learner", LEARNER] if LEARNER != "off" else []
CURATOR_ROOT = K.STATE / "learning" / "curator"       # one hash-chained lane per REAL year, shared by every window (trusted side only)
PAR, SCREEN_N, N_CAND = 3, 10, 24
MODEL_SEED = 7                    # random_state of every model the trader fits; recorded as the worker seed
WORKER_TIMEOUT_S, WORKER_MEM_MB, WORKER_HEARTBEAT_S, BEAT_EVERY_S = 3 * 3600, 6000, 900, 30
WORKER_FILES = ("result2.json", "blind_audit2.json", "health2.jsonl")
THIN_NAMES = 500                  # a window whose universe holds fewer names is THIN (survivor panel, canon C56): reported apart

# C22: the owner wants +/-200% years -> weekly swings near 15%. Aggressive settings dominate the search.
VOL_TARGET = 0.15
CFG_SPACE = {"k": [1, 1, 2, 2, 3, 4], "exit_q": [0.5, 0.7, 0.8, 0.9], "rebalance_weeks": [1, 1, 2], "brake": [None, None, 0.15],
             "max_per_sector": [None, 2], "w_model": [0.85, 1.0, 1.0], "pick": ["hivol", "hivol", "top"], "pool_q": [0.3, 0.5, 0.7, 0.9, 0.95],
             "liq_q": [0.0, 0.0, 0.2], "vol_filter": [False, False, True], "stress_thr": [None, 1.0, 1.05], "stress_k": [2, 3, 4],
             "trend_filter": [None, -0.05], "trend_gross": [0.0, 0.5],
             "w_move": [0.0, 0.3, 0.5, 0.7], "w_mom": [0.0, 0.2, 0.4]}
META_SPACE = B.META_SPACE          # single source (engine/basis_search.py); tests pin it to the Bible Phase 19 list

# C56: the starting basis is data-free (the middle grid value of every knob), never the sensitivity-study defaults, which
# were tuned on the outcomes of 37 real years (80% of possible windows are in-sample for them).
NEUTRAL_CFG = LA.neutral_default_cfg(CFG_SPACE)
# F06 (C69 W-10): the untrained start is data-free in the adaptation meta too. META_DEFAULT was tuned by the sensitivity study on
# real outcomes; NEUTRAL_META puts every searched meta knob at its middle grid value and lets the adapter move every knob it has a
# step grid for (engine.leak_audit.neutral_default_meta documents the choice). Structural switches (rails, dial, dates) keep
# META_DEFAULT's values, which encode no outcome.
NEUTRAL_META = LA.neutral_default_meta(META_SPACE, A.META_DEFAULT, list(A.STEPS))
assert not B.validate_meta(NEUTRAL_META), B.validate_meta(NEUTRAL_META)


def fresh_state():
    return {"windows": [], "version": 0, "cfg": dict(NEUTRAL_CFG), "meta": dict(A.META_DEFAULT), "phase": "volatility",
            "lineage": [], "rounds": []}


def migrate_state(st):
    """A state file written before the lineage existed carries a basis trained on windows of unknown real order and
    defaults from the sensitivity study: reset to the neutral start and mark every old window legacy (kept, but excluded from
    headline averages)."""
    if "lineage" in st:
        return st
    old = st.get("windows", [])
    st = {**fresh_state(), "windows": [{**w, "legacy": True} for w in old], "legacy_basis_version": st.get("version"),
          "legacy_cfg": st.get("cfg"), "phase": st.get("phase", "volatility")}
    return st


def quarantine_tuned_meta(state):
    """F06: a play that ran untrained on the sensitivity-tuned META_DEFAULT (before NEUTRAL_META) is quarantined - marked
    legacy='tuned_meta' so every headline average excludes it (its lessons remain in the memory bank: reported, not removed).
    Trained plays are marked when their meta still carries a tuned non-searched key (the adaptive knob list). Idempotent."""
    for w in state.get("windows", []):
        if w.get("legacy"):
            continue
        searched = () if (w.get("basis_version") in (0, None) or w.get("untrained_basis")) else tuple(META_SPACE)
        if w.get("meta") is not None and LA.tuned_meta_keys(w["meta"], A.META_DEFAULT, NEUTRAL_META, searched):
            w["legacy"], w["tuned_meta"] = LA.TUNED_META_LEGACY, True
    return state


def publish_bank_exclusions(state, root=None):
    """F06 channel 4: the windows quarantined as tuned_meta are listed for livesim.Feed.long_term_memory, which drops their
    lessons from every later trader's long-term memory (the rows stay in the bank: reported, not removed)."""
    ids = sorted({str(w.get("run_id") or w.get("window")) for w in state.get("windows", [])
                  if w.get("legacy") == LA.TUNED_META_LEGACY and (w.get("run_id") or w.get("window"))})
    f = Path(root or DIR) / livesim.BANK_EXCLUSIONS
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"windows": ids, "reason": LA.TUNED_META_LEGACY}), encoding="utf-8")
    tmp.replace(f)
    return ids


st = quarantine_tuned_meta(migrate_state(json.loads(STATE.read_text()))) if STATE.exists() else fresh_state()
save = lambda: STATE.write_text(json.dumps(st, indent=1, default=str))


def load_window(a):
    a = Path(a)
    ws = {f.stem[6:]: pd.read_parquet(f) for f in sorted(a.glob("wsnap_*.parquet"))}
    closes = pd.read_parquet(a / ("closes_v2.parquet" if (a / "closes_v2.parquet").exists() else "closes.parquet"))
    opens = pd.read_parquet(a / "opens_v2.parquet") if (a / "opens_v2.parquet").exists() else None
    ltm = pd.read_parquet(a / "ltm.parquet") if (a / "ltm.parquet").exists() else None
    sc = pd.read_parquet(a / "sic.parquet")
    n_names = int(closes.iloc[0].notna().sum()) if len(closes) else 0
    return {"id": a.name, "n_names": n_names, "thin": n_names < THIN_NAMES, "snaps": ws, "closes": closes, "opens": opens, "ltm": ltm, "bps": json.loads((a / "meta.json").read_text())["cost_bps"],
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


def real_span(run_id):
    """(real start, real end) of a window: referee side only - the trader never sees a real date."""
    st0 = livesim.SealedYear.start_of(livesim.SealedYear(run_id)._read())
    return st0, blind_gates.window_end(st0)


def lineage_from_state(state):
    """The BasisLineage recorded in the state file (versions >= 1; version 0 is the neutral start and is never registered)."""
    lin = LA.BasisLineage()
    for v in state.get("lineage", []):
        lin.register(v["version"], v["cfg"], v["meta"], [LA.TrainedOn(i, pd.Timestamp(a), pd.Timestamp(b)) for i, a, b in v["trained_on"]])
    return lin


def plan_round(ids, lineage, span=real_span):
    """Per window, the basis it plays with: the newest version trained ONLY on windows that ended before this window's real start
    (canon C56), else the neutral untrained start. Returns {id: {cfg, meta, version, untrained, real_start}}."""
    out = {}
    for r in ids:
        start, _ = span(r)
        rec = lineage.basis_for(start)
        if rec is None:
            out[r] = {"cfg": dict(NEUTRAL_CFG), "meta": dict(NEUTRAL_META), "version": 0, "untrained": True, "real_start": start}
        else:
            out[r] = {"cfg": rec["cfg"], "meta": rec["meta"], "version": rec["version"], "untrained": False, "real_start": start}
    return out


def register_basis(state, lineage, version, cfg, meta, wins, span=real_span):
    """Record a newly adopted basis with every window it was trained on (real dates), in memory and in the state file."""
    trained = []
    for w in wins:
        a, b = span(w["id"])
        trained.append([w["id"], str(a.date()), str(b.date())])
    lineage.register(version, cfg, meta, [LA.TrainedOn(i, pd.Timestamp(a), pd.Timestamp(b)) for i, a, b in trained])
    state["lineage"].append({"version": version, "cfg": cfg, "meta": meta, "trained_on": trained})


def round_summary(rnd, plan, done, thin_ids):
    """Per-round accounting asked for by the owner: how many played windows used an untrained basis, and which were THIN."""
    played = [r for r in plan if r in done]
    un = [r for r in played if plan[r]["untrained"]]
    return {"round": rnd, "windows": played, "untrained_basis": un, "n_untrained": len(un), "n_played": len(played),
            "versions": {r: plan[r]["version"] for r in played}, "thin": [r for r in played if r in thin_ids]}


def train_basis(wins, cfg, meta, seed, evaluate=None, as_of=None, config=None, used_from=None, span=real_span):
    """Phase 19 outer step: 24 random configs screened on 10 random archived windows, top 3 (+ incumbent) confirmed on all,
    adopted only through the firewall + held-out bootstrap (engine.basis_search). `evaluate` defaults to run_window.
    F06 (canon C56): `used_from` is the first REAL day the resulting basis may be played; the call itself refuses
    (LateTrainingWindow) if any window it is handed had not ended before that day - main splits the archive first, so a refusal
    means the split was skipped. `span(id)` -> (real start, real end) is the referee's lookup."""
    if used_from is not None:
        LA.refuse_late_training(wins, used_from, span)
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
    records `invalid` and writes no result, so its window is excluded (and never enters the long-term memory bank).
    Canon C56: the network is closed for the whole worker (a socket guard plus the data-refresh functions poisoned) before
    anything else runs; the guard is lifted on exit so an in-process caller (a test) gets its network back."""
    guard = LA.NetworkGuard().install()
    from engine import data as _data
    unpoison = LA.poison(_data, ["update", "download"])
    try:
        _worker(run_id, cfg, meta)
    finally:
        unpoison()
        guard.uninstall()


def _worker(run_id, cfg, meta):
    a = DIR / run_id
    a.mkdir(exist_ok=True)
    for f in ("result2.json", "blind_audit2.json"):
        (a / f).unlink(missing_ok=True)                      # never let an older run's result stand in for this one
    wl = health.WorkerLog(a / "health2.jsonl", run_id)
    wl.begin(expected_config(cfg, meta), run_id, MODEL_SEED, code=provenance.code_stamp())
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(wl, stop), daemon=True).start()
    try:
        hook = None
        if LEARNER == "legit":
            from engine.learning import test_path as TP
            from engine.learning import wiring as W
            W.configure_production(run_id)                           # the hub's sinks persist under state/learning/hub/<run_id>/
            hook = TP.hook_factory(TP.PathConfig(store_root=str(CURATOR_ROOT), seed=MODEL_SEED))
        extra = {} if hook is None else {"hook_factory": hook}          # the off path calls livesim.run exactly as before
        feed, trader, sealed, wall = livesim.run(cfg, run_id, log=lambda *x: print(f"[{run_id}]", *x, flush=True),
                                                 adaptive=True, meta=meta, **extra)
        findings = feed.audit()
        if getattr(trader, "hook", None) is not None:
            findings += [blind_gates.Finding(f.gate, f.severity, f.message) for f in trader.hook.findings()]
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
    n_names = int(feed._stocks["Close"].loc[feed.first_live].notna().sum())     # names with a price when the window opens
    r["n_names"], r["thin"] = n_names, n_names < THIN_NAMES
    r["dropped_crypto"] = len(getattr(feed, "dropped_crypto", []))
    r["provenance"] = provenance.stamp({"cfg": cfg, "meta": meta}, seed=run_id)
    if getattr(trader, "hook", None) is not None:
        r["legit"] = trader.hook.report()                                # the shadow learner's own record; never part of the traded result
    r.update({"window": run_id, "seed": MODEL_SEED, "run_id": run_id, "prior_cfg": cfg, "meta": meta, "preseason": trader.preseason, "clock_s": wall,
              "ms_per_day": 1000 * wall / max(1, len(trader.session.days)), "used_cfg": trader.cfg})
    (a / "result2.json").write_text(json.dumps(r, default=str))
    wl.done(a / "result2.json", code=provenance.code_stamp())


def run_workers(ids, cfg, meta, supervise=health.supervise, per_id=None):
    """One supervised child per window (timeout, memory ceiling, heartbeat), all in parallel; worker stdout is inherited.
    `per_id` = {window: (cfg, meta)} gives each window its own past-only basis (canon C56); without it all share cfg/meta."""
    out = {}

    def launch(r):
        c, m = per_id[r] if per_id else (cfg, meta)
        out[r] = supervise([sys.executable, "-u", str(SRC_FILE), "--worker", r, json.dumps(c), json.dumps(m), *LEARNER_ARGS],
                           DIR / r / "health2.jsonl", r, WORKER_TIMEOUT_S, WORKER_MEM_MB, WORKER_HEARTBEAT_S, stdout="inherit")
    threads = [threading.Thread(target=launch, args=(r,)) for r in ids]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return out


def classify_round(ids, cfg, meta, rnd=None, root=None, stale_check=None, per_id=None):
    """Phase 24: classify every worker; anything not OK is EXCLUDED (never replaced, reused or averaged)."""
    root = Path(root) if root else DIR
    stale_check = stale_check or (lambda row: provenance.stale({k: row.get(k) for k in ("code_hash", "code_files", "code_mixed")}))
    workers = {}
    for r in ids:
        want = expected_config(*per_id[r]) if per_id else expected_config(cfg, meta)
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
    publish_bank_exclusions(st)
    lineage = lineage_from_state(st)
    rnd = len(st["windows"]) // PAR + 1
    while len(st["windows"]) < MAXW:
        ids = [f"w{rnd:02d}{x}" for x in "abc"[:PAR]]
        for r in ids:
            livesim.SealedYear(r)                                # sealed one at a time: no duplicate draws
        plan = plan_round(ids, lineage)                          # C56: each window's basis is trained on windows that ended before it began
        per_id = {r: (p["cfg"], p["meta"]) for r, p in plan.items()}
        print(f"\n=== round {rnd} ({st['phase']} phase): {PAR} sealed 12-month windows, latest basis v{st['version']}; "
              f"{sum(p['untrained'] for p in plan.values())} of {PAR} windows play the UNTRAINED neutral basis "
              f"(basis versions used: {[p['version'] for p in plan.values()]}) ===", flush=True)
        t0 = time.perf_counter()
        for r in ids:
            (DIR / r).mkdir(exist_ok=True)
            reset_worker_files(DIR / r)
        run_workers(ids, st["cfg"], st["meta"], per_id=per_id)
        print(f"  round wall time {time.perf_counter() - t0:.0f}s", flush=True)
        report = classify_round(ids, st["cfg"], st["meta"], rnd, per_id=per_id)
        for attempt in range(2):                                 # a worker that outlived a code edit is rerun, not judged
            stale_ids = [e["worker"] for e in report["excluded"] if e["status"] == health.STALE]
            if not stale_ids:
                break
            for r in stale_ids:
                print(f"  [{r}] STALE CODE - rerunning the window under current code", flush=True)
                reset_worker_files(DIR / r)
            run_workers(stale_ids, st["cfg"], st["meta"], per_id=per_id)
            report = classify_round(ids, st["cfg"], st["meta"], rnd, per_id=per_id)
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
                subprocess.run([sys.executable, "-u", __file__, "--worker", r, json.dumps(res["prior_cfg"]), json.dumps(res["meta"]),
                                *LEARNER_ARGS])
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
                                  "basis_version": plan[r]["version"], "untrained_basis": plan[r]["untrained"], "thin": w["thin"],
                                  "n_names": w["n_names"], "phase": st["phase"]})
        if not gate_ok:
            save(); print("  ANTI-CHEAT GATE FAILED - stopping before any retraining", flush=True); break
        # ---- train the training basis for the NEXT round (F06, canon C56) ----
        # The next round is sealed now, so the first real day the new basis may be played on is known (referee side). The
        # archive is split there: only windows that ENDED before it are trained on, the search starts from the newest basis
        # that is itself past-only for that day (never the latest global one), and train_basis refuses any late window.
        loaded = [load_window(a) for a in archive_dirs(DIR)]
        wins = [w for w in loaded if not w["thin"]]               # THIN windows (survivor universes) never train a basis
        thin_ids = {w["id"] for w in loaded if w["thin"]}
        nxt = [f"w{rnd + 1:02d}{x}" for x in "abc"[:PAR]]
        for r in nxt:
            livesim.SealedYear(r)
        used_from = min(real_span(r)[0] for r in nxt)
        wins, late = LA.split_training_windows(wins, used_from, real_span)
        base = lineage.basis_for(used_from)
        base_cfg, base_meta = (base["cfg"], base["meta"]) if base else (dict(NEUTRAL_CFG), dict(NEUTRAL_META))
        print(f"  basis training set: {len(wins)} windows ({len(thin_ids)} THIN excluded: {sorted(thin_ids)}; {len(late)} not ended "
              f"before the next round begins, refused)", flush=True)
        t1 = time.perf_counter()
        res = train_basis(wins, base_cfg, base_meta, seed=1000 * st["version"] + rnd, used_from=used_from)
        inc, win_ = res.incumbent.confirm, res.winner.confirm
        print(f"  training basis: {res.n_evals} replays; {N_CAND} candidates screened on {res.n_screen} of {res.n_windows} "
              f"windows, top 3 confirmed on all ({time.perf_counter() - t1:.0f}s)", flush=True)
        if res.adopted:
            st["version"] += 1
            st["cfg"], st["meta"] = res.cfg, res.meta
            register_basis(st, lineage, st["version"], res.cfg, res.meta, wins)   # every trained-on window, with its real dates
            st["lineage"][-1]["used_from"] = str(pd.Timestamp(used_from).date())     # the first real day it may be played (F06)
            print(f"  NEW BASIS v{st['version']}: weeks in band {inc.t1:.0%} -> {win_.t1:.0%}, risk {inc.risk:+.3f} -> {win_.risk:+.3f} "
                  f"({res.reason})", flush=True)
            print(f"     defaults {res.cfg}", flush=True)
            print(f"     adaptation {res.meta}", flush=True)
        else:
            print(f"  kept basis v{st['version']} (weeks in band {inc.t1:.0%}, risk {inc.risk:+.3f}): {res.reason}", flush=True)
        headline = [w for w in st["windows"] if not w.get("thin") and not w.get("legacy")]      # THIN and pre-lineage windows reported apart
        fresh = [w["mean_week"] for w in headline]
        thin_rows = [w["mean_week"] for w in st["windows"] if w.get("thin") and not w.get("legacy")]
        summ = round_summary(rnd, plan, done, thin_ids)
        st["rounds"].append(summ)
        if st["phase"] == "volatility" and fresh and np.mean(fresh[-6:]) >= 0.01:
            st["phase"] = "direction"
        # the basis decision above is final: only now may the true periods be revealed (RevealGate, Phase 21)
        for w, y in zip(st["windows"][-len(done):], reveal_round([w["run_id"] for w in st["windows"][-len(done):]], True).values()):
            w["revealed"] = y
            w["real_start"] = str(pd.Timestamp(plan[w["run_id"]]["real_start"]).date())   # lets the lineage audit resolve every trained play
        print("  revealed:", {w["run_id"]: w["revealed"] for w in st["windows"][-len(done):]}, flush=True)
        print(f"  round {rnd}: {summ['n_untrained']} of {summ['n_played']} windows used an UNTRAINED basis; THIN windows this round: {summ['thin']}", flush=True)
        print(f"  running average over {len(fresh)} headline windows (THIN and legacy excluded): "
              f"{(np.mean(fresh) if fresh else float('nan')):+.2%}/week (target +7.00%); THIN windows apart: "
              f"{len(thin_rows)} at {(np.mean(thin_rows) if thin_rows else float('nan')):+.2%}/week", flush=True)
        seed = 1000 * st["version"] + rnd
        log_experiment({**res.to_record(), "event": "loop2_basis_search", "round": rnd},
                       cfg={"cfg": st["cfg"], "meta": st["meta"]}, seed=seed, outcome="adopt" if res.adopted else "reject",
                       reason=str(res.reason), window_ids=[w["run_id"] for w in st["windows"]],
                       train_range=f"Test windows that ended before {pd.Timestamp(used_from).date()} (next round's first real day)",
                       validation_range="basis_search held-out windows",
                       test_range="next round's fresh sealed windows")
        rw = st["windows"][-len(done):]
        metrics = {w["run_id"]: {k: w.get(k) for k in ("mean_week", "in_band", "sd_week", "max_dd", "year_return")} for w in rw}
        # Phase 0.2: the round record carries every field; Phase 0.3: each round is bundled as a checkpoint
        log_experiment({"event": "loop2_round", "round": rnd, "basis": st["version"], "phase": st["phase"],
                        "fresh_avg": float(np.mean(fresh)) if fresh else None, "n_untrained_basis": summ["n_untrained"],
                        "untrained_windows": summ["untrained_basis"], "thin_windows": summ["thin"]},
                       cfg={"cfg": st["cfg"], "meta": st["meta"]}, seed=seed,
                       window_ids=[w["run_id"] for w in rw], metrics=metrics,
                       gates={w["run_id"]: "passed re-tester, future-scramble, fill audit" for w in rw},
                       outcome="continue_testing", reason="blind Test round (measurement)",
                       train_range="history before each sealed year", validation_range="pre-season nested split",
                       test_range="the sealed 12-month windows")
        try:
            from engine.checkpoint import checkpoint_run
            checkpoint_run(f"loop2_round{rnd:03d}", {"cfg": st["cfg"], "meta": st["meta"], "basis": st["version"]},
                           metrics, {"basis_search_seed": seed, "windows": [w["run_id"] for w in rw]},
                           {"round": rnd, "adopted": bool(res.adopted), "reason": str(res.reason),
                            "fresh_avg": float(np.mean(fresh)) if fresh else None, "n_untrained": summ["n_untrained"]},
                           logs={"round.txt": "\n".join(f"{w['run_id']}: {metrics[w['run_id']]}" for w in rw)})
        except Exception as e:                                    # a checkpoint failure is reported, never fatal
            print(f"  checkpoint failed: {e}", flush=True)
        save()
        if any(w["mean_week"] >= 0.07 and not w.get("thin") for w in st["windows"][-len(done):]):
            print("TARGET REACHED", flush=True); break
        rnd += 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4]))
        sys.exit(0)
    main()
