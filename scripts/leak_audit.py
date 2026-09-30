"""Real-data runner for the future-leak audit (B23; canon C56 / C55; Bible phases 1, 2, 21, 26).

usage: python scripts/leak_audit.py [--parts static,survivorship,adjusted,metadata,fingerprint,causality,assemble]
                                    [--net] [--seed 20260928]

Each part measures its channels on the real caches and writes state/research/leak_audit/parts/<part>.json (a checkpoint:
a killed run costs one part). `assemble` reads every part, builds the channel records and writes report.md + summary.json
with provenance and a registry record. Nothing here touches state/livesim or writes to data/cache. `--net` allows the
audit ITSELF (never a blind worker) to pull split history from Yahoo for a seeded ticker sample. Run detached:
  Start-Process .venv/Scripts/python -ArgumentList '-u','scripts/leak_audit.py','--net' -RedirectStandardOutput data/leak_audit.log
RAM: every part that loads panels first waits until >= 2.5 GB is free (polls every 60 s, gives up after 20 min)."""
import argparse
import ast
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from engine import config as K, leak_audit as L, provenance

OUT = K.STATE / "research" / "leak_audit"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
CACHE = K.CACHE
SEED = 20260928
ORDER = ["static", "runtime", "defaults", "survivorship", "adjusted", "metadata", "fingerprint", "causality", "lookup", "assemble"]


def wait_for_ram(need_gb=2.5, patience_s=1200):
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available / 1e9 < need_gb:
        if time.time() - t0 > patience_s:
            raise SystemExit(f"less than {need_gb} GB free for {patience_s}s - giving up (reported, not forced)")
        print(f"  waiting for RAM ({psutil.virtual_memory().available / 1e9:.1f} GB free)", flush=True)
        time.sleep(60)


def save_part(name, obj):
    (PARTS / f"{name}.json").write_text(json.dumps(L._jsonable(obj), indent=1, default=str))
    print(f"[{name}] saved", flush=True)


def load_part(name):
    p = PARTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stitched(field, since=None):
    """One field of the stitched price history exactly as engine.replay._all_prices builds it (pre-2000 file up to 1999,
    then the 2000+ file), float32, without loading the other fields."""
    fname = field.lower()
    new = pd.read_parquet(CACHE / f"stocks_{fname}.parquet")
    try:
        old = pd.read_parquet(CACHE / f"stocks_pre2000_{fname}.parquet")
        out = pd.concat([old, new.loc["2000-01-01":]]).sort_index()
        del old
    except FileNotFoundError:
        out = new
    del new
    out = out.loc[~out.index.duplicated(keep="last")].astype("float32")
    return out.loc[since:] if since else out


def market_close():
    """SPY (S&P 500 index scaled before 1993) and VIX as replay._all_prices splices them - the exposure the feed serves."""
    from engine import data
    mk = data.load("market")["Close"]
    ix = data.load("index_hist")["Close"]
    close = mk.reindex(ix.index.union(mk.index))
    g = ix["^GSPC"].reindex(close.index)
    spy = close["SPY"]
    f = spy.first_valid_index()
    close["SPY"] = spy.where(spy.notna(), g * (spy.loc[f] / g.loc[f]))
    close["^VIX"] = close["^VIX"].where(close["^VIX"].notna(), ix["^VIX"].reindex(close.index))
    return close


# =====================================================================================================================
def part_static(args):
    """Reachability, network, learned-state lineage, text/news, labels, defaults contamination, memory bank."""
    out = {}
    eng = K.ROOT / "engine"
    entries = [eng / "livesim.py", eng / "adaptive.py", eng / "memory.py", K.ROOT / "scripts" / "livesim_loop2.py"]
    cl = L.import_closure(entries)
    out["closure_modules"] = sorted(cl)
    out["not_reachable_from_blind_path"] = sorted(p.stem for p in eng.glob("*.py") if p.stem not in cl and p.stem != "__init__")
    out["research_only_modules_reachable"] = sorted({"pattern_bank", "lessons", "analogs", "patterns", "trust_store", "trust", "analog_weighting",
                                                     "pattern_movers", "pattern_lifecycle", "pattern_identity", "pattern_stats"} & set(cl))
    net = {}
    for m in sorted(cl):
        p = (eng / f"{m}.py") if (eng / f"{m}.py").exists() else K.ROOT / "scripts" / f"{m}.py"
        mk = L.network_markers(p)
        if mk:
            net[m] = mk
    out["network_markers"] = net
    reads = L.trader_side_reads(cl)
    out["trader_side_reads"] = reads
    # loop2: does the basis search get an as_of? (channel 4)
    src = (K.ROOT / "scripts" / "livesim_loop2.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "train_basis":
            calls.append({"line": n.lineno, "kwargs": [k.arg for k in n.keywords], "passes_as_of": any(k.arg == "as_of" for k in n.keywords)})
    out["loop2_train_basis_calls"] = calls
    out["loop2_trains_on_all_archived_windows"] = "archive_dirs(DIR)" in src and "split_training_windows" not in src
    out["loop2_training_gated_at_first_use"] = "split_training_windows" in src and "used_from=used_from" in src   # F06 (verdict reads the AST)
    out["loop2_hands_worker_the_global_basis"] = 'run_workers(ids, st["cfg"], st["meta"])' in src
    # simulate the loop's draws: how much of a played window's basis training set is its own future or overlap
    sims = []
    for seed in range(8):
        r = L.basis_future_share(L.simulate_window_draws(20, par=3, seed=SEED + seed))
        sims.append(r)
    out["basis_future_share_by_seed"] = sims
    out["basis_future_share_mean"] = float(np.mean([s["mean_future_share"] for s in sims]))
    out["basis_windows_touched_mean"] = float(np.mean([s["share_of_windows_touched"] for s in sims]))
    # cost of the fix: how many played windows would fall back to the untrained defaults under BasisLineage
    fb = []
    for seed in range(8):
        rounds = L.simulate_window_draws(20, par=3, seed=SEED + seed)
        lin, seen, played = L.BasisLineage(), [], []
        T = lambda w: L.TrainedOn(w["id"], w["start"], w["end"])
        for r, rnd in enumerate(rounds):
            for w in rnd:
                played.append(lin.basis_for(w["start"]) is None)
            seen += rnd
            lin.register(r + 1, {}, {}, [T(w) for w in seen])
        fb.append(float(np.mean(played)))
    out["lineage_fallback_to_defaults_share"] = float(np.mean(fb))
    # defaults derived from the sensitivity study over these calendar years
    try:
        sens = json.loads((K.STATE / "research" / "sensitivity.json").read_text())
        starts = pd.date_range("1965-01-01", "2025-09-01", freq="MS")
        out["sensitivity_tuned_years"] = sorted(set(sens["years"]))
        out["defaults_contamination"] = L.defaults_contamination(sens["years"], starts)
    except FileNotFoundError:
        out["defaults_contamination"] = None
    # text / news / speeches: every table the blind feed serves
    ev = pd.read_parquet(CACHE / "events.parquet")
    ins = pd.read_parquet(CACHE / "insider.parquet")
    sic = pd.read_parquet(CACHE / "sic.parquet")
    out["free_text_columns"] = {"events": L.free_text_columns(ev), "insider": L.free_text_columns(ins),
                                "sic_served_to_trader(sic)": L.free_text_columns(sic[["ticker", "sic"]])}
    out["events_columns"], out["insider_columns"] = list(ev.columns), list(ins.columns)
    out["events_kinds"] = ev["kind"].value_counts().to_dict()
    # label embargo vs the label horizon (5 sessions + next-open entry)
    from engine import model
    out["model_embargo_sessions"], out["label_reach_sessions"] = int(model.EMBARGO), 6
    # cost bps by era: what the trader is told about its own costs
    out["cost_bps_by_start_year"] = {"<1997": 40, "1997-2000": 20, ">=2001": 10}
    starts = pd.date_range("1965-01-01", "2025-09-01", freq="MS")
    cls = pd.Series(pd.cut(starts.year, [0, 1996, 2000, 3000], labels=["40bps", "20bps", "10bps"])).value_counts(normalize=True)
    out["cost_class_share_of_start_months"] = cls.to_dict()
    out["warmup_shorter_than_6y_for_starts_before"] = "1968-01-01"
    # memory bank filter (real_end < start) proven on a planted late row
    from engine import blind_gates as BG
    bank = pd.DataFrame({"real_end": ["2001-12-31", "2019-06-30"], "arm": 1, "ctx": 0, "outcome": 0.0})
    out["memory_bank_filter_catches_planted_late_row"] = bool(BG.check_memory_bank_causality(bank, "2010-01-01"))
    save_part("static", out)


# =====================================================================================================================
def part_defaults(args):
    """S18: what the DEFAULT blind path actually is, read from the source (AST) - not from a comment, not from an earlier audit
    - plus the proofs that push a planted leak through each fix, plus the lineage check over the loop's own state file
    (state/livesim/loop2.json: the referee's ledger; no sealed window is opened). No caches, no RAM."""
    facts = L.default_path_facts()
    proofs = L.run_proofs()
    sp = K.STATE / "livesim" / "loop2.json"
    state = json.loads(sp.read_text()) if sp.exists() else None
    check = L.lineage_state_check(state, facts.get("loop2_cfg_space"), facts.get("meta_default_literal"))
    out = {"facts": facts, "proofs": proofs, "state_check": check, "state_file": str(sp.relative_to(K.ROOT)) if sp.exists() else None,
           "state_file_mtime": pd.Timestamp(sp.stat().st_mtime, unit="s").isoformat() if sp.exists() else None}
    if check:
        print(f"  [defaults] plays: {len(check['untrained'])} untrained, {len(check['trained'])} trained, {len(check['legacy'])} legacy; "
              f"violations {len(check['violations'])}, unresolved {len(check['unresolved'])}", flush=True)
    save_part("defaults", out)


# =====================================================================================================================
def synthetic_data(start, n=60, seed=3):
    """A small synthetic world around a window (business days, `n` names) - enough for a full adaptive blind run."""
    start = pd.Timestamp(start)
    idx = pd.bdate_range(start - pd.DateOffset(years=6), start + pd.DateOffset(months=12) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    tick = sorted(f"AB{chr(65 + i % 26)}{i}" for i in range(n))
    ret = rng.normal(0.0004, 0.02, (len(idx), n)) + 0.004 * rng.standard_t(3, (len(idx), n)).clip(-6, 6)
    close = pd.DataFrame(40 * np.exp(ret.cumsum(0)), index=idx, columns=tick)
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, close.shape))
    vol = pd.DataFrame(rng.integers(3e5, 3e6, close.shape).astype(float), index=idx, columns=tick)
    stocks = {"Close": close, "Open": opn, "High": close * 1.015, "Low": close * 0.985, "Volume": vol}
    mk = pd.DataFrame({"SPY": 100 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum()), "^VIX": 18 + rng.normal(0, 2, len(idx)).cumsum() * 0.1}, index=idx)
    market = {f: mk for f in ("Close", "Open", "High", "Low", "Volume")}
    at = pd.DatetimeIndex(idx[::7], tz="UTC") + pd.Timedelta(hours=13)
    kinds = ["EARN", "OFFERING", "AGREEMENT", "PERIODIC"]
    ev = pd.DataFrame({"ticker": [tick[i % n] for i in range(len(at))], "accepted": at, "kind": [kinds[i % 4] for i in range(len(at))], "form": "8-K"})
    fd = idx[::11]
    ins = pd.DataFrame({"symbol": [tick[i % n] for i in range(len(fd))], "filed": fd, "tdate": fd - pd.Timedelta(days=2), "owner_cik": "1",
                        "value": 50_000.0, "relation": "officer", "title": "ceo"})
    return stocks, market, ev, ins, pd.DataFrame({"ticker": tick, "sic": 3570})


def part_runtime(args):
    """A whole adaptive blind window on the hardened feed, under the network guard and a file-access recorder: which files
    were actually opened, whether anything tried the network, how long it took."""
    import tempfile
    from engine import livesim, blind_gates as BG
    tmp = Path(tempfile.mkdtemp(prefix="leakaudit_rt_"))
    old = livesim.DIR
    livesim.DIR = tmp
    cfg = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0, "pick": "hivol", "pool_q": 0.7,
           "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2, "trend_filter": None, "trend_gross": 0.0}
    try:
        rec_seal = BG.seal_window([], 99, "2026-01-01", tag="rt")
        (tmp / "sealed_rt.json").write_text(json.dumps(rec_seal))
        data = synthetic_data(rec_seal["start"])
        guard = L.NetworkGuard()
        t0 = time.time()
        with L.FileAccessRecorder() as rec, guard:
            feed = L.hardened_feed_class()(livesim.SealedYear("rt"), data=data)
            feed.precompute_features()
            livesim.parity_test(feed, seed=1)
            trader = livesim.BlindTrader(feed, cfg, adaptive=True)
            trader.train()
            livesim.drive(feed, trader.on_tick)
        files = rec.project_files()
        out = {"wall_s": time.time() - t0, "sessions": int(len(feed.clock.sessions)), "blocked_network_attempts": guard.blocked,
               "blind_gate_fails": [str(f) for f in feed.audit() if f.severity == "fail"],
               "project_code_files_opened": files["code"], "data_cache_files_opened": files["data_cache"],
               "state_files_opened": [f for f in files["state"] if not f.startswith("state/livesim/")], "other_files_opened": files["other"],
               "livesim_state_files_opened": [f for f in files["state"] if f.startswith("state/livesim/")],
               "weeks_traded": int(len(trader.session.weeks)), "year_return": float(trader.session.result()["year_return"])}
        print(f"  [runtime] {out['sessions']} sessions in {out['wall_s']:.0f}s; blocked network attempts: {len(guard.blocked)}; cache files: {out['data_cache_files_opened']}", flush=True)
    finally:
        livesim.DIR = old
    save_part("runtime", out)


# =====================================================================================================================
def part_survivorship(args):
    wait_for_ram()
    out = {}
    C = stitched("Close")
    prof = L.attrition_profile(C)
    out["panel"] = {"rows": int(C.shape[0]), "cols": int(C.shape[1]), "first": str(C.index[0].date()), "last": str(C.index[-1].date())}
    out["alive_by_year"] = {int(y): int(v) for y, v in prof["n_alive"].items()}
    out["exits_by_year"] = {int(y): int(v) for y, v in prof["n_exit"].items() if v}
    last = C.apply(lambda s: s.last_valid_index())
    out["names_ending_before_panel_end"] = {"n": int((last < C.index[-21]).sum()), "of": int(C.shape[1])}
    out["reference_us_listed_firms_approx"] = {"1996": 7322, "2016": 4331,
                                               "note": "Doidge-Karolyi-Stulz 2017 figures as recalled, not re-verified; order of magnitude only"}
    out["panel_vs_reference"] = {"alive_1996": int(prof.loc[1996, "n_alive"]), "alive_2016": int(prof.loc[2016, "n_alive"])}
    # hazard from the SEC terminal-event table
    ev = pd.read_parquet(CACHE / "delisted_events.parquet")
    ev = ev[pd.to_datetime(ev["delist_date"]).dt.year.between(2001, 2025)]
    haz = L.delisting_hazard(ev, prof["n_alive"].loc[2001:2025])
    out["registry_hazard_by_year"] = {int(y): float(v) for y, v in haz["rate"].items()}
    reg_upper = float(haz["rate"].mean())
    out["hazard_bands"] = {"registry_upper": reg_upper, "literature_low": L.LITERATURE_HAZARD[0], "literature_high": L.LITERATURE_HAZARD[1]}
    reg = pd.DataFrame(json.loads((CACHE / "delisted_registry.json").read_text()))
    last_close = C.iloc[-1]
    conc = L.concentration_from_registry(reg, last_close)
    out["concentration"] = conc
    out["registry_rows_with_last_close"] = int(pd.to_numeric(reg["last_close"], errors="coerce").notna().sum())
    dp = pd.read_parquet(CACHE / "delisted_prices.parquet")
    out["dead_names_with_recovered_prices"] = int(dp.iloc[:, 0].nunique()) if len(dp) else 0
    # analytic haircut
    hair = {}
    for name, h in (("literature_low", 0.03), ("literature_mid", 0.045), ("literature_high", 0.06), ("registry_upper", min(0.5, reg_upper))):
        for c in (1.0, 3.0):
            hair[f"{name}|conc={c:g}"] = L.survivorship_haircut(h, -0.45, c)
    out["analytic_haircut"] = hair
    # simulation: a high-volatility basket on the survivor panel vs the same panel plus injected dead names
    sl = C.loc["2011-01-01":]
    sl = sl.loc[:, sl.notna().sum() > 250]
    del C
    gc.collect()
    base = L.vol_basket_weekly(sl, top_frac=0.10, min_price=1.0)
    out["vol_basket"] = {"weeks": int(len(base)), "mean_weekly": float(base.mean()), "sd_weekly": float(base.std())}
    alive_y = sl.notna().groupby(sl.index.year).any().sum(axis=1)
    sims = {}
    for name, h in (("literature_low", 0.03), ("literature_mid", 0.045), ("literature_high", 0.06)):
        drags = []
        for rep in range(4):
            rng = np.random.default_rng(SEED + rep + int(h * 1000))
            dates = []
            for y, n in alive_y.items():
                yd = sl.index[(sl.index.year == y)][:-60] if y == sl.index.year.max() else sl.index[sl.index.year == y]
                if len(yd) > 5:
                    dates += list(rng.choice(yd, size=int(round(h * n)), replace=True))
            aug = L.inject_dead_names(sl, dates, seed=SEED + rep, terminal_mean=-0.45, terminal_sd=0.30, high_vol_bias=0.7)
            with_dead = L.vol_basket_weekly(aug, top_frac=0.10, min_price=1.0)
            drags.append(float(with_dead.mean() - base.mean()))
            del aug
            gc.collect()
            print(f"  [survivorship] {name} rep {rep}: drag {drags[-1] * 1e4:+.1f} bps/week", flush=True)
        sims[name] = {"drags_bps_per_week": [d * 1e4 for d in drags], "mean_bps_per_week": float(np.mean(drags) * 1e4),
                      "annual_hazard": h, "n_dead_injected_per_rep": int(sum(int(round(h * n)) for n in alive_y))}
    out["simulated_vol_basket_drag"] = sims
    starts = pd.date_range("1965-01-01", "2025-09-01", freq="MS")
    out["thin_universe"] = {str(n): L.thin_universe_share(prof["n_alive"], starts, min_names=n) for n in (200, 500, 1000)}
    out["assumptions"] = "dead names copy a donor path (70% from the top volatility third) then print one terminal return ~ N(-45%, 30%); an assumption model, not recovered history"
    save_part("survivorship", out)


# =====================================================================================================================
def part_adjusted(args):
    wait_for_ram()
    out = {}
    C = stitched("Close")
    drift = L.price_level_drift(C.iloc[::5])
    out["median_close_by_year"] = {int(y): float(v) for y, v in drift["median_close"].items()}
    out["share_below_3_by_year"] = {int(y): float(v) for y, v in drift["share_below_3"].items()}
    out["median_close_first5y_vs_last5y"] = float(drift["median_close"].iloc[:5].mean() / drift["median_close"].iloc[-5:].mean())
    rng = np.random.default_rng(SEED)
    out["price_filter_in_features.build"] = "relative mode: C.rank(axis=1,pct=True) >= 0.2 on ADJUSTED close (price-rank clause)"
    V = stitched("Volume")
    chg = {}
    for y in (1985, 2005, 2018):
        Cy, Vy = C.loc[f"{y - 1}-06-01":f"{y}-12-31"], V.loc[f"{y - 1}-06-01":f"{y}-12-31"].reindex(columns=C.columns)
        dv20 = (Cy * Vy).rolling(20, min_periods=15).median()
        old = (Cy.rank(axis=1, pct=True) >= 0.2) & (dv20.rank(axis=1, pct=True) >= 0.4) & Cy.notna()
        new = L.split_invariant_tradable(Cy, Vy, 0.4)
        o, n = old.loc[str(y)], new.loc[str(y)]
        chg[y] = {"tradable_cells_old_rule": int(o.sum().sum()), "tradable_cells_new_rule": int(n.sum().sum()),
                  "cells_that_change_status_share_of_old": float((o != n).sum().sum() / max(1, o.sum().sum())),
                  "names_ever_tradable_old": int(o.any().sum()), "names_ever_tradable_new": int(n.any().sum())}
    out["tradable_rule_swap_effect"] = chg
    del V
    gc.collect()
    if args.net:
        out["net_sample"] = net_split_sample(C, rng)
    else:
        out["net_sample"] = load_part("adjusted_net") or "not run (--net)"
    del C
    gc.collect()
    save_part("adjusted", out)


def net_split_sample(C, rng, n=150):
    """Pull Yahoo's split-only-adjusted Close, dividends and splits for a seeded sample; rebuild as-traded prices and measure
    where the level rules decide differently on the adjusted panel. Cached: a rerun reuses the raw pulls."""
    import yfinance as yf
    cache = OUT / "net"
    cache.mkdir(exist_ok=True)
    pool = C.loc["2005-01-01":].columns[C.loc["2005-01-01":].notna().sum() > 3000]
    tick = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    adj, traded, info = {}, {}, {}
    for i, t in enumerate(tick):
        f = cache / f"{t}.json"
        if f.exists():
            raw = json.loads(f.read_text())
        else:
            try:
                h = yf.Ticker(t).history(period="max", auto_adjust=False, actions=True)
                raw = {"close": {str(d.date()): float(v) for d, v in h["Close"].dropna().items()},
                       "adj": {str(d.date()): float(v) for d, v in h["Adj Close"].dropna().items()},
                       "splits": {str(d.date()): float(v) for d, v in h["Stock Splits"][h["Stock Splits"] > 0].items()}}
                f.write_text(json.dumps(raw))
            except Exception as e:                                       # noqa: BLE001 - a dead ticker is skipped, not fatal
                info[t] = f"failed: {type(e).__name__}"
                continue
            time.sleep(0.4)
        if not raw["close"]:
            continue
        cs = pd.Series(raw["close"]); cs.index = pd.to_datetime(cs.index)
        sp = pd.Series(raw["splits"], dtype=float); sp.index = pd.to_datetime(sp.index)
        tr = cs * L.cum_future_split_factor(sp, cs.index)
        traded[t] = tr
        adj[t] = C[t].dropna()
        info[t] = {"n_splits": int(len(sp)), "max_cum_factor": float(L.cum_future_split_factor(sp, cs.index).max())}
        if (i + 1) % 25 == 0:
            print(f"  [adjusted] {i + 1}/{len(tick)}", flush=True)
    A, T = pd.DataFrame(adj), pd.DataFrame(traded)
    T = T.reindex(A.index)
    A = A.loc["2005-01-01":]
    T = T.loc["2005-01-01":]
    res = L.level_rule_disagreement(A, T, floor=3.0, rank_q=0.2)
    fac = pd.Series({t: v["max_cum_factor"] for t, v in info.items() if isinstance(v, dict)})
    by_year = {}
    for y in (2005, 2010, 2015, 2020, 2025):
        a, t = A[A.index.year == y], T[T.index.year == y]
        if len(a):
            by_year[y] = L.level_rule_disagreement(a, t)
    r05 = (T.loc["2005-01-03":"2005-01-31"].mean() / A.loc["2005-01-03":"2005-01-31"].mean()).dropna()
    return {"traded_over_adjusted_price_jan2005": {"median": float(r05.median()), "p10": float(r05.quantile(0.1)), "p90": float(r05.quantile(0.9)),
                                                   "share_ge_2": float((r05 >= 2).mean()), "n": int(len(r05))},
            "n_tickers": int(len(tick)), "n_used": int(A.shape[1]), "disagreement": res, "by_year": by_year,
            "share_with_any_split": float((fac > 1.0).mean()) if len(fac) else float("nan"),
            "median_max_cum_split_factor": float(fac.median()) if len(fac) else float("nan"),
            "share_with_cum_split_ge_4": float((fac >= 4.0).mean()) if len(fac) else float("nan"),
            "failed": {t: v for t, v in info.items() if not isinstance(v, dict)}}


# =====================================================================================================================
def part_metadata(args):
    out = {}
    u = pd.read_csv(CACHE / "universe.csv")
    sic = pd.read_parquet(CACHE / "sic.parquet")
    cols = pd.read_parquet(CACHE / "stocks_close.parquet").columns
    old_cols = pd.read_parquet(CACHE / "stocks_pre2000_close.parquet").columns
    allcols = cols.union(old_cols)
    C = pd.concat([pd.read_parquet(CACHE / "stocks_pre2000_close.parquet"), pd.read_parquet(CACHE / "stocks_close.parquet").loc["2000-01-01":]]).sort_index()
    first, last = C.apply(lambda s: s.first_valid_index()), C.apply(lambda s: s.last_valid_index())
    del C
    ce = L.crypto_exposure(first, last, u.set_index("ticker")["name"])
    out["crypto_matches_in_panel"] = ce.assign(first=ce["first"].astype(str), last=ce["last"].astype(str)).to_dict("records")
    out["n_universe"], out["n_panel_columns"] = int(len(u)), int(len(allcols))
    from engine.universe import CRYPTO_NAME, CRYPTO_TICKERS
    out["universe_names_matching_crypto_regex_still_in_universe"] = int(u["name"].str.contains(CRYPTO_NAME, na=False).sum())
    out["hard_listed_crypto_tickers_in_panel"] = sorted(set(CRYPTO_TICKERS) & set(allcols))
    out["hard_listed_crypto_tickers_traded_before_2018"] = sorted(t for t in CRYPTO_TICKERS if t in first.index and pd.notna(first[t]) and first[t] < pd.Timestamp("2018-01-01"))
    from engine import policy
    codes = pd.Index([f"S{n:04d}" for n in range(200)])
    out["not_crypto_on_blind_codes_filters_nothing"] = bool(policy.not_crypto(codes).all())
    out["blind_universe_note"] = ("the panel columns (universe.csv) still hold the crypto-named/hard-listed tickers above; the C11 crypto rule is applied only at pick time by "
                                  "policy.not_crypto on REAL tickers, so in blind runs (code names) it is inert and those names are tradable")
    out["sic"] = {"rows": int(len(sic)), "source": "SEC company_tickers / submissions, CURRENT classification", "n_distinct_2digit": int(sic["sic"].astype(str).str[:2].nunique()),
                  "n_distinct_4digit": int(sic["sic"].astype(str).nunique()), "n_distinct_division_1digit": int(sic["sic"].astype(str).str[:1].nunique())}
    out["sic_uses"] = "features.build groups ind_mom20/60 and rel_ind20/60 by 2-digit SIC; policy.sic_division (division) caps sectors; both from today's codes"
    out["universe_membership_is_todays_listing"] = {"exchange_listed_today": True, "n": int(len(u))}
    save_part("metadata", out)


# =====================================================================================================================
def build_daily():
    wait_for_ram(3.0)
    mk = market_close()
    C = stitched("Close")
    V = stitched("Volume")
    d1 = L.daily_fingerprint_series(C, None, None, None, V, mk)
    del V
    gc.collect()
    H, Lo = stitched("High"), stitched("Low")
    d2 = L.daily_fingerprint_series(C, None, H, Lo, None)
    del H, Lo
    gc.collect()
    O = stitched("Open")
    d3 = L.daily_fingerprint_series(C, O, None, None, None)
    del O
    gc.collect()
    reg = L.regime_series(C, mk)                         # the m_* context the model and the memory actually consume
    del C
    gc.collect()
    daily = d1.copy()
    daily["hl_equal"], daily["open_nan"] = d2["hl_equal"], d3["open_nan"]
    for c in reg.columns:
        daily[c] = reg[c]
    return daily


def part_fingerprint(args):
    cached = OUT / "daily_fingerprint.parquet"
    if args.reuse_daily and cached.exists():
        daily = pd.read_parquet(cached)
    else:
        daily = build_daily()
        daily.to_parquet(cached)
    out = {"daily_rows": int(len(daily)), "first": str(daily.index[0].date()), "last": str(daily.index[-1].date())}
    starts = pd.date_range("1965-01-01", "2025-09-01", freq="MS")
    for name, warm in (("exposed_6y_warmup_plus_window", 6), ("hidden_12_months_only", 0)):
        F = pd.DataFrame([L.window_features(daily, s, warm_years=warm) for s in starts], index=starts)
        groups = L.GROUPS
        probe = L.FingerprintProbe(seed=SEED, n_trees=300)
        res = probe.run(F, starts, groups)
        for g, r in res.items():
            r["verdict"] = L.fingerprint_verdict(r)
        out[name] = res
        # permutation control: shuffle the year labels of the training windows - must land at chance
        rng = np.random.default_rng(SEED)
        Fp = F.copy()
        perm = rng.permutation(len(Fp))
        Fp[:] = F.to_numpy()[perm]
        out[name + "__shuffled_control"] = {g: probe.score(Fp, starts, cols) for g, cols in groups.items()}
        out[name + "__cost_bps_only"] = probe.score(
            pd.DataFrame({"c": np.where(starts.year < 1997, 40.0, np.where(starts.year < 2001, 20.0, 10.0))}, index=starts), starts, ["c"])
        out[name + "__trader_input_single_feature"] = {c: probe.score(F, starts, [f"{c}_mean", f"{c}_sd"]) for c in L.TRADER_COLS}
        print(f"  [fingerprint] {name}: " + ", ".join(f"{g}={r.get('skill', float('nan')):.2f}" for g, r in res.items() if isinstance(r, dict) and "skill" in r), flush=True)
    # the regular-grid calendar scrub, measured on the same windows
    Fg = []
    for s in starts:
        d = daily.loc[s - pd.DateOffset(years=min(6, (s - pd.Timestamp("1962-01-01")).days // 365)): s + pd.DateOffset(months=12) - pd.Timedelta(days=1)]
        d = d[d["n_names"] > 0]
        Fg.append(L.calendar_features(L.regular_grid_index(len(d))) if len(d) > 30 else {k: np.nan for k in L.CAL_FEATS})
    Fg = pd.DataFrame(Fg, index=starts)
    out["regular_grid_calendar_probe"] = L.FingerprintProbe(seed=SEED, n_trees=300).score(Fg, starts, [c for c in L.CAL_FEATS])
    out["regular_grid_calendar_probe"]["verdict"] = L.fingerprint_verdict(out["regular_grid_calendar_probe"])
    save_part("fingerprint", out)


# =====================================================================================================================
def part_causality(args):
    """Feature future-invariance on a real sample and the feed exposure on real windows (plain vs hardened feed)."""
    wait_for_ram(4.0)
    from engine import features, livesim
    out = {}
    rng = np.random.default_rng(SEED)
    C = stitched("Close", since="2010-06-01")
    live = C.columns[C.notna().sum() > 700]
    tick = sorted(rng.choice(live, size=60, replace=False).tolist())
    stocks = {f: stitched(f, since="2010-06-01").loc[:"2013-06-30", tick] for f in ("Close", "Open", "High", "Low", "Volume")}
    del C
    mk = market_close()
    market = {"Close": mk.loc["2010-06-01":"2013-06-30", ["SPY", "^VIX"]]}
    for f in ("Open", "High", "Low", "Volume"):
        market[f] = market["Close"]
    ev = pd.read_parquet(CACHE / "events.parquet")
    ev = ev[ev["ticker"].isin(tick) & (ev["accepted"] >= pd.Timestamp("2010-01-01", tz="UTC")) & (ev["accepted"] <= pd.Timestamp("2013-07-01", tz="UTC"))]
    ins = pd.read_parquet(CACHE / "insider.parquet")
    ins = ins[ins["symbol"].isin(tick) & (ins["filed"] >= "2010-01-01") & (ins["filed"] <= "2013-07-01")]
    sic = pd.read_parquet(CACHE / "sic.parquet")
    sic = sic[sic["ticker"].isin(tick)]
    t0 = time.time()
    res = L.truncation_invariance(features.build, stocks, market, ev, ins, sic, ["2012-01-18", "2012-07-11", "2013-02-13"], window=15, start="2011-06-01")
    res["seconds"] = time.time() - t0
    res["n_tickers"], res["n_events"], res["n_insider"] = len(tick), int(len(ev)), int(len(ins))
    out["features_truncation_invariance_2012_sample"] = res
    print(f"  [causality] features clean={res['clean']} leaky={list(res['leaky_features'])[:5]}", flush=True)
    # feed exposure on real windows (a temporary seal in a temp directory; state/livesim is not touched)
    import tempfile
    from engine import blind_gates as BG
    exposure = {}
    Hard = L.hardened_feed_class()
    Cfull, Ofull, Hfull, Lfull, Vfull = (stitched(f) for f in ("Close", "Open", "High", "Low", "Volume"))
    evf = pd.read_parquet(CACHE / "events.parquet")
    insf = pd.read_parquet(CACHE / "insider.parquet")
    sicf = pd.read_parquet(CACHE / "sic.parquet")
    for start in ("1975-03-01", "1995-06-01", "2018-03-01"):
        tmp = Path(tempfile.mkdtemp(prefix="leakaudit_"))
        old_dir = livesim.DIR
        livesim.DIR = tmp
        try:
            rec = BG.seal_window([], 4242, "2026-01-01", tag="audit")
            rec["start"] = start
            rec["digest"] = BG.seal_digest(rec)
            (tmp / "sealed_audit.json").write_text(json.dumps(rec))
            data = ({"Close": Cfull, "Open": Ofull, "High": Hfull, "Low": Lfull, "Volume": Vfull}, {f: mk.rename(columns=str) for f in ("Close", "Open", "High", "Low", "Volume")}, evf, insf, sicf)
            rec2 = BG.seal_window([], 4243, "2026-01-02", tag="audit2")
            rec2["start"] = start
            rec2["digest"] = BG.seal_digest(rec2)
            (tmp / "sealed_audit2.json").write_text(json.dumps(rec2))
            row = {}
            for label, cls in (("plain", livesim.Feed), ("hardened", Hard)):
                feed = cls(livesim.SealedYear("audit"), data=data, enforce=True)
                feed.i = feed.sessions.get_loc(feed.first_live)
                row[label] = L.feed_exposure(feed)
                feed2 = cls(livesim.SealedYear("audit2"), data=data, enforce=True)
                row[label + "_rerun_linkability"] = L.rerun_linkability(feed, feed2)
                del feed, feed2
                gc.collect()
            exposure[start] = row
            print(f"  [causality] {start}: plain {row['plain']} | hardened {row['hardened']}", flush=True)
        finally:
            livesim.DIR = old_dir
    out["feed_exposure_real_windows"] = exposure
    save_part("causality", out)


# =====================================================================================================================
# F06 lookup part (channel 6 with the C64 curator path running): these probes import the curator and engine.research, so they
# live here on the audit's trusted side - engine/leak_audit.py is on the trader's import closure and must never name them.
# Synthetic and seeded: no cache, no sealed window, seconds. The verdict is engine.leak_audit.verdict_year_lookup.
# =====================================================================================================================
LOOKUP_START, LOOKUP_END = pd.Timestamp("2011-03-01"), pd.Timestamp("2012-02-29")


def lookup_states(days, seed):
    """m_* market state per real day: a seeded random walk (the curator's era similarity needs a state that moves)."""
    rng = np.random.default_rng(seed)
    cols = ("m_vix", "m_breadth", "m_dispersion", "m_spy_r5")
    walk = np.cumsum(rng.normal(0, 0.2, (len(days), len(cols))), axis=0)
    return {d: {c: float(v) for c, v in zip(cols, row)} for d, row in zip(days, walk)}


def lookup_item(name, feats=None):
    from engine.learning.trader_view import opaque_token
    return {"item_id": opaque_token("lookup-" + name), "kind": "pattern", "weight": 1.0,
            "features": dict(feats or {"vol20": 1.2, "r5": -0.4}), "lean": 0.6, "horizon": 5}


def curator_run(store, days, states, code, file_every=0, extra=()):
    """One replay of the window through a Curator on `store` (None = a fresh one): release day by day and, every `file_every`
    days, file an episode that operated 8 sessions earlier and matured one session ago (what the blind learner's filer does).
    `extra` = [(day index, name, operated, matured)] plants. Returns (release digests, [(day, item ids)], the store used).
    Note: Curator(store=<empty store>) builds a new store (an empty store is falsy), so the store actually used is returned."""
    from engine.learning.curator import Curator
    cur = Curator(store=store, code_hash=code)
    digests, released = [], []
    for i, d in enumerate(days):
        _, rel = cur.run_day(d, states[d])
        digests.append(rel.digest())
        released.append((d, [it.item_id for it in rel.items]))
        if file_every and i >= 8 and i % file_every == 0:
            op, mat = days[i - 8], days[i - 1]
            cur.file(lookup_item(f"ep{i}", {"vol20": round(0.1 * (i % 7), 3), "r5": round(0.05 * (i % 5) - 0.1, 3)}),
                     op, op.year, states[op], matured_at=mat, reliability=0.8)
        for when, name, op, mat in extra:
            if when == i:
                cur.file(lookup_item(name), op, pd.Timestamp(op).year, states[pd.Timestamp(op)], matured_at=mat, reliability=0.8)
    return digests, released, cur.store


def curator_lookup_probe(seed=0, end=None):
    """The C64 curator as a year-keyed store, driven over a synthetic 12-month window (real dates 2011-03..2012-02). Planted: a
    memory from before the window (control: must be released), memories filed under the replayed year that mature inside it
    (never released on or before their maturity) and one that matures after the window (never released). Then the SAME window
    is replayed twice on one shared store (run 1 files episodes as it goes; run 2 re-files the same ones): on every day of the
    rerun, the release must equal the release from a store truncated to the memories matured before that day (prefix
    invariance). Controls: a curator that ignores maturity must fail that check, and one extra memory planted by run 1 must
    change run 2's releases (the rerun does see its first run's past - that is live information, not a leak). Also: planted year/date content refused at filing and in a trader item; whether a hindsight-named feature key is
    refused (latent - no writer files such keys today)."""
    from engine.learning.curator import Curator
    from engine.learning import curator as CU
    from engine.learning import trader_view as TV
    from engine.learning.core import FirewallBreach
    end = pd.Timestamp(end) if end is not None else LOOKUP_END            # a shorter window keeps unit tests fast
    days = pd.bdate_range(LOOKUP_START, end)
    states = lookup_states(pd.bdate_range(LOOKUP_START - pd.Timedelta(days=200), LOOKUP_END + pd.Timedelta(days=200)), seed)
    code = "lookup-audit-code"
    pre = Curator(store_root=None, code_hash=code)
    plants = {"before_window": ("2010-11-01", "2010-11-08"), "replayed_year_early": ("2011-03-10", "2011-03-17"),
              "replayed_year_late": ("2011-09-01", "2011-09-08"), "after_window": ("2012-06-01", "2012-06-08")}
    for name, (op, mat) in plants.items():
        pre.file(lookup_item(name), op, int(op[:4]), states[pd.Timestamp(op)], matured_at=mat, reliability=0.9)
    key = {TV.opaque_token("lookup-" + n): n for n in plants}
    _, rel, used = curator_run(pre.store, days, states, code)
    mat_of = {m.key: pd.Timestamp(m.matured_at) for m in used.memories()}
    first, unmatured = {}, 0
    for d, ids in rel:
        for i in ids:
            unmatured += int(mat_of.get(i) is None or mat_of[i] >= d)
            if i in key:
                first.setdefault(key[i], d)
    run1, _, shared = curator_run(None, days, states, code, file_every=5)
    n_after_run1 = len(shared)
    run2, rel2, _ = curator_run(shared, days, states, code, file_every=5)
    _, _, planted = curator_run(None, days, states, code, file_every=5, extra=[(120, "run1_only", days[20], days[40])])
    run2p, _, _ = curator_run(planted, days, states, code, file_every=5)
    mat2 = {m.key: min(pd.Timestamp(x.matured_at) for x in shared.memories() if x.key == m.key) for m in shared.memories()}
    rerun_unmatured = sum(1 for d, ids in rel2 for i in ids if mat2.get(i) is None or mat2[i] >= d)
    # the rerun invariant: on every day, releasing from the store run 1 left behind equals releasing from a store truncated to the
    # memories that had matured before that day (curator.prefix_invariance); a curator that ignores maturity must fail it
    judge = Curator(store=shared, code_hash=code)
    prefix_bad = CU.prefix_invariance(judge, list(days), states.__getitem__)

    class IgnoresMaturity(Curator):
        def relevance(self, real_now, market_state):
            far = pd.Timestamp("2100-01-04")
            return Curator.relevance(self, far, market_state)
    broken_bad = CU.prefix_invariance(IgnoresMaturity(store=shared, code_hash=code), list(days[::10]), states.__getitem__)
    lag_days = sum(a != b for a, b in zip(run1, run2))
    probe = Curator(store_root=None, code_hash=code)
    refusals = {}
    for name, item in (("date_key", lookup_item("dk", {"filed_2011": 1.0})), ("year_value", lookup_item("yv", {"vol20": 2011.0}))):
        try:
            probe.file(item, "2011-03-10", 2011, states[pd.Timestamp("2011-03-10")], matured_at="2011-03-17")
            refusals[name] = False
        except FirewallBreach:
            refusals[name] = True
    try:
        TV.TraderMemoryItem.make(TV.opaque_token("yr"), "pattern", 1.0, {"year": 0.3}, 0.1, 5)
        refusals["trader_item_year_key"] = False
    except FirewallBreach:
        refusals["trader_item_year_key"] = True
    try:
        probe.file(lookup_item("hs", {"knowability_unpredictable": 0.25, "vol20": 1.0}), "2011-03-10", 2011,
                   states[pd.Timestamp("2011-03-10")], matured_at="2011-03-17")
        hindsight_refused = False
    except FirewallBreach:
        hindsight_refused = True
    hits = sum(sum(TV.leak_scan_text(json.dumps(ids)).values()) for _, ids in rel + rel2)
    return {"window_real": [str(LOOKUP_START.date()), str(end.date())], "days": len(days),
            "first_release_real_day": {k: str(v.date()) for k, v in first.items()},
            "unmatured_releases": unmatured + rerun_unmatured, "after_window_released": "after_window" in first,
            "past_control_released": "before_window" in first,
            "replayed_year_late_first_release_after_maturity": ("replayed_year_late" not in first) or first["replayed_year_late"] > pd.Timestamp("2011-09-08"),
            "rerun_prefix_mismatches": len(prefix_bad), "rerun_prefix_examples": prefix_bad[:3], "rerun_days_compared": len(days),
            "control_curator_ignoring_maturity_mismatches": len(broken_bad),
            "rerun_days_differing_from_first_run": int(lag_days), "rerun_days_with_items": sum(1 for _, ids in rel2 if ids),
            "rerun_difference_explained": "run 1 files an episode after that day's release (filing latency), a rerun sees it at the next release; still matured strictly before the day",
            "memories_filed_by_run1": n_after_run1, "memories_after_rerun": len(shared), "planted_extra_changes_releases": run2p != run1,
            "planted_date_content_refused": refusals, "planted_year_feature_refused": all(refusals.values()),
            "hindsight_feature_refused": hindsight_refused, "release_text_date_hits": int(hits)}


def research_lookup_probe():
    """Research records filed by year, through the one road to the trader (engine.research.firewall). The planted-leak suite must
    catch the same-year rerun and hindsight-label channels with its clean control released; and the same-year rule, isolated on
    one clean record learned inside the replayed year: refused DURING the replay of that year, admitted for a replay of a later
    year (so the refusal is the same-year rule), and admitted with no ReplayContext (the rule is only as good as its caller)."""
    from engine.research import firewall as FWL
    from engine.research.namespaces import ResearchStore
    suite = FWL.run_planted_suite()
    res = {str(c): {"caught": r.caught, "channels": list(r.channels)} for c, r in suite.results.items()}
    st = ResearchStore("lookup-probe")
    st.put(FWL.clean_object("sameyr", learned="2011-04-15"))
    fw = FWL.ResearchTraderFirewall(st, "lookup-probe")
    during = fw.inspect("sameyr", "2011-09-01", FWL.ReplayContext("w-probe", str(LOOKUP_START.date()), str(LOOKUP_END.date()), run_index=1))
    later = fw.inspect("sameyr", "2013-09-02", FWL.ReplayContext("w-later", "2013-03-01", "2014-02-28", run_index=0))
    bare = fw.inspect("sameyr", "2011-09-01", None)
    same = FWL.LeakChannel.SAME_YEAR_RERUN
    return {"suite_passed": suite.passed, "suite_void": suite.void, "suite_missed": suite.missed,
            "same_year_caught_by_suite": res.get(str(same), {}).get("caught"),
            "hindsight_caught": res.get(str(FWL.LeakChannel.HINDSIGHT_LABEL), {}).get("caught"),
            "research_only_caught": res.get(str(FWL.LeakChannel.RESEARCH_ONLY_KNOWLEDGE), {}).get("caught"),
            "same_year_refused_during_replay": (not during.admitted) and same in during.channels,
            "admitted_outside_replay": later.admitted, "admitted_without_replay_context": bare.admitted,
            "during_channels": sorted(str(c) for c in during.channels)}


def lookup_static(root=None):
    """Can the trader's import closure reach the curator or engine.research; who writes into the curator; does the research loop
    pass a ReplayContext; is the curator on the DEFAULT blind path (loop2 --learner default)?"""
    from engine.learning import trader_view as TV
    from engine.research import firewall as FWL
    closure = TV.trader_closure(root=root)
    bad = sorted(m for m in closure if m.startswith("engine.research") or m in ("engine.learning.curator", "engine.learning.test_path"))
    viol = FWL.trader_research_violations(root=root)
    facts = L.default_path_facts(root)
    return {"trader_closure_modules": len(closure), "trader_closure_reaches": bad, "leak_audit_on_trader_path": "engine.leak_audit" in closure, "trader_research_violations": [str(v) for v in viol],
            "trader_closure_clean": bool(closure) and not bad and not viol, "curator_writers": L._curator_writers(root),
            "research_loop_release_passes_replay": L._research_loop_release_passes_replay(root),
            "loop2_learner_default": facts.get("loop2_learner_default")}


def year_lookup_audit(seed=0, root=None, end=None):
    """The computed channel-6 lookup part (`end` shortens the synthetic window; the audit uses the full 12 months)."""
    return {"curator": curator_lookup_probe(seed, end), "research": research_lookup_probe(), "memory_bank": L.memory_bank_lookup_probe(),
            "static": lookup_static(root), "seed": seed}


def part_lookup(args):
    """F06 / C69 W-10: with the C64 curator path running, can anything LOOKED UP by year reach the trader in a disguised replay?"""
    out = year_lookup_audit(args.seed if getattr(args, "seed", None) is not None else 0)
    v = L.verdict_year_lookup(out)
    out["verdict"] = v.as_evidence()
    print(f"  [lookup] {v.status}: {list(v.reasons)[:3]}", flush=True)
    save_part("lookup", out)


# =====================================================================================================================
def assemble(args):
    P = {n: load_part(n) for n in ORDER[:-1]}
    miss = [n for n, v in P.items() if v is None]
    if miss:
        print(f"[assemble] parts missing: {miss} - their channels are reported as unmeasured", flush=True)
    A = L.Audit()
    S, V, J, M, F, Cz, R = (P[k] or {} for k in ("static", "survivorship", "adjusted", "metadata", "fingerprint", "causality", "runtime"))
    D = P["defaults"] or {}
    facts, proofs, sc = D.get("facts") or {}, D.get("proofs") or {}, D.get("state_check")
    VD = L.compute_verdicts({k: v for k, v in P.items() if v}, facts, proofs, sc)     # every status below comes from here
    chan = lambda key, name, status, evidence, **kw: L.Channel(key, name, status, {"computed_verdict": VD[key].as_evidence(), **evidence}, **kw)

    # 1 survivorship
    sim = V.get("simulated_vol_basket_drag", {})
    A.add(chan("1", "Survivorship (universe = today's listings; dead names absent)", VD["1"].status, {
        "panel": V.get("panel"), "names_ending_before_panel_end": V.get("names_ending_before_panel_end"),
        "alive_by_year_1965_1985_2005_2025": {y: V.get("alive_by_year", {}).get(str(y)) for y in (1965, 1985, 2005, 2025)},
        "panel_vs_reference_listed_firms": V.get("panel_vs_reference"), "reference": V.get("reference_us_listed_firms_approx"),
        "hazard_bands": V.get("hazard_bands"), "concentration_of_dead_in_cheap_names": V.get("concentration"),
        "thin_universe_windows(share of the 729 possible start months whose window has fewer than N names)": {k: {"share_below": v.get("share_below"), "first_clean_start": v.get("first_clean_start"), "by_decade": v.get("share_below_by_decade")} for k, v in (V.get("thin_universe") or {}).items()},
        "simulated_high_vol_basket_drag_bps_per_week(4 seeded reps each; noise about +-3)": {k: round(v["mean_bps_per_week"], 1) for k, v in sim.items()},
        "analytic_weekly_drag_bps_(hazard 4.5%, terminal loss -45%)": {c: round(1e4 * (V.get("analytic_haircut", {}).get(f"literature_mid|conc={c}") or {}).get("weekly_drag", float("nan")), 1) for c in ("1", "3")},
        "for_scale": "the sensitivity study's model edge is +0.32%/week (32 bps); the dominant survivorship effect is not this drag but the thin early universe (see thin_universe_windows)",
        "vol_basket_mean_weekly_on_survivors": (V.get("vol_basket") or {}).get("mean_weekly"), "assumptions": V.get("assumptions")},
        test="test_inject_dead_names_creates_terminal_prints_and_lowers_vol_basket; test_pit_universe_excludes_future_ipo_and_effective_delistings; test_attrition_profile_sees_dead_names_and_flags_survivor_panel",
        fix="engine.leak_audit: attrition_profile (detector), delisting_hazard + survivorship_haircut (size), inject_dead_names (correction, seeded, assumption-labelled), pit_universe (as-of membership)",
        hook="(a) blind_gates.FIRST_START / seal_window: windows starting before the first month that clears `thin_universe_share` (see evidence) hold a few dozen survivors - exclude them from learning and from any average, or start the draw later. (b) Real dead-name price history cannot be recovered offline (registry has 244 tickers, 77 priced). Until it is, every blind result is upward-biased by the drag above for volatility baskets: report results net of `survivorship_haircut(0.045, -0.45, conc)` or run windows on `inject_dead_names(...)` panels. In livesim.Feed.__init__ after `stocks, market = _all_prices()`: `stocks['Close'] = leak_audit.inject_dead_names(...)` (all five fields, same seed) if the owner wants the correction inside the run.",
        measured_on="real caches"))

    # 2 adjusted prices
    ns = J.get("net_sample")
    ev2 = {"median_close_first5y_vs_last5y": J.get("median_close_first5y_vs_last5y"),
           "median_close_by_year_sample": {y: J.get("median_close_by_year", {}).get(str(y)) for y in (1965, 1980, 2000, 2012, 2025)},
           "rule": J.get("price_filter_in_features.build"), "effect_of_swapping_to_split_invariant_tradable": J.get("tradable_rule_swap_effect")}
    if isinstance(ns, dict):
        ev2["as_traded_vs_adjusted_150_name_sample"] = {"traded_over_adjusted_price_jan2005": ns.get("traded_over_adjusted_price_jan2005"), "disagreement": ns.get("disagreement"), "share_with_any_split": ns.get("share_with_any_split"),
                                                       "share_with_cum_split_ge_4": ns.get("share_with_cum_split_ge_4"), "by_year": ns.get("by_year")}
    A.add(chan("2", "Split/dividend back-adjusted prices (levels encode later corporate actions)", VD["2"].status, ev2,
        test="test_level_rule_disagreement_catches_split_and_is_zero_when_unadjusted; test_split_invariant_tradable_ignores_a_later_split_but_price_rank_does_not; test_price_level_drift_detects_back_adjustment",
        fix="engine.leak_audit.split_invariant_tradable ranks 20-day dollar volume only (split-invariant); reconstruct_as_traded + level_rule_disagreement size the effect",
        hook=(("IMPLEMENTED on the default path: livesim.Feed defaults to tradable_rule='split_invariant' and features.build routes it to leak_audit.split_invariant_tradable "
               "(price-rank clause removed on the Test path; the Live rule keeps it). The non-relative branch is not on the blind path.") if VD["2"].status == L.FIXED else
              "engine/features.py build(): in relative mode replace `tradable = (C.rank(axis=1, pct=True) >= rel_q[0]) & (dv20.rank(...) >= rel_q[1]) & C.notna()` with `tradable = leak_audit.split_invariant_tradable(C, V, rel_q[1])`; the non-relative branch (MIN_PRICE=3) is not on the blind path. Every other feature is a ratio and split-invariant; log_dv is rank-normalised in model.normalise."),
        measured_on="real caches + Yahoo split sample" if isinstance(ns, dict) else "real caches"))

    # 3 metadata
    A.add(chan("3", "Today's metadata (names, SIC, crypto filter, exchange listing)", VD["3"].status, {
        "crypto_matches_in_panel": M.get("crypto_matches_in_panel"), "hard_listed_crypto_tickers_traded_before_2018": M.get("hard_listed_crypto_tickers_traded_before_2018"),
        "not_crypto_inert_on_blind_codes": M.get("not_crypto_on_blind_codes_filters_nothing"), "note": M.get("blind_universe_note"),
        "sic": M.get("sic"), "sic_uses": M.get("sic_uses"), "trader_sees_sic_codes_only_no_names": True},
        test="test_crypto_exposure_flags_name_and_list_matches_with_dates; test_crypto_filter_is_inert_on_disguised_codes; test_coarsen_sic_reduces_codes",
        fix="the crypto rule cannot leak in blind runs (it is inert on codes); the panel columns are today's exchange listings (channel 1); SIC can be coarsened with leak_audit.coarsen_sic",
        hook="C11 COMPLIANCE GAP (not a leak): 18 crypto tickers are in the blind panel and tradable because the C11 rule keys on real tickers; a point-in-time rule (crypto only from the date a firm became crypto) needs the owner's ruling. engine/livesim.py Feed.__init__: `sic = leak_audit.coarsen_sic(sic, 2)` (2-digit is what features use; 4-digit adds nothing) and pass coarsened codes; company names/exchange listing never reach the trader. Historical SIC is not available offline: the residual (a firm that changed industry) is unmeasured.",
        measured_on="real caches"))

    # 4 learned state
    dc = S.get("defaults_contamination") or {}
    lg = proofs.get("lineage_gate", {})
    scs = sc or {}
    A.add(chan("4", "Learned state from the future (basis cfg/meta trained on later or same windows; sensitivity-derived defaults)",
        VD["4"].status, {
        "train_basis_calls_in_main(source)": facts.get("loop2_train_basis_calls"),
        "register_basis_calls_in_main(source)": facts.get("loop2_register_basis_calls"),
        "training_still_uses_every_archived_window": S.get("loop2_trains_on_all_archived_windows"),
        "training_call_gated_at_first_use(F06, source AST)": {k: facts.get(k) for k in ("loop2_every_train_call_passes_used_from", "loop2_train_basis_refuses_late_windows",
                                                                                     "loop2_main_splits_training_set", "loop2_train_call_uses_global_basis")},
        "training_gate_proof(planted late window + seeded draw process)": proofs.get("training_gate"),
        "untrained_meta(F06)": {"data_free_NEUTRAL_META": facts.get("loop2_neutral_meta_from_neutral_default_meta"),
                                "plan_round_untrained_meta_is_neutral": facts.get("loop2_plan_round_untrained_meta_is_neutral"),
                                "loop_quarantines_tuned_meta_plays": facts.get("loop2_quarantines_tuned_meta_plays"),
                                "plays_on_tuned_META_DEFAULT_in_state_file": scs.get("tuned_meta_plays"),
                                "of_which_already_marked_by_the_loop": scs.get("tuned_meta_quarantined"),
                                "untrained_on_NEUTRAL_META": scs.get("untrained_on_neutral_meta"),
                                "versions_trained_past_their_first_use": scs.get("used_from_violations")},
        "play_calls_missing_a_per_window_basis": facts.get("loop2_play_calls_without_per_id"),
        "global_basis_reaching_a_play_call": facts.get("loop2_global_basis_reaches_a_play_call"),
        "real_state_file": {"file": (D.get("state_file")), "modified": D.get("state_file_mtime"), "windows": scs.get("n_windows"), "legacy_windows_(pre-lineage, excluded from headline)": scs.get("legacy"),
                            "untrained_plays": len(scs.get("untrained", [])), "trained_plays": len(scs.get("trained", [])), "violations": scs.get("violations"), "unresolved": scs.get("unresolved"),
                            "lineage_versions": scs.get("versions"), "untrained_cfg_mismatch": scs.get("untrained_cfg_mismatch"), "untrained_meta_mismatch": scs.get("untrained_meta_mismatch")},
        "control_if_the_loop_handed_every_window_the_newest_basis(seeded replay of the draw process)": lg.get("naive_share_of_plays_touched_by_future_training"),
        "gated_replay": {k: lg.get(k) for k in ("gated_plays_checked", "gated_violations", "gated_share_of_plays_on_untrained_neutral_basis")},
        "measured_before_the_fix_(static part, old loop design)": {"share_of_played_windows_whose_basis_touched_a_later_window": S.get("basis_windows_touched_mean"),
                                                                    "mean_future_share_of_training_set": S.get("basis_future_share_mean")},
        "starting_cfg": "NEUTRAL_CFG = neutral_default_cfg(CFG_SPACE): the middle grid value of every knob, no outcome used" if facts.get("loop2_neutral_cfg_from_neutral_default_cfg") else "NOT data-free",
        "starting_meta_META_DEFAULT_is_tuned_on_real_outcomes": facts.get("meta_default_data_tuned"),
        "sensitivity_defaults_calendar_years": S.get("sensitivity_tuned_years"), "share_of_possible_windows_in_sample_for_those_tuned_defaults": dc.get("contaminated_share"),
        "memory_bank": "filtered real_end < start in Feed.long_term_memory and re-checked by BG.check_memory_bank_causality",
        "memory_bank_planted_late_row_caught": S.get("memory_bank_filter_catches_planted_late_row"),
        "pattern_bank_lessons_analogs_reachable_from_blind_path": S.get("research_only_modules_reachable"),
        "same_window_rerun": "BasisLineage(allow_same_window=False) is the strict default: a rerun of a real window never plays a basis trained on its own first run (C56 over C54)"},
        test="test_channel4_verdict_flips_when_the_loop_passes_the_global_basis; test_channel4_verdict_flips_when_the_registered_set_differs_from_the_trained_set; test_lineage_state_check_flags_a_planted_future_trained_play; test_current_loop_design_trains_on_windows_from_the_future; test_violations_flags_planted_future_training; test_lineage_filter_removes_the_measured_leak; test_leak_closure.py: test_the_loops_train_basis_call_itself_refuses_a_planted_future_window, test_channel4_verdict_flips_when_any_link_of_the_training_gate_is_removed, test_training_gate_proof_holds_and_fails_when_the_gate_admits_everything, test_channel4_meta_residual_fixed_only_when_no_play_ran_on_the_tuned_meta",
        fix=("scripts/livesim_loop2.py: plan_round (BasisLineage.basis_for(real start), else NEUTRAL_CFG + NEUTRAL_META), register_basis (every trained-on window with real dates, "
             "plus used_from), run_workers/classify_round per_id; F06: main seals the NEXT round before training, splits the archive at its earliest real start "
             "(leak_audit.split_training_windows), starts the search from lineage.basis_for(that day), and train_basis(used_from=...) refuses any late window "
             "(leak_audit.refuse_late_training -> LateTrainingWindow); quarantine_tuned_meta marks old META_DEFAULT plays legacy='tuned_meta'"),
        hook=(("; ".join(VD["4"].reasons) + ". Owner decision still open: C54 (learn across reruns) vs C56 for reruns of the same window; and whether to drop the "
               "tuned-meta windows' rows from memory_bank.parquet (engine/livesim.py Feed.long_term_memory: exclude windows the loop marked legacy='tuned_meta') - "
               "that would let this channel reach FIXED.") if VD["4"].status == L.QUARANTINED else
              "Closed on the default path: the training call is gated at first use and every untrained play starts from data-free cfg and meta." if VD["4"].status == L.FIXED else
              ("; ".join(VD["4"].reasons) or "see the failed checks in computed_verdict")),
        measured_on="loop2 + livesim source (AST), the loop's own state file (referee side; no sealed window opened), seeded replay of its draw process"))

    # 5 macro
    A.add(chan("5", "Macro revisions (FRED current vintage vs first release)", VD["5"].status, {
        "analogs_or_parity_reachable_from_blind_path": bool({"analogs", "parity"} & set(S.get("closure_modules", []))),
        "consumers": "engine/analogs.py and engine/parity.py only (neither is on the blind path)",
        "revised_series_in_macro_parquet": sorted(L.MACRO_REVISED), "unrevised_series": sorted(L.MACRO_UNREVISED),
        "vintage_measurement": "not measurable offline: FRED's fredgraph.csv ignores vintage_date (verified); ALFRED needs an API key"},
        test="test_pit_macro_drops_revised_lags_monthly_and_never_shows_the_future; test_macro_revision_risk_and_unknown_series_treated_as_revised",
        fix="engine.leak_audit.pit_macro: drops revised series unless a first-release vintage is supplied, applies publication lags",
        hook="If analogs ever enter the blind path: engine/analogs.py fingerprints(): `M = leak_audit.pit_macro(M, as_of)` per as-of date (current code uses session lags on current-vintage UNRATE/CPIAUCSL/INDPRO/NFCI/STLFSI4: a residual revision leak, plus USREC with a 460-session lag)",
        measured_on="static reachability"))

    # 6 fingerprints
    LK = P.get("lookup") or {}
    exp = F.get("exposed_6y_warmup_plus_window", {})
    hid = F.get("hidden_12_months_only", {})
    def sk(d, g):
        return None if not d or g not in d or not isinstance(d[g], dict) else {"skill": d[g].get("skill"), "mae_years": d[g].get("mae_years"), "baseline_mae": d[g].get("baseline_mae_years"),
                                                                             "decade_acc": d[g].get("decade_acc"), "majority_decade_acc": d[g].get("majority_decade_acc"), "verdict": d[g].get("verdict")}
    GN6 = ("calendar", "levels_raw", "levels_after_hardening", "levels_scrubbed", "market_state", "trader_inputs", "scrubbed_all")
    open6 = [g for g in GN6 if exp.get(g, {}).get("verdict") == "identifiable"]
    trader_ok = exp.get("trader_inputs", {}).get("verdict") == "not identifiable" and hid.get("trader_inputs", {}).get("verdict") == "not identifiable"
    A.add(chan("6", "Year fingerprints (C55): could the exposed feed identify the real year?", VD["6"].status, {
        "exposed_window(6y warm-up + 12 months; upper bound, neighbouring windows share warm-up data)": {g: sk(exp, g) for g in GN6},
        "hidden_12_months_only(strict)": {g: sk(hid, g) for g in GN6},
        "shuffled_label_control(must be ~0)": {g: (F.get("exposed_6y_warmup_plus_window__shuffled_control", {}).get(g) or {}).get("skill") for g in GROUPS_NAMES},
        "trader_inputs_not_identifiable_in_either_view": trader_ok,
        "trader_input_single_feature_skill_exposed": {c: (v or {}).get("skill") for c, v in F.get("exposed_6y_warmup_plus_window__trader_input_single_feature", {}).items()},
        "trader_input_single_feature_skill_hidden_only": {c: (v or {}).get("skill") for c, v in F.get("hidden_12_months_only__trader_input_single_feature", {}).items()},
        "consumer_of_the_fingerprint": "the memory bank recalls lessons by market context (memory.CTX = vix, vix term, spy ma50/ma200, breadth, dispersion, spy r5); recall is by design and causal (real_end < start filter, BG.check_memory_bank_causality). A rerun of the same real window cannot recall its own lessons: its rows end on/after its start and are excluded",
        "cost_bps_only": F.get("exposed_6y_warmup_plus_window__cost_bps_only"), "regular_grid_calendar": F.get("regular_grid_calendar_probe"),
        "groups_identifiable_in_exposed_window": open6,
        "by_design": "dates shift by whole weeks so holidays/closures survive (check_calendar demands it); VIX/SPY-state is live information; universe size and price/volume levels are data",
        "cost_bps": "the trader is told its era-coded cost (40/20/10 bps): three classes, see static part",
        "warmup": "starts before 1968 get a shorter warm-up",
        "year_lookup_part(F06: can anything keyed by the year be looked up by the trader in a disguised replay?)": {
            "status": (VD["6"].checks or {}).get("year_lookup_status"),
            "curator": {k: (LK.get("curator") or {}).get(k) for k in ("unmatured_releases", "after_window_released", "past_control_released", "first_release_real_day",
                                                                      "rerun_prefix_mismatches", "control_curator_ignoring_maturity_mismatches", "rerun_days_differing_from_first_run",
                                                                      "rerun_difference_explained", "planted_date_content_refused", "hindsight_feature_refused")},
            "research": {k: (LK.get("research") or {}).get(k) for k in ("suite_passed", "same_year_refused_during_replay", "admitted_outside_replay",
                                                                        "admitted_without_replay_context", "hindsight_caught", "research_only_caught")},
            "memory_bank": LK.get("memory_bank"),
            "static": {k: (LK.get("static") or {}).get(k) for k in ("trader_closure_modules", "trader_closure_clean", "curator_writers",
                                                                    "research_loop_release_passes_replay", "loop2_learner_default")}}},
        test="test_probe_identifies_year_from_a_planted_level_channel_and_not_after_scrub; test_probe_split_never_overlaps_train_and_test_windows; test_calendar_features_see_a_midweek_closure_and_the_regular_grid_removes_it; test_hardened_feed_closes_the_three_exposures; test_leak_closure.py: test_lookup_verdict_is_clean_on_a_good_part_and_flips_on_each_planted_defect, test_real_lookup_probes_run_and_are_clean_with_live_controls, test_w11_no_research_filed_under_the_replayed_year_is_released_during_the_rerun, test_w11_no_hindsight_label_or_research_item_reaches_the_trader_side",
        fix="HardenedFeed removes the absolute SPY/market volume level and real column order (levels_after_hardening barely moves: universe size, price and dollar-volume levels and data artefacts stay identifiable); regular_grid_index removes closure signatures but only cuts the calendar probe from 0.79 to 0.54 skill (the session COUNT survives) and changes what a 'week' is: opt-in, not wired",
        hook=("Computed from the fingerprint part: the trader's OWN inputs (ranks, ratios and the m_* market context) identify the real period on their own - the m_* columns carry most of it "
              "(single-feature skill exposed/hidden-only: m_dispersion 0.74/0.56, m_vix_chg5 0.52/0.33, m_breadth 0.51/0.00). Market state is live information by design, but it is also a date. "
              "A fingerprint can only act where something can be looked up by it: (a) learned state - closed by BasisLineage (no basis contains the played window's year); (b) the memory bank, which "
              "recalls by exactly these m_* columns - closed by the real_end < start filter (BG.check_memory_bank_causality); (c) the model itself, which fits its own warm-up. So the status stays LEAK "
              "until the owner rules that identifiable-but-unusable market context is acceptable (then QUARANTINED) or coarsens the m_* inputs. Not resolved here. "
              "F06 re-audit with the C64 curator path: " + "; ".join(r for r in VD["6"].reasons if not r.startswith("the trader's own inputs")) +
              ". Latent hooks: engine/learning/curator.py Curator._coerce_item: refuse `leak_audit.hindsight_feature_keys(feats)` before knowability's regime prior is ever "
              "filed; engine/research/loop.py st_release: pass `replay=` (the window's ReplayContext) to FWL.run_day whenever the loop runs beside a disguised replay.") if VD["6"].status == L.LEAK else
             ("Level and calendar fingerprints are identifiable in the exposed frames but the trader consumes only ranks, ratios and the m_* context, which the probe finds not identifiable. "
              "Learned state is the only place a year fingerprint can act; with BasisLineage a window's state contains nothing from its own year."),
        measured_on="real caches, 729 window starts 1965-2025"))

    # 7 network
    nm = S.get("network_markers", {})
    A.add(chan("7", "Network (blind worker must have none)", VD["7"].status, {
        "full_adaptive_window_under_guard": {k: R.get(k) for k in ("sessions", "wall_s", "blocked_network_attempts", "data_cache_files_opened", "state_files_opened", "livesim_state_files_opened", "blind_gate_fails")},
        "network_imports_or_calls_reachable_from_blind_path": {m: [f"{x['kind']}:{x['what']}@{x['line']}" for x in v] for m, v in nm.items()},
        "why_it_matters": "engine.data imports yfinance at import time and exposes update()/download(); edgar/universe use requests; livesim imports data",
        "guard": "NetworkGuard patches socket connect/getaddrinfo (covers requests, urllib, yfinance); poison() stubs refresh functions"},
        test="test_network_reachable_without_guard_and_blocked_with_it; test_guard_allow_loopback_still_blocks_remote; test_poison_raises_and_restores; test_network_markers_find_planted_refresh",
        fix="leak_audit.NetworkGuard / poison; hardened_run() runs the whole window under the guard",
        hook="scripts/livesim_loop2.py worker(): first line `leak_audit.NetworkGuard().install()` (workers are separate processes, so it never touches the loop or the audit); optional `leak_audit.poison(data, ['update','download'])`",
        measured_on="static + runtime tests"))

    # 8 other
    fe = Cz.get("feed_exposure_real_windows", {})
    ti = Cz.get("features_truncation_invariance_2012_sample", {})
    tr = S.get("trader_side_reads", {})
    A.add(chan("8a", "Text, news, speeches, headlines in served data", VD["8a"].status, {
        "free_text_columns_found": S.get("free_text_columns"), "events_columns": S.get("events_columns"), "insider_columns": S.get("insider_columns"),
        "events_kinds": S.get("events_kinds"), "note": "events carry only ticker/form/accepted/kind; insider forms carry titles of a few words; no prose exists to research"},
        test="test_free_text_columns_finds_a_planted_headline_and_ignores_codes", measured_on="real caches"))
    A.add(chan("8b", "Feature look-ahead (fast path vs truncated data) and label embargo", VD["8b"].status, {
        "truncation_invariance_real_sample": {k: ti.get(k) for k in ("n_tickers", "n_cuts", "n_features", "clean", "leaky_features", "n_events", "n_insider")},
        "model_embargo_sessions": S.get("model_embargo_sessions"), "label_reach_sessions": S.get("label_reach_sessions"),
        "note": "training rows are only warm-up rows, whose stocks frame ends at `now`: labels past now are NaN and dropped"},
        test="test_real_feature_builder_is_future_invariant_on_synthetic_panel; test_truncation_invariance_catches_a_planted_look_ahead_feature", measured_on="real 2010-2013 sample, 60 tickers"))
    plain = {k: v.get("plain") for k, v in fe.items()}
    hard = {k: v.get("hardened") for k, v in fe.items()}
    link = {k: {"plain": v.get("plain_rerun_linkability"), "hardened": v.get("hardened_rerun_linkability")} for k, v in fe.items()}
    A.add(chan("8c", "What the feed shows: real alphabetical column order, columns for future IPOs, absolute SPY level", VD["8c"].status, {
        "plain_feed": plain, "hardened_feed": hard, "rerun_of_the_same_real_window_(two seals): names re-identified by position / by return correlation": link,
        "cost_bps_and_warmup_are_era_coded": "see channel 6"},
        test="test_plain_feed_shows_real_alphabetical_order_future_ipo_and_absolute_spy; test_hardened_feed_closes_the_three_exposures; test_hardened_feed_changes_no_decision_on_clean_data",
        fix="leak_audit.hardened_feed_class() (columns sorted by code, market prices rebased to 100, names hidden until first price); clean-data decisions bit-identical",
        hook=(("IMPLEMENTED on the default path: livesim.run(hardened=True) builds the feed through blind_feed_class -> leak_audit.hardened_feed_class; the worker passes no opt-out. "
               "Residual (data, not shape): the same real window re-identifies by return correlation across reruns - see channel 6 and channel 4.") if VD["8c"].status == L.FIXED else
              "scripts/livesim_loop2.py: `livesim.Feed = leak_audit.hardened_feed_class()` before livesim.run (or call leak_audit.hardened_run)"), measured_on="three real windows (1975, 1995, 2018)"))
    A.add(chan("8d", "Files the trader side reads at run time / reachable modules", VD["8d"].status, {
        "files_actually_opened_during_a_full_blind_window": {k: R.get(k) for k in ("data_cache_files_opened", "state_files_opened", "livesim_state_files_opened", "other_files_opened")},
        "import_reachable_file_reads_(static, over-approximate: reachable != executed)": {m: [f"{r['function']}: {r['call']}" for r in v] for m, v in tr.items()},
        "not_reachable_from_blind_path": S.get("not_reachable_from_blind_path"),
        "policy.not_crypto": "reads data/cache/universe.csv (today's names) - inert on code names, but it is a read of today's data on the trader side"},
        test="test_import_closure_follows_lazy_imports_and_finds_reads; test_real_blind_path_reaches_no_pattern_bank_or_lessons",
        fix="static inventory (import_closure, data_access, trader_side_reads); pattern_bank/lessons/analogs/trust are unreachable (tested)",
        hook="engine/policy.py not_crypto(): make the ticker set an argument (empty in blind runs) instead of reading universe.csv", measured_on="static"))
    A.add(chan("8e", "Universe filters / labels using future volume, ordering by future info, analog fingerprints with full-sample stats", VD["8e"].status, {
        "tradable": "per-date cross-sectional ranks of price (channel 2) and of trailing 20-day dollar volume (8b: identical when the future is cut off)",
        "ever_column_selection": "features.build keeps tickers tradable at any time in the window; rows exist only on dates they were tradable (8b covers it)",
        "analogs_full_sample_stats": "analog_weighting.pit_moments uses expanding moments; analogs is not on the blind path",
        "file_timestamps_cached_later_runs": "the trader reads nothing under state/ (8d); worker result files are written after the run"},
        test="test_real_feature_builder_is_future_invariant_on_synthetic_panel", measured_on="static + 8b"))

    stamp = provenance.stamp({"audit": "leak_audit", "parts": [n for n in ORDER[:-1] if P[n]]}, seed=SEED)
    A.save(OUT, stamp)
    from engine.improve import log_experiment
    log_experiment({"event": "leak_audit", "task": "B23", "counts": A.counts(), "open_leaks": A.open_leaks(), "report": "state/research/leak_audit/report.md",
                    "outcome": "continue_testing", "reason": "audit complete; fixes are opt-in hooks pending main-session integration"}, seed=SEED)
    print(json.dumps(A.counts()), "open leaks:", A.open_leaks(), flush=True)


GROUPS_NAMES = ("calendar", "levels_raw", "levels_after_hardening", "levels_scrubbed", "market_state", "trader_inputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(ORDER))
    ap.add_argument("--net", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reuse-daily", action="store_true", help="fingerprint part: reuse the cached daily series instead of rebuilding it")
    args = ap.parse_args()
    fn = {"static": part_static, "runtime": part_runtime, "defaults": part_defaults, "survivorship": part_survivorship, "adjusted": part_adjusted, "metadata": part_metadata,
          "fingerprint": part_fingerprint, "causality": part_causality, "lookup": part_lookup, "assemble": assemble}
    for p in args.parts.split(","):
        t = time.time()
        print(f"=== {p} ===", flush=True)
        fn[p](args)
        print(f"=== {p} done in {time.time() - t:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
