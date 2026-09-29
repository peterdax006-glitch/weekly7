"""Bible Phases 21-24 exercised on the REAL close caches (read-only, seeded, small sample, well under 1.5 GB).

What it does, in order (each section writes numbers to state/research/blind_gates/):
  1. seals     draw N seeded windows one after another; audit each (12 months, overlap, sealed-before-workers),
               report coverage per decade and per cost era.
  2. disguise  for a sample of windows: shift + seeded remap of a random ticker sample from the real closes, then run
               shift / map / frame / duplicate-path / calendar / warm-up gates. Counts real duplicate price paths.
  3. probe     lookahead_probe on a real momentum-ranking function (must pass) and on a peeking twin (must be caught).
  4. retest    an equal-weight top-k momentum rule run on the real closes as a stand-in "blind run": archive it, replay
               it (must PASS), replay it with a one-day-early information leak and with drifting fills (must FAIL),
               and replay under a different code hash (must be STALE_CODE).
  5. health    classification of synthetic worker logs over the real window list, showing the exclusion report.
Nothing under state/livesim or the sealed windows is read: seals here are drawn fresh from the seed."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import config as K, provenance, blind_gates as G, retester as R, health as H   # noqa: E402

OUT = K.STATE / "research" / "blind_gates"
SEED = 20260928
N_SEALS, N_DISGUISE, N_TICKERS = 40, 12, 250


def load_sample(rng):
    """Random column sample from both close caches, joined on date; only the sampled columns are read."""
    parts = []
    for n in ("stocks_pre2000_close", "stocks_close"):
        f = pq.ParquetFile(K.CACHE / f"{n}.parquet")
        cols = [c for c in f.schema_arrow.names if c != "Date"]
        pick = sorted(rng.choice(cols, size=min(N_TICKERS // 2, len(cols)), replace=False).tolist())
        df = f.read(columns=["Date"] + pick).to_pandas()
        df = df.set_index("Date") if "Date" in df.columns else df
        parts.append(df.loc[:, ~df.columns.duplicated()])
    df = pd.concat(parts, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else pd.DatetimeIndex(df.index)
    return df[~df.index.duplicated()]


def section_seals():
    used, rows, findings = [], [], []
    for i in range(N_SEALS):
        rec = G.seal_window(used, SEED + i, "2026-09-28 00:00", tag=f"run{i}")
        f = G.check_seal(rec, other_starts=used, first_worker_start="2026-09-28 01:00")
        findings += f
        used.append(rec["start"])
        rows.append({"i": i, "start": rec["start"], "era": G.era_of(rec["start"]), "shift_weeks": rec["shift_days"] // 7,
                     "fails": sum(x.severity == "fail" for x in f)})
    cov = G.coverage_report(used)
    pair_fail = G.check_no_overlap_across(used)
    return {"n": N_SEALS, "seal_fails": sum(r["fails"] for r in rows), "pairwise_overlap_fails": len(pair_fail),
            "coverage": cov, "coverage_findings": [str(x) for x in G.check_coverage(used)], "rows": rows}, used


def disguise_one(closes, start, seed_i):
    rec = G.seal_window([], seed_i, "2026-09-28 00:00")
    rec["start"] = str(pd.Timestamp(start).date())
    rec["digest"] = G.seal_digest(rec)
    lo = pd.Timestamp(start) - pd.DateOffset(years=6)
    hi = G.window_end(start)
    sub = closes.loc[lo:hi]
    live = sub.columns[sub.loc[start:hi].notna().any()]
    sub = sub[live]
    m = G.make_ticker_map(list(live), rec["shift_days"])
    shown = sub.copy()
    shown.index = shown.index + pd.Timedelta(days=rec["shift_days"])
    shown.columns = [m[c] for c in live]
    f = []
    f += G.check_seal(rec)
    f += G.check_shift(rec["shift_days"], sub.index, shown.index)
    f += G.check_ticker_map(m, list(live), seed=rec["shift_days"])
    f += G.check_disguised_frame(shown, m, real_dates_known=sub.index)
    f += G.check_calendar(sub.index, shown.index)
    dup = G.check_disguise_signature(shown)
    have = (closes.loc[:lo].dropna(how="all").index[0] if closes.loc[:lo].notna().any().any() else pd.Timestamp(start))
    warm = (pd.Timestamp(start) - sub.index[0]).days / 365.25
    f += G.check_warmup(start, warm, first_data=closes.index[0])
    return {"start": str(pd.Timestamp(start).date()), "era": G.era_of(start), "n_live": len(live), "sessions": len(sub),
            "warmup_years": round(warm, 2), "hard_fails": [str(x) for x in f if x.severity == "fail"],
            "duplicate_paths": [str(x) for x in dup]}


def momentum_rank(closes, now, k=10, look=20):
    """Top-k by trailing return, honest: sees only rows <= now."""
    c = closes.loc[:now]
    r = (c.iloc[-1] / c.iloc[-1 - look] - 1).dropna()
    return r.nlargest(k).index.tolist()


def momentum_rank_peeking(closes, now, k=10, look=20):
    """Twin that ranks on the return over the NEXT `look` rows: the planted look-ahead."""
    c = closes.loc[:now]
    i = len(c) - 1
    r = (closes.iloc[i + look] / closes.iloc[i] - 1).dropna()
    return r.nlargest(k).index.tolist()


def section_probe(closes):
    idx = closes.index
    res = []
    for now in idx[np.linspace(len(idx) * 0.3, len(idx) * 0.9, 6).astype(int)]:
        good = G.lookahead_probe(lambda d: pd.Series(momentum_rank(d, now)), closes.copy(), now, seed=SEED, trials=2)
        bad = G.lookahead_probe(lambda d: pd.Series(momentum_rank_peeking(d, now)), closes.copy(), now, seed=SEED, trials=2)
        res.append({"now": str(now.date()), "honest_passes": G.passed(good), "peeker_caught": not G.passed(bad)})
    return res


def run_rule(closes, weeks, code_hash, leak_days=0, drift=0.0):
    """Stand-in blind run: each Friday-ish decision date holds the top-k momentum names for the next 5 sessions,
    filled at the next session (close-to-close proxy here; the point is replay parity, not the strategy)."""
    idx = closes.index
    dec_i = list(range(40, 40 + 5 * weeks, 5))
    hold, trades, wk = [], [], {}
    prev = set()
    for w, i in enumerate(dec_i):
        now = idx[i - leak_days]
        names = momentum_rank(closes, now)
        for t in names:
            hold.append({"date": str(idx[i].date()), "ticker": t, "weight": 0.1})
        for t in set(names) - prev:
            trades.append({"date": str(idx[i + 1].date()), "ticker": t, "side": "buy", "qty": 1.0,
                           "price": float(closes[t].iloc[i + 1]) * (1 + drift)})
        prev = set(names)
        fwd = closes[names].iloc[i + 1] if i + 1 < len(idx) else None
        end = closes[names].iloc[min(i + 6, len(idx) - 1)]
        wk[f"w{w:02d}"] = float((end / fwd - 1).mean())
    return {"provenance": {"code_hash": code_hash}, "holdings": pd.DataFrame(hold), "trades": pd.DataFrame(trades),
            "scores": pd.DataFrame({"date": [str(idx[i].date()) for i in dec_i], "ticker": ["-"] * len(dec_i), "score": [1.0] * len(dec_i)}),
            "weekly_returns": wk, "adaptation_events": [], "pattern_activation": {}, "memory_state": {"n": len(dec_i)}}


def section_retest(closes):
    sub = closes.loc["2012-01-01":"2014-12-31"].dropna(axis=1, how="any")
    a = run_rule(sub, 40, "codeA")
    same = run_rule(sub, 40, "codeA")
    leak = run_rule(sub, 40, "codeA", leak_days=1)
    drift = run_rule(sub, 40, "codeA", drift=0.02)
    stale = run_rule(sub, 40, "codeB")
    out = {}
    for name, rep in (("faithful", same), ("one_day_leak", leak), ("fill_drift_2pct", drift), ("other_code", stale)):
        v = R.compare_runs(a, rep)
        out[name] = {"status": v.status, "summary": v.summary(), "compared": v.compared,
                     "worst": {c: [n, (None if w == float("inf") else round(w, 5))] for c, (n, w) in R.worst_by_component(v).items()},
                     "noise_floor": R.noise_floor(a, rep) if v.status != R.STALE else None,
                     "first_divergent_week": R.first_divergence_week(a["weekly_returns"], rep["weekly_returns"]),
                     "turnover_archive_vs_replay": [round(R.turnover(a["holdings"]), 4), round(R.turnover(rep["holdings"]), 4)]}
    return out


def section_health(starts):
    rng = np.random.default_rng(SEED)
    workers = {}
    fates = ["ok"] * 8 + ["crash", "oom", "timeout", "invalid", "stale"]
    for i, st in enumerate(starts[:len(fates)]):
        fate, wid = fates[i], f"w{i:02d}"
        rows = [{"t": 1000.0, "worker": wid, "event": "start", "code_hash": "cur" if fate != "stale" else "old"},
                {"t": 1000.0, "worker": wid, "event": "config", "config": {"k": 5}},
                {"t": 1000.0, "worker": wid, "event": "window", "window": st},
                {"t": 1000.0, "worker": wid, "event": "seed", "seed": i},
                {"t": 1010.0, "worker": wid, "event": "memory", "rss_mb": float(rng.uniform(200, 900))}]
        if fate in ("ok", "invalid", "stale"):
            rows.append({"t": 1500.0, "worker": wid, "event": "complete"})
        elif fate == "crash":
            rows.append({"t": 1200.0, "worker": wid, "event": "crash", "returncode": -11})
        elif fate == "oom":
            rows.append({"t": 1200.0, "worker": wid, "event": "oom", "rss_mb": 7900.0})
        else:
            rows.append({"t": 4600.0, "worker": wid, "event": "timeout"})
        wr = list(rng.normal(0.01, 0.03, 52))
        if fate == "invalid":
            wr[10] = float("nan")
        workers[wid] = {"log": rows, "result": {"window": st, "seed": i, "weekly_returns": wr},
                        "config": {"k": 5}, "window": st, "seed": i}
    rep = H.exclusion_report(workers, current_code_hash="cur")
    return {"counts": rep["counts"], "excluded": rep["excluded"], "window_breakdown": H.window_breakdown(workers, rep),
            "rerun_plan": H.plan_rerun(rep, range(len(fates)), len(fates)),
            "aggregate_ready": H.aggregate_ready(rep, min_ok_fraction=0.6)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    closes = load_sample(rng)
    print(f"sample closes {closes.shape} {closes.index[0].date()}..{closes.index[-1].date()} "
          f"{closes.memory_usage(deep=True).sum() / 2**20:.0f} MB", flush=True)
    seals, used = section_seals()
    print(f"seals: {seals['seal_fails']} fails, {seals['pairwise_overlap_fails']} overlap fails, coverage {seals['coverage']['per_bin']}", flush=True)
    starts = [pd.Timestamp(s) for s in used
              if pd.Timestamp(s) >= closes.index[0] + pd.DateOffset(years=1) and G.window_end(s) <= closes.index[-1]]
    pick = [starts[i] for i in np.random.default_rng(SEED).permutation(len(starts))[:N_DISGUISE]]
    dis = [disguise_one(closes, s, SEED + 100 + i) for i, s in enumerate(sorted(pick))]
    print(f"disguise: {sum(len(d['hard_fails']) for d in dis)} hard fails over {len(dis)} windows", flush=True)
    probe = section_probe(closes)
    print(f"probe: honest passed {sum(p['honest_passes'] for p in probe)}/{len(probe)}, peeker caught {sum(p['peeker_caught'] for p in probe)}/{len(probe)}", flush=True)
    retest = section_retest(closes)
    for k, v in retest.items():
        print(f"retest {k}: {v['status']}", flush=True)
    health = section_health([str(d.date()) for d in sorted(pick)] + ["x"] * 5)
    print(f"health: {health['counts']}", flush=True)
    result = {"stamp": provenance.stamp({"seed": SEED, "n_tickers": N_TICKERS}, SEED), "seconds": round(time.time() - t0, 1),
              "sample": {"shape": list(closes.shape), "first": str(closes.index[0].date()), "last": str(closes.index[-1].date())},
              "seals": seals, "disguise": dis, "probe": probe, "retest": retest, "health": health}
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str))
    per_era = {}
    for d in dis:
        e = per_era.setdefault(d["era"], {"windows": 0, "hard_fails": 0, "dup_paths": 0})
        e["windows"] += 1; e["hard_fails"] += len(d["hard_fails"]); e["dup_paths"] += len(d["duplicate_paths"])
    md = ["# Blind gates on real caches", "", f"seed {SEED}, {result['seconds']} s, sample {closes.shape}", "",
          "## Seals", f"{seals['n']} seals: {seals['seal_fails']} failing, {seals['pairwise_overlap_fails']} overlapping pairs.",
          f"Per decade: {seals['coverage']['per_bin']}; per era: {seals['coverage']['per_era']}", "",
          "## Disguise per era", *[f"- {e}: {v}" for e, v in sorted(per_era.items())], "",
          "## Look-ahead probe", *[f"- {p['now']}: honest passes {p['honest_passes']}, peeker caught {p['peeker_caught']}" for p in probe], "",
          "## Re-test", *[f"- {k}: {v['summary']}" for k, v in retest.items()], "",
          "## Health", f"{health['counts']}"]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print(f"done in {result['seconds']} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
