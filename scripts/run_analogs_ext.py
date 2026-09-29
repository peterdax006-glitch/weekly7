"""Phase 8 runner: market / sector / stock analog engines and weighting on the REAL caches (read-only, seeded sample).

Reads only the columns of a seeded ticker sample (default 250) plus the index, VIX and macro caches, keeps peak memory
well under 1.5 GB, and never touches state/livesim or writes into data/cache. Writes results.json (stamped with
engine.provenance.stamp), report.md and per-level CSVs to state/research/analogs_ext/.

Sections: rule audit of every engine -> 8.8 ablation at market level (walk-forward weights, per-decade breakdown, confidence
calibration, metric/k sweep) -> sector level (per-sector ablation, feature report, cross-sector rank IC) -> stock level
(own-history ablation on long-history names, pooled cross-stock rank IC by era).
Survivorship: the stock cache holds today's survivors, so sector/stock outcomes are optimistic; read relative results only."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from engine import config as K
from engine import analog_weighting as W
from engine import analogs_sector as S
from engine import analogs_stock as T
from engine.provenance import stamp

OUT = K.STATE / "research" / "analogs_ext"


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return float("nan")


def load(seed, n_tickers):
    """Seeded sample of tickers with a SIC code; long history = pre-2000 file (to 1999) + main file (2000 on)."""
    sic = pd.read_parquet(K.CACHE / "sic.parquet").set_index("ticker")["sic"].astype(str)
    import pyarrow.parquet as pq
    have_new = set(pq.ParquetFile(K.CACHE / "stocks_close.parquet").schema_arrow.names) - {"Date"}
    have_old = set(pq.ParquetFile(K.CACHE / "stocks_pre2000_close.parquet").schema_arrow.names) - {"Date"}
    pool = sorted(t for t in sic.index if t in have_new and str(sic[t]).strip() not in ("", "nan"))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(pool, min(n_tickers, len(pool)), replace=False))
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=pick)
    V = pd.read_parquet(K.CACHE / "stocks_volume.parquet", columns=pick)
    old = [t for t in pick if t in have_old]
    if old:
        Co = pd.read_parquet(K.CACHE / "stocks_pre2000_close.parquet", columns=old)
        Vo = pd.read_parquet(K.CACHE / "stocks_pre2000_volume.parquet", columns=old)
        C = pd.concat([Co, C.loc["2000-01-01":]]).sort_index()
        V = pd.concat([Vo, V.loc["2000-01-01":]]).sort_index()
        C, V = C.loc[~C.index.duplicated(keep="last")], V.loc[~V.index.duplicated(keep="last")]
    ix = pd.read_parquet(K.CACHE / "index_hist_close.parquet")
    if "Date" in ix.columns:
        ix = ix.set_index("Date")
    spx = ix["^GSPC"].dropna()
    vix = ix["^VIX"].dropna()
    mk = pd.read_parquet(K.CACHE / "market_close.parquet")
    macro = pd.read_parquet(K.CACHE / "macro.parquet")
    vix3m = mk["^VIX3M"].dropna() if "^VIX3M" in mk else None
    return C.reindex(spx.index), V.reindex(spx.index), spx, vix, vix3m, macro, sic.reindex(pick)


def frame(tab):
    return tab.reset_index().rename(columns={"index": "mode"})


def summarise_ablation(tab):
    return {m: {k: (None if pd.isna(v) else float(v)) for k, v in tab.loc[m].items()} for m in tab.index}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tickers", type=int, default=250)
    ap.add_argument("--refit", type=int, default=504, help="weight refit spacing in sessions")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--max-sectors", type=int, default=8)
    ap.add_argument("--stock-names", type=int, default=12)
    ap.add_argument("--pooled-names", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    peak = 0.0
    result = {"provenance": stamp({"tickers": a.tickers, "refit": a.refit, "n_random": a.n_random}, a.seed), "args": vars(a)}
    tables = {}

    C, V, spx, vix, vix3m, macro, sic = load(a.seed, a.tickers)
    peak = max(peak, rss_mb())
    print(f"loaded {C.shape[1]} tickers x {C.shape[0]:,} sessions ({spx.index[0].date()} -> {spx.index[-1].date()}) rss={peak:.0f}MB", flush=True)

    # ---- market level ----------------------------------------------------------------------------------------------------
    ctx = W.market_context(spx, vix, vix3m, macro).join(S.breadth_dispersion_crowding(C, V, sic))
    coverage = W.spec_coverage(ctx)
    cut_pts = [int(len(spx) * f) for f in (0.35, 0.6, 0.85)]
    causal = W.causality_audit(lambda d: W.market_context(d[0], d[1], d[2], d[3]).join(S.breadth_dispersion_crowding(C.loc[:d[0].index[-1]], V.loc[:d[0].index[-1]], sic)),
                               (spx, vix, vix3m, macro), cut_pts,
                               lambda d, c: (d[0].iloc[:c], d[1].loc[:d[0].index[c - 1]], None if d[2] is None else d[2].loc[:d[0].index[c - 1]],
                                             d[3].loc[:d[0].index[c - 1]]))
    print(f"fingerprint coverage {int(coverage['present'].sum())}/18 items; causality offenders: {len(causal)}", flush=True)
    Om = W.forward_outcomes(spx, 21)
    eng = W.KNNEngine(ctx, Om, "fwd_ret_21d", horizon=21)
    lo = eng.min_hist + eng.gap + eng.horizon + 8 * eng.sep
    W.learn_weights_walk_forward(eng, list(range(lo, len(ctx), a.refit)), seed=a.seed)
    pos = W.eval_positions(eng, "1975-01-01", None, 21)
    audit = W.audit_engine(eng, pos[:: max(1, len(pos) // 60)])
    table, preds = W.compare_methods(eng, pos, 5, a.seed, a.n_random)
    verdict = W.weighting_verdict(table, preds, seed=a.seed)
    eras = W.era_breakdown(preds)
    cal_tab, rho = W.confidence_calibration(preds["weighted"])
    sweep = W.sweep_settings(eng, pos[::2])
    refits = list(range(lo, len(ctx), a.refit))
    learners, _ = W.compare_learners(eng, refits, pos, 5, a.seed, a.n_random)
    null_ic = W.weight_learner_null(eng, refits[::2], n_shuffles=3, seed=a.seed, method="ic")
    null_desc = W.weight_learner_null(eng, refits[::3], n_shuffles=2, seed=a.seed, method="descent")
    ledger_n = 0
    led_dates = [ctx.index[ctx.index.searchsorted(d)] for d in pd.date_range("2024-01-01", "2026-06-01", freq="MS")]
    led = [eng.query(int(ctx.index.get_loc(t)), 5) for t in led_dates]
    ledger_n = W.append_ledger(OUT / "ledger.parquet", W.results_to_frame(led, "market", [str(t.date()) for t in led_dates]))
    feat = W.feature_report(eng.schedule, eng.cols)
    tables.update({"Market ablation (fwd 21d return, vs no-analog)": table, "Market per-decade skill": eras.pivot(index="era", columns="mode", values="skill_ref"),
                   "Market confidence calibration": cal_tab, "Market metric/k sweep": sweep, "Market feature weights": feat,
                   "Market weight refits": eng.schedule.diagnostics,
                   "Market weight learners (uniform vs descent vs IC)": learners, "Fingerprint coverage (Bible 8.1)": coverage})
    result["market"] = {"n_eval": len(pos), "ablation": summarise_ablation(table), "verdict": verdict, "confidence_spearman": rho,
                        "refits": int(len(eng.schedule.dates)), "refits_accepted": int(eng.schedule.diagnostics["accepted"].sum()),
                        "audit_dirty_rows": len(audit), "causality_offenders": [list(map(str, c)) for c in causal],
                        "fingerprint_items_present": int(coverage["present"].sum()), "learner_null_ic": null_ic,
                        "learner_null_descent": null_desc, "ledger_rows": ledger_n}
    print(f"market: n={len(pos)} audit_dirty={len(audit)} verdict={verdict} rho={rho:.3f} [{time.time()-t0:.0f}s]", flush=True)
    peak = max(peak, rss_mb())

    # ---- sector level ------------------------------------------------------------------------------------------------------
    groups = S.sector_groups(sic, min_members=6)
    groups = dict(sorted(groups.items(), key=lambda kv: -len(kv[1]))[: a.max_sectors])
    keep = ["drawdown", "ret_3m", "vol_1m", "vix", "vix_term", "rate_chg_1y", "credit_chg_3m"]
    fp = S.sector_fingerprints(C, V, groups, spx, ctx[[c for c in keep if c in ctx]], horizon=21)
    sa = S.SectorAnalogs(fp, horizon=21)
    sa.learn(refit_every=a.refit)
    rows, per_era, sec_dirty = [], [], 0
    all_preds = {}
    for sec in sa.sectors():
        e = sa.engines[sec]
        p = W.eval_positions(e, "1975-01-01", None, 21)
        if len(p) < 30:
            continue
        sec_dirty += len(W.audit_engine(e, p[:: max(1, len(p) // 25)]))
        tab, pr, ver = sa.ablation(sec, "1975-01-01", None, 21, 5, a.seed, a.n_random)
        all_preds[sec] = pr
        er = W.era_breakdown(pr)
        er["sector"] = sec
        per_era.append(er[er["mode"] == "weighted"])
        rows.append({"sector": sec, "n_names": len(groups[sec]), "n_eval": int(tab.loc["weighted", "n"]),
                     "mse_none": tab.loc["none", "mse"], "mse_unweighted": tab.loc["unweighted", "mse"], "mse_weighted": tab.loc["weighted", "mse"],
                     "skill_unweighted": tab.loc["unweighted", "skill_ref"], "skill_weighted": tab.loc["weighted", "skill_ref"],
                     "p_weighted_vs_none": tab.loc["weighted", "p_ref"], "skill_random": tab.loc["random", "skill_ref"],
                     "skill_shuffled": tab.loc["shuffled", "skill_ref"], "skill_nn_random": tab.loc["nn_random", "skill_ref"],
                     "beats_random": ver["beats_random"], "beats_unweighted": ver["beats_unweighted"]})
    sec_tab = pd.DataFrame(rows)
    tables["Sector ablation by sector"] = sec_tab.set_index("sector") if len(sec_tab) else sec_tab
    if per_era:
        pe = pd.concat(per_era)
        tables["Sector weighted skill by decade (mean over sectors)"] = pe.groupby("era")[["n", "skill_ref"]].mean()
    mdates = spx.loc["1990-01-01":].resample("MS").first().index
    mdates = [spx.index[spx.index.searchsorted(d)] for d in mdates][:-2]
    _, ic_s = sa.rank_ic(mdates)
    result["sector"] = {"n_sectors": len(sec_tab), "audit_dirty_rows": sec_dirty, "rank_ic": ic_s,
                        "mean_skill_weighted": float(sec_tab["skill_weighted"].mean()) if len(sec_tab) else None,
                        "share_sectors_weighted_beats_none_p05": float((sec_tab["p_weighted_vs_none"] < 0.05).mean()) if len(sec_tab) else None,
                        "share_sectors_beat_random": float(sec_tab["beats_random"].mean()) if len(sec_tab) else None,
                        "share_sectors_beat_unweighted": float(sec_tab["beats_unweighted"].mean()) if len(sec_tab) else None}
    print(f"sector: {result['sector']} [{time.time()-t0:.0f}s]", flush=True)
    peak = max(peak, rss_mb())

    # ---- stock level -------------------------------------------------------------------------------------------------------
    keep_st = ["drawdown", "vix", "rate_chg_1y"]
    ctx_st = ctx[[c for c in keep_st if c in ctx]]
    span = C.notna().sum().sort_values(ascending=False)
    longs = list(span.index[: a.stock_names])
    fps = {tk: T.stock_fingerprint(C[tk].dropna(), V[tk].reindex(C[tk].dropna().index), spx, 5, ctx_st) for tk in longs}
    st = T.StockAnalogs(fps, horizon=5)
    srows, st_dirty = [], 0
    for tk in list(st.engines):
        e = st.engines[tk]
        p = W.eval_positions(e, None, None, 5)
        if len(p) < 40:
            continue
        p = p[:: max(1, len(p) // 250)]
        st_dirty += len(W.audit_engine(e, p[:: max(1, len(p) // 20)]))
        preds_tk = {m: W.predict_series(e, p, m, 5, a.seed, a.n_random if m in ("random", "shuffled", "nn_random") else 1)
                    for m in ("none", "unweighted", "random", "shuffled", "nn_random")}
        tab = W.ablation_table(preds_tk, ref="none", control="random", seed=a.seed)
        srows.append({"ticker": tk, "n_eval": int(tab.loc["unweighted", "n"]), "mse_none": tab.loc["none", "mse"],
                      "skill_unweighted": tab.loc["unweighted", "skill_ref"], "p_unweighted": tab.loc["unweighted", "p_ref"],
                      "skill_random": tab.loc["random", "skill_ref"], "skill_shuffled": tab.loc["shuffled", "skill_ref"],
                      "skill_nn_random": tab.loc["nn_random", "skill_ref"]})
    stk = pd.DataFrame(srows)
    tables["Stock own-history ablation (fwd 5d return)"] = stk.set_index("ticker") if len(stk) else stk
    # pooled cross-stock analogs on the recent era (memory bounded: pooled-names x sessions x features)
    pn = list(span.index[: a.pooled_names])
    fpp = {}
    for tk in pn:
        s = C[tk].loc["2005-01-01":].dropna()
        if len(s) > 800:
            fpp[tk] = T.stock_fingerprint(s, V[tk].reindex(s.index), spx, 5, ctx_st)
    pooled_summary = {}
    if len(fpp) >= 10:
        pa = T.PooledStockAnalogs(fpp, horizon=5, stride=5)
        pdates = [pa.index[i] for i in range(pa.min_hist + pa.gap + 5, len(pa.index) - 6, 21)]
        ic, ic_s = pa.rank_ic(pdates, k=8)
        pooled_summary = {"n_names": len(fpp), **ic_s}
        by = ic.groupby(ic.index.year // 5 * 5).agg(["mean", "count"])
        tables["Pooled stock analog rank IC by half-decade"] = by
    result["stock"] = {"own_history_names": len(stk), "audit_dirty_rows": st_dirty,
                       "mean_skill_unweighted": float(stk["skill_unweighted"].mean()) if len(stk) else None,
                       "share_beat_none_p05": float((stk["p_unweighted"] < 0.05).mean()) if len(stk) else None,
                       "share_beat_random": float((stk["skill_unweighted"] > stk["skill_random"]).mean()) if len(stk) else None,
                       "pooled": pooled_summary, "insufficient": st.insufficient}
    print(f"stock: {result['stock']} [{time.time()-t0:.0f}s]", flush=True)
    peak = max(peak, rss_mb())

    result["peak_rss_mb"] = peak
    result["seconds"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "report.md").write_text(W.render_report("Phase 8 analog engines on real caches", tables, [
        f"seed {a.seed}, {a.tickers} sampled tickers, refit every {a.refit} sessions, peak RSS {peak:.0f} MB",
        "survivor universe: sector/stock outcomes are optimistic, compare levels only against controls on the same rows",
        f"audit dirty rows (must be 0): market {len(audit)}, sector {sec_dirty}, stock {st_dirty}",
        f"causality offenders (must be empty): {len(causal)}",
        f"weight-learner false acceptance: real vs shuffled outcomes, IC {null_ic}, descent {null_desc}"]), encoding="utf-8")
    for name, tb in tables.items():
        if len(tb):
            tb.to_csv(OUT / (name.split("(")[0].strip().lower().replace(" ", "_").replace("/", "_") + ".csv"))
    print(f"done in {time.time()-t0:.0f}s peak rss {peak:.0f}MB -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
