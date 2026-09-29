"""Bible Phases 6, 7 and 34 on the REAL caches (read-only): does the pattern miner hold up out of sample across four
eras, and do its patterns beat the controls as mover features?

Builds a seeded ticker sample from the 1962-1999 and 2000-2026 daily bar caches, computes a price-only weekly feature
panel per segment (no feature or label ever crosses the 1999/2000 seam, whose two downloads may be adjusted
differently), attaches market context from ^GSPC, then runs
  * engine.heavy_tests.run_heavy      - rolling-origin walk-forward, seeds, noise control, eras, regimes, decay, costs;
  * engine.heavy_tests.ablation_across_eras - the six-arm mover ablation with each era as the test window;
  * engine.pattern_movers.ablation_sweep    - the same ablation at several mover thresholds for the latest era.
Peak memory stays well under 1.5 GB (only the sampled ticker columns are read).  Results go to
state/research/heavy_algo/ and state/research/pattern_movers/ with engine.provenance.stamp.

usage: heavy_algo_real.py [--tickers 350] [--seeds 7,11] [--step 78] [--min-train 260] [--train-max 520] [--tag name]"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import config as K, heavy_tests as ht, pattern_movers as pm
from engine.provenance import stamp

CACHE = K.CACHE
SEGMENTS = {"pre2000": "stocks_pre2000_{}.parquet", "post2000": "stocks_{}.parquet"}


def sample_tickers(seg, n, seed, min_frac=0.4):
    """Seeded sample of tickers that are listed for at least `min_frac` of the segment (non-null counts come from the
    parquet column statistics, so nothing but metadata is read).  Long-lived names only: the sample is survivor-biased,
    which the output says."""
    f = CACHE / SEGMENTS[seg].format("close")
    pf = pq.ParquetFile(f)
    names = pf.schema_arrow.names
    rows = pf.metadata.num_rows
    nn = np.zeros(len(names))
    for g in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(g)
        for i in range(len(names)):
            st = rg.column(i).statistics
            nn[i] += rows if st is None else rg.num_rows - st.null_count
    ok = [nm for nm, c in zip(names, nn) if nm != "Date" and c >= min_frac * rows]
    rng = np.random.default_rng([seed, 5150 if seg == "pre2000" else 5151])
    return sorted(rng.choice(ok, min(n, len(ok)), replace=False).tolist())


def load_segment(seg, tickers):
    out = {}
    for field in ("close", "open", "high", "low"):
        f = CACHE / SEGMENTS[seg].format(field)
        have = [c for c in tickers if c in pq.ParquetFile(f).schema_arrow.names]
        df = pd.read_parquet(f, columns=have + ["Date"]) if "Date" in pq.ParquetFile(f).schema_arrow.names else pd.read_parquet(f, columns=have)
        if "Date" in df.columns:
            df = df.set_index("Date")
        out[field] = df.astype("float32")
    return out


def weekly_dates(idx):
    idx = pd.DatetimeIndex(idx)
    wk = idx.to_series().groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).max()
    return pd.DatetimeIndex(wk.values)


def build_segment(bars, gspc, min_names=40):
    """Price-only features for one segment, sampled at week-end dates.  Every feature uses data up to and including
    the decision day; the label is the return from the NEXT open to the close 5 sessions later (C33)."""
    C, O, H, L = bars["close"], bars["open"], bars["high"], bars["low"]
    ret = C.pct_change(fill_method=None)
    F = {
        "r5": C.pct_change(5, fill_method=None), "r20": C.pct_change(20, fill_method=None),
        "r60": C.pct_change(60, fill_method=None), "r120": C.pct_change(120, fill_method=None),
        "mom_12_1": C.shift(21) / C.shift(252) - 1,
        "vol20": ret.rolling(20, min_periods=15).std(),
        "atr_pct": ((H - L) / C).rolling(14, min_periods=10).mean(),
        "dist_ma50": C / C.rolling(50, min_periods=40).mean() - 1,
        "dist_ma200": C / C.rolling(200, min_periods=150).mean() - 1,
        "dist_52wh": C / C.rolling(252, min_periods=200).max() - 1,
        "max20": C / C.rolling(20, min_periods=15).max() - 1, "min20": C / C.rolling(20, min_periods=15).min() - 1,
        "gap_today": O / C.shift(1) - 1, "close_loc": ((C - L) / (H - L).replace(0, np.nan)),
        "range_compress": ((H - L) / C).rolling(5).mean() / ((H - L) / C).rolling(60, min_periods=40).mean(),
        "skew60": ret.rolling(60, min_periods=45).skew(),
    }
    fwd = C.shift(-5) / O.shift(-1) - 1
    wk = weekly_dates(C.index)
    wk = wk[wk.isin(C.index)]
    g = gspc.reindex(C.index).ffill()
    gr = g.pct_change(fill_method=None)
    ctx = pd.DataFrame({
        "m_vix": gr.rolling(20, min_periods=15).std() * np.sqrt(252),
        "m_vix_term": gr.rolling(20, min_periods=15).std() / gr.rolling(120, min_periods=90).std(),
        "m_spy_ma200": g / g.rolling(200, min_periods=150).mean() - 1,
        "m_breadth": (C > C.rolling(50, min_periods=40).mean()).where(C.notna()).mean(axis=1),
        "m_dispersion": C.pct_change(5, fill_method=None).std(axis=1),
    })
    parts, ys = [], []
    for name, f in F.items():
        parts.append(f.loc[wk].stack(future_stack=True).rename(name))
    X = pd.concat(parts, axis=1)
    X.index.names = ["date", "ticker"]
    y = fwd.loc[wk].stack(future_stack=True).reindex(X.index)
    X = X.dropna(subset=["vol20", "r20"])
    y = y.reindex(X.index)
    n_per = X.groupby(level=0).size()
    X = X[X.index.get_level_values(0).map(n_per) >= min_names]
    y = y.reindex(X.index)
    for c in ctx:
        X[c] = ctx[c].reindex(X.index.get_level_values(0)).to_numpy()
    y = y.where(np.isfinite(y))
    y = y.clip(-0.9, 3.0)                                            # one bad tick must not own a week
    return X.astype("float32"), y.astype("float64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=350)
    ap.add_argument("--seeds", default="7,11")
    ap.add_argument("--step", type=int, default=78)
    ap.add_argument("--min-train", type=int, default=260)
    ap.add_argument("--train-max", type=int, default=520)
    ap.add_argument("--sample-seed", type=int, default=3)
    ap.add_argument("--tag", default="real")
    a = ap.parse_args()
    t0 = time.time()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    gspc = pd.read_parquet(CACHE / "index_hist_close.parquet")
    gspc = (gspc.set_index("Date") if "Date" in gspc.columns else gspc)["^GSPC"].dropna()
    Xs, ys = [], []
    for seg in SEGMENTS:
        tk = sample_tickers(seg, a.tickers, a.sample_seed)
        bars = load_segment(seg, tk)
        X, y = build_segment(bars, gspc)
        print(f"[{time.time() - t0:5.0f}s] {seg}: {len(tk)} tickers -> {len(X):,} rows, {X.index.get_level_values(0).nunique()} weeks "
              f"({X.index.get_level_values(0).min().date()} to {X.index.get_level_values(0).max().date()})", flush=True)
        Xs.append(X); ys.append(y)
        del bars
    X, y = pd.concat(Xs), pd.concat(ys)
    X = X.sort_index(); y = y.reindex(X.index)
    print(f"[{time.time() - t0:5.0f}s] panel {X.shape}, {X.memory_usage(deep=True).sum() / 1e6:.0f} MB, y NaN {y.isna().mean():.3f}", flush=True)

    cfg = {"min_train_dates": a.min_train, "step_dates": a.step, "train_max_dates": a.train_max, "gap_days": 10, "topk": 25,
           "min_era_dates": 20, "movement": True,
           "miner": {"max_pairs": 1200, "max_unless": 150, "null_reps": 1, "min_n": 300, "half_life_years": 6.0, "max_rows": 400_000}}
    out_h = K.STATE / "research" / "heavy_algo"
    out_p = K.STATE / "research" / "pattern_movers"
    out_p.mkdir(parents=True, exist_ok=True)
    S = ht.run_heavy(X, y, cfg, seeds=seeds, null_seeds=(101,), n_hidden=30, hidden_len=52, out_dir=out_h, cache_dir=out_h / "cache")
    print(f"[{time.time() - t0:5.0f}s] heavy run done: {S.get('provenance', {}).get('run_id')}", flush=True)
    if S.get("ok"):
        print(ht.render_markdown(S)[:6000], flush=True)

    abl_cfg = {"n_controls": 3, "boot": 400, "miner": {**cfg["miner"], "max_rows": 250_000}}
    res = {}
    for label, base_cols in (("all_features", None),
                             ("lean_mover_inputs", ["atr_pct", "vol20", "r5", "r20", "gap_today", "m_vix", "m_breadth"])):
        tbl, full = ht.ablation_across_eras(X, y, cfg=({**abl_cfg, "base_cols": base_cols}), seed=seeds[0],
                                            min_train_dates=a.min_train)
        print(f"\nablation across eras, mover model sees: {label}\n{tbl.to_string()}", flush=True)
        for era, r in full.items():
            if r.get("ok"):
                print(f"--- {label} / {era}\n{pm.format_ablation(r)}", flush=True)
        res[label] = {"table": json.loads(tbl.reset_index().to_json(orient="records")),
                      "eras": {e: {"deploy": r.get("deploy"), "reasons": r.get("reasons"), "gain": r.get("gain_vs_base"),
                                   "arms": r.get("arms"), "direction": r.get("direction"),
                                   "detectable_gain": pm.detectable_gain(r["daily_diff_real_vs_base"].values) if r.get("ok") else None,
                                   "attribution_by_kind": json.loads(pm.attribute_by_kind(r["attribution"]).to_json(orient="records"))
                                   if r.get("ok") else None,
                                   "attribution_top": json.loads(r["attribution"].head(10).to_json(orient="records")) if r.get("ok") else None}
                               for e, r in full.items()}}
    last = X.index.get_level_values(0).unique()
    split = last[int(len(last) * 0.85)]
    sweep = pm.ablation_sweep(X, y, split, qs=(0.7, 0.8, 0.9), cfg={**abl_cfg, "base_cols": None}, seed=seeds[0])
    print(f"\nmover-threshold sweep, test from {split.date()}\n{sweep.to_string()}", flush=True)
    doc = {"provenance": stamp({"heavy": cfg, "ablation": abl_cfg, "tickers": a.tickers, "sample_seed": a.sample_seed}, list(seeds)),
           "panel": {"rows": int(len(X)), "features": [c for c in X.columns if not c.startswith("m_")],
                     "first": str(last.min().date()), "last": str(last.max().date()),
                     "note": "price-only features from a seeded ticker sample; survivorship of the cache is not corrected"},
           "heavy_run_id": S.get("provenance", {}).get("run_id"), "ablation": res, "threshold_sweep": json.loads(sweep.to_json(orient="records")),
           "elapsed_s": round(time.time() - t0)}
    p = out_p / f"ablation_{a.tag}_{doc['provenance']['config_hash']}.json"
    p.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print(f"[{time.time() - t0:5.0f}s] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
