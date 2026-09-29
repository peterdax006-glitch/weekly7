"""Real-cache study for Phases 9, 14 and 18 (B15): does the memory predict, and does learning from missed winners help?

usage: run_b15_memory_adapter.py [--tickers 600] [--seed 0] [--start 2017-01-01] [--controls 20] [--out state/research/memory_adapter]

Reads (read-only) data/cache/oos_preds.parquet (mu_raw), panel.parquet (feature columns + market context) and
stocks_close.parquet (prices), for a seeded sample of tickers and the LAST session of every week. Nothing under
state/livesim is touched. Forward return of decision day d = close at the last session of the NEXT week / close(d) - 1
(a decision at a close fills at the next open, so this is generous to the detector: the honest number is lower; the
comparison against controls is what matters, and both sides get the same generosity).

Part A  missed-winner detector (Phase 14): engine.missed_winners.evaluate on the weekly panels - detector alone,
        detector + base, shuffled detector, future-scrambled, shuffled-winners control - plus top-k curve, calibration,
        coefficient stability, threshold sweep and a per-era breakdown.
Part B  memory (Phase 9): the adapter's 'ic' arms rebuilt on real weeks - each evidence indicator's weekly rank IC with
        the market context of the decision day - then engine.memory_diagnostics: walk-forward skill with a block-bootstrap
        interval, factor ablation, half-life sweep, honest nested tuning, stability by block, calibration slope.
Results go to <out>/ as JSON + CSV with engine.provenance.stamp. RAM: waits until 2.5 GB is free (polls 60 s, gives up
after 20 min), reads one parquet row group at a time, float32."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import config as K, memory as MM, memory_diagnostics as D, missed_winners as W, provenance

FEATS = {"e_ear": "ear", "e_ins_buyers30": "ins_buyers30", "e_ins_officer30": "ins_officer30", "e_dist_52wh": "dist_52wh",
         "e_ind_mom60": "ind_mom60", "e_frog": "frog", "e_mom_12_1": "mom_12_1", "e_r5_nonews": "r5_nonews",
         "e_max20": "max20", "e_skew60": "skew60"}
RAW = ["vol20", "max20", "log_dv", "r5"]                # DET_FEATS that are used as-is
CTXC = MM.CTX


def wait_for_ram(need_gb=2.5, poll=60, give_up=1200):
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available / 1e9 < need_gb:
        if time.time() - t0 > give_up:
            raise SystemExit(f"only {psutil.virtual_memory().available / 1e9:.1f} GB free after {give_up}s; giving up")
        time.sleep(poll)


def weekly_last_sessions(index):
    """The last session of every ISO week present in `index` (a DatetimeIndex of trading days)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.isocalendar().year.values, index.isocalendar().week.values]).max().values)


def load(tickers_n, seed, start):
    closes = pd.read_parquet(K.CACHE / "stocks_close.parquet")
    closes = closes.loc[start:]
    if "Date" in closes.columns:
        closes = closes.drop(columns=["Date"])
    ok = closes.columns[closes.notna().mean() > 0.9]
    rng = np.random.default_rng(seed)
    tk = sorted(rng.choice(np.array(ok), size=min(tickers_n, len(ok)), replace=False).tolist())
    closes = closes[tk].astype("float32")
    wk = weekly_last_sessions(closes.index)
    fwd_dates = dict(zip(wk[:-1], wk[1:]))
    parts = []
    want_cols = sorted(set(FEATS.values()) | set(RAW) | set(CTXC))
    pf = pq.ParquetFile(K.CACHE / "panel.parquet")
    for g in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(g, columns=[c for c in want_cols if c in pf.schema.names] + ["date", "ticker"]).to_pandas()
        df = df[df.index.get_level_values(0).isin(wk) & df.index.get_level_values(1).isin(tk)]
        parts.append(df.astype("float32"))
    panel = pd.concat(parts)
    parts = []
    of = pq.ParquetFile(K.CACHE / "oos_preds.parquet")
    for g in range(of.metadata.num_row_groups):
        df = of.read_row_group(g, columns=["mu_raw", "date", "ticker"]).to_pandas()
        df = df[df.index.get_level_values(0).isin(wk) & df.index.get_level_values(1).isin(tk)]
        parts.append(df.astype("float32"))
    mu = pd.concat(parts)
    panel = panel.join(mu, how="inner")
    return panel, closes, fwd_dates


def build_weeks(panel, closes, fwd_dates, min_names=60):
    weeks = []
    for d in sorted(set(panel.index.get_level_values(0))):
        if d not in fwd_dates:
            continue
        p = panel.xs(d, level=0)
        fwd = (closes.loc[fwd_dates[d]] / closes.loc[d] - 1).reindex(p.index)
        p = p.rename(columns={v: k for k, v in FEATS.items()})
        ok = fwd.notna() & np.isfinite(fwd)
        if ok.sum() < min_names:
            continue
        weeks.append((pd.Timestamp(d), p[ok], fwd[ok].astype("float64")))
    return weeks


def part_a(weeks, controls, seed, out):
    res = W.evaluate(weeks, None, seed=seed, n_controls=controls)
    ctl = res.pop("control")
    res["control"] = {k: v for k, v in ctl.items() if k != "uplifts"}
    pd.Series(ctl["uplifts"], name="control_uplift").to_csv(out / "control_uplifts.csv", index=False)
    (out / "missed_winners_summary.txt").write_text(W.summary_text({**res, "control": ctl}), encoding="utf-8")
    W.topk_curve(weeks, ks=(5, 10, 20, 40)).to_csv(out / "topk_curve.csv", index=False)
    cal = W.detector_calibration(weeks)
    res["calibration"] = {k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in cal.items() if isinstance(v, dict)}
    res["calibration"]["base_rate"] = cal["base_rate"]
    cal["corrected"]["table"].to_csv(out / "calibration_corrected.csv", index=False)
    W.coef_stability(weeks).to_csv(out / "coef_stability.csv", index=False)
    W.threshold_sweep(weeks).to_csv(out / "threshold_sweep.csv", index=False)
    era = W.evaluate_by_group(weeks, lambda d: MM.era_of(d))
    era.to_csv(out / "by_era.csv", index=False)
    yr = W.evaluate_by_group(weeks, lambda d: str(d.year))
    yr.to_csv(out / "by_year.csv", index=False)
    return res


def ic_records(weeks):
    """The adapter's 'ic' arms on real weeks: (arm, week index, market context, rank IC of the indicator vs next week)."""
    rec = []
    for i, (d, p, fwd) in enumerate(weeks):
        ctx = np.array([float(p[c].iloc[0]) if c in p and p[c].notna().any() else 0.0 for c in CTXC])
        ctx = np.nan_to_num(ctx)
        for name in FEATS:
            if name in p and p[name].nunique() >= 3:
                rec.append((("ic", name[2:]), float(i), ctx, W.rank_ic(p[name], fwd)))
    return rec


def part_b(weeks, seed, out):
    rec = ic_records(weeks)
    dates = {float(i): d for i, (d, _, _) in enumerate(weeks)}
    params = {"mem_half_life": 8.0}
    df, s = D.walk_forward_skill(rec, params)
    lo, hi = D.skill_ci(df, seed=seed)
    tab, slope = D.calibration_table(df)
    blocks, share = D.stability_by_block(df)
    result = {"records": len(rec), "weeks": len(weeks), "walk_forward": {**s, "ci": [lo, hi]}, "calibration_slope": slope,
              "positive_block_share": share}
    D.factor_ablation(rec, params).to_csv(out / "memory_factor_ablation.csv", index=False)
    D.sweep(rec, "mem_half_life", [2.0, 4.0, 8.0, 16.0, 52.0, 1e6], params).to_csv(out / "memory_half_life_sweep.csv", index=False)
    D.sweep(rec, "mem_shrink", [0.0, 2.0, 6.0, 12.0, 24.0], params).to_csv(out / "memory_shrink_sweep.csv", index=False)
    D.sweep(rec, "mem_bandwidth", [0.5, 1.0, 1.5, 3.0, 1e6], params).to_csv(out / "memory_bandwidth_sweep.csv", index=False)
    tune = D.nested_tune(rec, {"mem_half_life": [2.0, 4.0, 8.0, 16.0, 52.0, 1e6]}, {}, split=0.6)
    tune["train_table"].to_csv(out / "memory_nested_tune_train.csv", index=False)
    result["nested_tune"] = {k: v for k, v in tune.items() if k != "train_table"}
    blocks.to_csv(out / "memory_block_stability.csv", index=False)
    tab.to_csv(out / "memory_calibration.csv", index=False)
    # per-era: the same walk-forward skill inside each era's stretch of predictions
    df = df.assign(era=[MM.era_of(dates.get(w)) for w in df["week"]])
    rows = []
    for e, g in df.groupby("era"):
        z = float((g["outcome"] ** 2).mean())
        rows.append({"era": e, "n": len(g), "skill": 1 - float(((g["pred"] - g["outcome"]) ** 2).mean()) / z if z > 0 else 0.0})
    pd.DataFrame(rows).to_csv(out / "memory_by_era.csv", index=False)
    # capacity: what forgetting costs on this stream
    result["eviction_regret"] = {str(c): D.eviction_regret(rec, c, params) for c in (50, 200)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--out", default=str(K.STATE / "research" / "memory_adapter"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wait_for_ram()
    t0 = time.time()
    panel, closes, fwd_dates = load(a.tickers, a.seed, a.start)
    weeks = build_weeks(panel, closes, fwd_dates)
    del panel
    print(f"{len(weeks)} weeks, {a.tickers} tickers sampled, loaded in {time.time() - t0:.0f}s", flush=True)
    res = {"stamp": provenance.stamp({"tickers": a.tickers, "start": a.start, "controls": a.controls}, a.seed),
           "weeks": len(weeks), "first": str(weeks[0][0].date()), "last": str(weeks[-1][0].date())}
    res["missed_winners"] = part_a(weeks, a.controls, a.seed, out)
    print("part A done", round(time.time() - t0), "s", flush=True)
    res["memory"] = part_b(weeks, a.seed, out)
    res["seconds"] = round(time.time() - t0)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("done", res["seconds"], "s ->", out, flush=True)


if __name__ == "__main__":
    main()
