"""Conditional gap-risk model (engine/gaprisk.py) on the REAL caches, read-only: walk-forward coverage per era versus
the old per-ticker model, event-week diagnostics, and the loss-cap analysis for "no position worse than -20%".
Outputs -> state/research/gaprisk/ (summary.json is provenance-stamped). No network; events.parquet read only.
    python scripts/run_gaprisk.py [--n-tickers 1500] [--per-week 25]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engine import config as K  # noqa: E402
from engine import gaprisk as G  # noqa: E402
from engine import stops as S  # noqa: E402
from engine.provenance import stamp  # noqa: E402
from run_exits_stops import build_paths, clean, load_fields, sample_tickers  # noqa: E402

OUT = K.STATE / "research" / "gaprisk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickers", type=int, default=1500)
    ap.add_argument("--per-week", type=int, default=25)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-train-weeks", type=int, default=104)
    ap.add_argument("--block-weeks", type=int, default=26)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tickers = sample_tickers(a.n_tickers, a.seed)
    p, skipped = build_paths(load_fields(tickers, a.start, a.end), a.per_week)
    p, dq = clean(p)
    ev = pd.read_parquet(K.CACHE / "events.parquet")
    ev = ev[ev.ticker.isin(set(tickers)) & (ev.accepted <= pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=10))]
    cal = G.EventCalendar(ev)
    print(f"positions={len(p)} weeks={len(np.unique(p.week))} tickers_with_events={len(cal.by_ticker)} {dq}", flush=True)

    fac = {"old_per_ticker": lambda tr, pr: S.fit_gap_model(tr, q=pr).tail,
           "conditional_evt": lambda tr, pr: (lambda m: (lambda x: m.night_quantile(x, pr)))(G.ConditionalGapModel(cal).fit(tr)),
           "conditional_evt_uncalibrated": lambda tr, pr: (lambda m: (lambda x: m.night_quantile(x, pr)))(
               G.ConditionalGapModel(cal, calibrate=False).fit(tr))}
    kw = dict(min_train_weeks=a.min_train_weeks, block_weeks=a.block_weeks)
    cov = G.walk_forward_coverage(p, fac, **kw)
    cov.to_csv(OUT / "coverage_by_era.csv", index=False)
    summ = G.coverage_summary(cov)
    summ.to_csv(OUT / "coverage_summary.csv", index=False)
    evc = G.walk_forward_coverage(p, fac, label_fn=lambda t: np.where(cal.expected_in_week(t), "event_week", "normal_week"), **kw)
    evc.to_csv(OUT / "coverage_event_vs_normal.csv", index=False)
    hv = G.walk_forward_coverage(p, fac, label_fn=lambda t: np.where(pd.Series(t.atr).groupby(pd.factorize(t.week)[0]).transform("median").values
                                                                   > np.quantile(p.atr, 0.75), "high_vol_week", "normal_vol_week"), **kw)
    hv.to_csv(OUT / "coverage_vol_regime.csv", index=False)

    exp = cal.expected_in_week(p)
    real = cal.realized(p).any(1)
    g = S.adverse_gaps(p)
    big = (g >= 0.10).any(1)
    diag = {"expected_flag_share": float(exp.mean()), "realized_event_week_share": float(real.mean()),
            "expected_given_realized": float(exp[real].mean()) if real.any() else None,
            "big_gap_10pct_positions": int(big.sum()), "big_gap_share_in_expected_event_weeks": float(exp[big].mean()) if big.any() else None,
            "big_gap_rate_event_week": float(big[exp].mean()) if exp.any() else None,
            "big_gap_rate_normal_week": float(big[~exp].mean())}
    print(diag, flush=True)

    lc = G.loss_cap_analysis(p, cal, cap=0.20, stop_k=3.0, **kw)
    pol = G.policy_table(lc, taus=(0.05, 0.03, 0.02, 0.01), target=0.01)
    pol.to_csv(OUT / "loss_cap_policies_by_era.csv", index=False)
    G.calibration_of_p(lc).to_csv(OUT / "loss_cap_model_calibration.csv", index=False)
    req = G.required_avoidance(lc, 0.01)
    req_3 = G.required_avoidance(lc, 0.02)
    lc10 = G.loss_cap_analysis(p, cal, cap=0.10, stop_k=3.0, **kw)
    G.policy_table(lc10, taus=(0.20, 0.10, 0.05), target=0.01).to_csv(OUT / "loss_cap10_policies_by_era.csv", index=False)
    summary = {**stamp({"n_tickers": a.n_tickers, "per_week": a.per_week, "range": [a.start, a.end]}, a.seed), "data_quality": dq,
               "positions": len(p), "coverage_summary": summ.to_dict("records"), "event_diagnostics": diag,
               "loss_cap_20": {"required_avoidance_1pct": req, "required_avoidance_2pct": req_3, "base_breach": float(lc.realized.mean()),
                               "weight_cap_for_3pct_portfolio_hit": G.weight_cap_for_portfolio_hit(0.03, 0.20)},
               "seconds": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print(evc.groupby(["model", "prob", "era"])[["exceed_rate", "ratio"]].mean().to_string())
    print(pol[pol.era == "ALL"].to_string(index=False))
    print("required avoidance (1%):", req, "\nrequired (2%):", req_3, f"\ndone {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
