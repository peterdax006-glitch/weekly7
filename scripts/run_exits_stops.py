"""Run the exit learner (Bible Phase 15) and the stop engine (Phase 16) on the REAL price caches, read-only.

Candidates are the volatility finder's proxy: each week, the `per-week` most volatile eligible stocks by trailing ATR
(everything known at the prior close), entered at the first session's open and followed for five sessions. Types are
trailing-vol tercile x 20-day direction. Everything is evaluated walk-forward (each block traded with a policy learned
only from earlier, finished weeks). Weeks with a market holiday (fewer than 5 sessions) are skipped, not padded.

Outputs -> state/research/exits_stops/: summary.json (provenance-stamped), *.csv tables, report.txt.
No network, does not touch state/livesim or write to data/cache. Pattern-failure exits are NOT run here: there is no
per-close pattern score in the caches, so that family is reported as unavailable rather than faked.

    python scripts/run_exits_stops.py [--n-tickers 1200] [--per-week 25] [--start 2005-01-01] [--end 2022-12-31]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import config as K  # noqa: E402
from engine import exits as E  # noqa: E402
from engine import stops as S  # noqa: E402
from engine.provenance import stamp  # noqa: E402

OUT = K.STATE / "research" / "exits_stops"
D = 5


def load_fields(tickers, start, end):
    fr = {}
    for f in ("open", "high", "low", "close", "volume"):
        df = pd.read_parquet(K.CACHE / f"stocks_{f}.parquet", columns=tickers)
        fr[f] = df.loc[start:end].astype("float64")
    return fr


def sample_tickers(n, seed):
    import pyarrow.parquet as pq
    names = [c for c in pq.ParquetFile(K.CACHE / "stocks_close.parquet").schema_arrow.names if c != "Date"]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(names, min(n, len(names)), replace=False).tolist())


def build_paths(fr, per_week, min_price=K.MIN_PRICE, min_dollar_vol=K.MIN_DOLLAR_VOL):
    """Weekly candidate positions from wide frames. Every feature at entry uses data through the prior close only."""
    o, h, l, c, v = (fr[k] for k in ("open", "high", "low", "close", "volume"))
    idx = c.index
    prev = c.shift(1)
    tr = ((h - l) / prev)
    atr20 = tr.rolling(20, min_periods=15).mean().shift(1)                 # known at entry (through prior close)
    vol20 = c.pct_change().rolling(20, min_periods=15).std().shift(1)
    ret20 = (c.shift(1) / c.shift(21) - 1)
    dv20 = (c * v).rolling(20, min_periods=15).median().shift(1)
    iso = idx.isocalendar()
    key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    pos = np.arange(len(idx))
    rows = {k: [] for k in ("o", "h", "l", "c", "prev", "vol", "atr", "week", "end", "kind", "ticker")}
    skipped_weeks = 0
    for _, grp in pd.Series(pos, index=key.values).groupby(level=0):
        g = grp.values
        if len(g) != D or g[0] == 0 or (idx[g[-1]] - idx[g[0]]).days > 6:
            skipped_weeks += 1
            continue
        t0 = g[0]
        ok = (atr20.iloc[t0].notna() & vol20.iloc[t0].notna() & ret20.iloc[t0].notna() & (prev.iloc[t0] >= min_price)
              & (dv20.iloc[t0] >= min_dollar_vol) & o.iloc[g].notna().all() & h.iloc[g].notna().all()
              & l.iloc[g].notna().all() & c.iloc[g].notna().all())
        cand = atr20.iloc[t0][ok].nlargest(per_week).index
        if len(cand) < 6:
            continue
        vt = vol20.iloc[t0][cand]
        bucket = pd.qcut(vt.rank(method="first"), 3, labels=["vlow", "vmid", "vhigh"]).astype(str)
        direction = np.where(ret20.iloc[t0][cand] > 0, "up", "dn")
        for tk, b, dr in zip(cand, bucket, direction):
            rows["o"].append(o[tk].values[g]); rows["h"].append(h[tk].values[g]); rows["l"].append(l[tk].values[g])
            rows["c"].append(c[tk].values[g]); rows["prev"].append(prev[tk].values[t0]); rows["vol"].append(vol20[tk].values[t0])
            rows["atr"].append(atr20[tk].values[t0]); rows["week"].append(idx[g[0]].to_datetime64()); rows["end"].append(idx[g[-1]].to_datetime64())
            rows["kind"].append(f"{b}_{dr}"); rows["ticker"].append(tk)
    p = E.Paths(np.array(rows["o"]), np.array(rows["h"]), np.array(rows["l"]), np.array(rows["c"]), np.array(rows["prev"]),
                np.array(rows["vol"]), np.array(rows["atr"]), np.array(rows["week"]), np.array(rows["end"]),
                np.array(rows["kind"]), np.array(rows["ticker"]))
    return p, skipped_weeks


def clean(p: E.Paths, max_gap=0.60):
    """Drop rows the bar-consistency check rejects or whose overnight gap is implausible (unadjusted split artefact)."""
    n0 = len(p)
    hi, lo = np.maximum(p.o, p.c), np.minimum(p.o, p.c)
    bad = ((p.h < hi - 1e-9) | (p.l > lo + 1e-9) | (p.o <= 0)).any(1)
    gap_bad = (np.abs(p.o / np.concatenate([p.prev_close[:, None], p.c[:, :-1]], 1) - 1) > max_gap).any(1)
    keep = ~(bad | gap_bad)
    q = p.take(np.flatnonzero(keep))
    return q.check(), {"rows": n0, "dropped_inconsistent_bar": int(bad.sum()), "dropped_gap_gt60pct": int((gap_bad & ~bad).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickers", type=int, default=1200)
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
    fr = load_fields(tickers, a.start, a.end)
    p, skipped = build_paths(fr, a.per_week)
    del fr
    p, dq = clean(p)
    print(f"positions={len(p)} weeks={len(np.unique(p.week))} types={sorted(set(p.kind))} skipped_holiday_weeks={skipped} {dq}", flush=True)

    kw = dict(min_train_weeks=a.min_train_weeks, block_weeks=a.block_weeks, seed=a.seed)
    rules = E.default_rules(has_fail=False)
    wf = E.walk_forward(p, rules, **kw)
    rep = wf.report()
    rep.to_csv(OUT / "exit_oos_by_type.csv", index=False)
    E.era_breakdown(wf).to_csv(OUT / "exit_oos_by_era.csv", index=False)
    E.rule_usage(wf).to_csv(OUT / "exit_rule_usage.csv", index=False)
    train0 = p.until(np.unique(p.week)[a.min_train_weeks] - np.timedelta64(1, "D"))
    E.evaluate_rules(train0, rules).to_csv(OUT / "exit_families_insample_first_train.csv", index=False)
    hold = p.take(np.flatnonzero(wf.tested))
    E.evaluate_rules(hold, rules[:1] + [r for r in rules if r.name in ("target_10", "trail_6", "volt_1.0", "t10_trail6")]).to_csv(
        OUT / "exit_families_oos_window_untuned.csv", index=False)
    cs = pd.concat({n: E.cost_sensitivity(hold, s) for n, s in
                    (("target_10", E.ExitSpec(target=0.10)), ("trail_6", E.ExitSpec(trail=0.06, arm=0.03)), ("week_end", E.ExitSpec()))})
    cs.to_csv(OUT / "exit_cost_sensitivity.csv")
    audit = E.audit_no_lookahead(p.take(np.arange(min(3000, len(p)))), E.ExitSpec(target=0.10, trail=0.06, arm=0.03, stop=0.08))

    swf = S.walk_forward_stops(p, **kw)
    S.risk_report(swf).to_csv(OUT / "stop_oos_by_type.csv", index=False)
    S.stop_era_breakdown(swf).to_csv(OUT / "stop_oos_by_era.csv", index=False)
    E.rule_usage(swf).to_csv(OUT / "stop_rule_usage.csv", index=False)
    S.type_gap_table(p).to_csv(OUT / "gap_by_type.csv", index=False)
    S.stop_distance_curve(hold).to_csv(OUT / "stop_distance_curve.csv", index=False)
    gm = S.fit_gap_model(p.until(np.unique(p.week)[a.min_train_weeks] - np.timedelta64(1, "D")))
    test = p.take(np.flatnonzero(swf.tested))
    tails = gm.tail(test)                                            # learned from earlier weeks only
    eff = {f"drop_worst_{int(100 - q * 100)}pct(tail>{np.quantile(tails, q):.3f})": S.filter_effect(
        run_exit_none(test), S.filter_candidates(test, gm, float(np.quantile(tails, q)))) for q in (0.5, 0.75, 0.9, 0.95)}
    dist = np.full(len(test), 0.08)
    S.sizing_effect(test, E.run_exit(test, E.ExitSpec(), stop_dist=dist), dist, tails).to_csv(OUT / "gap_sizing_effect.csv", index=False)
    S.gap_calibration(gm, test).to_csv(OUT / "gap_model_calibration.csv", index=False)
    pd.DataFrame([{"max_gap_tail": k, "kept_share": v["kept_share"], "kept_cat": v["kept"]["cat_rate"], "dropped_cat": v["dropped"]["cat_rate"],
                   "kept_expected_loss": v["kept"]["expected_loss"], "dropped_expected_loss": v["dropped"]["expected_loss"]}
                  for k, v in eff.items()]).to_csv(OUT / "gap_filter_effect.csv", index=False)

    text = "\n\n".join([E.format_report(wf), S.format_risk_report(swf)])
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    summary = {**stamp({"n_tickers": a.n_tickers, "per_week": a.per_week, "range": [a.start, a.end]}, a.seed),
               "data_quality": dq, "skipped_holiday_weeks": skipped, "positions": len(p), "weeks": int(len(np.unique(p.week))),
               "oos_positions_exit": int(wf.tested.sum()), "lookahead_audit_violations": audit,
               "exit_gain_all": E.oos_gain_interval(wf), "exit_report": rep.to_dict("records"),
               "stop_report": S.risk_report(swf).to_dict("records"), "gap_stats_all": S.gap_stats(p),
               "filter_effect_by_gap_tail_quantile": eff, "fail_family": "unavailable: no per-close pattern scores in caches",
               "seconds": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(text)
    print("gap filter (kept share / kept cat / dropped cat): " + "; ".join(
        f"{k}: {v['kept_share']:.0%}/{v['kept']['cat_rate']:.2%}/{v['dropped']['cat_rate']:.2%}" for k, v in eff.items()))
    print(f"audit violations={audit}; done in {time.time() - t0:.0f}s -> {OUT}")


def run_exit_none(p):
    return E.run_exit(p, E.ExitSpec())


if __name__ == "__main__":
    main()
