"""Phases 10-11 on the REAL caches (read-only): weekly decisions from the out-of-sample model scores, lessons learned
on window A, firewalled against a disguised rerun of A and an unseen window B, for every consecutive pair of
two-year eras. Small seeded ticker sample so memory stays well under 1.5 GB.

usage: lessons_real.py [--tickers 400] [--seed 0] [--era-years 2] [--tag name]
Writes state/research/lessons/<tag>.json and <tag>.md (provenance-stamped). Never writes to data/cache or state/livesim."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import config as K
from engine.antimemo import report_markdown, run_experiment, tercile_labeller
from engine.lessons import usable_features
from engine.provenance import stamp

FEATS = ["r5", "r20", "mom_12_1", "dist_52wh", "dist_ma50", "vol20", "vol_ratio", "atr_pct", "log_dv", "close_loc",
         "gap_today", "rel_ind20", "days_to_earn", "news5", "ins_buyers30",
         "m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion", "m_spy_r5"]
OOS = ["score", "p_stop", "evidence"]


def _groups(path, cols, start):
    f = pq.ParquetFile(path)
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=cols + ["date", "ticker"]).to_pandas()
        t = t[t.index.get_level_values(0) >= start]
        if len(t):
            yield t


def weekly_dates(dates):
    s = pd.Series(pd.DatetimeIndex(sorted(set(dates))))
    return pd.DatetimeIndex(s.groupby([s.dt.isocalendar().year, s.dt.isocalendar().week]).max().to_numpy())


def load(n_tk, seed, start="2017-01-01"):
    rng = np.random.default_rng(seed)
    oos = pd.concat([g for g in _groups(K.CACHE / "oos_preds.parquet", OOS, pd.Timestamp(start))])
    oos = oos[~oos.index.duplicated()]
    wk = weekly_dates(oos.index.get_level_values(0))
    oos = oos[oos.index.get_level_values(0).isin(wk)]
    cnt = oos.index.get_level_values(1).value_counts()
    pool = sorted(cnt[cnt >= 0.6 * cnt.max()].index)
    keep = set(rng.choice(pool, size=min(n_tk, len(pool)), replace=False))
    oos = oos[oos.index.get_level_values(1).isin(keep)]
    parts = []
    for g in _groups(K.CACHE / "panel.parquet", FEATS, pd.Timestamp(start)):
        g = g[g.index.get_level_values(0).isin(wk) & g.index.get_level_values(1).isin(keep)]
        parts.append(g.astype("float32"))
    X = oos.join(pd.concat(parts), how="inner")
    O = pd.read_parquet(K.CACHE / "stocks_open.parquet", columns=sorted(keep & set(pq.ParquetFile(K.CACHE / "stocks_open.parquet").schema_arrow.names)))
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=list(O.columns))
    fwd = (C.shift(-5) / O.shift(-1) - 1).stack(future_stack=True)      # bought at the next open, sold 5 sessions on
    fwd.index.names = ["date", "ticker"]
    y = fwd.reindex(X.index)
    ok = y.notna() & X["score"].notna()
    return X[ok].sort_index(), y[ok].sort_index(), pool


def score_fn(Xd):
    return Xd["score"].groupby(level=0).rank(pct=True)               # percentile rank: positive, so x factor is a re-weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--era-years", type=int, default=2)
    ap.add_argument("--tag", default="real_v1")
    a = ap.parse_args()
    t0 = time.time()
    X, y, pool = load(a.tickers, a.seed)
    d = X.index.get_level_values(0)
    print(f"rows {len(X):,} weeks {d.nunique()} tickers {X.index.get_level_values(1).nunique()} "
          f"[{d.min().date()} .. {d.max().date()}] {time.time() - t0:.0f}s", flush=True)
    edges = pd.date_range(d.min().normalize().replace(month=1, day=1), d.max(), freq=f"{a.era_years}YS")
    wins = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if ((d >= lo) & (d < hi)).sum() > 500]
    cfg = {"k": 5, "cost": 0.0005, "horizon_days": 8, "n_disguises": 2, "boot": 300, "block": 4, "long_only": True, "null_reps": 3}
    out = {"stamp": stamp(cfg, a.seed), "cfg": cfg, "tickers_sampled": a.tickers, "pairs": []}
    for i in range(len(wins) - 2):
        (alo, ahi), (blo, bhi), (clo, chi) = wins[i], wins[i + 1], wins[i + 2]
        m = lambda lo, hi: (d >= lo) & (d < hi)
        XA, yA, XB, yB = X[m(alo, ahi)], y[m(alo, ahi)], X[m(blo, bhi)], y[m(blo, bhi)]
        XC, yC = X[m(clo, chi)], y[m(clo, chi)]                       # never used for a decision: the honest figure
        label = tercile_labeller(XA["atr_pct"])
        cols, bad = usable_features(XA)
        r = run_experiment(XA, yA, XB, yB, score_fn, cfg=cfg, seed=a.seed + i, type_fn=lambda Z: label(Z["atr_pct"]), XC=XC, yC=yC)
        r.pop("final_book", None)
        r["window_A"], r["window_B"] = [str(alo.date()), str(ahi.date())], [str(blo.date()), str(bhi.date())]
        r["window_C"] = [str(clo.date()), str(chi.date())]
        r["proxy_excluded"] = bad
        out["pairs"].append(r)
        print(f"A {r['window_A'][0]}..  B {r['window_B'][0]}..: lessons {r['lesson_count']} kept {len(r['accepted_lessons'])} "
              f"impA {r['improvement_A'] * 100:+.3f}% impDisA {r['improvement_A_disguised'] * 100:+.3f}% "
              f"impB {r['improvement_B'] * 100:+.3f}% finalB {r['final_improvement_B'] * 100:+.3f}% "
              f"C {r['holdout_C']['mean'] * 100:+.3f}% nullmax {r['null_control']['null_max'] * 100:+.3f}% "
              f"[{time.time() - t0:.0f}s]", flush=True)
    kept = sum(len(p["accepted_lessons"]) for p in out["pairs"])
    total = sum(p["lesson_count"] for p in out["pairs"])
    out["summary"] = {"pairs": len(out["pairs"]), "lessons": total, "kept": kept,
                      "mean_final_B": float(np.mean([p["final_improvement_B"] for p in out["pairs"]])) if out["pairs"] else None,
                      "mean_holdout_C": float(np.mean([p["holdout_C"]["mean"] for p in out["pairs"]])) if out["pairs"] else None,
                      "pairs_beating_null": sum(p["null_control"]["exceeds_null"] for p in out["pairs"]),
                      "mean_memorisation_gap": float(np.mean([p["memorisation_gap"] for p in out["pairs"]])) if out["pairs"] else None,
                      "seconds": round(time.time() - t0)}
    dest = K.STATE / "research" / "lessons"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{a.tag}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    md = [f"# Lessons on real data ({a.tag})", "", f"Summary: {json.dumps(out['summary'])}", ""]
    for p in out["pairs"]:
        md += [report_markdown(p, f"A {p['window_A'][0]}..{p['window_A'][1]} -> B {p['window_B'][0]}..{p['window_B'][1]}"), ""]
    (dest / f"{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(out["summary"]), flush=True)


if __name__ == "__main__":
    main()
