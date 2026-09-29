"""Bible Phases 26/33/34 on the REAL cache: anti-overfitting battery, reproducibility and ablation, read-only.

Loads a seeded sample of tickers from data/cache (panel.parquet columns + open/close for the labels; peak memory stays
well under 1.5 GB), builds week-end rows with the label the Algorithm uses (next open -> 5 sessions later, minus that
week's average), and reports:
  * the full battery per ERA and, for the reference evaluator, per stock TYPE (volatility tercile);
  * the same battery with the real PatternMiner as the evaluator (--miner; slow);
  * reproducibility of the reference evaluator, of a PatternMiner run, and across PYTHONHASHSEED values;
  * remove-one feature-GROUP ablation and the six-arm feature ablation for three candidate features.
Writes state/research/antioverfit/<tag>/{results.json,summary.txt} stamped with engine.provenance.stamp.

usage: antioverfit_real.py [--tickers 300] [--seed 7] [--reps 6] [--miner] [--tag NAME]"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import ablation as ab
from engine import antioverfit as ao
from engine import config as K
from engine import provenance, repro

GROUPS = {"momentum": ["r5", "r20", "mom_12_1", "dist_52wh", "dist_ma50"],
          "volatility": ["vol20", "atr_pct", "max20", "min20", "skew60"],
          "volume": ["vol_ratio", "log_dv", "vol_surge5"],
          "intraday": ["overnight20", "intraday20", "close_loc", "gap_today"],
          "industry": ["rel_ind20", "ind_mom20"],
          "earnings": ["days_since_earn", "ear", "earn_in_week"]}
CTX = ["m_vix", "m_breadth", "m_spy_ma200", "m_vix_term", "m_dispersion"]
CANDIDATES = ["frog", "range_compress", "ins_buyers30"]
EVAL = partial(ao.wf_evaluate, horizon=1, n_folds=4, min_train_dates=20)      # weekly rows: labels do not overlap


T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def load_sample(n_tickers, seed):
    """Week-end panel rows for a seeded ticker sample, with the demeaned next-open -> +5 session label."""
    names = [n for n in pq.read_schema(K.CACHE / "stocks_close.parquet").names if n != "Date"]
    pan = set(pd.read_parquet(K.CACHE / "panel.parquet", columns=["r5"]).index.get_level_values(1).unique()) \
        if False else None
    rng = np.random.default_rng(seed)
    tk = sorted(rng.choice(names, min(n_tickers, len(names)), replace=False).tolist())
    feats = sorted({c for g in GROUPS.values() for c in g} | set(CANDIDATES))
    X = pd.read_parquet(K.CACHE / "panel.parquet", columns=feats + CTX, filters=[("ticker", "in", tk)])
    d = X.index.get_level_values(0)
    ud = pd.DatetimeIndex(sorted(d.unique()))
    wk = ud[[i for i in range(len(ud) - 1) if ud[i + 1].isocalendar().week != ud[i].isocalendar().week]]
    X = X[d.isin(wk)].astype("float32")
    have = sorted(set(X.index.get_level_values(1)))
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=have)
    O = pd.read_parquet(K.CACHE / "stocks_open.parquet", columns=have)
    fwd = C.shift(-5) / O.shift(-1) - 1
    y = fwd.stack(future_stack=True).reindex(X.index)
    y = y - y.groupby(level=0).transform("mean")
    ok = y.notna().values
    X, y = X[ok], y[ok]
    return X.sort_index(), y.sort_index()


def era_bounds(X, n=4):
    ud = np.sort(X.index.get_level_values(0).unique())
    cuts = np.array_split(np.arange(len(ud)), n)
    return [(pd.Timestamp(ud[c[0]]), pd.Timestamp(ud[c[-1]])) for c in cuts]


def subset(X, y, a, b):
    d = X.index.get_level_values(0)
    m = (d >= a) & (d <= b)
    return X[m], y[m]


def vol_types(X, n=3):
    """Stock type = tercile of the ticker's mean 20-day volatility."""
    v = X["vol20"].groupby(level=1).mean().dropna()
    q = pd.qcut(v.rank(method="first"), n, labels=["quiet", "middle", "wild"][:n])
    return {t: q.index[q == t] for t in q.cat.categories}


def brief(rep):
    return {"overall": rep["overall"], "real_metric": rep["real"]["metric"], "real_t": rep["real"]["t"],
            "status": {v["name"]: v["status"] for v in rep["verdicts"]}, "reasons": {v["name"]: v["reason"] for v in rep["verdicts"]
                                                                                    if v["status"] not in ("pass",)}}


# ------------------------------------------------------------------ light experiment for fresh-process checks
def light_experiment(cfg, seed):
    """Importable by run_in_subprocess: same evaluator on a tiny slice, all seven required artefacts."""
    X, y = load_sample(cfg["tickers"], seed)
    X, y = subset(X, y, pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"]))
    r = EVAL(X, y, pd.Timestamp(cfg["end"]))
    imp = sorted(r["importance"].items(), key=lambda kv: (-kv[1], kv[0]))
    return {"candidate_order": [k for k, _ in imp], "model_config": cfg, "predictions": r["predictions"],
            "trades": r["predictions"].groupby(level=0).idxmax().map(lambda t: t[1]).reset_index(),
            "metrics": {"ic": r["metric"], "t": r["t"]}}


def miner_experiment(cfg, seed):
    from engine.patterns import PatternMiner
    X, y = load_sample(cfg["tickers"], seed)
    X, y = subset(X, y, pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"]))
    ud = np.sort(X.index.get_level_values(0).unique())
    cut = pd.Timestamp(ud[int(len(ud) * 0.8)])
    d = X.index.get_level_values(0)
    M = PatternMiner(cfg.get("miner", {})).fit(X[d < cut], y[d < cut], now=cut)
    last = X.xs(pd.Timestamp(ud[-1]), level=0)
    s = M.score(last)
    top = s.sort_values(ascending=False, kind="mergesort").index[:10].tolist()
    return {"candidate_order": M.patterns["key_named"].tolist() if len(M.patterns) else [], "model_config": M.p,
            "predictions": s, "trades": top, "metrics": M.report}


# ------------------------------------------------------------------ sections
def battery_by_era(X, y, reps, seed):
    out = {}
    for i, (a, b) in enumerate(era_bounds(X)):
        Xe, ye = subset(X, y, a, b)
        rep = ao.run_battery(EVAL, Xe, ye, b, seed=seed + i, n_rep=reps, cfg={"era": [str(a.date()), str(b.date())]})
        out[f"{a.year}-{b.year}"] = {"rows": int(len(Xe)), **brief(rep)}
        log(f"era {a.date()}..{b.date()} rows={len(Xe):,} IC={rep['real']['metric']:.4f} -> {rep['overall']} {rep.get('failed')}")
    return out


def battery_by_type(X, y, reps, seed):
    out = {}
    for name, tks in vol_types(X).items():
        m = X.index.get_level_values(1).isin(tks)
        rep = ao.run_battery(EVAL, X[m], y[m], X.index.get_level_values(0).max(), seed=seed, n_rep=reps,
                             tests=("label_permutation", "randomized_outcomes", "feature_shuffle", "regime_split",
                                    "walk_forward", "date_disguise"), cfg={"type": name})
        out[name] = {"rows": int(m.sum()), "n_tickers": int(len(tks)), **brief(rep)}
        log(f"type {name} rows={m.sum():,} IC={rep['real']['metric']:.4f} -> {rep['overall']}")
    return out


def battery_miner(X, y, seed, reps=3):
    a, b = era_bounds(X)[2]
    Xe, ye = subset(X, y, a, b)
    tk = sorted(set(Xe.index.get_level_values(1)))[:100]
    m = Xe.index.get_level_values(1).isin(tk)
    Xe, ye = Xe[m], ye[m]
    ev = ao.miner_evaluator({"max_pairs": 800, "max_unless": 100, "min_n": 200})
    rep = ao.run_battery(ev, Xe, ye, b, seed=seed, n_rep=reps,
                         tests=("label_permutation", "randomized_outcomes", "date_disguise", "future_scramble"),
                         cfg={"miner": True})
    return {"rows": int(len(Xe)), **brief(rep)}


def reproducibility(seed):
    out = {}
    cfg = {"tickers": 150, "start": "2016-01-01", "end": "2019-12-31"}
    r = repro.run_twice(light_experiment, cfg, seed)
    out["reference_evaluator"] = {"reproducible": r["reproducible"], "problems": r["problems"]}
    log(f"repro reference evaluator: {r['reproducible']} {r['problems']}")
    sub = repro.run_in_subprocess("antioverfit_real:light_experiment", cfg, seed, pythonpath=[Path(__file__).parent])
    out["fresh_process_hashseeds"] = {"reproducible": sub["reproducible"], "differs": sub["differs"]}
    log(f"repro across PYTHONHASHSEED: {sub['reproducible']} {sub['differs']}")
    mc = {"tickers": 120, "start": "2016-01-01", "end": "2020-12-31", "miner": {"max_pairs": 600, "max_unless": 80, "min_n": 200}}
    r2 = repro.run_twice(miner_experiment, mc, seed)
    out["pattern_miner"] = {"reproducible": r2["reproducible"], "problems": r2["problems"],
                            "diagnoses": [{k: d[k] for k in ("name", "first_difference", "causes")} for d in r2["diagnoses"]]}
    log(f"repro PatternMiner: {r2['reproducible']} {r2['problems']}")
    return out


def ablations(X, y, seed):
    now = X.index.get_level_values(0).max()

    def group_eval(active):
        cols = [c for g in active for c in GROUPS[g]]
        return EVAL(X[cols + CTX], y, now)["ic_series"] if cols else pd.Series(dtype=float)
    rem = ab.ablate(group_eval, list(GROUPS), seed=seed, n_boot=1000)
    log("ablation remove-one done")
    out = {"remove_group": rem["table"].round(5).to_dict("records"), "reference_ic": rem["reference_score"]}
    base_cols = [c for g in ("momentum", "volatility", "volume") for c in GROUPS[g]]
    feats = {}
    for f in CANDIDATES:
        rep = ab.feature_ablation(X, y, now, f, baseline_cols=base_cols, evaluate=EVAL, seed=seed, n_rep=3, n_boot=800)
        feats[f] = {"verdict": rep["verdict"], "arm_means": rep["arm_means"], "reasons": rep["reasons"], "notes": rep["notes"],
                    "incremental": rep["checks"]["incremental"]}
        log(f"feature ablation {f}: {rep['verdict']} gain={rep['checks']['incremental']['mean']:+.4f}")
    out["features"] = feats
    return out


def summary(res):
    L = [f"Weekly7 anti-overfitting / reproducibility / ablation on real cache  ({res['sample']})"]
    for era, r in res["eras"].items():
        L.append(f"  era {era:<10} IC {r['real_metric']:+.4f} t {r['real_t']:+.1f}  {r['overall'].upper()}  "
                 + " ".join(f"{k}={v}" for k, v in r["status"].items() if v != "pass"))
    for t, r in res["types"].items():
        L.append(f"  type {t:<7} IC {r['real_metric']:+.4f} t {r['real_t']:+.1f}  {r['overall'].upper()}")
    for k, v in res["repro"].items():
        L.append(f"  repro {k}: {'OK' if v['reproducible'] else 'FAIL ' + str(v.get('problems') or v.get('differs'))}")
    for r in res["ablation"]["remove_group"]:
        L.append(f"  remove {r['component']:<11} {r['verdict']:<11} dIC {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]")
    for f, r in res["ablation"]["features"].items():
        L.append(f"  candidate {f:<15} {r['verdict']}  gain {r['incremental']['mean']:+.4f}  {'; '.join(r['reasons'])}")
    if "miner" in res:
        L.append(f"  miner evaluator: {res['miner']['overall'].upper()} {res['miner']['status']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--miner", action="store_true")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d"))
    a = ap.parse_args()
    X, y = load_sample(a.tickers, a.seed)
    log(f"sample: {len(X):,} week-end rows, {X.index.get_level_values(1).nunique()} tickers, "
        f"{X.index.get_level_values(0).min().date()}..{X.index.get_level_values(0).max().date()}, "
        f"{X.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    res = {"sample": {"rows": int(len(X)), "tickers": a.tickers, "seed": a.seed},
           "stamp": provenance.stamp({"tickers": a.tickers, "reps": a.reps, "miner": a.miner}, a.seed)}
    res["eras"] = battery_by_era(X, y, a.reps, a.seed)
    res["types"] = battery_by_type(X, y, a.reps, a.seed)
    res["ablation"] = ablations(X, y, a.seed)
    res["repro"] = reproducibility(a.seed)
    if a.miner:
        res["miner"] = battery_miner(X, y, a.seed)
        log(f"miner battery: {res['miner']['overall']}")
    out = ROOT / "state" / "research" / "antioverfit" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ao.write_report(res, out / "results.json")
    txt = summary({**res, "sample": f"{len(X):,} rows, {a.tickers} tickers, seed {a.seed}"})
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
