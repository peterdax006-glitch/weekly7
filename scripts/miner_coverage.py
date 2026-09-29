"""Bible Phase 3.1 coverage audit: are all 54 engine features, all 25 candle signals and the market context really
candidates of the pattern miner?  Uses the REAL names - the columns of data/cache/panel.parquet (schema only) and the
signals engine.candles.build produces on a tiny synthetic bar set - never a hand-typed list alone.

    .venv/Scripts/python scripts/miner_coverage.py [--seeds 20] [--occupancy-groups 6] [--out DIR]

Writes coverage.json, coverage_table.csv and reach_by_seed.csv to state/research/miner_coverage/ with provenance.
Exit code 1 if the universe disagrees with the real sources or any feature is unreachable; the audit is a gate, not a
report. Memory: names only, plus (optionally) a few parquet row groups for the quintile occupancy check."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import candidates as C            # noqa: E402
from engine import candles                    # noqa: E402
from engine import config as K                # noqa: E402
from engine import provenance                 # noqa: E402

BLUEPRINT_LOW, BLUEPRINT_HIGH = 4300, 4400


def panel_columns():
    import pyarrow.parquet as pq
    return [c for c in pq.ParquetFile(K.CACHE / "panel.parquet").schema.names]


def candle_names():
    """Signal names from the real builder run on 160 synthetic sessions (names do not depend on the data)."""
    rng = np.random.default_rng(0)
    n, k = 160, 4
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
    o = c * np.exp(rng.normal(0, 0.01, (n, k)))
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    cols = [f"T{i}" for i in range(k)]
    bars = {nm: pd.DataFrame(v, index=idx, columns=cols) for nm, v in
            (("Open", o), ("High", h), ("Low", l), ("Close", c))}
    return sorted(candles.build(bars))


def reach_by_seed(uni, seeds):
    """Does every feature appear in >=1 pair for every seed, and how thin is the thinnest feature?"""
    rows = []
    for s in range(seeds):
        g = C.enumerate_static(uni, seed=s)
        rep = C.coverage_audit(uni, g.set)
        rows.append({"seed": s, "n_candidates": len(g.set), "min_pairs_per_feature": rep["min_pairs_per_feature"],
                     "median_pairs_per_feature": rep["median_pairs_per_feature"], "ok": rep["ok"],
                     "digest": rep["digest"][:12], **{f"n_{k}": v for k, v in rep["candidate_counts"].items()}})
    return pd.DataFrame(rows)


def occupancy(uni, groups):
    """Quintile occupancy on a few real row groups (whole dates only). Flags features that cannot fill 5 levels."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(K.CACHE / "panel.parquet")
    pick = sorted({int(x) for x in np.linspace(0, pf.num_row_groups - 1, min(groups, pf.num_row_groups))})
    cols = [c for c in uni.stock_features + uni.context if c in pf.schema.names]
    df = pd.concat([pf.read_row_group(i, columns=cols + ["date", "ticker"]).to_pandas() for i in pick])
    if "date" in df.columns:
        df = df.set_index(["date", "ticker"])
    df = df.astype("float32")
    d = df.index.get_level_values(0)
    df = df[(d > d.min()) & (d < d.max())]                 # the first and last date of a slice may be partial
    qm = C.quantise(df, uni.restrict(df.columns), min_history=20)
    return qm.occupancy(), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--occupancy-groups", type=int, default=0, help="0 = skip the real-data occupancy check")
    ap.add_argument("--out", default=str(K.STATE / "research" / "miner_coverage"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pcols = panel_columns()
    cnames = candle_names()
    uni = C.universe_from_columns(pcols + cnames, cnames)   # the panel has no candle columns; candles.build adds them
    # today's miner: every non-m_ column it is handed; the panel has no candle columns, algo_test merges them in
    miner_panel_only = [c for c in pcols if c not in ("date", "ticker") and not c.startswith("m_")]
    miner_with_candles = miner_panel_only + cnames
    g = C.enumerate_static(uni, seed=7)
    rep = C.coverage_audit(uni, g.set, pcols, cnames, miner_with_candles,
                           expected={"engine": 46, "context": 8, "candle": 25})
    plan = C.plan_size(uni)
    cur_feats = len(miner_panel_only)
    cur_singles, cur_total = cur_feats * 5, cur_feats * 5 + 4000 + 600
    by_seed = reach_by_seed(uni, a.seeds)

    summary = {
        "universe": rep["sizes"], "engine_plus_context": rep["sizes"]["engine"] + rep["sizes"]["context"],
        "panel_columns": len(pcols) - 2, "candle_signals_built": len(cnames),
        "plan": plan, "blueprint_range": [BLUEPRINT_LOW, BLUEPRINT_HIGH],
        "plan_within_blueprint_range": BLUEPRINT_LOW <= plan["total"] <= BLUEPRINT_HIGH,
        "enumerated_seed7": {"n": rep["n_candidates"], **rep["candidate_counts"], "digest": rep["digest"]},
        "miner_today": {"features_panel_only": cur_feats, "singles": cur_singles, "max_total_incl_pairs_unless": cur_total,
                        "candle_signals_reached_from_panel_alone": 0,
                        "context_columns_as_candidates": 0,
                        "context_columns_used_only_for_weights_and_scope": [c for c in pcols if c.startswith("m_") and c in
                                                                              ("m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion")]},
        "unreached_by_miner_today_panel_only": sorted(set(uni.all) - set(miner_panel_only)),
        "reach_all_seeds_ok": bool(by_seed["ok"].all()), "seeds": a.seeds,
        "min_pairs_per_feature_over_seeds": int(by_seed["min_pairs_per_feature"].min()),
        "problems": rep["problems"], "ok": rep["ok"] and bool(by_seed["ok"].all()),
    }
    if a.occupancy_groups:
        occ, nrows = occupancy(uni, a.occupancy_groups)
        thin = occ[occ["levels_occupied"] < 5]
        summary["occupancy"] = {"rows": nrows, "features_below_5_levels": thin["feature"].tolist(),
                                "levels_occupied": {r.feature: int(r.levels_occupied) for r in thin.itertuples()}}
        occ.to_csv(out / "occupancy.csv", index=False)
    rep["table"].to_csv(out / "coverage_table.csv", index=False)
    by_seed.to_csv(out / "reach_by_seed.csv", index=False)
    (out / "coverage.json").write_text(json.dumps({"provenance": provenance.stamp({"seeds": a.seeds}, 7),
                                                   "summary": summary}, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1, default=str))
    if summary["problems"]:
        print("PROBLEMS:", *summary["problems"], sep="\n  ")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
