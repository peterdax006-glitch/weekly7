"""Bible Phases 4-5 on the REAL caches (read-only): walk forward through eras; at each era end retest the bank on all data
that had closed by then, then let the miner learn on the same data (re-testing the bank's own prior) and fold its run into
the bank. Afterwards judge every pattern the bank held at an era end on the NEXT era - which nobody had seen.

Reports, per era: lifecycle state counts, transitions this era, rescope/discard reasons, trust; and, per state and per
pattern kind, whether the pattern's sign persisted out of sample (signed weekly excess return, t, hit rate) against a
control of patterns the miner rejected. Also proves the earlier-windows-only read: every era's snapshot re-read at the
end must be byte-identical to what it was when it was current.

usage: run_pattern_bank.py [--tickers 500] [--seed 7] [--eras 2017-01-01,2019-01-01,...] [--tag name]
Memory: a seeded ticker sample (default 500 of ~5,000) of weekly rows, ~150 MB; well under 1.5 GB."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import config as K
from engine.pattern_bank import PatternBank
from engine.pattern_lifecycle import Panel, parse_key_named
from engine.patterns import PatternMiner
from engine.provenance import stamp

HORIZON = 7


def load(n_tickers, seed):
    schema = pq.ParquetFile(K.CACHE / "panel.parquet").schema_arrow.names
    feats = [c for c in schema if c not in ("date", "ticker") and not c.startswith("ev_activist")]
    tick = pq.read_table(K.CACHE / "panel.parquet", columns=["ticker"]).column("ticker").unique().to_pylist()
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(tick, min(n_tickers, len(tick)), replace=False).tolist())
    X = pq.read_table(K.CACHE / "panel.parquet", columns=feats + ["date", "ticker"],
                      filters=[("ticker", "in", pick)]).to_pandas().sort_index()
    d = X.index.get_level_values(0)
    ud = pd.DatetimeIndex(sorted(d.unique()))
    wk = ud[[i for i in range(len(ud) - 1) if ud[i + 1].isocalendar().week != ud[i].isocalendar().week]]
    X = X[d.isin(wk)].astype("float32")
    O = pd.read_parquet(K.CACHE / "stocks_open.parquet", columns=[t for t in pick if t])
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=[t for t in pick if t])
    fwd = C.shift(-5) / O.shift(-1) - 1                                    # C33: bought at the next open
    y = fwd.stack(future_stack=True).reindex(X.index)
    return X, y, pick


def kind_of(name):
    return {"s": "single", "p": "pair", "u": "unless"}[parse_key_named(name)[0]]


def persistence(records, X, y, t0, t1, params):
    """Signed mean excess weekly return of each pattern on (t0, t1] with the pattern's own sign and scope. One row each."""
    panel = Panel.build(X, y, t1, params)
    win = (panel.dates > t0) & (panel.dates <= t1 - pd.Timedelta(days=HORIZON))
    out = []
    for r in records:
        m = panel.means(r["names"])
        sel = panel.scope_dates(r["scope"])
        if m is None or sel is None:
            continue
        v = m[win & sel & np.isfinite(m)]
        s = float(np.sign(r["effect"])) or 1.0
        if len(v) < 5:
            out.append({"id": r["id"], "state": r["state"], "kind": kind_of(r["name"]), "n": len(v), "signed": np.nan})
            continue
        out.append({"id": r["id"], "state": r["state"], "kind": kind_of(r["name"]), "n": len(v),
                    "signed": float(s * v.mean()), "hit": float((s * v > 0).mean())})
    return pd.DataFrame(out)


def digest(recs):
    return hashlib.sha256(json.dumps(recs, sort_keys=True, default=str).encode()).hexdigest()


def group_stats(df, by):
    if df.empty:
        return []
    rows = []
    for k, g in df.dropna(subset=["signed"]).groupby(by):
        n = len(g)
        sd = g["signed"].std(ddof=1) if n > 1 else np.nan
        rows.append({**(dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,)))),
                     "patterns": n, "mean_signed_weekly": float(g["signed"].mean()),
                     "t": float(g["signed"].mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None,
                     "mean_hit_rate": float(g["hit"].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eras", default="2017-01-01,2019-01-01,2021-01-01,2023-01-01,2025-01-01,2026-09-01")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--max_pairs", type=int, default=1500)
    ap.add_argument("--p_real_min", type=float, default=0.8, help="miner confirmation bar; lower it to stress the lifecycle")
    a = ap.parse_args()
    t0 = time.time()
    out = K.STATE / "research" / "pattern_bank"
    out.mkdir(parents=True, exist_ok=True)
    work = out / f"bank_{a.tag}"
    if work.exists():
        shutil.rmtree(work)
    X, y, pick = load(a.tickers, a.seed)
    print(f"loaded {len(X):,} rows x {X.shape[1]} features, {len(pick)} tickers [{time.time()-t0:.0f}s]", flush=True)
    eras = [pd.Timestamp(e) for e in a.eras.split(",")]
    bank = PatternBank(work)
    d = X.index.get_level_values(0)
    ud = pd.DatetimeIndex(sorted(d.unique()))
    era_rows, snaps, oos_rows = [], {}, []
    for k, T in enumerate(eras):
        step = time.time()
        summ = bank.retest(X, y, T, f"retest_{T.date()}") if bank.head()[0] else {"reviewed": 0}
        n_log = sum(len(r["history"]) for r in bank.read(T + pd.Timedelta(days=1)))
        cut = ud[ud.searchsorted(T) - 2] if ud.searchsorted(T) >= 2 else ud[0]      # label closes before T
        tr = d <= cut
        yv = y[tr]
        yv = yv - yv.groupby(level=0).transform("mean")
        M = PatternMiner({"max_pairs": a.max_pairs, "max_unless": 100, "null_reps": 1, "seed": a.seed,
                          "max_rows": 400_000, "p_real_min": a.p_real_min}).fit(X[tr], yv, now=T, prior=bank.prior_frame(T))
        ing = bank.ingest_miner(M.patterns, T, f"miner_{T.date()}")
        after = T + pd.Timedelta(days=1)
        recs = bank.read(after)
        snaps[k] = (after, digest(recs))
        S = bank.summary(after)
        counts = S["state"].value_counts().to_dict() if len(S) else {}
        new = {"era_end": str(T.date()), "bank_version": ing["version"], "patterns_in_bank": len(recs),
               "retest": summ, "miner": {kk: v for kk, v in M.report.items() if isinstance(v, (int, float))},
               "states": counts, "trusted": int(len(bank.trusted(after))),
               "median_trust_active": float(S[S["state"] == "active"]["trust"].median()) if (S["state"] == "active").any() else None,
               "seconds": round(time.time() - step)}
        disc = [r for r in recs if r["state"] == "discarded"]
        new["discard_reasons"] = pd.Series([str(r["discard_reason"]).split(" (")[0] for r in disc]).value_counts().head(5).to_dict()
        new["rescoped_scopes"] = pd.Series([f"{r['scope']['col']}:{r['scope']['label']}" for r in recs
                                            if r["state"] in ("rescoped", "active") and r["scope"]]).value_counts().head(8).to_dict()
        era_rows.append(new)
        print(json.dumps(new, default=str), flush=True)
        if k + 1 < len(eras):
            P = persistence(recs, X, y, T, eras[k + 1], {})
            P["era"] = str(T.date())
            oos_rows.append(P)
    oos = pd.concat(oos_rows) if oos_rows else pd.DataFrame()
    # earlier-windows-only proof: every snapshot re-read now equals the one taken when it was current
    stable = {str(snaps[k][0].date()): digest(bank.read(snaps[k][0])) == snaps[k][1] for k in snaps}
    res = {"run": stamp({"tag": a.tag, "eras": a.eras, "tickers": a.tickers, "max_pairs": a.max_pairs, "p_real_min": a.p_real_min}, a.seed),
           "rows": int(len(X)), "eras": era_rows, "verify": bank.verify(), "audit": bank.audit(),
           "past_reads_unchanged": stable,
           "oos_by_state": group_stats(oos, ["state"]), "oos_by_kind": group_stats(oos, ["kind"]),
           "oos_by_state_kind": group_stats(oos, ["state", "kind"]), "oos_by_era_state": group_stats(oos, ["era", "state"]),
           "seconds": round(time.time() - t0)}
    (out / f"run_{a.tag}.json").write_text(json.dumps(res, indent=1, default=str))
    if len(oos):
        oos.to_csv(out / f"oos_{a.tag}.csv", index=False)
    lines = [f"# Pattern lifecycle + bank on real caches ({a.tag})", "",
             f"{len(X):,} weekly rows, {len(pick)} tickers, seed {a.seed}. Bank versions: {bank.head()[0]}. "
             f"Integrity ok: {res['verify']['ok']}. Audit problems: {len(res['audit'])}. "
             f"Past reads unchanged: {all(stable.values())}.", "", "## Out-of-sample sign persistence (next era, unseen)", "",
             "| state | patterns | mean signed weekly excess | t | hit rate |", "|---|---|---|---|---|"]
    lines += [f"| {r['state']} | {r['patterns']} | {r['mean_signed_weekly']:+.5f} | {r['t'] and round(r['t'], 2)} | {r['mean_hit_rate']:.3f} |"
              for r in res["oos_by_state"]]
    lines += ["", "## By pattern kind", "", "| kind | patterns | mean signed weekly excess | t |", "|---|---|---|---|"]
    lines += [f"| {r['kind']} | {r['patterns']} | {r['mean_signed_weekly']:+.5f} | {r['t'] and round(r['t'], 2)} |"
              for r in res["oos_by_kind"]]
    (out / f"report_{a.tag}.md").write_text("\n".join(lines) + "\n")
    print("done", json.dumps({k: res[k] for k in ("verify", "audit", "past_reads_unchanged", "oos_by_state")}, default=str))


if __name__ == "__main__":
    main()
