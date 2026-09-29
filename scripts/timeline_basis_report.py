"""Exercise engine.objective / engine.timeline / engine.basis_search on the REAL price cache (read-only).

usage: timeline_basis_report.py [--seed 0] [--tickers 600] [--out state/research/timeline_basis]

Real weekly returns of volatility baskets (trailing-vol ranked, k names, no forecasting model) are cut into 52-week
windows over 2000-2026. On them we report: (1) tier scores by basket type and by era, (2) the dial trained on earlier
windows and gated on later ones, per era, (3) the basis search run with as_of=2012 and then scored out of sample on the
windows after it. This is a PROXY portfolio (weekly close-to-close, equal weight, no costs, survivor-biased universe),
not the trading system: it shows the modules behave sensibly on real return distributions, nothing more.
Memory: a seeded sample of tickers only (float32), well under 1.5 GB. Never writes outside --out."""
import argparse, json, sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, objective as O, timeline as T, basis_search as B, provenance

ERAS = [("2000-2008", 2000, 2008), ("2009-2015", 2009, 2015), ("2016-2026", 2016, 2026)]
WEEKS = 52
TRAIL = 12


def era_of(ts):
    y = pd.Timestamp(ts).year
    return next(n for n, a, b in ERAS if a <= y <= b)


def load_weekly(n_tickers, seed):
    import pyarrow.parquet as pq
    path = K.CACHE / "stocks_close.parquet"
    names = [n for n in pq.ParquetFile(path).schema_arrow.names if n != "Date"]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(names, size=min(n_tickers, len(names)), replace=False))
    px = pd.read_parquet(path, columns=pick).astype("float32")
    px.index = pd.DatetimeIndex(px.index)
    wk = px.resample("W-FRI").last()
    ret = wk.pct_change(fill_method=None)
    bad = int((ret > 3.0).sum().sum())                     # +300% in one week: treat as a data error, drop
    ret = ret.where(ret <= 3.0)
    return wk, ret, bad


class Basket:
    """Weekly volatility basket. Decision at close t uses returns up to t only; it earns week t+1."""
    def __init__(self, wk, ret):
        self.ret = ret.to_numpy(np.float32)
        self.dates = ret.index
        vol = ret.rolling(TRAIL, min_periods=TRAIL).std()
        self.vol = vol.to_numpy(np.float32)
        self.ok = (wk.to_numpy(np.float32) >= 5.0) & np.isfinite(self.vol)   # price floor: no sub-$5 quote noise

    def week(self, t, k, pool_q, rng):
        if t + 1 >= len(self.dates):
            return None
        idx = np.flatnonzero(self.ok[t] & np.isfinite(self.ret[t + 1]))
        if len(idx) < 30:
            return None
        v = self.vol[t, idx]
        pool = idx[v >= np.quantile(v, pool_q)]
        chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
        return float(np.mean(self.ret[t + 1, chosen]))

    def window_weeks(self, start, k, pool_q, seed):
        rng = np.random.default_rng(seed)
        out = [self.week(t, k, pool_q, rng) for t in range(start, start + WEEKS)]
        return np.array([x for x in out if x is not None])


def make_windows(bk):
    starts = list(range(TRAIL + 1, len(bk.dates) - WEEKS - 1, WEEKS))
    return [{"id": f"w{s}", "start": s, "end": bk.dates[s + WEEKS - 1]} for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tickers", type=int, default=600)
    ap.add_argument("--out", default=str(K.STATE / "research" / "timeline_basis"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    wk, ret, bad = load_weekly(a.tickers, a.seed)
    bk = Basket(wk, ret)
    wins = make_windows(bk)
    rep = {"seed": a.seed, "tickers": int(wk.shape[1]), "weeks": int(len(wk)), "windows": len(wins),
           "dropped_weekly_returns_over_300pct": bad, "caveats": ["survivor-biased universe", "close-to-close, no costs",
                                                                  "equal-weight vol basket, no forecasting model"]}

    # 1. tier scores by basket type and era
    rows, labels, table = [], [], {}
    for k in (1, 2, 3, 5, 10, 25):
        for pq_ in (0.5, 0.9):
            r = [O.week_row(bk.window_weeks(w["start"], k, pq_, a.seed + i)) for i, w in enumerate(wins)]
            table[f"k{k}_pool{pq_}"] = {"overall": O.summary(O.evaluate(r)),
                                        "by_era": O.by_group(r, [era_of(w["end"]) for w in wins]),
                                        "mean_week": float(np.mean([x["mean_week"] for x in r])),
                                        "sd_week": float(np.mean([x["sd_week"] for x in r]))}
            rows.append(O.evaluate(r))
    rep["tiers_by_basket"] = table
    rep["tier1_tier2_rank_corr"] = O.rank_correlation_of_tiers(rows)

    # 2. dial: train on the earliest 60% of windows, gate on the rest, then per era on the unseen part
    dial = {}
    for k, pq_ in ((1, 0.9), (3, 0.9), (10, 0.5)):
        dw = [{"id": w["id"], "end": w["end"], "weeks": bk.window_weeks(w["start"], k, pq_, a.seed + i)} for i, w in enumerate(wins)]
        dw = [w for w in dw if len(w["weeks"]) == WEEKS]
        cut = int(len(dw) * 0.6)
        train, test = dw[:cut], dw[cut:]
        params, tr_score = T.train_dial(train, n_draws=30, seed=a.seed)
        gate = T.promotion_gate(train, test, params, seed=a.seed)
        base_rows = [O.week_row(w["weeks"]) for w in test]
        dial_rows = T._rows(test, params)
        lab = [era_of(w["end"]) for w in test]
        dial[f"k{k}_pool{pq_}"] = {
            "n_train": len(train), "n_test": len(test), "gate": gate.summary(), "checks": gate.checks, "numbers": gate.numbers,
            "test_base": O.by_group(base_rows, lab), "test_dial": O.by_group(dial_rows, lab),
            "test_in_band": [O.summary(O.evaluate(base_rows))["in_band"], O.summary(O.evaluate(dial_rows))["in_band"]]}
    rep["dial"] = dial

    # 3. basis search on the cfg space that this proxy understands; as_of=2012, then out of sample after it
    def ev(w, cfg, meta):
        return O.week_row(bk.window_weeks(w["start"], cfg["k"], cfg["pool_q"], zlib_seed(w["id"], a.seed)))
    zlib_seed = lambda wid, s: zlib.crc32(wid.encode()) + s
    cs = {"k": [1, 2, 3, 5, 10, 25], "pool_q": [0.5, 0.7, 0.9, 0.95]}
    inc_cfg, inc_meta = {"k": 25, "pool_q": 0.5}, {"half_life": 6}
    cutoff = pd.Timestamp("2012-12-31")
    res = B.BasisSearch(ev, wins, inc_cfg, inc_meta, B.SearchConfig(seed=a.seed, n_start=24), cs, {}).run(as_of=cutoff)
    later = [w for w in wins if w["end"] > cutoff]
    oos_inc = O.summary(O.evaluate([ev(w, inc_cfg, inc_meta) for w in later]))
    oos_new = O.summary(O.evaluate([ev(w, res.cfg, res.meta) for w in later]))
    rep["basis_search"] = {"record": res.to_record(), "adopted_cfg": res.cfg, "windows_used": res.n_windows,
                           "windows_out_of_sample": len(later), "oos_incumbent": oos_inc, "oos_adopted": oos_new,
                           "note": "meta-parameters are not exercised by this proxy (no adaptation engine in the loop)"}
    rep["provenance"] = provenance.stamp({"tickers": a.tickers}, a.seed)
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    (out / f"report_seed{a.seed}.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = [f"# timeline / objective / basis-search on real weekly baskets (seed {a.seed})", "",
             f"{rep['tickers']} sampled tickers, {rep['windows']} windows of {WEEKS} weeks. Proxy portfolio, see caveats in JSON.", "",
             "## share of weeks in the 5-10% band, by basket and era", "", "| basket | overall | " + " | ".join(n for n, _, _ in ERAS) + " |",
             "|---|---|" + "---|" * len(ERAS)]
    for name, t in table.items():
        lines.append(f"| {name} | {t['overall']['in_band']:.0%} | " + " | ".join(
            f"{t['by_era'].get(n, {}).get('in_band', float('nan')):.0%}" for n, _, _ in ERAS) + " |")
    lines += ["", f"tier1-vs-tier2 rank correlation across baskets: {rep['tier1_tier2_rank_corr']:.2f}", "", "## dial gate (train early, test later)", ""]
    for name, d in dial.items():
        lines.append(f"- {name}: {d['gate']} (in-band share on test, base -> dial: {d['test_in_band'][0]:.0%} -> {d['test_in_band'][1]:.0%})")
    bs = rep["basis_search"]
    lines += ["", "## basis search (as_of 2012)", "", f"- adopted: {res.adopted} ({res.reason}); cfg {res.cfg}",
              f"- out of sample in-band share: incumbent {bs['oos_incumbent']['in_band']:.0%} -> adopted {bs['oos_adopted']['in_band']:.0%}",
              f"- risk: {bs['oos_incumbent']['risk']:+.3f} -> {bs['oos_adopted']['risk']:+.3f}"]
    (out / f"report_seed{a.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {rep['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
