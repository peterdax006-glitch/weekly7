"""Phase 27/28 runner: exercises engine.data_sources and engine.isolation on the REAL caches, read-only.

    python scripts/data_live_audit.py [--seed 7] [--sample 150] [--perms 20]

Writes state/research/data_live_safety/report.json and report.md (stamped with engine.provenance.stamp). Memory stays
well under 1.5 GB: only Close is loaded in full; OHLCV are read for a seeded ticker sample. Nothing under data/cache
or state/livesim is modified, nothing is fetched."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import config as K, data_sources as D, isolation as I, provenance  # noqa: E402

OUT = K.STATE / "research" / "data_live_safety"


def read_cols(name, field, cols=None):
    return pd.read_parquet(K.CACHE / f"{name}_{field}.parquet", columns=cols)


def long_from_cache(name, tickers):
    out = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        w = read_cols(name, f.lower())
        out[f] = w[[t for t in tickers if t in w.columns]]
    return D.wide_to_long(out)


def isolation_section():
    a = I.full_audit(ROOT)
    tz = I.ET
    from datetime import datetime, timedelta
    from engine import broker
    ts = [datetime(y, 1, 1, tzinfo=tz) + timedelta(minutes=30 * i) for y in range(2019, 2027) for i in range(0, 366 * 48, 1)
          if (datetime(y, 1, 1, tzinfo=tz) + timedelta(minutes=30 * i)).year == y]
    unsafe_by_year, kinds = {}, {"holiday": 0, "half_day": 0}
    r = I.audit_hours_firewall(broker, ts)
    for s in r["unsafe"]:
        t = datetime.fromisoformat(s)
        unsafe_by_year[t.year] = unsafe_by_year.get(t.year, 0) + 1
        kinds["holiday" if t.date() in I.nyse_holidays(t.year) else "half_day"] += 1
    return {"static": {"ok": a["ok"], "import_violations": a["imports"]["violations"],
                       "known_couplings": a["imports"]["known_couplings"],
                       "state_write_violations": a["state_writes"]["violations"],
                       "known_state_writes": a["state_writes"]["known"], "credential_leaks": a["credentials"],
                       "paper_only": {k: v["ok"] for k, v in a["paper_only"].items()}},
            "hours_firewall": {"half_hours_tested": r["n"], "unsafe_total": len(r["unsafe"]),
                               "unsafe_by_year": unsafe_by_year, "unsafe_by_kind": kinds,
                               "over_blocked": len(r["over_blocked"])}}


def data_section(seed, sample, perms):
    rng = np.random.default_rng(seed)
    res = {}
    close = read_cols("stocks", "close")
    res["stocks_shape"] = list(close.shape)
    res["stocks_range"] = [str(close.index[0].date()), str(close.index[-1].date())]
    as_of = close.index[-1]
    reg = D.DelistedRegistry.load(OUT / "delisted_registry.json")
    stops = D.unexplained_stops(close, as_of, reg)
    res["unexplained_stops"] = {"count": len(stops), "of_tickers": close.shape[1],
                                "by_guess": stops["guess"].value_counts().to_dict(),
                                "by_kind": stops["kind"].value_counts().to_dict(),
                                "by_last_year": stops["last_date"].dt.year.value_counts().sort_index().to_dict(),
                                "sample": stops.head(8).assign(last_date=stops["last_date"].dt.strftime("%Y-%m-%d"))
                                .to_dict("records")}
    stops.assign(last_date=stops["last_date"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "registry_candidates.csv", index=False)
    tick = sorted(rng.choice(close.columns, min(sample, close.shape[1]), replace=False))
    res["sample_tickers"] = len(tick)
    census_parts, splits, stale, wrn = [], [], [], []
    for name in ("stocks", "stocks_pre2000"):
        try:
            long = long_from_cache(name, tick if name == "stocks" else
                                   sorted(rng.choice(read_cols(name, "close").columns, sample, replace=False)))
        except (FileNotFoundError, ValueError) as e:
            res[f"{name}_error"] = str(e)
            continue
        census_parts.append(D.defect_census(long).assign(cache=name))
        clean, rep = D.validate_prices(long, name, as_of, fetched_at="cache")
        res[f"validate_{name}"] = {k: v for k, v in rep.as_dict().items() if k != "warnings"} | \
                                  {"n_warnings": len(rep.warnings), "warning_kinds": _kinds(rep.warnings)}
        w_c, w_v = D.to_wide(clean, "close"), D.to_wide(clean, "volume")
        sp = D.detect_splits(w_c, w_v)
        sp["cache"] = name
        splits.append(sp)
        sr = D.stale_runs(clean, min_run=10)
        sr["cache"] = name
        stale.append(sr)
        cov = D.coverage(clean)
        res[f"coverage_{name}"] = {"tickers": len(cov), "median_missing_share": float(cov["missing_share"].median()),
                                   "p95_missing_share": float(cov["missing_share"].quantile(0.95)),
                                   "worst": cov["missing_share"].nlargest(3).round(4).to_dict()}
    if census_parts:
        c = pd.concat(census_parts)
        res["defect_census"] = json.loads(c.reset_index().to_json(orient="records"))
    sp = pd.concat(splits) if splits else pd.DataFrame()
    res["split_suspects"] = {"total": len(sp), "volume_supported": int(sp["supported"].sum()) if len(sp) else 0,
                             "by_era": D.era_of(sp["date"]).value_counts().to_dict() if len(sp) else {}}
    st = pd.concat(stale) if stale else pd.DataFrame()
    res["stale_runs"] = {"total": len(st), "by_kind": st["kind"].value_counts().to_dict() if len(st) else {},
                         "longest": int(st["length"].max()) if len(st) else 0}
    # market + sector ETFs: full validation, per-ETF inception, and incremental information of sector-relative momentum
    mk = {f: read_cols("market", f.lower()) for f in ("Open", "High", "Low", "Close", "Volume")}
    mlong = D.wide_to_long(mk)
    etfs = mlong[mlong["ticker"].isin(D.SECTOR_ETFS)]
    eclean, erep = D.validate_prices(etfs, "sector_etf", as_of, "cache", first_date=D.ETF_FIRST_DATE)
    res["sector_etf"] = {"validate": {k: v for k, v in erep.as_dict().items() if k != "warnings"},
                         "warnings": erep.warnings[:6],
                         "first_dates": {t: str(g["date"].min().date()) for t, g in eclean.groupby("ticker")},
                         "coverage_median_missing": float(D.coverage(eclean)["missing_share"].median())}
    res["incremental_information"] = info_value(close, eclean, tick, seed, perms)
    return res, stops


def _kinds(ws):
    out = {}
    for w in ws:
        k = w.split(":", 1)[1].strip().split(" ", 2)[1:] if ":" in w else [w]
        key = "gaps" if "gaps" in w else "moves" if "moves" in w else "holiday" if "holiday" in w else "other"
        out[key] = out.get(key, 0) + 1
    return out


def info_value(close, etf_long, tick, seed, perms):
    """Does 'stock 5d return minus its point-in-time best-fit sector ETF 5d return' add rank information beyond
    [r5, r20, vol20] for the next 5 days? ETF assignment uses only the prior 252 sessions (no look-ahead)."""
    cw = D.to_wide(etf_long)
    px = close[[t for t in tick if t in close.columns]]
    px = px.loc["2016-01-01":]
    cw = cw.reindex(px.index).ffill()
    ret = px.pct_change()
    er = cw.pct_change()
    r5, r20 = px.pct_change(5), px.pct_change(20)
    rel = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    e5 = cw.pct_change(5)
    bestetf = {}
    for i in range(252, len(px), 21):                       # re-assign monthly from trailing 252 sessions only
        win = slice(i - 252, i)
        corr = ret.iloc[win].apply(lambda s: er.iloc[win].corrwith(s))
        bestetf.update({(i, t): corr[t].idxmax() for t in corr.columns if corr[t].notna().any()})
        for t in corr.columns:
            if corr[t].notna().any():
                rel.iloc[i:i + 21, rel.columns.get_loc(t)] = (r5[t].iloc[i:i + 21] - e5[corr[t].idxmax()].iloc[i:i + 21]).values
    fwd = px.pct_change(5).shift(-5)
    vol20 = ret.rolling(20).std()
    def stack(x, n):
        return x.stack().rename(n)
    base = pd.concat([stack(r5, "r5"), stack(r20, "r20"), stack(vol20, "vol20")], axis=1)
    new = stack(rel, "rel_sector5").to_frame()
    y = stack(fwd, "y")
    idx = base.index.intersection(new.index).intersection(y.index)
    base, new, y = base.loc[idx].dropna(), new.loc[idx].dropna(), y.loc[idx].dropna()
    idx = base.index.intersection(new.index).intersection(y.index)
    r = D.incremental_information(base.loc[idx], new.loc[idx], y.loc[idx], seed=seed, n_perm=perms)
    r["rows"] = len(idx)
    r["assignments"] = len(bestetf)
    return r


def render(rep):
    L = ["# Data expansion and live safety audit (Phase 27/28)", "", f"Seed {rep['seed']}, {rep['minutes']:.1f} min. "
         f"Code hash `{rep['stamp']['code_hash']}`.", ""]
    s = rep["isolation"]["static"]
    L += ["## Isolation and paper-only", f"- static audit ok: **{s['ok']}**",
          f"- import violations: {len(s['import_violations'])} -> {[(v['module'], v['kind']) for v in s['import_violations']]}",
          f"- known couplings (research reads engine.live constants): {sorted({c['module'] for c in s['known_couplings']})}",
          f"- unreviewed state writes: {len(s['state_write_violations'])}; reviewed: {[(k['module'], k['file']) for k in s['known_state_writes']]}",
          f"- credential reads outside live side: {len(s['credential_leaks'])}; paper-only clients: {s['paper_only']}", ""]
    h = rep["isolation"]["hours_firewall"]
    L += ["## Trading-hours firewall (engine.broker.regular_hours vs NYSE calendar)",
          f"- half-hours tested: {h['half_hours_tested']}; broker allows while market closed: **{h['unsafe_total']}** "
          f"({h['unsafe_by_kind']}); by year {h['unsafe_by_year']}; over-blocked {h['over_blocked']}", ""]
    d = rep["data"]
    L += ["## Data", f"- stocks cache {d['stocks_shape']} {d['stocks_range']}; sample {d['sample_tickers']} tickers",
          f"- unexplained stops (registry candidates): {d['unexplained_stops']['count']} of {d['unexplained_stops']['of_tickers']}; "
          f"{d['unexplained_stops']['by_guess']} {d['unexplained_stops']['by_kind']}",
          f"- split suspects (sample): {d['split_suspects']}", f"- stale runs >=10: {d['stale_runs']}",
          f"- sector ETFs: {d['sector_etf']['validate']['ok']}, first dates {d['sector_etf']['first_dates']}", "",
          "### Defect census per era (raw rows)", ""]
    if "defect_census" in d:
        c = pd.DataFrame(d["defect_census"])
        L += ["```", c.to_string(index=False), "```"]
    ii = d["incremental_information"]
    L += ["", "### Incremental information: sector-relative 5d return", f"- {ii}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--perms", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = {"seed": a.seed, "stamp": provenance.stamp({"sample": a.sample, "perms": a.perms}, a.seed)}
    rep["isolation"] = isolation_section()
    rep["data"], _ = data_section(a.seed, a.sample, a.perms)
    rep["minutes"] = (time.time() - t0) / 60
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (OUT / "report.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))


if __name__ == "__main__":
    main()
