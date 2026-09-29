"""Bible PHASE 12/13 real-data study: does the direction engine's calibration and >=80% gate hold up walk-forward?

Read-only on data/cache (a seeded sample of tickers is read column-wise; peak memory is logged and capped).
Population: weekly decision dates (week-ends); a row is a MOVER when the movers label fires (canon C23/C33: entry at the
next open, +/-10% touched within 5 sessions - same definition as scripts/movers.py `labels`, asserted equal in
tests). Target: direction of the move = sign of the 5-session close-to-entry return.
Inputs are simple, fully point-in-time stand-ins for the real sources (the real pattern/model/analog wiring is not
attached): pattern = 5d reversal rank, model = 20d momentum rank, evidence = gap rank, analog = the stock's own
previously RESOLVED mover history (fraction up). Pattern/model/evidence are oriented and neutralized by a per-type
TrustTable refit every January on earlier data only (a type without reliable evidence contributes 0).
Walk-forward: for each test year Y the engine is fitted with now = Jan 1 Y (rows whose 10-day label window had not
closed are dropped), then scored on year Y's movers. Alternative calibrations/gates are compared on the same rows.
Caveats stated in the output: survivorship (today's tickers), realised movers rather than predicted movers, stand-in
inputs.

usage: direction_study.py [--seed 7] [--n-tickers 500] [--first-year 2005] [--last-year 2025] [--out DIR]"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import config as K, provenance, direction as D, direction_calib as C, direction_ablate as A
from engine import trust as T, trust_store as S

H, MOVE = 5, 0.10
HORIZON_DAYS = 10                    # calendar days a 5-session label window can span (holidays)
MEM_CAP_MB = 1400
CACHE = K.DATA / "cache"
SIGNALS = ["pattern", "model", "evidence"]


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def check_mem(tag):
    m = rss_mb()
    print(f"[mem] {tag}: {m:.0f} MB", flush=True)
    if m == m and m > MEM_CAP_MB:
        raise MemoryError(f"{tag}: {m:.0f} MB exceeds cap {MEM_CAP_MB}")


# ---- data ---------------------------------------------------------------------------------------------------------
def sample_tickers(n, seed, min_rows=750):
    """Seeded sample among tickers with enough history. Reads one column at a time so memory stays small."""
    cols = [c for c in pq.ParquetFile(CACHE / "stocks_close.parquet").schema.names if c != "Date"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cols))
    out = []
    for i in order:
        c = cols[i]
        s = pq.read_table(CACHE / "stocks_close.parquet", columns=[c]).column(0).to_pandas()
        if s.notna().sum() >= min_rows:
            out.append(c)
        if len(out) >= n:
            break
    return sorted(out)


def load_bars(tickers):
    bars = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d = pd.read_parquet(CACHE / f"stocks_{f.lower()}.parquet", columns=tickers)
        d.index = pd.DatetimeIndex(d.index)
        bars[f] = d.astype("float64")
    return bars


def labels(stocks):
    """Identical to scripts/movers.py labels (kept in sync by tests/test_direction_study.py)."""
    C_, Hh, L, O = stocks["Close"], stocks["High"], stocks["Low"], stocks["Open"]
    entry = O.shift(-1)
    hi = pd.concat([Hh.shift(-k) for k in range(1, H + 1)]).groupby(level=0).max()
    lo = pd.concat([L.shift(-k) for k in range(1, H + 1)]).groupby(level=0).min()
    return hi / entry - 1, lo / entry - 1, C_.shift(-H) / entry - 1


def week_ends(idx):
    return [d for i, d in enumerate(idx[:-1]) if idx[i + 1].isocalendar().week != d.isocalendar().week]


def _rank(df):
    return (df.rank(axis=1, pct=True) - 0.5) * 2


def build_panel(bars, sic: pd.DataFrame, min_price=2.0, min_dv=2e5):
    """Weekly panel: index (date, ticker); columns pattern/model/evidence/analog (signed ranks / prob), type info
    (sic, size, vol, trend, theme_mom, attention), y_ret (5d close-from-entry return), up (direction), mover (bool).
    Every feature at date t uses bars <= t only; labels look forward by construction and are only ever read after
    `now` by the fitters (they drop rows whose window had not closed)."""
    Cl, O, V = bars["Close"], bars["Open"], bars["Volume"]
    up_t, dn_t, ret = labels(bars)
    ret1 = Cl.pct_change(fill_method=None)
    r5, r20 = Cl / Cl.shift(5) - 1, Cl / Cl.shift(20) - 1
    gap = O / Cl.shift(1) - 1
    dv = (Cl * V).rolling(20, min_periods=15).mean()
    vol20 = ret1.rolling(20, min_periods=15).std()
    surge = (V / V.rolling(60, min_periods=40).mean())
    touch = ((up_t >= MOVE) | (dn_t <= -MOVE)) & ret.notna() & (ret.abs() < 2.0)
    was_up = (touch & (ret > 0)).astype("float64")
    was_dn = (touch & (ret <= 0)).astype("float64")
    # own resolved mover history: an event at s is known once its 5-session window closed -> shift 6 sessions
    cu, cd = was_up.cumsum().shift(H + 1), was_dn.cumsum().shift(H + 1)
    n_ev = cu + cd
    analog = ((cu + 1) / (n_ev + 2)).where(n_ev >= 3)
    wk = pd.DatetimeIndex(week_ends(Cl.index))
    ok = (Cl >= min_price) & (dv >= min_dv) & r5.notna() & r20.notna() & gap.notna() & vol20.notna()
    div = sic.set_index("ticker")["sic"].map(T.sic_division).reindex(Cl.columns).fillna("NA")
    theme = r20.T.groupby(div.values).transform("mean").T
    P = {"pattern": -_rank(r5.where(ok)), "model": _rank(r20.where(ok)), "evidence": _rank(gap.where(ok)),
         "analog": analog.where(ok), "size": np.log(dv.where(ok)), "vol": vol20.where(ok), "trend": r20.where(ok),
         "theme_mom": theme.where(ok), "attention": surge.where(ok), "y_ret": ret.where(ok),
         "mover": touch.where(ok)}
    P = {k: v.loc[wk] for k, v in P.items()}
    df = pd.DataFrame({k: v.stack(future_stack=True) for k, v in P.items()})
    df.index.names = ["date", "ticker"]
    df = df[df["y_ret"].notna() & df["pattern"].notna()]
    df["mover"] = df["mover"].fillna(False).astype(bool)
    df["up"] = (df["y_ret"] > 0).astype(float)
    df["sic"] = pd.Series(div).reindex(df.index.get_level_values(1)).values
    return df


# ---- trust ---------------------------------------------------------------------------------------------------------
INFO = ["sic", "size", "vol", "trend", "theme_mom", "attention"]


def yearly_trust(panel, first_year, last_year, store=None, min_train_years=1):
    """For each January fit a TrustTable on earlier rows only; orient/neutralize that year's signals with it. Returns
    the panel with oriented pattern/model/evidence plus `trust` (1 if the pattern is reliable for the row's type)."""
    out = panel.copy()
    out["trust"] = 0.0
    dates = out.index.get_level_values(0)
    tabs = {}
    for Y in range(panel.index.get_level_values(0).min().year + min_train_years, last_year + 1):
        now = pd.Timestamp(f"{Y}-01-01")
        past = panel[dates < now]
        tt = T.TrustTable(min_weeks=52, min_obs=400, half_life_weeks=260, horizon_days=HORIZON_DAYS)
        tt.fit(past[SIGNALS], past["y_ret"], past[INFO], now)
        tabs[Y] = tt
        if store is not None and not tt.table.empty:
            try:
                store.save(tt, note=f"January {Y}")
            except S.StoreError:
                pass
        rows = (dates >= now) & (dates < pd.Timestamp(f"{Y + 1}-01-01"))
        yr = out[rows]
        for d, g in yr.groupby(level=0):
            info_day = g.droplevel(0)[INFO]
            W = tt.weights(info_day, SIGNALS) if not tt.table.empty else pd.DataFrame(0.0, index=info_day.index, columns=SIGNALS)
            sgn = np.sign(W.values) * (W.values != 0)
            idx = out.index.get_indexer(g.index)
            for j, c in enumerate(SIGNALS):
                out.iloc[idx, out.columns.get_loc(c)] = g[c].values * sgn[:, j]
            out.iloc[idx, out.columns.get_loc("trust")] = (W["pattern"].values != 0).astype(float)
        print(f"[trust] {Y}: {int(tt.table['reliable'].sum()) if len(tt.table) else 0}/{len(tt.table)} reliable cells", flush=True)
    # years before the first fitted table have no trust: their rows stay trust=0 and signals are zeroed (neutral)
    first_fit = panel.index.get_level_values(0).min().year + min_train_years
    early = dates < pd.Timestamp(f"{first_fit}-01-01")
    for c in SIGNALS:
        out.loc[early, c] = 0.0
    return out, tabs


# ---- walk-forward --------------------------------------------------------------------------------------------------
def inputs(panel):
    return D.build_inputs(pattern=panel["pattern"], analog=panel["analog"], trust=panel["trust"], model=panel["model"],
                          evidence=panel["evidence"], index=panel.index)


def fold(panel, F, Y, gate=0.80):
    now = pd.Timestamp(f"{Y}-01-01")
    dates = panel.index.get_level_values(0)
    eng = D.DirectionEngine(gate=gate, horizon_days=HORIZON_DAYS, min_rows=400).fit(F, panel["up"], now, panel["mover"])
    te = (dates >= now) & (dates < pd.Timestamp(f"{Y + 1}-01-01")) & panel["mover"].values
    rec = dict(year=Y, n_test=int(te.sum()), engine_open=bool(eng.open), reason=eng.reason,
               n_train_rows=eng.diag.get("n_rows", 0), calibrator=eng.diag.get("calibrator"))
    if eng.stack is None or te.sum() < 20:
        return rec, None
    Fte, yte = F[te], panel["up"][te].values
    dec = eng.decide(Fte)
    p = dec["p_up"].values
    base = float(panel["up"][(dates < now) & panel["mover"].values].mean())
    bets = dec["bet"].values
    hit = int(((dec["side"].values[bets] > 0) == (yte[bets] > 0.5)).sum())
    lo, hi = D.wilson(hit, int(bets.sum()))
    rec.update(brier=D.brier(p, yte), brier_base=D.brier(np.full(len(yte), base), yte), log_loss=D.log_loss(p, yte),
               ece=D.ece(p, yte), ece_floor=D.ece_noise_floor(p), acc=float(((p >= 0.5) == (yte > 0.5)).mean()),
               base_rate_up=float(yte.mean()), n_bet=int(bets.sum()), coverage=float(bets.mean()),
               acc_bet=hit / bets.sum() if bets.any() else float("nan"), acc_bet_lo=lo if bets.any() else float("nan"),
               conf_ge_gate=int((dec["conf"] >= gate).sum()),
               acc_if_gate_ignored=direction_acc_at(dec, yte, gate))
    # alternative routes on identical rows: raw stacker scores of the calibration block vs this year
    d0 = pd.Timestamp(eng.diag["calib_start"])
    d1 = pd.Timestamp(eng.diag["test_start"])
    cal = (dates >= d0) & (dates < d1) & panel["mover"].values & (dates + pd.Timedelta(days=HORIZON_DAYS) <= now)
    alt = C.compare_gates(eng.raw(F[cal]), panel["up"][cal].values, eng.raw(Fte), yte, gate,
                          panel["sic"][cal].values, panel["sic"][te].values)
    alt.insert(0, "year", Y)
    return rec, alt


def direction_acc_at(dec, y, gate):
    """Accuracy among rows whose calibrated confidence reached the gate, ignoring whether the engine was open. Shows
    what was being held back when the gate refused."""
    m = (dec["conf"] >= gate).values
    if not m.any():
        return float("nan")
    return float(((dec["p_up"].values[m] >= 0.5) == (y[m] > 0.5)).mean())


def era_summary(folds: pd.DataFrame, eras):
    rows = []
    for name, (a, b) in eras.items():
        f = folds[(folds["year"] >= a) & (folds["year"] <= b) & folds["brier"].notna()]
        if f.empty:
            continue
        w = f["n_test"]
        nb = f["n_bet"].sum()
        hits = (f["acc_bet"].fillna(0) * f["n_bet"]).sum()
        lo, hi = D.wilson(hits, nb)
        rows.append(dict(era=name, folds=len(f), n=int(w.sum()), brier=np.average(f["brier"], weights=w),
                         brier_base=np.average(f["brier_base"], weights=w), ece=np.average(f["ece"], weights=w),
                         acc=np.average(f["acc"], weights=w), open_folds=int(f["engine_open"].sum()),
                         n_bet=int(nb), coverage=nb / w.sum(), acc_bet=hits / nb if nb else float("nan"),
                         acc_bet_lo=lo if nb else float("nan")))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-tickers", type=int, default=500)
    ap.add_argument("--first-year", type=int, default=2005)
    ap.add_argument("--last-year", type=int, default=2025)
    ap.add_argument("--out", default=str(K.STATE / "research" / "direction"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = sample_tickers(a.n_tickers, a.seed)
    print(f"sampled {len(tick)} tickers (seed {a.seed})", flush=True)
    bars = load_bars(tick)
    check_mem("bars")
    sic = pd.read_parquet(CACHE / "sic.parquet")
    panel = build_panel(bars, sic)
    del bars
    print(f"panel: {len(panel)} weekly rows, {int(panel['mover'].sum())} movers, "
          f"{panel.index.get_level_values(0).min().date()}..{panel.index.get_level_values(0).max().date()}", flush=True)
    check_mem("panel")
    store = S.TrustStore(out / "trust_store")
    for p in store.root.glob("v*.json"):            # a study run owns its store: start clean so versions are this run
        p.unlink()
    panel, tabs = yearly_trust(panel, a.first_year, a.last_year, store)
    check_mem("trust")
    F = inputs(panel)
    recs, alts = [], []
    for Y in range(a.first_year, a.last_year + 1):
        r, alt = fold(panel, F, Y)
        recs.append(r)
        if alt is not None:
            alts.append(alt)
        print(f"[fold {Y}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                                        if k in ("n_test", "engine_open", "brier", "brier_base", "ece", "acc", "n_bet",
                                                 "acc_bet", "acc_if_gate_ignored")), flush=True)
        check_mem(f"fold {Y}")
    folds = pd.DataFrame(recs)
    folds.to_csv(out / "folds.csv", index=False)
    alt = pd.concat(alts) if alts else pd.DataFrame()
    alt.to_csv(out / "alt_routes.csv", index=False)
    eras = {"2005-09": (2005, 2009), "2010-14": (2010, 2014), "2015-19": (2015, 2019), "2020-25": (2020, 2025),
            "all": (a.first_year, a.last_year)}
    eras_df = era_summary(folds, eras)
    eras_df.to_csv(out / "eras.csv", index=False)
    alt_pool = pd.DataFrame()
    if len(alt):
        g = alt.assign(hits=lambda d: d["acc_bet"].fillna(0) * d["n_bet"]).groupby("method")
        alt_pool = g.apply(lambda d: pd.Series(dict(
            n=d["n"].sum(), brier=np.average(d["brier"], weights=d["n"]), ece=np.average(d["ece"], weights=d["n"]),
            n_bet=d["n_bet"].sum(), coverage=d["n_bet"].sum() / d["n"].sum(),
            acc_bet=d["hits"].sum() / d["n_bet"].sum() if d["n_bet"].sum() else np.nan)), include_groups=False)
        alt_pool.to_csv(out / "alt_routes_pooled.csv")
    # ablation on three held-out blocks
    abl = []
    for Y in (2012, 2018, 2024):
        if a.first_year <= Y <= a.last_year:
            r = A.ablate(F, panel["up"], pd.Timestamp(f"{Y}-01-01"), panel["mover"], n_boot=1000, seed=a.seed,
                         horizon_days=HORIZON_DAYS, min_rows=400)
            r.insert(0, "asof_year", Y)
            abl.append(r)
    abl = pd.concat(abl) if abl else pd.DataFrame()
    abl.to_csv(out / "ablation.csv", index=False)
    # trust drift between the last two yearly tables, plus the HTML report
    drift_df = pd.DataFrame()
    vs = store.versions()
    if len(vs) >= 2:
        drift_df = S.drift(store.load(vs[-2][0]), store.load(vs[-1][0]))
        drift_df.to_csv(out / "trust_drift_last.csv", index=False)
    if vs:
        docs = K.ROOT / "docs"
        S.render_html(store.load(), docs / "trust_report_direction.html", drift_df,
                      title=f"Per-type trust, study seed {a.seed}")
    summary = dict(stamp=provenance.stamp(vars(a), a.seed), args=vars(a), n_tickers=len(tick), n_rows=len(panel),
                   n_movers=int(panel["mover"].sum()), minutes=round((time.time() - t0) / 60, 1),
                   peak_mb=rss_mb(), eras=json.loads(eras_df.to_json(orient="records")),
                   alt_routes_pooled=json.loads(alt_pool.reset_index().to_json(orient="records")) if len(alt_pool) else [],
                   caveats=["survivorship: sample drawn from today's tickers", "realised movers, not predicted movers",
                            "inputs are simple point-in-time stand-ins, not the production pattern/model/analog sources"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("\n=== ERAS ===\n" + eras_df.round(4).to_string(index=False))
    if len(alt_pool):
        print("\n=== ALTERNATIVE ROUTES (pooled, same rows) ===\n" + alt_pool.round(4).to_string())
    if len(abl):
        print("\n=== ABLATION ===\n" + abl.round(4).to_string(index=False))
    print(f"\ndone in {summary['minutes']} min, peak {summary['peak_mb']:.0f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
