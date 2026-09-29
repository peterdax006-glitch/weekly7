"""Bible Phase 1 - run the point-in-time firewall on the REAL caches (read-only, seeded, small memory).

    .venv/Scripts/python scripts/pit_audit_real.py [--seed 0] [--sample 60] [--skip features,panel]

Sections (each one answers a Phase-1 clause on real data and is independent: one failing does not stop the rest):
  insider    1.1  Form-4 publication lag by year; engine rule (filed + 1 day) vs strict PIT vs a naive trade-date join
  events     1.1  SEC filing acceptance time (ET): after-close share by kind, engine rule (15:30 ET) vs PIT (16:00 ET)
  macro      1.1  FRED series: engine LAG (sessions) vs an approximated real release calendar, per series and decade
  survivor   1.1  is the price panel survivor-biased? attrition per era from first/last bar, vs universe.csv
  panel      1.1  panel / oos_preds dates are sessions, no duplicates, no rows outside a ticker's price life
  entry_gap  1.4  overnight gap a close-entry label ignores (a decision at close can only fill at the next open)
  features   1.3  engine.features.build on a seeded sample: future-invariance at several cuts, with a planted canary
  decisions  1.4  live decisions.jsonl/orders.jsonl timestamps against decide-after-close

Writes state/research/pit/pit_audit.json (stamped with engine.provenance.stamp) and PIT_AUDIT.md. Never writes to
data/cache or state/livesim. Release dates in `macro` are APPROXIMATE (documented rules, not real vintages); they show
where the engine's lag is clearly too short, not proof that it is exact."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import config as K                      # noqa: E402
from engine import pit                              # noqa: E402
from engine.provenance import stamp                 # noqa: E402

OUT = K.STATE / "research" / "pit"
ET = "America/New_York"
ERAS = [("<2013", None, "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"), ("2020+", "2020-01-01", None)]


def _era(dates: pd.Series) -> np.ndarray:
    y = pd.DatetimeIndex(dates).year
    return np.where(y < 2013, "<2013", np.where(y < 2020, "2013-2019", "2020+"))


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:                               # noqa: BLE001
        return float("nan")


def _tab(df: pd.DataFrame) -> dict:
    return json.loads(df.to_json(orient="index")) if len(df) else {}


def sessions_calendar() -> pit.Calendar:
    c = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=["A"])
    return pit.Calendar(c.index)


# ------------------------------------------------------------------------------------------------- insider
def sec_insider(cal: pit.Calendar, rng, **_):
    d = pd.read_parquet(K.CACHE / "insider.parquet", columns=["filed", "tdate", "symbol", "acc", "owner_cik", "shares", "price"])
    d = d.dropna(subset=["filed", "tdate"])
    filed, tdate = pd.DatetimeIndex(d["filed"]).normalize(), pd.DatetimeIndex(d["tdate"]).normalize()
    era, year = _era(d["tdate"]), pd.DatetimeIndex(d["tdate"]).year
    res = {"rows": int(len(d)), "filed_before_trade": int((filed < tdate).sum())}
    res["lag_by_era"] = _tab(pit.lag_profile(tdate, filed, cal, by=era))
    res["lag_by_year"] = _tab(pit.lag_profile(tdate, filed, cal, by=year))
    strict = cal.strict_next(filed)                                # filing time unknown -> known next session
    engine_vis = pd.DatetimeIndex(cal.sessions[np.minimum(cal.sessions.searchsorted(filed + pd.Timedelta(days=1)),
                                                          len(cal.sessions) - 1)])
    ok = np.asarray(filed + pd.Timedelta(days=1) <= cal.sessions[-1])
    res["engine_vs_pit"] = _tab(pit.visibility_gap(engine_vis[ok], strict[ok], by=era[ok], calendar=cal))
    res["naive_tdate_join_vs_pit"] = _tab(pit.visibility_gap(cal.on_or_after(tdate), strict, by=era, calendar=cal))
    # identical transactions repeated within a filing would double-count buyers
    res["exact_duplicate_transactions"] = int(d.duplicated(["acc", "tdate", "symbol", "owner_cik", "shares", "price"]).sum())
    res["trade_year_outside_1990_2026"] = int(((d["tdate"].dt.year < 1990) | (d["tdate"].dt.year > 2026)).sum())
    samp = d.sample(min(100_000, len(d)), random_state=int(rng.integers(1 << 31)))
    st = pit.PITStore(cal).add_insider("ins", samp.rename(columns={"symbol": "ticker", "tdate": "trade_date",
                                                                   "filed": "filing_date"}))
    v = st.validate()
    res["validate_sample"] = {"ok": v.ok, "errors": sorted(v.error_codes())}
    return res


# ------------------------------------------------------------------------------------------------- events
def sec_events(cal: pit.Calendar, rng, **_):
    e = pd.read_parquet(K.CACHE / "events.parquet", columns=["ticker", "kind", "accepted"]).dropna(subset=["accepted"])
    loc = e["accepted"].dt.tz_convert(ET)
    day = pd.DatetimeIndex(loc.dt.tz_localize(None).dt.normalize())
    minute = (loc.dt.hour * 60 + loc.dt.minute).values
    is_sess = cal.is_session(day)
    # strict PIT: known at the close of the acceptance day only if accepted before 16:00 on a session
    pit_vis = pd.DatetimeIndex(np.where(is_sess & (minute < 16 * 60), cal.on_or_after(day).values,
                                        cal.strict_next(day).values))
    sess = cal.sessions
    i = sess.searchsorted(day.values)
    same = (i < len(sess)) & (sess[np.minimum(i, len(sess) - 1)] == day.values)
    i = np.where((minute >= 15 * 60 + 30) & same, i + 1, i)         # engine rule from features._event_calendar
    ok = i < len(sess)
    engine_vis = pd.DatetimeIndex(sess[np.minimum(i, len(sess) - 1)])
    kind = e["kind"].values
    res = {"rows": int(len(e))}
    res["engine_vs_pit_by_kind"] = _tab(pit.visibility_gap(engine_vis[ok], pit_vis[ok], by=kind[ok], calendar=cal))
    res["naive_date_join_vs_pit_by_kind"] = _tab(pit.visibility_gap(cal.on_or_after(day), pit_vis, by=kind, calendar=cal))
    res["naive_date_join_vs_pit_by_era"] = _tab(pit.visibility_gap(cal.on_or_after(day), pit_vis, by=_era(day), calendar=cal))
    g = pd.DataFrame({"kind": kind, "after_close": (minute >= 16 * 60) | ~is_sess, "window_1530_1600":
                      (minute >= 15 * 60 + 30) & (minute < 16 * 60) & is_sess})
    res["after_close_share_by_kind"] = _tab(g.groupby("kind").agg(n=("kind", "size"), after_close=("after_close", "mean"),
                                                                  window_1530_1600=("window_1530_1600", "mean")))
    res["weekend_or_holiday_share"] = float((~is_sess).mean())
    # is the accepted stamp itself plausible? (future-dated acceptance would be a corrupt feed)
    res["accepted_after_last_session_close"] = int((day > cal.sessions[-1]).sum())
    return res


# ------------------------------------------------------------------------------------------------- macro
def _first_friday(ts: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(ts.year, ts.month, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def _last_friday(ts: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(ts.year, ts.month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - 4) % 7)


# approximate real release: (obs -> release date). Monthly obs are dated the 1st of the month they describe.
RELEASE = {
    "UNRATE": lambda o: _first_friday(o + pd.offsets.MonthBegin(1)),
    "CPIAUCSL": lambda o: pd.Timestamp((o + pd.offsets.MonthBegin(1)).year, (o + pd.offsets.MonthBegin(1)).month, 15),
    "INDPRO": lambda o: pd.Timestamp((o + pd.offsets.MonthBegin(1)).year, (o + pd.offsets.MonthBegin(1)).month, 17),
    "UMCSENT": lambda o: _last_friday(o),
}
WEEKLY = {"NFCI": 3, "STLFSI4": 4}          # Friday observation -> Wed / Thu of the next week, in sessions
DAILY = ["DGS10", "DGS2", "VIXCLS", "BAMLH0A0HYM2"]


def macro_lags(cal: pit.Calendar, **_):
    from engine.analogs import LAG
    M = pd.read_parquet(K.CACHE / "macro.parquet")
    res = {"engine_LAG": {k: int(v) for k, v in LAG.items()}, "series": {}}
    start = pd.Timestamp("1990-01-01")
    for s, rule in RELEASE.items():
        obs = pd.DatetimeIndex(M[s].dropna().index)
        obs = obs[(obs >= start) & (obs <= cal.sessions[-1] - pd.Timedelta(days=120))]
        rel = pd.DatetimeIndex([rule(o) for o in obs])
        pit_vis = cal.on_or_after(rel)
        eng = cal.shift(cal.on_or_after(obs), LAG.get(s, 1))
        naive = cal.on_or_after(obs)
        dec = [f"{(o.year // 10) * 10}s" for o in obs]
        res["series"][s] = {
            "engine_vs_approx_release": _tab(pit.visibility_gap(eng, pit_vis, by=dec, calendar=cal)),
            "naive_obs_date_vs_release": _tab(pit.visibility_gap(naive, pit_vis, by=dec, calendar=cal)),
            "engine_lag_median_calendar_days": float(np.median((eng - obs).days)),
            "approx_release_lag_median_calendar_days": float(np.median((rel - obs).days))}
    for s, n in WEEKLY.items():
        obs = pd.DatetimeIndex(M[s].dropna().index)
        obs = obs[(obs >= start) & (obs <= cal.sessions[-1] - pd.Timedelta(days=30))]
        pit_vis = cal.shift(cal.on_or_after(obs), n)
        eng = cal.shift(cal.on_or_after(obs), LAG.get(s, 1))
        res["series"][s] = {"engine_vs_approx_release": _tab(pit.visibility_gap(eng, pit_vis, calendar=cal)),
                            "naive_obs_date_vs_release": _tab(pit.visibility_gap(cal.on_or_after(obs), pit_vis, calendar=cal))}
    for s in DAILY:
        obs = pd.DatetimeIndex(M[s].dropna().index)
        obs = obs[(obs >= start) & (obs <= cal.sessions[-1] - pd.Timedelta(days=10))]
        pit_vis = cal.strict_next(obs)
        eng = cal.shift(cal.on_or_after(obs), LAG.get(s, 1))
        res["series"][s] = {"engine_vs_next_session": _tab(pit.visibility_gap(eng, pit_vis, calendar=cal)),
                            "naive_same_day": _tab(pit.visibility_gap(cal.on_or_after(obs), pit_vis, calendar=cal))}
    u = pd.DatetimeIndex(M["USREC"].dropna().index)
    res["USREC_note"] = ("NBER announces turning points months after the fact (2020 peak: 4 months, 2020 trough: 15 "
                         "months); engine lag is %d sessions (~%d calendar days) so late trough dates can leak"
                         % (LAG.get("USREC", 1), int(np.median((cal.shift(cal.on_or_after(u[u > start][:200]), LAG.get("USREC", 1))
                                                                - u[u > start][:200]).days))))
    return res


# ------------------------------------------------------------------------------------------------- survivorship
def survivorship(cal: pit.Calendar, rng, **_):
    res = {}
    uni = set(pd.read_csv(K.CACHE / "universe.csv")["ticker"].astype(str))
    for name in ("stocks", "stocks_pre2000"):
        C = pd.read_parquet(K.CACHE / f"{name}_close.parquet").astype("float32")
        first = C.apply(lambda s: s.first_valid_index())
        last = C.apply(lambda s: s.last_valid_index())
        end = C.index[-1]
        first, last = first.dropna(), last.dropna()
        ended = last[last < end - pd.Timedelta(days=10)]
        block = {"tickers": int(len(last)), "start": str(C.index[0].date()), "end": str(end.date()),
                 "ended_before_end": int(len(ended)), "share_ended": float(len(ended) / C.shape[1]),
                 "ended_by_year": {int(k): int(v) for k, v in pd.DatetimeIndex(ended).year.value_counts().sort_index().items()},
                 "listed_by_year": {int(k): int(v) for k, v in pd.DatetimeIndex(first.dropna()).year.value_counts().sort_index().items()}}
        if name == "stocks":
            block["in_current_universe_csv"] = float(np.mean([t in uni for t in C.columns]))
            block["ended_and_in_universe_csv"] = float(np.mean([t in uni for t in ended.index])) if len(ended) else None
        lst = pd.DataFrame({"ticker": last.index, "list_date": first.reindex(last.index).values,
                            "delist_date": [l if (pd.notna(l) and l < end - pd.Timedelta(days=10)) else pd.NaT for l in last.values]})
        L = pit.Listings(lst.dropna(subset=["list_date"]))
        eras = {}
        for a in (2002, 2008, 2014, 2020):
            if name == "stocks_pre2000" and a > 1990:
                continue
            start = pd.Timestamp(f"{a}-01-01") if name == "stocks" else pd.Timestamp("1975-01-01")
            m0 = L.members(start)
            samp = list(rng.choice(m0, min(1500, len(m0)), replace=False)) if m0 else []
            if not samp or (end - start).days < 700:
                continue
            rep = pit.survivorship_report(C[samp], pit.Listings(lst[lst["ticker"].isin(samp)].dropna(subset=["list_date"])),
                                          end, start=start)
            eras[str(a)] = {"errors": sorted(rep.error_codes()), **{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                                  for k, v in rep.stats.items()}}
        block["eras"] = eras
        res[name] = block
        del C
    return res


# ------------------------------------------------------------------------------------------------- panel
def _dt_tk(path: Path, probe: str):
    df = pd.read_parquet(path, columns=[probe])
    if "date" in df.columns:
        return pd.DatetimeIndex(df["date"]), df["ticker"].astype("category")
    idx = df.index
    return pd.DatetimeIndex(idx.get_level_values("date" if "date" in idx.names else 0)), \
        pd.Series(idx.get_level_values("ticker" if "ticker" in idx.names else 1)).astype("category")


def panel_integrity(cal: pit.Calendar, **_):
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet").astype("float32")
    first, last = C.apply(lambda s: s.first_valid_index()), C.apply(lambda s: s.last_valid_index())
    res = {}
    for nm, probe in (("panel", "r1"), ("oos_preds", "score")):
        dts, tk = _dt_tk(K.CACHE / f"{nm}.parquet", probe)
        r = {"rows": int(len(dts)), "start": str(dts.min().date()), "end": str(dts.max().date())}
        r["non_session_rows"] = int((~cal.is_session(dts)).sum())
        keyed = pd.DataFrame({"d": dts.values, "t": tk.values})
        r["duplicate_date_ticker"] = int(keyed.duplicated().sum())
        f, l = first.reindex(tk.astype(str)).values, last.reindex(tk.astype(str)).values
        r["rows_ticker_not_in_prices"] = int(pd.isna(f).sum())
        r["rows_after_last_bar"] = int((dts.values > l).sum())
        r["rows_before_first_bar"] = int((dts.values < f).sum())
        r["rows_by_era"] = {k: int(v) for k, v in pd.Series(_era(dts)).value_counts().sort_index().items()}
        res[nm] = r
        del dts, tk, keyed
    return res


# ------------------------------------------------------------------------------------------------- entry gap
def entry_gap(cal: pit.Calendar, rng, sample=800, **_):
    cols = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=["A"]).columns
    allc = pd.read_parquet(K.CACHE / "stocks_close.parquet").columns
    take = list(rng.choice(allc, min(sample, len(allc)), replace=False))
    O = pd.read_parquet(K.CACHE / "stocks_open.parquet", columns=take).astype("float32")
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=take).astype("float32")
    V = pd.read_parquet(K.CACHE / "stocks_volume.parquet", columns=take).astype("float32")
    liquid = ((C >= K.MIN_PRICE) & ((C * V).rolling(20, min_periods=15).median() >= K.MIN_DOLLAR_VOL))
    all_prof = pit.entry_gap_profile(O, C)
    liq_prof = pit.entry_gap_profile(O.where(liquid), C.where(liquid))
    fwd5 = np.log(C.shift(-5) / C)                                      # close-entry label
    fwd5_open = np.log(C.shift(-5) / O.shift(-1))                       # what a next-open fill earns
    d = (fwd5 - fwd5_open).where(liquid)
    return {"sample_tickers": len(take), "gap_all": _tab(all_prof), "gap_liquid": _tab(liq_prof),
            "label5_minus_openentry5_mean_liquid": float(np.nanmean(d.values)),
            "label5_minus_openentry5_mean_abs_liquid": float(np.nanmean(np.abs(d.values)))}


# ------------------------------------------------------------------------------------------------- features
def feature_invariance(cal: pit.Calendar, rng, sample=60, n_sessions=800, **_):
    from engine import data as D
    from engine import features as F
    t0 = time.time()
    C_all = pd.read_parquet(K.CACHE / "stocks_close.parquet", columns=None)
    alive = C_all.iloc[-n_sessions:].notna().mean() > 0.98
    pick = list(rng.choice(list(alive[alive].index), sample, replace=False))
    del C_all
    stocks = {f: pd.read_parquet(K.CACHE / f"stocks_{f.lower()}.parquet", columns=pick).iloc[-n_sessions:] for f in D.FIELDS}
    market = {f: v.iloc[-n_sessions:] for f, v in D.load("market").items()}
    ev = pd.read_parquet(K.CACHE / "events.parquet")
    ev = ev[ev["ticker"].isin(pick)]
    ins = pd.read_parquet(K.CACHE / "insider.parquet")
    ins = ins[ins["symbol"].isin(pick)]
    sic = pd.read_parquet(K.CACHE / "sic.parquet")
    data = {**{f"s_{k}": v for k, v in stocks.items()}, **{f"m_{k}": v for k, v in market.items()}, "ev": ev, "ins": ins}
    start = stocks["Close"].index[300]

    def fn(d):
        st = {k[2:]: v for k, v in d.items() if k.startswith("s_")}
        mk = {k[2:]: v for k, v in d.items() if k.startswith("m_")}
        X, _ = F.build(st, mk, d["ev"], d["ins"], sic, start=start)
        return X

    masks = {"ev": lambda df, t: (df["accepted"].dt.tz_convert(ET).dt.tz_localize(None) > t + pd.Timedelta(hours=16)).values,
             "ins": lambda df, t: np.asarray(pd.DatetimeIndex(df["filed"]).normalize() >= t)}
    dates = stocks["Close"].index
    cuts = [dates[i] for i in (450, 520, 600, 680, 760)]
    rep = pit.future_invariance(fn, data, cuts=cuts, mask_fns=masks, seed=int(rng.integers(1 << 31)), atol=1e-5, rtol=1e-4)
    canary = pit.future_invariance(
        lambda d: np.log(d["s_Close"].shift(-5) / d["s_Close"]).iloc[300:], {"s_Close": data["s_Close"]},
        cuts=cuts, seed=1)
    Xfull = fn(data)
    return {"sample_tickers": sample, "sessions": n_sessions, "cuts": [str(c.date()) for c in cuts],
            "build_columns": int(Xfull.shape[1]), "build_rows": int(len(Xfull)),
            "real_features_passed": rep.passed, "real_features_summary": rep.summary(),
            "leaking_columns": rep.columns, "canary_shift_minus5_detected": (not canary.passed),
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------- decisions
def live_decisions(cal: pit.Calendar, **_):
    rows = []
    for name in ("decisions.jsonl", "orders.jsonl"):
        p = K.STATE / name
        if not p.exists():
            continue
        for k, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            j = json.loads(line)
            t = pd.Timestamp(j["t"])
            t = t.tz_convert(ET) if t.tzinfo else t.tz_localize("UTC").tz_convert(ET)
            rows.append({"order_id": f"{name}:{k}", "ticker": j.get("ticker", "*"), "decision_date": t.tz_localize(None).normalize(),
                         "decision_time": t.hour + t.minute / 60, "src": name})
    if not rows:
        return {"decisions": 0}
    dec = pd.DataFrame(rows)
    empty = pd.DataFrame(columns=["order_id", "ticker", "fill_date", "fill_price"])
    rep = pit.audit_fills(dec, empty, pd.DataFrame(), None, cal)
    return {"decisions": len(dec), "decision_hours_et": sorted({round(h, 2) for h in dec["decision_time"]}),
            "error_codes": sorted(rep.error_codes()), "findings": [f"{f.code} {f.subject} {f.detail}" for f in rep.findings[:8]]}


SECTIONS = {"insider": sec_insider, "events": sec_events, "macro": macro_lags, "survivor": survivorship,
            "panel": panel_integrity, "entry_gap": entry_gap, "features": feature_invariance,
            "decisions": live_decisions}


def render_md(out: dict) -> str:
    L = [f"# PIT audit on real caches", f"seed {out['seed']}, peak RSS {out['peak_rss_mb']:.0f} MB, {out['seconds']:.0f}s", ""]
    for name, r in out["sections"].items():
        L.append(f"## {name}")
        if "error" in r:
            L += [f"FAILED: {r['error']}", ""]
            continue
        L += ["```", json.dumps(r, indent=1, default=str)[:6000], "```", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--skip", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    skip = set(filter(None, a.skip.split(",")))
    only = set(filter(None, a.only.split(",")))
    OUT.mkdir(parents=True, exist_ok=True)
    cal = sessions_calendar()
    rng = np.random.default_rng(a.seed)
    out = {"seed": a.seed, "provenance": stamp({"sample": a.sample}, a.seed), "sections": {}}
    t0, peak = time.time(), _rss_mb()
    for name, fn in SECTIONS.items():
        if name in skip or (only and name not in only):
            continue
        t = time.time()
        try:
            out["sections"][name] = fn(cal=cal, rng=np.random.default_rng(rng.integers(1 << 31)), sample=a.sample)
        except Exception as e:                       # noqa: BLE001 - one section failing must not hide the others
            out["sections"][name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        out["sections"][name]["_seconds"] = round(time.time() - t, 1)
        peak = max(peak, _rss_mb())
        print(f"[{name}] {out['sections'][name]['_seconds']}s  rss {_rss_mb():.0f} MB", flush=True)
    out["seconds"], out["peak_rss_mb"] = time.time() - t0, peak
    (OUT / "pit_audit.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (OUT / "PIT_AUDIT.md").write_text(render_md(out), encoding="utf-8")
    print(f"wrote {OUT / 'pit_audit.json'}")
    return out


if __name__ == "__main__":
    main()
