"""Build the delisted-company history the survivor-only price panel lacks (Phase 27, foundation for PIT work).

    python scripts/fetch_delisted.py [--from-year 1996] [--stages harvest,classify,resolve,prices,report]
                                     [--max-resolve 4000] [--max-prices 2000] [--seed 7]

Sources (all free, no keys):
  EDGAR full-index form.gz  every Form 25 (exchange delisting), Form 15 (deregistration) and 10-K since 1994.
                            This is the event history: WHO exited and WHEN. It carries no ticker.
  Yahoo search + chart      ticker resolution by company name (kept only above a name-similarity floor) and price
                            history for whatever Yahoo still serves. Yahoo removes most dead tickers, so recovery is
                            MEASURED and reported, never assumed.
Outputs (new delisted_* files only; existing caches are never touched, and guard() refuses any other name):
  data/cache/delisted_events.parquet   classified terminal/continuing Form 25 events, one row per CIK
  data/cache/delisted_map.parquet      cik -> ticker candidates with similarity and how they were found
  data/cache/delisted_registry.json    DelistedRegistry (terminal exits with a resolved ticker), point-in-time
  data/cache/delisted_prices.parquet   long OHLCV for recovered names, validated by engine.data_sources
  state/research/delisted/report.{md,json}   coverage per era against 3-6%/yr expected attrition
Checkpoints live in data/cache/delisted_raw/ so an interrupted run resumes without re-downloading."""
import argparse, gzip, json, sys, threading, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import config as K, data_sources as D, provenance  # noqa: E402

RAW = K.CACHE / "delisted_raw"
OUT = K.STATE / "research" / "delisted"
EV, MAP, PX = (K.CACHE / f"delisted_{n}.parquet" for n in ("events", "map", "prices"))
SEC = {"User-Agent": K.SEC_UA, "Accept-Encoding": "gzip, deflate"}
WEB = {"User-Agent": "Mozilla/5.0"}
_last = {}
_lock = threading.Lock()


def get(url, headers, min_gap, retries=4):
    """Rate-limited GET with backoff. Returns the response or None (404 / repeated failure)."""
    host = url.split("/")[2]
    for a in range(retries):
        with _lock:
            wait = min_gap - (time.time() - _last.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            _last[host] = time.time()
        try:
            r = requests.get(url, headers=headers, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** a)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(2 ** a * (3 if r.status_code == 429 else 1))
    return None


def guard(path):
    """This script may only ever write its own delisted_* files; it can never clobber another cache."""
    if not path.name.startswith("delisted_") or path.parent != K.CACHE:
        sys.exit(f"refusing to write {path}")


# ---------------------------------------------------------------- stage 1: harvest
def harvest(from_year):
    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow()
    parts, fetched, failed = [], 0, []
    for y in range(from_year, now.year + 1):
        for q in range(1, 5):
            if y == now.year and q > (now.month - 1) // 3 + 1:
                break
            cp = RAW / f"idx_{y}Q{q}.parquet"
            if cp.exists() and not (y == now.year and q == (now.month - 1) // 3 + 1):     # current quarter: refetch
                parts.append(pd.read_parquet(cp))
                continue
            r = get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.gz", SEC, 0.15)
            if r is None:
                failed.append(f"{y}Q{q}")
                continue
            body = gzip.decompress(r.content).decode("latin-1") if r.content[:2] == b"\x1f\x8b" else r.text
            df = D.keep_relevant_forms(D.parse_form_idx(body))
            df.to_parquet(cp)
            parts.append(df)
            fetched += 1
            print(f"harvest {y}Q{q}: {len(df)} relevant rows", flush=True)
    ev = pd.concat(parts, ignore_index=True).drop_duplicates(["form", "cik", "date", "path"])
    print(f"harvest done: {len(ev)} rows, fetched {fetched}, failed {failed}")
    return ev, failed


# ---------------------------------------------------------------- stage 2: classify
def classify(ev, as_of):
    c = D.classify_delistings(ev, as_of)
    guard(EV)
    c.to_parquet(EV)
    print("classified:", c["status"].value_counts().to_dict())
    return c


# ---------------------------------------------------------------- stage 3: resolve tickers
def survivors_by_cik():
    u = pd.read_csv(K.CACHE / "universe.csv")
    return dict(zip(u["cik"].astype(int), u["ticker"]))


def resolve(c, max_n, seed):
    ck = RAW / "resolve.jsonl"
    done = {}
    if ck.exists():
        for l in ck.read_text().splitlines():
            r = json.loads(l)
            done[r["cik"]] = r
    surv = survivors_by_cik()
    todo = c[c["terminal"] & ~c["cik"].isin(done)].copy()
    todo["y"] = todo["delist_date"].dt.year
    rng = np.random.default_rng(seed)
    # stratify by year so a capped run still covers every era evenly
    todo = todo.iloc[rng.permutation(len(todo))]
    todo = todo.groupby("y").head(max(1, max_n // max(1, todo["y"].nunique()))).head(max_n)
    n429 = 0
    with open(ck, "a") as f:
        for i, r in enumerate(todo.itertuples(), 1):
            rec = {"cik": int(r.cik), "company": r.company, "ticker": None, "sim": None, "via": None}
            if r.cik in surv:
                rec.update(ticker=surv[r.cik], sim=1.0, via="universe")
            else:
                q = D.clean_name(r.company) or r.company
                resp = get("https://query2.finance.yahoo.com/v1/finance/search?quotesCount=8&newsCount=0&q=" +
                           requests.utils.quote(q), WEB, 0.35)
                if resp is None:
                    n429 += 1
                    if n429 > 40:
                        print("too many search failures; stopping resolve (checkpoint kept)")
                        break
                else:
                    pick = D.pick_symbol(r.company, resp.json().get("quotes", []))
                    if pick:
                        rec.update(ticker=pick["symbol"], sim=round(pick["sim"], 3), via="yahoo_search")
            f.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"resolve {i}/{len(todo)}", flush=True)
    m = pd.DataFrame([json.loads(l) for l in ck.read_text().splitlines()]).drop_duplicates("cik", keep="last")
    guard(MAP)
    m.to_parquet(MAP)
    print("resolved:", int(m["ticker"].notna().sum()), "of", len(m))
    return m


# ---------------------------------------------------------------- stage 4: prices
def chart(ticker):
    r = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=max&interval=1d&events=div,splits",
            WEB, 0.35, retries=2)
    if r is None:
        return None
    try:
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
        ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    df = pd.DataFrame({"date": ts, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                       "volume": q["volume"]})
    if adj and len(adj) == len(df):                       # split/dividend-adjust like the rest of the panel
        k = pd.Series(adj, index=df.index) / df["close"]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * k
    return df.dropna(subset=["close"])


def prices(c, m, max_n, as_of):
    ck = RAW / "prices_done.jsonl"
    done = {}
    if ck.exists():
        done = {json.loads(l)["cik"]: json.loads(l) for l in ck.read_text().splitlines()}
    good = m[m["ticker"].notna() & (m["sim"].fillna(0) >= 0.85)]
    good = good[good["ticker"].astype(str).str.fullmatch(r"[A-Z]{1,5}")]          # no foreign suffixes (.KS, .NS, .T)
    term = c[c["terminal"]].merge(good[["cik", "ticker"]], on="cik")
    todo = term[~term["cik"].isin(done)].head(max_n)
    frames = [pd.read_parquet(RAW / f"px_{k}.parquet") for k in done if (RAW / f"px_{k}.parquet").exists()]
    with open(ck, "a") as f:
        for i, r in enumerate(todo.itertuples(), 1):
            df = chart(r.ticker)
            ok = False
            if df is not None and len(df):
                df = df.assign(ticker=r.ticker)[D.PRICE_COLS]
                clean, rep = D.validate_prices(df, "yahoo_delisted", as_of, fetched_at=str(datetime.utcnow()))
                # keep the history up to the exchange delisting (OTC tails are not tradable in the universe) and
                # require that it really is this company's run: starts long before, reaches the delist date, and is
                # neither a later re-use of the ticker (SPAC shells) nor a live company (Form 25 for a partial class)
                cut = clean[clean["date"] <= r.delist_date + pd.Timedelta(days=30)]
                if rep.ok and len(cut) >= 60 and cut["date"].min() <= r.delist_date - pd.Timedelta(days=250) \
                        and cut["date"].max() >= r.delist_date - pd.Timedelta(days=45):
                    cut = cut.assign(cik=int(r.cik), delist_date=r.delist_date)
                    cut.to_parquet(RAW / f"px_{int(r.cik)}.parquet")
                    frames.append(cut)
                    ok = True
            f.write(json.dumps({"cik": int(r.cik), "ticker": r.ticker, "ok": ok}) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"prices {i}/{len(todo)} recovered so far {len(frames)}", flush=True)
    if frames:
        allp = pd.concat(frames, ignore_index=True)
        ncik = allp.groupby("ticker")["cik"].nunique()
        amb = sorted(ncik[ncik > 1].index)          # one symbol claimed by two CIKs (reuse, successors): drop, do not guess
        allp = allp[~allp["ticker"].isin(amb)]
        survivors = set(pd.read_csv(K.CACHE / "universe.csv")["ticker"])
        allp["successor_in_panel"] = allp["ticker"].isin(survivors)
        clean, vrep = D.validate_prices(allp[D.PRICE_COLS], "yahoo_delisted", as_of)
        if not vrep.ok:
            sys.exit(f"recovered prices fail validation: {vrep.errors}")
        print("ambiguous tickers dropped:", amb, "| validation dropped:", vrep.dropped, "| warnings:", len(vrep.warnings))
        allp = allp.drop_duplicates(["ticker", "date"])
        guard(PX)
        allp.to_parquet(PX)
        print("price rows:", len(allp), "names:", allp["ticker"].nunique())
        return allp
    return pd.DataFrame(columns=D.PRICE_COLS + ["cik"])


# ---------------------------------------------------------------- stage 5: report
def report(c, m, px, failed, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    close = pd.read_parquet(K.CACHE / "stocks_close.parquet")
    alive = close.notna().groupby(close.index.year).any().sum(axis=1)          # names alive in each year (survivors only)
    resolved = set(m.loc[m["ticker"].notna(), "cik"])
    priced_all = set(px["cik"]) if len(px) else set()
    priced = set(px.loc[~px["successor_in_panel"], "cik"]) if len(px) else set()   # names the survivor panel lacks
    cov = D.attrition_coverage(c, resolved, priced, alive)
    cov = cov.loc[cov.index >= 2000]
    era = cov.groupby(pd.cut(cov.index, [1999, 2004, 2009, 2014, 2019, 2030],
                             labels=["2000-04", "2005-09", "2010-14", "2015-19", "2020+"]), observed=True)
    era_tab = era[["terminal_events", "resolved", "priced", "expected_low", "expected_high"]].sum()
    era_tab["events_vs_expected_mid"] = era_tab["terminal_events"] / ((era_tab["expected_low"] + era_tab["expected_high"]) / 2)
    era_tab["priced_share_of_events"] = era_tab["priced"] / era_tab["terminal_events"]
    rep = {"seed": seed, "stamp": provenance.stamp({"from": "fetch_delisted"}, seed), "failed_quarters": failed,
           "status_counts": c["status"].value_counts().to_dict(),
           "terminal": int(c["terminal"].sum()), "resolved_tickers": len(resolved), "priced_names": len(priced), "priced_but_ticker_lives_on": len(priced_all) - len(priced),
           "resolution_attempted": int(len(m)),
           "recovery_rate_of_resolved": round(len(priced_all) / max(1, int(m["ticker"].notna().sum())), 3),
           "resolution_via": m["via"].value_counts().to_dict() if len(m) else {},
           "panel_survivors_alive_by_year": alive.to_dict(),
           "by_year": json.loads(cov.round(4).reset_index().rename(columns={"index": "year"}).to_json(orient="records")),
           "by_era": json.loads(era_tab.round(4).reset_index().rename(columns={"index": "era"}).to_json(orient="records"))}
    tick = dict(zip(m["cik"], m["ticker"].where(m["sim"].fillna(0) >= 0.85)))
    last = px.sort_values("date").groupby("ticker")["close"].last().to_dict() if len(px) else {}
    reg = D.to_registry(c, {k: v for k, v in tick.items() if isinstance(v, str)}, last)
    guard(K.CACHE / "delisted_registry.json")
    reg.save(K.CACHE / "delisted_registry.json")
    rep["registry_records"], rep["registry_rejected"] = len(reg.frame()), len(reg.rejected)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    md = ["# Delisted-company coverage", "", f"terminal exits found (Form 25 + Form 15 / went dark): **{rep['terminal']}**; "
          f"status counts {rep['status_counts']}", f"tickers resolved: {rep['resolved_tickers']} {rep['resolution_via']}; "
          f"names with recovered prices: {rep['priced_names']}", f"failed index quarters: {failed}", "",
          "## Per era (events vs 3-6%/yr of names alive in the survivor panel)", "```", era_tab.round(3).to_string(), "```",
          "", "## Per year", "```", cov.round(3).to_string(), "```", ""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1996)
    ap.add_argument("--stages", default="harvest,classify,resolve,prices,report")
    ap.add_argument("--max-resolve", type=int, default=4000)
    ap.add_argument("--max-prices", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ARGS = ap.parse_args()
    st = set(ARGS.stages.split(","))
    as_of = pd.Timestamp(datetime.utcnow().date())
    ev, failed = harvest(ARGS.from_year) if "harvest" in st else (None, [])
    if ev is None:
        ev = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("idx_*.parquet"))], ignore_index=True)
    c = classify(ev, as_of) if "classify" in st else pd.read_parquet(EV)
    empty_map = pd.DataFrame(columns=["cik", "company", "ticker", "sim", "via"])
    m = resolve(c, ARGS.max_resolve, ARGS.seed) if "resolve" in st else (pd.read_parquet(MAP) if MAP.exists() else empty_map)
    px = prices(c, m, ARGS.max_prices, as_of) if "prices" in st else (pd.read_parquet(PX) if PX.exists() else
                                                                    pd.DataFrame(columns=D.PRICE_COLS + ["cik"]))
    if "report" in st:
        report(c, m, px, failed, ARGS.seed)


if __name__ == "__main__":
    main()
