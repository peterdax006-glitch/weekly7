"""SEC EDGAR: point-in-time corporate events and insider purchases (all free, 10 req/s limit).

events.parquet  one row per filing that matters: ticker, form, items, accepted (UTC), kind
insider.parquet one row per open-market purchase (code P) from the DERA Form 3/4/5 data sets
Every row carries the moment it became public; features may only use rows accepted before
the decision time (after-close filings count from the next session)."""
import io, json, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

from .config import CACHE, SEC_UA

H = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
_last = [0.0]
_lock = threading.Lock()
_session = threading.local()


def _get(url, **kw):
    """Thread-safe GET held under SEC's 10 requests/second (we use ~8)."""
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
    for attempt in range(5):
        with _lock:
            wait = 0.125 - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = _session.s.get(url, headers=H, timeout=60, **kw)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
        if r.status_code == 200:
            return r
        if r.status_code in (404,):
            return None
        time.sleep(2 ** attempt)
    return None


EQUITY_424 = ("424B1", "424B4", "424B5", "424B7")


def classify(form: str, items: str) -> list:
    items = set((items or "").split(","))
    k = []
    if form in ("8-K", "8-K/A"):
        if "2.02" in items: k.append("EARN")
        if "1.01" in items: k.append("AGREEMENT")
        if "2.01" in items: k.append("ACQ_DONE")
        if "1.03" in items: k.append("BANKRUPTCY")
        if "3.01" in items: k.append("DELIST_NOTICE")
        if "4.01" in items: k.append("AUDITOR_CHANGE")
        if "4.02" in items: k.append("RESTATEMENT")
        if "3.02" in items: k.append("UNREG_SALE")
    elif form in EQUITY_424:           # 424B2/B3 are mostly bank structured notes / resales
        k.append("OFFERING")
    elif form in ("S-3", "S-1", "F-1", "F-3"):
        k.append("SHELF")
    elif form in ("SC 13D", "SCHEDULE 13D"):
        k.append("ACTIVIST")
    elif form in ("SC 13D/A", "SCHEDULE 13D/A"):
        k.append("ACTIVIST_AMEND")
    elif form in ("NT 10-K", "NT 10-Q"):
        k.append("LATE_FILING")
    elif form in ("10-Q", "10-K"):
        k.append("PERIODIC")
    return k


def _rows(block, ticker, since):
    out = []
    for i, form in enumerate(block["form"]):
        if block["filingDate"][i] < since:
            continue
        for kind in classify(form, block.get("items", [""] * len(block["form"]))[i]):
            out.append((ticker, form, block["acceptanceDateTime"][i], kind))
    return out


def _one(t, cik, since):
    r = _get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    if r is None:
        return [], None
    d = r.json()
    rows = _rows(d["filings"]["recent"], t, since)
    for f in d["filings"].get("files", []):
        if f["filingTo"] >= since:
            r2 = _get("https://data.sec.gov/submissions/" + f["name"])
            if r2 is not None:
                rows += _rows(r2.json(), t, since)
    return rows, (t, int(cik), d.get("sic") or "", d.get("sicDescription") or "")


def fetch_events(universe: pd.DataFrame, since="2011-06-01") -> pd.DataFrame:
    rows, meta = [], []
    with ThreadPoolExecutor(10) as ex:
        futs = [ex.submit(_one, t, c, since) for t, c in zip(universe["ticker"], universe["cik"])]
        for n, fu in enumerate(futs):
            rr, mm = fu.result()
            rows += rr
            if mm:
                meta.append(mm)
            if n % 250 == 0:
                print(f"  edgar {n}/{len(universe)}", flush=True)
    ev = pd.DataFrame(rows, columns=["ticker", "form", "accepted", "kind"])
    ev["accepted"] = pd.to_datetime(ev["accepted"], utc=True)
    ev.to_parquet(CACHE / "events.parquet")
    pd.DataFrame(meta, columns=["ticker", "cik", "sic", "sic_desc"]).to_parquet(CACHE / "sic.parquet")
    return ev


# ---------- insider purchases (Cohen-Malloy-Pomorski) ----------
DERA = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"


def _read(z, name):
    with z.open(name) as f:
        return pd.read_csv(f, sep="\t", dtype=str, on_bad_lines="skip", low_memory=False)


def fetch_insider(first_year=2009, last=(2026, 3)) -> pd.DataFrame:
    parts = []
    for y in range(first_year, last[0] + 1):
        for q in range(1, 5):
            if (y, q) > last:
                break
            r = _get(DERA.format(y=y, q=q))
            if r is None:
                print("  missing", y, q); continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            names = {n.split("/")[-1].upper(): n for n in z.namelist()}
            tr = _read(z, names["NONDERIV_TRANS.TSV"])
            tr = tr[tr["TRANS_CODE"] == "P"]
            sub = _read(z, names["SUBMISSION.TSV"])[["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]]
            own = _read(z, names["REPORTINGOWNER.TSV"])[["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]]
            own = own.drop_duplicates("ACCESSION_NUMBER")
            m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
            m = m[["ACCESSION_NUMBER", "FILING_DATE", "TRANS_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "RPTOWNERCIK",
                   "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]]
            parts.append(m)
            print(f"  insider {y}q{q}: {len(m)} purchases", flush=True)
    ins = pd.concat(parts, ignore_index=True)
    ins.columns = ["acc", "filed", "tdate", "issuer_cik", "symbol", "owner_cik", "relation", "title", "shares", "price"]
    for c in ("filed", "tdate"):
        ins[c] = pd.to_datetime(ins[c], format="mixed", dayfirst=False, errors="coerce")
    for c in ("shares", "price"):
        ins[c] = pd.to_numeric(ins[c], errors="coerce")
    ins["value"] = ins["shares"] * ins["price"]
    ins = ins.dropna(subset=["filed", "tdate", "value"])
    ins = ins[ins["value"] > 0].drop_duplicates(["acc", "owner_cik", "tdate", "shares"])
    ins.to_parquet(CACHE / "insider.parquet")
    return ins


if __name__ == "__main__":
    import sys
    if "insider" in sys.argv:
        fetch_insider()
    else:
        fetch_events(pd.read_csv(CACHE / "universe.csv"))


# ---------- live incremental refresh (daily index) ----------
import re
from datetime import date, timedelta

RELEVANT = re.compile(r"^(8-K|8-K/A|424B\d|S-3|S-1|SC 13D|SCHEDULE 13D|NT 10-K|NT 10-Q|10-Q|10-K|4)\s*$")


def _daily_index(d: date):
    q = (d.month - 1) // 3 + 1
    r = _get(f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx")
    if r is None:
        return []
    out = []
    for line in r.text.splitlines():
        m = re.match(r"^(.{12})\s+(.*?)\s{2,}(\d+)\s+(\d{8})\s+(edgar/\S+)$", line.strip()) or \
            re.match(r"^(\S+(?: \S+)?)\s+(.*?)\s+(\d{3,10})\s+(\d{8})\s+(edgar/\S+)$", line.strip())
        if m and RELEVANT.match(m.group(1).strip()):
            out.append((m.group(1).strip(), int(m.group(3)), m.group(5)))
    return out


def _parse_form4(txt):
    sym = re.search(r"<issuerTradingSymbol>\s*([^<]+)", txt)
    icik = re.search(r"<issuerCik>\s*(\d+)", txt)
    ocik = re.search(r"<rptOwnerCik>\s*(\d+)", txt)
    title = re.search(r"<officerTitle>\s*([^<]*)", txt)
    rel = " ".join(k for k in ("isDirector", "isOfficer", "isTenPercentOwner")
                   if re.search(rf"<{k}>\s*(1|true)", txt, re.I))
    rows = []
    for blk in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", txt, re.S):
        if not re.search(r"<transactionCode>\s*P\s*<", blk):
            continue
        def g(tag):   # the <value> strictly inside this tag's own element
            el = re.search(rf"<{tag}>(.*?)</{tag}>", blk, re.S)
            v = re.search(r"<value>\s*([^<]+?)\s*</value>", el.group(1), re.S) if el else None
            return v.group(1) if v else None
        sh, px, dt = g("transactionShares"), g("transactionPricePerShare"), g("transactionDate")
        try:
            sh, px = float(sh), float(px)
        except (TypeError, ValueError):
            continue
        if dt:
            rows.append({"symbol": sym.group(1).strip().upper() if sym else None,
                         "issuer_cik": icik.group(1) if icik else None, "owner_cik": ocik.group(1) if ocik else None,
                         "relation": rel, "title": title.group(1).strip() if title else "",
                         "shares": sh, "price": px, "tdate": dt[:10]})
    return rows


def refresh_recent(universe: pd.DataFrame, days=4):
    """Pull the last few days of filings: re-fetch submissions for universe companies that filed
    anything relevant, and parse new Form 4 open-market purchases. Merges into the caches."""
    cik2t = dict(zip(universe["cik"].astype(int), universe["ticker"]))
    hits, f4 = set(), []
    d = date.today()
    for k in range(days + 3):
        day = d - timedelta(days=k)
        if day.weekday() >= 5:
            continue
        for form, cik, path in _daily_index(day):
            if cik in cik2t:
                if form == "4":
                    f4.append((path, day))
                else:
                    hits.add(cik)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: _one(cik2t[c], c, "2011-06-01"), hits))
    new = [r for rows, _ in res for r in rows]
    evp = CACHE / "events.parquet"
    if new:
        nd = pd.DataFrame(new, columns=["ticker", "form", "accepted", "kind"])
        nd["accepted"] = pd.to_datetime(nd["accepted"], utc=True)
        ev = pd.concat([pd.read_parquet(evp), nd]).drop_duplicates(["ticker", "form", "accepted", "kind"])
        ev.to_parquet(evp)
    rows = []
    for path, day in dict(f4).items():
        r = _get("https://www.sec.gov/Archives/" + path)
        if r is not None:
            for x in _parse_form4(r.text):
                x.update({"acc": path.rsplit("/", 1)[-1], "filed": pd.Timestamp(day)})
                rows.append(x)
    if rows:
        ip = CACHE / "insider.parquet"
        nd = pd.DataFrame(rows)
        nd["tdate"] = pd.to_datetime(nd["tdate"], errors="coerce")
        nd["value"] = nd["shares"] * nd["price"]
        ins = pd.concat([pd.read_parquet(ip), nd]) if ip.exists() else nd
        ins = ins[ins["value"] > 0].drop_duplicates(["acc", "owner_cik", "tdate", "shares"])
        ins.to_parquet(ip)
    return len(hits), len(rows)


def backfill_form4(start: date, universe: pd.DataFrame, end: date = None):
    """Fill the gap between the last quarterly DERA data set and today from daily Form 4 filings."""
    cik2t = dict(zip(universe["cik"].astype(int), universe["ticker"]))
    end = end or date.today()
    paths = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            paths += [(p, d) for f, c, p in _daily_index(d) if f == "4" and c in cik2t]
        d += timedelta(days=1)
    paths = list(dict(paths).items())
    print(f"  form4 backfill: {len(paths)} filings", flush=True)

    def one(pd_):
        path, day = pd_
        r = _get("https://www.sec.gov/Archives/" + path)
        out = []
        if r is not None:
            for x in _parse_form4(r.text):
                x.update({"acc": path.rsplit("/", 1)[-1], "filed": pd.Timestamp(day)})
                out.append(x)
        return out
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for i, rr in enumerate(ex.map(one, paths)):
            rows += rr
            if i % 2000 == 0:
                print(f"  form4 {i}/{len(paths)}", flush=True)
    if rows:
        ip = CACHE / "insider.parquet"
        nd = pd.DataFrame(rows)
        nd["tdate"] = pd.to_datetime(nd["tdate"], errors="coerce")
        nd["value"] = nd["shares"] * nd["price"]
        ins = pd.concat([pd.read_parquet(ip), nd]) if ip.exists() else nd
        ins = ins[ins["value"] > 0].drop_duplicates(["acc", "owner_cik", "tdate", "shares"])
        ins.to_parquet(ip)
    return len(rows)


# ---------- 13D attribution fix ----------
def _subject_cik(cik, acc):
    """Read the filing header; return the SUBJECT COMPANY's CIK (13D filings list the filer too)."""
    nodash = acc.replace("-", "")
    r = _get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{acc}-index-headers.html")
    if r is None:
        return None
    m = re.search(r"SUBJECT COMPANY:.*?CENTRAL INDEX KEY:\s*(\d+)", r.text, re.S)
    return int(m.group(1)) if m else None


def fix_activist(universe: pd.DataFrame, since="2011-06-01"):
    """Keep ACTIVIST / ACTIVIST_AMEND events only for the company that is the 13D's subject."""
    evp = CACHE / "events.parquet"
    ev = pd.read_parquet(evp)
    act = ev[ev["kind"].isin(["ACTIVIST", "ACTIVIST_AMEND"])]
    tick2cik = dict(zip(universe["ticker"], universe["cik"].astype(int)))
    tickers = sorted(set(act["ticker"]))
    print(f"  13D fix: {len(act)} events across {len(tickers)} companies", flush=True)

    def one(t):
        cik = tick2cik.get(t)
        if cik is None:
            return []
        r = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if r is None:
            return []
        d = r.json()
        blocks = [d["filings"]["recent"]]
        for f in d["filings"].get("files", []):
            if f["filingTo"] >= since:
                r2 = _get("https://data.sec.gov/submissions/" + f["name"])
                if r2 is not None:
                    blocks.append(r2.json())
        keep = []
        for b in blocks:
            for i, form in enumerate(b["form"]):
                if form in ("SC 13D", "SCHEDULE 13D", "SC 13D/A", "SCHEDULE 13D/A") and b["filingDate"][i] >= since:
                    subj = _subject_cik(cik, b["accessionNumber"][i])
                    if subj == cik:
                        keep.append(b["acceptanceDateTime"][i])
        return [(t, a) for a in keep]
    # checkpoint per company: a run killed at 3,400 of 3,737 (28 Sep 2026) lost ~2 hours of SEC requests.
    # Progress is keyed by `since` so a different window never reuses stale answers.
    prog_path = CACHE / f"13d_fix_progress_{since}.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    todo = [t for t in tickers if t not in prog]
    print(f"  13D fix: {len(prog)} companies already checked, {len(todo)} to go", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, (t, rows) in enumerate(zip(todo, ex.map(one, todo))):
            prog[t] = [a for _, a in rows]
            if i % 50 == 0 or i == len(todo) - 1:
                tmp = prog_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(prog))
                tmp.replace(prog_path)                              # atomic: a kill never leaves half a file
            if i % 100 == 0:
                print(f"  13D fix {len(prog)}/{len(tickers)}", flush=True)
    ok = [(t, a) for t, accs in prog.items() for a in accs]
    okset = {(t, pd.Timestamp(a).tz_convert("UTC") if pd.Timestamp(a).tzinfo else pd.Timestamp(a, tz="UTC")) for t, a in ok}
    is_act = ev["kind"].isin(["ACTIVIST", "ACTIVIST_AMEND"])
    keep_mask = ~is_act | pd.Series([(t, a) in okset for t, a in zip(ev["ticker"], ev["accepted"])], index=ev.index)
    print(f"  13D fix: kept {int((is_act & keep_mask).sum())} of {int(is_act.sum())} activist events", flush=True)
    ev[keep_mask].to_parquet(evp)
