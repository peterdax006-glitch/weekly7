"""Bible Phase 2: run the parity firewall (every feature family, engine/parity_suite.py) against the REAL caches on a
seeded sample, low RAM. Writes state/research/parity/parity_<UTC>.json (+ latest.json) with provenance and exits
nonzero on failure, so a wrapper can `abort dependent experiment`.

    python scripts/run_parity.py [--tickers 60] [--dates 12] [--seed 0] [--since 2008-01-01] [--max-mb 1400]

Exit codes: 0 pass, 1 parity failure, 2 crash or unusable data, 3 memory ceiling hit.
Only column subsets of the parquet caches are read; nothing in data/cache or state/livesim is written."""
import argparse
import ctypes
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import pyarrow.dataset, pyarrow.parquet   # import before the watchdog thread starts: a DLL load racing it stalls

from engine import config as K, provenance, parity as P, parity_suite as S

FIELDS = ("Open", "High", "Low", "Close", "Volume")


def rss_mb():
    """Working set of this process in MB (Windows; falls back to /proc)."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError
        return c.WorkingSetSize / 2**20
    except Exception:
        try:
            return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**20
        except Exception:
            return 0.0


def lower_priority():
    try:
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception:
        try:
            os.nice(10)
        except Exception:
            pass


class Watchdog(threading.Thread):
    def __init__(self, limit_mb, out_path):
        super().__init__(daemon=True)
        self.limit, self.peak, self.out = limit_mb, 0.0, out_path

    def run(self):
        while True:
            self.peak = max(self.peak, rss_mb())
            if self.peak > self.limit:
                self.out.write_text(json.dumps({"passed": False, "error": f"memory ceiling {self.limit} MB exceeded "
                                                f"({self.peak:.0f} MB)", "provenance": provenance.stamp()}, indent=1))
                print(f"ABORT: memory {self.peak:.0f} MB > {self.limit} MB", flush=True)
                os._exit(3)
            time.sleep(0.5)


def load_real(n_tickers, seed, since):
    """Seeded ticker sample with (near-)full history since `since`, all five bar fields, market, filings, insider,
    sic, macro. Columns are read selectively so RAM stays small."""
    cache = K.CACHE
    rng = np.random.default_rng(seed)
    import pyarrow.parquet as pq
    names = [n for n in pq.ParquetFile(cache / "stocks_close.parquet").schema.names if not n.startswith("__")]
    cand = list(rng.choice(names, size=min(len(names), n_tickers * 8), replace=False))
    C = pd.read_parquet(cache / "stocks_close.parquet", columns=cand).loc[since:]
    good = [c for c in C.columns if C[c].notna().mean() >= 0.97 and pd.notna(C[c].iloc[-1]) and C[c].median() > 5]
    if len(good) < 8:
        raise RuntimeError(f"only {len(good)} usable tickers in the candidate sample")
    pick = sorted(good[:n_tickers])
    stocks = {f: pd.read_parquet(cache / f"stocks_{f.lower()}.parquet", columns=pick).loc[since:] for f in FIELDS}
    market = {f: pd.read_parquet(cache / f"market_{f.lower()}.parquet").loc[since:] for f in ("Close",)}
    ev = pd.read_parquet(cache / "events.parquet")
    ev = ev[ev["ticker"].isin(pick)]
    ins = None
    if (cache / "insider.parquet").exists():
        ins = pd.read_parquet(cache / "insider.parquet")
        ins = ins[ins["symbol"].isin(pick)]
    sic = pd.read_parquet(cache / "sic.parquet")
    macro = pd.read_parquet(cache / "macro.parquet") if (cache / "macro.parquet").exists() else None
    return P.Inputs(stocks, market, ev, ins, sic, macro), pick


def fast_git_commit():
    """provenance.git_commit shells out to `git status` over the whole repo, which can stall for minutes while other
    jobs write state/; read HEAD directly instead (dirty flag not computed, stated in the value)."""
    try:
        g = K.ROOT / ".git"
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            f = g / ref
            head = f.read_text().strip() if f.exists() else next(
                (l.split()[0] for l in (g / "packed-refs").read_text().splitlines() if l.endswith(ref)), "unknown")
        return head + "+dirty-unchecked"
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--dates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--since", default="2008-01-01")
    ap.add_argument("--max-mb", type=float, default=1400)
    ap.add_argument("--out", default=str(K.STATE / "research" / "parity"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"parity_{stamp_utc}.json"
    provenance.git_commit = fast_git_commit
    lower_priority()
    wd = Watchdog(a.max_mb, path)
    wd.start()
    t0 = time.time()
    cfg = {"tickers": a.tickers, "dates": a.dates, "since": a.since, "tol": P.GATE_TOL}
    doc = {"config": cfg, "provenance": provenance.stamp(cfg, a.seed), "started": stamp_utc}
    try:
        inp, pick = load_real(a.tickers, a.seed, a.since)
        doc.update(tickers=pick, sessions=len(inp.dates), first=str(inp.dates[0].date()), last=str(inp.dates[-1].date()),
                   events=0 if inp.ev is None else len(inp.ev), insider=0 if inp.ins is None else len(inp.ins),
                   macro_series=0 if inp.macro is None else inp.macro.shape[1], loaded_mb=round(rss_mb()))
        print(f"loaded {len(pick)} tickers x {len(inp.dates)} sessions, {doc['loaded_mb']} MB", flush=True)
        suite = S.run_suite(inp, n_dates=a.dates, seed=a.seed)
        print(suite.summary(), flush=True)
        doc.update(suite.to_dict())
        code = 0 if suite.passed else 1
    except Exception as e:
        import traceback
        doc.update(passed=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(doc["traceback"], flush=True)
        code = 2
    doc.update(seconds=round(time.time() - t0, 1), peak_mb=round(wd.peak), code_stale=provenance.stale(doc["provenance"]["code_hash"]))
    text = json.dumps(doc, indent=1, default=str)
    path.write_text(text, encoding="utf-8")
    (out / "latest.json").write_text(text, encoding="utf-8")
    print(f"report: {path}  exit={code}  peak={doc['peak_mb']} MB  {doc['seconds']}s", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
