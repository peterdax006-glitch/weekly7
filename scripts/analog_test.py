"""A8 heavy test: do the nearest PAST analogs predict what the market does next (return, volatility, drawdown)?
Monthly checkpoints 1970-2025; every prediction uses only analogs ending >= 63 sessions before the checkpoint.
Also prints the analogs found for famous moments, as a sanity check the owner can read."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K
from engine.analogs import fingerprints, Analogs

t0 = time.time()
F, O = fingerprints()
print(f"fingerprints: {F.shape[0]:,} days x {F.shape[1]} features, {F.index[0].date()} -> {F.index[-1].date()} [{time.time()-t0:.0f}s]", flush=True)
A = Analogs(F, O)
checks = F.loc["1970":"2025-08"].resample("MS").first().index
checks = [F.index[F.index.searchsorted(c)] for c in checks]
rows = []
for t in checks:
    r = A.find(t)
    if r is None:
        continue
    rows.append({"t": t, **{f"pred_{k}": v for k, v in r["prediction"].items()}, **O.loc[t].to_dict(),
                 "uniqueness": r["uniqueness"]})
R = pd.DataFrame(rows).dropna()
res = {}
for c in ("fwd_ret_1m", "fwd_vol_1m", "fwd_maxdd_1m"):
    ic = R[f"pred_{c}"].rank().corr(R[c].rank())
    n = len(R)
    res[c] = {"rank_corr": float(ic), "t": float(ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-9))), "n_months": n}
    # does the analog beat the simple baseline "same as the last month"?
print(json.dumps(res, indent=1))
for label, day in (("dot-com peak", "2000-03-24"), ("Lehman", "2008-09-15"), ("Covid crash", "2020-03-16"),
                   ("Black Monday aftermath", "1987-10-20"), ("2022 rate shock", "2022-06-13")):
    t = F.index[F.index.searchsorted(pd.Timestamp(day))]
    r = A.find(t, k=3)
    if r:
        print(f"\n{label} ({t.date()}): uniqueness {r['uniqueness']:.2f}, close analogs {r['n_close']}")
        for a in r["analogs"].itertuples():
            print(f"   {a.date.date()}  distance {a.distance:.2f}  then: next month {a.fwd_ret_1m:+.1%}, vol {a.fwd_vol_1m:.0%}")
out = K.STATE / "research" / "algorithm"
out.mkdir(parents=True, exist_ok=True)
(out / "analog_test.json").write_text(json.dumps(res, indent=1))
F.to_parquet(K.CACHE / "fingerprints.parquet"); O.to_parquet(K.CACHE / "fingerprint_outcomes.parquet")
print(f"done [{time.time()-t0:.0f}s]")
