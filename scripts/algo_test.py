"""Algorithm heavy test: can the self-learning pattern miner find patterns that hold on UNSEEN later data?
Train on week-end rows whose 5-session label closed before CUT; score every later week blind; measure rank-IC of
the pattern score vs the next-open-to-5-day return, and top-decile excess. Also reports what it found.

usage: algo_test.py <cut> <end> [params_json]"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, data, candles
from engine.patterns import PatternMiner

t0 = time.time()
CUT = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2020-01-01")
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2022-01-01")
PARAMS = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
TAG = PARAMS.pop("tag", "")
stocks = data.load("stocks")
X = pd.read_parquet(K.CACHE / "panel.parquet")
d = X.index.get_level_values(0)
ud = pd.DatetimeIndex(sorted(d.unique()))
wk = ud[[i for i in range(len(ud) - 1) if ud[i + 1].isocalendar().week != ud[i].isocalendar().week]]
X = X[d.isin(wk)]
if PARAMS.pop("candles", True):
    cf = candles.build(stocks)
    for k, v in cf.items():
        X[k] = v.stack(future_stack=True).reindex(X.index).astype("float32").values
    del cf
O, C = stocks["Open"], stocks["Close"]
fwd = C.shift(-5) / O.shift(-1) - 1                              # C33: bought at the next open
y = fwd.stack(future_stack=True).reindex(X.index)
y = y - y.groupby(level=0).transform("mean")                     # excess vs that week's average stock
d = X.index.get_level_values(0)
last_train = ud[ud.searchsorted(CUT) - 7]                        # training labels close before CUT
tr, te = d <= last_train, (d >= CUT) & (d < END)
X = X.drop(columns=[c for c in X.columns if c.startswith("ev_activist")])
print(f"rows: train {tr.sum():,} (to {last_train.date()}), test {te.sum():,}; features {X.shape[1]}  [{time.time()-t0:.0f}s]", flush=True)
M = PatternMiner(PARAMS).fit(X[tr], y[tr], now=CUT)
print("miner:", M.report, f"[{time.time()-t0:.0f}s]", flush=True)
ics, tops = [], []
for day in sorted(set(d[te])):
    Xd = X.xs(day, level=0)
    s = M.score(Xd)
    yy = y.xs(day, level=0).reindex(Xd.index)
    ok = yy.notna()
    if ok.sum() < 50 or s[ok].nunique() < 3:
        continue
    ics.append(s[ok].rank().corr(yy[ok].rank()))
    tops.append(yy[ok][s[ok] >= s[ok].quantile(0.9)].mean())
ics = np.array(ics)
res = {"cut": str(CUT.date()), "tag": TAG, "params": PARAMS, "weeks": len(ics), "ic_mean": float(ics.mean()) if len(ics) else None,
       "ic_t": float(ics.mean() / ics.std() * np.sqrt(len(ics))) if len(ics) > 1 else None,
       "top_decile_excess_week": float(np.mean(tops)) if tops else None, **M.report}
print(json.dumps(res, indent=1))
P = M.patterns
for st in ("active", "rescoped", "benched"):
    sub = P[P["status"] == st]
    sub = sub.reindex(sub["effect"].abs().sort_values(ascending=False).index).head(6)
    print(f"\n{st.upper()} (strongest):")
    for r in sub.itertuples():
        print(f"  {r.effect*100:+.2f}%/wk  p(coincidence)={r.p_coincidence:.1e}  {r.key_named}" + (f"  [only when ctx {r.scope}]" if r.scope else ""))
out = K.STATE / "research" / "algorithm"
out.mkdir(parents=True, exist_ok=True)
(out / f"test_{CUT.date()}{TAG}.json").write_text(json.dumps(res, indent=1))
P.assign(key=P["key"].astype(str), scope=P["scope"].astype(str)).to_parquet(out / f"patterns_{CUT.date()}{TAG}.parquet")
print(f"done [{time.time()-t0:.0f}s]")
