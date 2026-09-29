"""A7 on real data: walk-forward self-tuning of the pattern miner on the weekly panel (same rows, labels and candle
inputs as scripts/algo_test.py), on a seeded ticker sample to fit in shared memory. Writes
state/research/algorithm/tuning/tune_<target>_<seed>.json and logs a registry record with the holdout verdict.

usage: miner_tune_real.py [target=move|direction] [seed=0] [tickers=600] [start=2008-01-01] [knobs=comma list]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, data, candles, miner_tuning as T
from engine.improve import log_experiment

TARGET = sys.argv[1] if len(sys.argv) > 1 else "move"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 600
START = pd.Timestamp(sys.argv[4] if len(sys.argv) > 4 else "2008-01-01")
KNOBS = sys.argv[5].split(",") if len(sys.argv) > 5 else None
BASE = {"max_pairs": 800, "max_unless": 100, "null_reps": 1}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


stocks = data.load("stocks")
X = pd.read_parquet(K.CACHE / "panel.parquet")
d = X.index.get_level_values(0)
X = X[d >= START]
tick = np.sort(X.index.get_level_values(1).unique())
keep = set(np.random.default_rng(SEED).choice(tick, min(NT, len(tick)), replace=False))
X = X[X.index.get_level_values(1).isin(keep)]
d = X.index.get_level_values(0)
ud = pd.DatetimeIndex(sorted(d.unique()))
wk = ud[[i for i in range(len(ud) - 1) if ud[i + 1].isocalendar().week != ud[i].isocalendar().week]]
X = X[d.isin(wk)]
cf = candles.build({k: v[[c for c in v.columns if c in keep]] for k, v in stocks.items()})
for k, v in cf.items():
    X[k] = v.stack(future_stack=True).reindex(X.index).astype("float32").values
del cf
O, C = stocks["Open"], stocks["Close"]
y = (C.shift(-5) / O.shift(-1) - 1).stack(future_stack=True).reindex(X.index)     # C33: bought at the next open
if TARGET == "move":
    y = y.abs()
y = y - y.groupby(level=0).transform("mean")
X = X.drop(columns=[c for c in X.columns if c.startswith("ev_activist")])
ok = y.notna()
X, y = X[ok], y[ok]
log(f"panel: {len(X):,} rows, {X.index.get_level_values(1).nunique()} tickers, {X.shape[1]} features, target {TARGET}")
R = T.tune(X, y, base=BASE, n_folds=4, holdout_frac=0.2, seed=SEED, knobs=KNOBS, log=log)
out = K.STATE / "research" / "algorithm" / "tuning"
out.mkdir(parents=True, exist_ok=True)
rec = {"target": TARGET, "seed": SEED, "tickers": NT, "start": str(START.date()), "base": BASE, **R.to_record(),
       "seconds": round(time.time() - t0)}
(out / f"tune_{TARGET}_{SEED}.json").write_text(json.dumps(rec, indent=1, default=float))
log(f"holdout: {R.holdout}")
log(f"chosen: {R.chosen}  (defaults {R.defaults})")
log_experiment({"event": "miner_self_tuning", "target": TARGET}, cfg={**BASE, **R.chosen}, seed=SEED,
               metrics={"holdout_default_ic": R.holdout["default_ic"], "holdout_chosen_ic": R.holdout["chosen_ic"],
                        "diff_lo": R.holdout["diff_lo"], "diff_hi": R.holdout["diff_hi"]},
               gates={"holdout_confirmed": R.holdout["verdict"] == "confirmed on holdout"},
               outcome="adopt" if R.adopted_any else "continue_testing", reason=R.holdout["verdict"],
               train_range=f"{START.date()}..holdout", test_range=f"holdout from {R.holdout['start']}",
               validation_range="rolling-origin folds", window_ids=[f"sample{SEED}"])
from engine.checkpoint import checkpoint_run                           # Bible P0.3
log(f"checkpoint: {checkpoint_run('miner_self_tuning', {'base': BASE, 'target': TARGET, 'tickers': NT, 'start': str(START.date())}, R.holdout, {'seed': SEED}, R.to_record(), artifacts={'result.json': str(out / f'tune_{TARGET}_{SEED}.json')})}")
