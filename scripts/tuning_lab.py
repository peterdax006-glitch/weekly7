"""Tuning lab (canon C7): hundreds of random historical 'months' per configuration.

Each configuration is simulated day by day over the whole out-of-sample period once; then
N random 4-week windows (a fresh $1,000 each) are drawn from it. Years are split so tuning never
sees the exam: configurations are ranked on TUNE years and the winner must also beat the
champion on the LOCKED years. Every configuration is logged to the experiment registry.

usage: tuning_lab.py <shard> <n_shards> <n_configs> [seed]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, data, backtest, policy

shard, n_shards, n_configs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
import os
SUFFIX = os.environ.get("W7_SUFFIX", "")
OUT = K.STATE / "research" / f"tuning{SUFFIX}"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed)
ROUND = os.environ.get("W7_ROUND", "1")
if ROUND == "1":
    SPACE = {"k": [3, 4, 5, 6], "exit_q": [0.6, 0.7, 0.8, 0.9], "brake": [None, 0.05, 0.08, 0.12],
             "max_per_sector": [None, 2, 3], "w_model": [0.3, 0.5, 0.7], "min_dv": [1e7, 2e7, 5e7],
             "vol_filter": [True, False], "pick": ["top", "hivol"], "pool_q": [0.9, 0.95, 0.98]}
else:   # round 2: refine around the round-1 winner region (hivol, few names)
    SPACE = {"k": [2, 3, 4], "exit_q": [0.8, 0.85, 0.9, 0.95], "brake": [None, 0.05, 0.08, 0.1],
             "max_per_sector": [None, 2], "w_model": [0.3, 0.5, 0.7], "min_dv": [2e7, 5e7, 1e8],
             "vol_filter": [True, False], "pick": ["hivol", "top"], "pool_q": [0.9, 0.95, 0.97, 0.99]}
configs = [{"k": 3, "exit_q": 0.9, "brake": 0.08, "max_per_sector": None, "w_model": 0.5, "min_dv": 5e7,
            "vol_filter": False, "pick": "hivol", "pool_q": 0.95, "name": "CHAMPION_v1.3"},
           {"k": 4, "exit_q": 0.8, "brake": 0.08, "max_per_sector": 2, "w_model": 0.5, "min_dv": 2e7,
            "vol_filter": True, "pick": "top", "pool_q": 0.95, "name": "CHAMPION_v1.2"}]
while len(configs) < n_configs:
    c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    c["name"] = "cfg_" + "_".join(str(c[k]) for k in SPACE)
    if c["name"] not in {x["name"] for x in configs}:
        configs.append(c)
mine = configs[shard::n_shards]

stocks = {"Close": data.load("stocks")["Close"]}
X = pd.read_parquet(K.CACHE / f"panel{SUFFIX}.parquet", columns=["vol20", "max20", "log_dv", "ev_red_flag", "ev_offering"])
P = pd.read_parquet(K.CACHE / f"oos_preds{SUFFIX}.parquet", columns=["mu_raw", "evidence"])
sic = pd.read_parquet(K.CACHE / "sic.parquet")
divs = {t: policy.sic_division(c) for t, c in zip(sic["ticker"], sic["sic"])}
spy = data.load("market")["Close"]["SPY"]

# cheating control (shard 0 only): a clairvoyant picker that sees next week's returns.
if shard == 0:
    C = stocks["Close"]
    fut = (C.shift(-5) / C - 1).stack(future_stack=True)
    configs_cheat = {**configs[0], "name": "CHEAT_CONTROL_sees_future"}
    mine = [configs_cheat] + mine
for c in mine:
    t0 = time.time()
    s = policy.score(P, c["w_model"])
    e, st = backtest.run_topk(s, X, stocks, k=c["k"], exit_q=c["exit_q"], brake=c["brake"],
                              params={"vol_filter": c["vol_filter"], "min_dv": c["min_dv"]},
                              sectors=divs if c["max_per_sector"] else None,
                              max_per_sector=c["max_per_sector"], name=c["name"], pick=c["pick"], pool_q=c["pool_q"],
                              peek=fut if c["name"].startswith("CHEAT") else None)
    e.to_frame().to_parquet(OUT / f"eq_{c['name']}.parquet")
    (OUT / f"cfg_{c['name']}.json").write_text(json.dumps({**c, "stats": st, "secs": round(time.time() - t0)}, default=float))
    print(c["name"], "final", round(float(e.iloc[-1])), f"{time.time() - t0:.0f}s", flush=True)
