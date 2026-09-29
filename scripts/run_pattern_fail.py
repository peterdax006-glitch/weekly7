"""Pattern-failure exits (Bible Phase 15, family 5) on REAL data, wired to B04's engine/pattern_movers.py.

1. mine a direction PatternBank from panel rows dated <= --train-end (labels: forward 5-session excess return),
2. score every later (date, ticker) row with that FIXED bank (no re-mining, no look-ahead),
3. a position's pattern is 'failing' at a close when its score has fallen by more than `drop` x the training score sd
   below the score it was entered on (entry = prior close); the exit fills at the next open (exits.run_exit),
4. run the exit walk-forward with the failure family enabled, plus two direct checks of whether the flag carries
   information at all: remaining return after a flag vs no flag, against a control where scores are shuffled across
   positions within the same week.
Outputs -> state/research/pattern_fail/. Read-only on caches; no network.
    python scripts/run_pattern_fail.py [--n-tickers 400] [--per-week 20]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engine import config as K  # noqa: E402
from engine import exits as E  # noqa: E402
from engine import pattern_movers as PM  # noqa: E402
from engine.provenance import stamp  # noqa: E402
from run_exits_stops import build_paths, clean, load_fields, sample_tickers  # noqa: E402

OUT = K.STATE / "research" / "pattern_fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickers", type=int, default=400)
    ap.add_argument("--per-week", type=int, default=20)
    ap.add_argument("--train-start", default="2013-01-02")
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--drop", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tickers = sample_tickers(a.n_tickers, a.seed)
    fr = load_fields(tickers, a.train_start, a.end)
    X = pd.read_parquet(K.CACHE / "panel.parquet", filters=[("ticker", "in", tickers)])
    if not isinstance(X.index, pd.MultiIndex):
        X = X.set_index(["date", "ticker"])
    X = X.loc[(X.index.get_level_values(0) >= a.train_start) & (X.index.get_level_values(0) <= a.end)].astype("float32")
    close = fr["close"]
    fwd = (close.shift(-5) / close - 1).stack()
    fwd.index.names = ["date", "ticker"]
    train_mask = X.index.get_level_values(0) <= pd.Timestamp(a.train_end) - pd.Timedelta(days=10)
    Xtr = X[train_mask]
    ytr = fwd.reindex(Xtr.index).dropna()
    Xtr = Xtr.loc[ytr.index]
    print(f"panel rows={len(X)} train rows={len(Xtr)} mining...", flush=True)
    dir_bank, _ = PM.mine_banks(Xtr, ytr, pd.Timestamp(a.train_end), seed=a.seed)
    print(f"bank patterns={len(dir_bank)} ({time.time() - t0:.0f}s)", flush=True)
    Xte = X[X.index.get_level_values(0) > pd.Timestamp(a.train_end)]
    score, nfire = PM.score_panel(dir_bank, Xte)
    train_sd = float(PM.score_panel(dir_bank, Xtr)[0].std())
    S = score.unstack("ticker").reindex(close.index)

    p, _ = build_paths(fr, a.per_week)
    p, dq = clean(p)
    idx = close.index
    pos = idx.get_indexer(pd.DatetimeIndex(p.week))
    col = S.columns.get_indexer(p.ticker)
    ok = (pos >= 1) & (col >= 0) & (p.week > np.datetime64(a.train_end))
    p = p.take(np.flatnonzero(ok))
    pos, col = pos[ok], col[ok]
    path = np.stack([S.values[pos + d, col] for d in range(5)], 1)
    entry = S.values[pos - 1, col]
    good = np.isfinite(path).all(1) & np.isfinite(entry)
    p, path, entry = p.take(np.flatnonzero(good)), path[good], entry[good]
    p.kind = np.where(entry > 0, "pat_pos", "pat_neg").astype(object)
    drop = a.drop * train_sd
    flags = path < (entry[:, None] - drop)
    p.fail = flags
    print(f"positions={len(p)} weeks={len(np.unique(p.week))} fail_flag_rate_per_night={flags.mean():.3f} train_score_sd={train_sd:.4f} {dq}", flush=True)

    # does the flag carry information? remaining return from the flag close to the last close, flagged vs not (same nights)
    rem = p.c[:, -1:] / p.c[:, :-1] - 1
    f4 = flags[:, :-1]
    info = {"remaining_ret_after_flag": float(rem[f4].mean()) if f4.any() else None, "remaining_ret_no_flag": float(rem[~f4].mean()),
            "n_flag_nights": int(f4.sum())}
    rng = np.random.default_rng(a.seed)
    ctrl = []
    for _ in range(50):                        # control: shuffle scores across positions inside each week
        sh = flags.copy()
        for w in np.unique(p.week):
            m = np.flatnonzero(p.week == w)
            sh[m] = flags[rng.permutation(m)]
        s4 = sh[:, :-1]
        if s4.any():
            ctrl.append(rem[s4].mean() - rem[~s4].mean())
    info["control_shuffled_diff_mean"] = float(np.mean(ctrl))
    info["control_shuffled_diff_sd"] = float(np.std(ctrl))
    info["real_diff"] = (info["remaining_ret_after_flag"] - info["remaining_ret_no_flag"]) if f4.any() else None
    info["real_diff_z_vs_control"] = float((info["real_diff"] - np.mean(ctrl)) / (np.std(ctrl) + 1e-12)) if f4.any() else None
    print(info, flush=True)

    rules = E.default_rules(has_fail=True)
    wf = E.walk_forward(p, rules, min_train_weeks=52, block_weeks=26, seed=a.seed)
    rep = wf.report()
    rep.to_csv(OUT / "exit_oos_with_fail_family.csv", index=False)
    E.rule_usage(wf).to_csv(OUT / "rule_usage.csv", index=False)
    hold = p.take(np.flatnonzero(wf.tested))
    tab = E.evaluate_rules(hold, [r for r in rules if r.name in ("week_end", "pattern_fail", "t10_fail", "target_10", "t10_trail6_fail")])
    tab.to_csv(OUT / "fail_family_on_oos_window.csv", index=False)
    E.era_breakdown(wf).to_csv(OUT / "era_breakdown.csv", index=False)
    txt = E.format_report(wf)
    (OUT / "report.txt").write_text(txt, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({**stamp({"n_tickers": a.n_tickers, "train_end": a.train_end, "drop_sd": a.drop}, a.seed),
                                                  "bank_patterns": len(dir_bank), "positions": len(p), "flag_information": info,
                                                  "report": rep.to_dict("records"), "seconds": round(time.time() - t0, 1)}, indent=1, default=float),
                                      encoding="utf-8")
    pd.set_option("display.width", 220)
    print(txt)
    print(tab[["rule", "n", "mean", "hit_rate", "tail5_mean", "days_held", "mean_week", "cvar5", "in_band"]].to_string(index=False))
    print(f"done {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
