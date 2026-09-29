"""Volatility targeting through the REAL adaptive re-tester (offline; engine/adaptive.py is not edited).

Canon C38/C39: most weeks should move about 7% (band 5-10%). Only 22-23% of weeks do: the weekly distribution is too
wide. The lever tested here: at every rebalance, scale the target weights so the portfolio's PREDICTED weekly move is
about `target` - exposure = min(1, target / predicted), never leverage. predicted = sqrt(w' S w) with per-name weekly
vol = vol20 * sqrt(5) (vol20 is the snapshot's 20-day daily sd) and a constant pairwise correlation rho.
The snapshot is all it reads (point-in-time). Selection: the target that scores best by the tiered objective
(engine.objective.evaluate) on the EARLY revealed windows; judged once on the LATER ones against no targeting, with a
paired bootstrap of the per-window in-band share. A reference for the learning system: if it helps, the lever goes
into the basis search space so the system can LEARN it (canon C54), not be hand-set.

usage: voltarget_offline.py [--workers 3] [--train-share 0.62] [--rho 0.3]"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from engine import objective as O, adaptive as A, provenance
import basis_offline as BO

TARGETS = [None, 0.05, 0.06, 0.07, 0.08, 0.10]


def predicted_week_vol(weights, vol20, rho):
    w = weights.values.astype(float)
    s = vol20.reindex(weights.index).fillna(vol20.median()).values.astype(float) * np.sqrt(5)
    cov = rho * np.outer(s, s) + (1 - rho) * np.diag(s * s)
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


class VTSession(A.Session):
    def __init__(self, *a, target=None, rho=0.3, **k):
        super().__init__(*a, **k)
        self.vt_target, self.rho, self.vol20, self.vt_log = target, rho, None, []

    def on_day(self, day, px, closes_to_now, next_is_new_week, snap=None, px_open=None):
        if snap is not None and "vol20" in snap:
            self.vol20 = snap["vol20"]
        return super().on_day(day, px, closes_to_now, next_is_new_week, snap, px_open)

    def _trade(self, target, px, val, day, reason, decision_date=None):
        if (self.vt_target and reason in ("rebalance", "initial build") and isinstance(target, pd.Series)
                and len(target) and self.vol20 is not None):
            pv = predicted_week_vol(target, self.vol20, self.rho)
            expo = min(1.0, self.vt_target / pv) if pv > 0 else 1.0
            self.vt_log.append((str(day.date()), round(pv, 4), round(expo, 3)))
            target = target * expo
        super()._trade(target, px, val, day, reason, decision_date)


def replay_vt(cfg, w, meta, target, rho):
    S = VTSession(cfg, w["divs"], w["bps"], adaptive=True, meta=meta, long_term=w["ltm"], target=target, rho=rho)
    dec = {pd.Timestamp(k): v for k, v in w["snaps"].items()}
    sessions, opens = w["closes"].index, w["opens"]
    for i, d in enumerate(sessions):
        nxt_new = i + 1 >= len(sessions) or sessions[i + 1].isocalendar().week != d.isocalendar().week
        snap = dec.get(d) if S.needs_snapshot(nxt_new) else None
        S.on_day(d, w["closes"].loc[d], w["closes"].loc[:d], nxt_new, snap,
                 opens.loc[d] if opens is not None and d in opens.index else None)
    return S


def row_of(S):
    r = S.result()
    wk = np.array(S.weeks)
    row = {**O.week_row(wk), "max_dd": r["max_dd"], "year_return": r["year_return"]}
    expos = [e for _, _, e in getattr(S, "vt_log", [])]
    row["mean_exposure"] = float(np.mean(expos)) if expos else 1.0
    return row


def _job(task):
    wid, target, rho = task
    L = BO.loop()
    w = BO.window(wid)
    return wid, target, row_of(replay_vt(L.st["cfg"], w, L.st["meta"], target, rho))


def paired_boot(a, b, reps=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(d.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--train-share", type=float, default=0.62)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    L = BO.loop()
    ids = BO.revealed_ids(L)
    ends = {i: str(BO.window(i)["closes"].index[-1]) for i in ids}
    ids = sorted(ids, key=lambda i: ends[i])                   # train on the earlier-ending windows only
    if a.limit:
        ids = ids[:a.limit]
    cut = int(len(ids) * a.train_share)
    train, test = ids[:cut], ids[cut:]
    tasks = [(w, t, a.rho) for w in ids for t in TARGETS]
    rows = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for wid, t, r in ex.map(_job, tasks):
            rows[(wid, t)] = r
            print(f"{wid} target {t}: in_band {r['in_band']:.0%} exposure {r['mean_exposure']:.2f}", flush=True)
    score = {t: O.evaluate([rows[(w, t)] for w in train]) for t in TARGETS}
    best = max(TARGETS, key=lambda t: score[t].key)
    base_te = [rows[(w, None)] for w in test]
    best_te = [rows[(w, best)] for w in test]
    band = paired_boot([r["in_band"] for r in best_te], [r["in_band"] for r in base_te])
    risk = paired_boot([r["worst5"] for r in best_te], [r["worst5"] for r in base_te])
    out = {"provenance": provenance.stamp({"targets": TARGETS, "rho": a.rho}, 0), "train": train, "test": test,
           "train_scores": {str(t): O.summary(score[t]) for t in TARGETS}, "chosen": best,
           "test_base": O.summary(O.evaluate(base_te)), "test_chosen": O.summary(O.evaluate(best_te)),
           "test_in_band_diff": {"mean": band[0], "lo90": band[1], "hi90": band[2]},
           "test_worst5_diff": {"mean": risk[0], "lo90": risk[1], "hi90": risk[2]},
           "per_window": {f"{w}|{t}": rows[(w, t)] for (w, t) in rows}, "seconds": round(time.time() - t0)}
    d = ROOT / "state" / "research" / "voltarget"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"voltarget_rho{a.rho}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("chosen", "test_base", "test_chosen", "test_in_band_diff", "test_worst5_diff")},
                     indent=1, default=float))
    from engine.improve import log_experiment
    helps = band[1] > 0
    log_experiment({"event": "voltarget_offline"}, cfg={"targets": TARGETS, "rho": a.rho, "chosen": best}, seed=0,
                   window_ids=ids, metrics={"in_band_diff": band[0], "worst5_diff": risk[0]},
                   gates={"in_band_lower_bound_above_0": helps}, outcome="continue_testing",
                   reason=("candidate lever for the basis search" if helps else "no demonstrated gain in band share"),
                   train_range="earlier revealed windows", validation_range="-", test_range="later revealed windows")


if __name__ == "__main__":
    main()
