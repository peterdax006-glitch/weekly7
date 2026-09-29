"""Timeline dial through the REAL adaptive re-tester (read-only offline experiment; engine/adaptive.py is not edited).

usage: dial_offline.py [--workers 3] [--train-share 0.62] [--seed 0] [--selfcheck]

The dial is applied from outside: `DialSession` is an adaptive.Session whose adapter.step is wrapped so that, at every
rebalance, the dial's k / pool_q / brake replace the adapter's for that week, and `_trade` scales rebalance weights by
the dial's exposure. The dial sees only completed weekly returns and the m_vix_term stress reading in the snapshot it
is already handed (no dates). Training = pick the best of a small fixed set of dial settings on the EARLY windows by
the tiered objective; the admission gate (engine.timeline.gate_rows) then judges it on the LATER windows.
Exact hook if adopted (not applied): engine/adaptive.py Session.on_day, right after
`self.cfg = self.adapter.step(...)`:  out, self.dial = timeline.step(self.weeks, {"stress": m_vix_term}, None, self.dial);
`self.cfg = {**self.cfg, **timeline.cfg_overrides(out)}`; and scale `target` by out.exposure before `self.pending = ...`.
`--selfcheck` proves DialSession with the dial off reproduces adaptive.replay exactly on one window."""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from engine import config as K, objective as O, timeline as T, adaptive as A, provenance
import basis_offline as BO

VARIANTS = [{}, {"gain": 0.75}, {"gain": 3.0}, {"max_step": 0.2, "deadband": 0.2}, {"brake_dd": 0.10, "over_limit": 0.3}]


class DialSession(A.Session):
    def __init__(self, *a, dial=None, **k):
        super().__init__(*a, **k)
        self.dial_params, self.dial_state, self.exposure, self.dial_log = dial, T.DialState(), 1.0, []
        if dial is not None and self.adapter is not None:
            inner = self.adapter.step

            def step(today, snap, closes_to_now, divs, held):
                cfg = inner(today, snap, closes_to_now, divs, held)
                vt = float(snap["m_vix_term"].iloc[0]) if "m_vix_term" in snap else float("nan")
                regime = {"stress": vt} if np.isfinite(vt) else None
                out, self.dial_state = T.step(self.weeks, regime, None, self.dial_state, self.dial_params)
                self.exposure = out.exposure
                self.dial_log.append((str(today.date()), out.k, round(out.exposure, 2), out.brake, round(out.aggr, 2)))
                return {**cfg, **T.cfg_overrides(out)}
            self.adapter.step = step

    def _trade(self, target, px, val, day, reason, decision_date=None):
        if self.dial_params is not None and reason in ("rebalance", "initial build") and isinstance(target, pd.Series):
            target = target * self.exposure
        super()._trade(target, px, val, day, reason, decision_date)


def replay_dial(cfg, w, meta, dial):
    """Same day loop as adaptive.replay (no scramble), on a DialSession. Keep in sync with A.replay."""
    S = DialSession(cfg, w["divs"], w["bps"], adaptive=True, meta=meta, long_term=w["ltm"], dial=dial)
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
    row = {**O.week_row(wk), "max_dd": r["max_dd"], "year_return": r["year_return"]}    # tier 2 reads the daily drawdown
    return row


def _job(task):
    wid, cfg, meta, dial = task
    w = BO.window(wid)
    S = replay_dial(cfg, w, meta, None if dial is None else T.DialParams(**dial))
    row = row_of(S)
    row["mean_k"] = float(np.mean([x[1] for x in S.dial_log])) if S.dial_log else float("nan")
    row["mean_exposure"] = float(np.mean([x[2] for x in S.dial_log])) if S.dial_log else 1.0
    row["brake_share"] = float(np.mean([x[3] is not None for x in S.dial_log])) if S.dial_log else 0.0
    return row


def selfcheck():
    L = BO.loop()
    wid = BO.revealed_ids(L)[0]
    w = BO.window(wid)
    ref = A.replay(L.st["cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=L.st["meta"], opens=w["opens"],
                   long_term=w["ltm"])
    mine = replay_dial(L.st["cfg"], w, L.st["meta"], None)
    ok = ref.decisions == mine.decisions and abs(ref.result()["year_return"] - mine.result()["year_return"]) < 1e-12
    print("selfcheck (dial off == adaptive.replay):", "OK" if ok else "MISMATCH", wid)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--train-share", type=float, default=0.62)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck and not selfcheck():
        sys.exit(1)
    L = BO.loop()
    t0 = time.perf_counter()
    ids = BO.revealed_ids(L)
    ends = {i: str(BO.window(i)["closes"].index[-1]) for i in ids}
    ids.sort(key=lambda i: ends[i])
    cut = int(len(ids) * a.train_share)
    train, test = ids[:cut], ids[cut:]
    cfg, meta = dict(L.st["cfg"]), dict(L.st["meta"])
    base_params = asdict(T.DialParams())
    with ProcessPoolExecutor(a.workers) as pool:
        run = lambda wids, d: list(pool.map(_job, [(i, cfg, meta, d) for i in wids]))
        base_tr, base_te = run(train, None), run(test, None)
        cands = []
        for v in VARIANTS:
            d = {**base_params, **v}
            rows = run(train, d)
            cands.append((O.evaluate(rows), d, rows))
            print("train variant", v, O.summary(cands[-1][0])["key"], f"in_band {cands[-1][0].t1:.0%}", flush=True)
        best = max(cands, key=lambda c: (c[0].key, c[0].soft))
        dial_te = run(test, best[1])
    gate = T.gate_rows(base_tr, best[2], base_te, dial_te, best[1], seed=a.seed)
    lab = [ends[i][:4] for i in test]
    rep = {"seed": a.seed, "train": train, "test": test, "variants_tried": len(VARIANTS), "chosen": best[1],
           "gate": gate.summary(), "checks": gate.checks, "numbers": gate.numbers,
           "train_base": O.summary(O.evaluate(base_tr)), "train_dial": O.summary(best[0]),
           "test_base": O.summary(O.evaluate(base_te)), "test_dial": O.summary(O.evaluate(dial_te)),
           "test_by_era_base": O.by_group(base_te, lab), "test_by_era_dial": O.by_group(dial_te, lab),
           "test_mean_k": float(np.mean([r["mean_k"] for r in dial_te])), "test_mean_exposure": float(np.mean([r["mean_exposure"] for r in dial_te])),
           "test_brake_share": float(np.mean([r["brake_share"] for r in dial_te])),
           "per_window_in_band": [[round(x["in_band"], 3), round(y["in_band"], 3)] for x, y in zip(base_te, dial_te)],
           "provenance": provenance.stamp({"cfg": cfg, "meta": meta, "dial": best[1]}, a.seed), "seconds": round(time.perf_counter() - t0, 1)}
    out = K.STATE / "research" / "timeline_basis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"dial_offline_seed{a.seed}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("gate", "test_base", "test_dial", "test_mean_k", "test_mean_exposure", "seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()
