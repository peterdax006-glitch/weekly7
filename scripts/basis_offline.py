"""Offline basis search on the archived REAL windows (read-only; nothing is adopted or written to the loop state).

usage: basis_offline.py [--workers 3] [--n-start 12] [--n-screen 8] [--n-top 2] [--train-share 0.62] [--seed 0]

Uses the loop's own legitimate re-tester path (livesim_loop2.load_window / run_window -> adaptive.replay). Windows: the
archived directories with weekly snapshots, minus backups (leading '_') and minus windows listed in loop2.json that are
not revealed yet (sealed = not evidence). Walk-forward: the search sees only windows ending on/before the cut; the
incumbent and the adopted basis are then both replayed on the LATER windows. Result: state/research/timeline_basis/
basis_offline_seed<N>.json (+ provenance stamp). One replay is ~30 s, so windows are spread over worker processes."""
import argparse, importlib.util, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import config as K, objective as O, basis_search as B, provenance

_LOOP = None
_CACHE = {}


def loop():
    """The loop module, imported with a neutral argv so that importing never starts anything."""
    global _LOOP
    if _LOOP is None:
        argv, sys.argv = sys.argv, ["livesim_loop2.py"]
        try:
            spec = importlib.util.spec_from_file_location("livesim_loop2", ROOT / "scripts" / "livesim_loop2.py")
            _LOOP = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_LOOP)
        finally:
            sys.argv = argv
    return _LOOP


def window(wid):
    L = loop()
    if wid not in _CACHE:
        if len(_CACHE) > 6:                                  # keep worker memory bounded
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[wid] = L.load_window(L.DIR / wid)
    return _CACHE[wid]


def _job(task):
    wid, cfg, meta = task
    return loop().run_window(window(wid), cfg, meta)


def revealed_ids(L):
    """Windows the loop itself has not yet revealed are still sealed: excluded from evidence."""
    sealed = {w["run_id"] for w in L.st["windows"] if "revealed" not in w}
    return [a.name for a in L.archive_dirs(L.DIR) if a.name not in sealed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-start", type=int, default=12)
    ap.add_argument("--n-screen", type=int, default=8)
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--train-share", type=float, default=0.62)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    L = loop()
    t0 = time.perf_counter()
    ids = revealed_ids(L)
    ends = {i: str(window(i)["closes"].index[-1]) for i in ids}
    ids.sort(key=lambda i: ends[i])
    wins = [{"id": i, "end": ends[i]} for i in ids]
    cut = int(len(wins) * a.train_share)
    as_of = wins[cut - 1]["end"]
    inc_cfg, inc_meta = dict(L.st["cfg"]), dict(L.st["meta"])
    rep = {"seed": a.seed, "windows": len(wins), "train_windows": cut, "test_windows": len(wins) - cut, "as_of": as_of,
           "excluded": sorted({w["run_id"] for w in L.st["windows"] if "revealed" not in w}),
           "search": {"n_start": a.n_start, "n_screen": a.n_screen, "n_top": a.n_top}}
    with ProcessPoolExecutor(a.workers) as pool:
        batch = lambda tasks: list(pool.map(_job, [(w["id"], c, m) for w, c, m in tasks]))
        sc = B.SearchConfig(n_start=a.n_start, n_screen=a.n_screen, n_top=a.n_top, seed=a.seed)
        res = B.BasisSearch(None, wins, inc_cfg, inc_meta, sc, L.CFG_SPACE, L.META_SPACE, evaluate_batch=batch).run(as_of=as_of)
        later = wins[cut:]
        oos_inc = batch([(w, inc_cfg, inc_meta) for w in later])
        oos_new = batch([(w, res.cfg, res.meta) for w in later]) if res.adopted else oos_inc
    ends_ = [w["end"] for w in later]
    rep["search_result"] = res.to_record()
    rep["adopted_cfg"], rep["adopted_meta_changes"] = res.cfg, B.changed_keys(inc_meta, res.meta)
    rep["oos"] = {"incumbent": O.summary(O.evaluate(oos_inc)), "adopted": O.summary(O.evaluate(oos_new)),
                  "firewall": O.firewall(O.evaluate(oos_inc), O.evaluate(oos_new))[1],
                  "per_window_in_band": [[round(x["in_band"], 3), round(y["in_band"], 3)] for x, y in zip(oos_inc, oos_new)],
                  "mean_week": [float(np.mean([r["mean_week"] for r in oos_inc])), float(np.mean([r["mean_week"] for r in oos_new]))]}
    # what the sealed-window round reported vs the proxy: fixed-basket proxy numbers live in report_seed*.json
    rep["provenance"] = provenance.stamp({"cfg": inc_cfg, "meta": inc_meta}, a.seed)
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    out = K.STATE / "research" / "timeline_basis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"basis_offline_seed{a.seed}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("windows", "train_windows", "test_windows", "excluded", "oos", "seconds")}, indent=1, default=str))
    print("search:", res.reason, "| replays:", res.n_evals)


if __name__ == "__main__":
    main()
