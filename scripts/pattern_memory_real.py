"""Timeline pattern memory on the REAL weekly panel (Bible phases 4, 5, 9; canon C55-C60). Read-only on data/cache.

Stage `mine`  (heavy, resumable): walk forward through half-year block ends T; at each T run the real PatternMiner on rows
              whose labels closed before T (seeded ticker sample), keep the active/rescoped patterns, checkpoint them to
              state/research/pattern_memory/real/blocks/miner_<T>.csv (+ .json with the try count). Existing blocks are skipped.
Stage `eval`  (light): rebuild the memory from the checkpoints - each pattern is REGISTERED at the block that found it and its
              evidence is the whole calendar quarters AFTER that block, ingested when their labels have matured - then report,
              over real time, how many patterns are universal / local / disregarded, first-noticed timelines, an era table,
              and the out-of-sample value of the view: signed next-window excess of patterns the view uses vs those it
              disregards vs (a) a random subset of the same registry and (b) random patterns. Then fit the memory thresholds
              on the EARLY blocks only and judge chosen vs default parameters on the LATER blocks. Prefix-invariance audit
              (0 violations required) and chain verification are part of the report.
Stage `all`   runs both.

usage: pattern_memory_real.py --stage {mine,eval,all} [--tickers 500] [--seed 7] [--first 2015-07-01] [--last 2026-07-01]
RAM: waits (polls every 60 s, gives up after 20 min) until >= 2.5 GB is free before loading the panel."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from engine import config as K
from engine import pattern_memory as pm
from engine import pattern_memory_eval as ev
from engine.pattern_lifecycle import Panel

OUT = K.STATE / "research" / "pattern_memory" / "real"
STORE = K.STATE / "pattern_memory" / "real"
ERAS = {"2015-2018": (2015, 2018), "2019-2022": (2019, 2022), "2023-2026": (2023, 2026)}
MIN_FREE_GB = 2.5


def wait_for_ram(min_gb=MIN_FREE_GB, poll=60, give_up=1200):
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available / 1e9 < min_gb:
        if time.time() - t0 > give_up:
            raise SystemExit(f"gave up after {give_up}s: free RAM stayed below {min_gb} GB")
        print(f"waiting for RAM ({psutil.virtual_memory().available / 1e9:.1f} GB free)", flush=True)
        time.sleep(poll)


def block_ends(a):
    b = ev.block_schedule(a.first, a.last, months=6)
    return b + [pd.Timestamp(a.end)]                  # final entry closes the last evaluation window; nothing is mined at it


def ckpt(T):
    return OUT / "blocks" / f"miner_{pd.Timestamp(T).date()}"


# ---------------------------------------------------------------- stage: mine
def mine(a):
    import run_pattern_bank as rpb
    from engine.patterns import PatternMiner
    (OUT / "blocks").mkdir(parents=True, exist_ok=True)
    todo = [T for T in block_ends(a)[:-1] if not ckpt(T).with_suffix(".json").exists()]
    print(f"{len(todo)} block(s) to mine", flush=True)
    if not todo:
        return
    wait_for_ram()
    t0 = time.time()
    X, y, pick = rpb.load(a.tickers, a.seed)
    print(f"loaded {len(X):,} rows, {len(pick)} tickers [{time.time() - t0:.0f}s]", flush=True)
    d = X.index.get_level_values(0)
    ud = pd.DatetimeIndex(sorted(d.unique()))
    for T in todo:
        step = time.time()
        cut = ud[max(ud.searchsorted(T) - 2, 0)]                                 # labels close before T
        tr = d <= cut
        yv = y[tr]
        yv = yv - yv.groupby(level=0).transform("mean")
        M = PatternMiner({"max_pairs": a.max_pairs, "max_unless": 100, "null_reps": 1, "seed": a.seed,
                          "max_rows": 400_000}).fit(X[tr], yv, now=T)
        P = M.patterns
        cols = [c for c in ("key_named", "effect", "t_conf", "t_disc", "status", "p_real", "p_coincidence") if c in P.columns]
        (P[cols] if len(P) else pd.DataFrame(columns=cols)).to_csv(ckpt(T).with_suffix(".csv"), index=False)
        rep = {"T": str(T.date()), "tested": int(M.report.get("tested", len(P))), "candidates_in_frame": int(len(P)),
               "by_status": P["status"].value_counts().to_dict() if len(P) else {}, "seconds": round(time.time() - step)}
        ckpt(T).with_suffix(".json").write_text(json.dumps(rep))                 # written last: its existence = block done
        print(json.dumps(rep), flush=True)


# ---------------------------------------------------------------- stage: eval
def select_candidates(fr, top, min_t):
    """Which of a block's mined candidates the memory is asked to follow. The miner's own status gate is strict enough that
    it keeps almost nothing on this panel, so the memory gets the top `top` candidates whose discovery and confirmation
    halves agree in sign and whose confirmation |t| >= min_t; the memory's timeline evidence, not the miner, decides."""
    if fr is None or len(fr) == 0 or "t_conf" not in fr:
        return pd.DataFrame(columns=["key_named", "effect"])
    fr = fr[fr["status"].ne("duplicate")] if "status" in fr else fr
    ok = (np.sign(fr["t_disc"]) == np.sign(fr["t_conf"])) & (fr["t_conf"].abs() >= min_t) & fr["effect"].notna()
    fr = fr[ok].assign(_a=lambda d: d["t_conf"].abs()).sort_values(["_a", "key_named"], ascending=[False, True]).head(top)
    return fr[["key_named", "effect"]]


def load_found(blocks, top=80, min_t=1.5):
    found = {}
    for k, T in enumerate(blocks[:-1]):
        j = ckpt(T).with_suffix(".json")
        if not j.exists():
            raise SystemExit(f"missing miner checkpoint for {T.date()}; run --stage mine first")
        rep = json.loads(j.read_text())
        c = ckpt(T).with_suffix(".csv")
        fr = pd.read_csv(c) if c.exists() and c.stat().st_size > 5 else pd.DataFrame(columns=["key_named", "effect"])
        found[k] = (select_candidates(fr, top, min_t), rep["tested"])
    return found


def load_panel(a):
    import run_pattern_bank as rpb
    wait_for_ram()
    X, y, pick = rpb.load(a.tickers, a.seed)
    panel = Panel.build(X, y, X.index.get_level_values(0).max() + pd.Timedelta(days=30))
    del X, y
    return panel, pick


def mode_series(mem, sc, blocks):
    rows = []
    for k, T in enumerate(blocks[:-1]):
        v = mem.view(T + pd.Timedelta(days=1), sc.ctx[k])
        c = pd.Series([w.mode for w in v.weights.values()]).value_counts().to_dict()
        rows.append({"T": str(T.date()), "registered": sum(it["T_reg"] <= T for it in sc.reg.items.values()),
                     "with_evidence": len(v.weights), "universal": c.get("universal", 0), "local": c.get("local", 0),
                     "disregarded": c.get("disregarded", 0), "cum_tries": mem.cumulative_tries()["total_tries"]})
    return pd.DataFrame(rows)


def first_noticed_table(mem, reg):
    an = pm.Analytics(mem)
    rows = []
    end = pd.Timestamp("2100-01-01")
    for key, it in reg.items.items():
        fn = an.first_noticed(key, end)
        rows.append({"key": key, "registered": str(it["T_reg"].date()), "first_evidence": fn["first_obs_date"] if fn else None,
                     "lag_days": (pd.Timestamp(fn["first_obs_date"]) - it["T_reg"]).days if fn else None})
    return pd.DataFrame(rows)


def evaluate(a):
    blocks = block_ends(a)
    found = load_found(blocks, a.top, a.min_t)
    panel, pick = load_panel(a)
    if STORE.exists():
        shutil.rmtree(STORE)
    t0 = time.time()
    mem, reg, cache, qs = ev.build_memory(STORE, panel, blocks[:-1] + [blocks[-1]], found, a.seed)
    print(f"memory built: {len(reg)} patterns, {len(mem.records())} observations [{time.time() - t0:.0f}s]", flush=True)
    sc = ev.Scorer(panel, reg, cache, blocks, seed=a.seed, n_random=a.n_random)
    n_eval = len(blocks) - 1
    split = int(n_eval * a.fit_share)
    fit_b, judge_b = list(range(0, split)), list(range(split, n_eval))
    defaults_tab = sc.score(mem)
    fitted = ev.fit_thresholds(mem, sc, fit_b, judge_b, n_configs=a.configs, seed=a.seed)
    chosen_tab = sc.score(mem, fitted["chosen"])
    modes = mode_series(mem, sc, blocks)
    fn = first_noticed_table(mem, reg)
    an = pm.Analytics(mem)
    end = blocks[-1]
    rep_end = an.report(end, sc.ctx[-1])
    eras = an.era_breakdown(end, ERAS)
    audit_dates = [blocks[len(blocks) // 3], blocks[len(blocks) // 2], blocks[-3]]
    audit = pm.audit_prefix_invariance(mem, audit_dates, sc.ctx[len(blocks) // 2])
    chain = mem.verify()
    OUT.mkdir(parents=True, exist_ok=True)
    defaults_tab.to_csv(OUT / "score_defaults.csv", index=False)
    chosen_tab.to_csv(OUT / "score_chosen.csv", index=False)
    modes.to_csv(OUT / "modes_over_time.csv", index=False)
    fn.to_csv(OUT / "first_noticed.csv", index=False)
    eras.to_csv(OUT / "era_breakdown.csv", index=False)
    rep_end.to_csv(OUT / "patterns_at_end.csv", index=False)
    tries = mem.cumulative_tries()
    tabs = fitted.pop("judge_tables")
    res = {"tickers": len(pick), "seed": a.seed, "blocks": [str(b.date()) for b in blocks],
           "fit_blocks": [str(blocks[i].date()) for i in fit_b], "judge_blocks": [str(blocks[i].date()) for i in judge_b],
           "patterns_registered": len(reg), "observations": len(mem.records()),
           "cumulative_tries": {k: tries[k] for k in ("total_tries", "distinct_keys", "runs", "expected_best_null_t")},
           "all_blocks_defaults": ev.summarise(defaults_tab), "all_blocks_chosen": ev.summarise(chosen_tab),
           "walk_forward_fit": fitted, "prefix_invariance_violations": audit, "chain": {k: v for k, v in chain.items() if k != "head"},
           "final_mode_counts": rep_end["mode"].value_counts().to_dict() if len(rep_end) else {},
           "first_noticed_lag_days": fn["lag_days"].describe().to_dict() if len(fn) else {},
           "seconds": round(time.time() - t0)}
    try:
        from engine.provenance import stamp
        res["provenance"] = stamp({"tickers": a.tickers, "first": a.first, "last": a.last, "configs": a.configs}, a.seed)
    except Exception as e:
        res["provenance"] = {"unavailable": str(e)}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    write_report(res, modes, defaults_tab, tabs, an, end, sc.ctx[-1], fn)
    print(json.dumps({k: res[k] for k in ("patterns_registered", "observations", "prefix_invariance_violations", "chain",
                                          "final_mode_counts")}, default=str))
    return 0 if not audit and chain["ok"] else 1


def _fmt(x, p=5):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{p}f}"


def write_report(res, modes, tab, judge_tabs, an, end, ctx_end, fn):
    L = ["# Timeline pattern memory on the real weekly panel", "",
         f"{res['tickers']} seeded tickers, {res['patterns_registered']} patterns registered from {len(res['blocks']) - 1} "
         f"half-year miner blocks, {res['observations']} post-registration quarterly observations. Cumulative candidate tries: "
         f"{res['cumulative_tries']['total_tries']} (best-of-many null |t| ~ {res['cumulative_tries']['expected_best_null_t']:.2f}). "
         f"Chain ok: {res['chain']['ok']}. Prefix-invariance violations: {len(res['prefix_invariance_violations'])}.", "",
         "## Patterns by mode over real time", "", modes.to_string(index=False), "",
         "## Out-of-sample value of the view (defaults, all blocks)", "",
         "Signed weekly excess return in the NEXT half-year for patterns the view uses (weight > 0), the ones it disregards,"
         " a random subset of the same size drawn from the view, and random patterns. Blocks are the independent unit.", ""]
    s = res["all_blocks_defaults"]
    L += [f"- blocks: {s.get('blocks')} (with >= 5 used patterns: {s.get('blocks_with_use')}); mean used {s.get('mean_n_use', 0):.1f}, "
          f"mean disregarded {s.get('mean_n_off', 0):.1f}",
          f"- used {_fmt(s.get('use'))}  weighted {_fmt(s.get('use_w'))}  disregarded {_fmt(s.get('off'))}  "
          f"random subset {_fmt(s.get('subset'))}  random patterns {_fmt(s.get('random'))}  registry baseline {_fmt(s.get('baseline'))}",
          f"- used minus disregarded {_fmt(s.get('use_minus_off'))} (t {_fmt(s.get('use_minus_off_t'), 2)}); used minus random subset "
          f"{_fmt(s.get('use_minus_subset'))} (t {_fmt(s.get('use_minus_subset_t'), 2)}); used minus random patterns "
          f"{_fmt(s.get('use_minus_random'))} (t {_fmt(s.get('use_minus_random_t'), 2)})", "",
          "## Walk-forward threshold fit", "",
          f"Fit blocks (early): {res['fit_blocks'][0]} .. {res['fit_blocks'][-1]}; judged on later blocks {res['judge_blocks'][0]} .. "
          f"{res['judge_blocks'][-1]}.", "", f"- chosen: {res['walk_forward_fit']['chosen']}", f"- defaults: {res['walk_forward_fit']['defaults']}",
          f"- fit objective chosen {_fmt(res['walk_forward_fit']['fit_objective_chosen'])} vs defaults "
          f"{_fmt(res['walk_forward_fit']['fit_objective_defaults'])}",
          f"- JUDGE objective chosen {_fmt(res['walk_forward_fit']['judge_objective_chosen'])} vs defaults "
          f"{_fmt(res['walk_forward_fit']['judge_objective_defaults'])}"]
    for nm in ("judge_chosen", "judge_defaults"):
        j = res["walk_forward_fit"][nm]
        L.append(f"- {nm}: used {_fmt(j.get('use'))} weighted {_fmt(j.get('use_w'))} disregarded {_fmt(j.get('off'))} "
                 f"random subset {_fmt(j.get('subset'))} random {_fmt(j.get('random'))}; used-minus-random-subset "
                 f"{_fmt(j.get('use_minus_subset'))} (t {_fmt(j.get('use_minus_subset_t'), 2)}); mean used {j.get('mean_n_use', 0):.1f}")
    L += ["", "## First-noticed lag (registration to first usable evidence, days)", "", str(res["first_noticed_lag_days"]), "",
          "## Final-state report", "", an.markdown(end, ctx_end, ERAS)]
    (OUT / "report.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mine", "eval", "all"], default="all")
    ap.add_argument("--tickers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--first", default="2015-07-01")
    ap.add_argument("--last", default="2026-07-01")
    ap.add_argument("--end", default="2026-09-25", help="close of the last evaluation window (no mining at it)")
    ap.add_argument("--top", type=int, default=80, help="candidates per block handed to the memory")
    ap.add_argument("--min_t", type=float, default=1.5)
    ap.add_argument("--max_pairs", type=int, default=1500)
    ap.add_argument("--configs", type=int, default=60)
    ap.add_argument("--fit_share", type=float, default=0.45)
    ap.add_argument("--n_random", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pid {os.getpid()} stage {a.stage}", flush=True)
    if a.stage in ("mine", "all"):
        mine(a)
    if a.stage in ("eval", "all"):
        return evaluate(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
