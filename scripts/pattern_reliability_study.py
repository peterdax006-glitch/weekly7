"""Pattern reliability study on the REAL weekly panel (Bible phases 3, 4, 9, 10; canon C56-C61). Read-only on data/cache and
on B26's mined block checkpoints; writes only to state/research/pattern_reliability/.

Question: can the system learn WHEN a pattern works, and does gating on that beat always-on and discard-after-break?

Stages (each checkpoints; a re-run skips finished stages unless --force):
  build    real patterns -> weekly signed pattern-return timelines. Patterns are the real PatternMiner's (B26's half-year
           block checkpoints state/research/pattern_memory/real/blocks): a pattern is REGISTERED at the block that found it
           (its timeline starts the week after, so nothing in-sample is used), its sign is fixed at discovery, its weekly
           return is the date-demeaned mean 5-day outcome (bought at the next open) of the stocks satisfying its condition.
           Context per week = market context + market-level liquidity/volatility + macro with publication lags, built
           point-in-time (truncation-audited). Crowding/share per pattern-week come from the pattern masks.
  analyse  run_reliability (guards, walk-forward models, health monitor, forced investigations, explanations, policies)
           for the main configuration; then sensitivity to the false-alarm design (arl0), gate mode and model family.
  memory   MemoryGate skill on B26's real store at several real_now dates (if the store exists).

usage: pattern_reliability_study.py [--stage build|analyse|memory|all] [--tickers 500] [--seed 7] [--per-block 4]
       [--max-patterns 60] [--force]      RAM: waits until >= 2.5 GB is free (polls 60 s, gives up after 20 min)."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd

from engine import config as K
from engine import pattern_reliability as PR

OUT = K.STATE / "research" / "pattern_reliability"
BLOCKS = K.STATE / "research" / "pattern_memory" / "real" / "blocks"
MEM_STORE = K.STATE / "pattern_memory" / "real"
MACRO = {"T10Y2Y": ("m_curve", 1), "DFF": ("m_ffr", 1), "T10YIE": ("m_infl", 1), "DGS10": ("m_y10", 1)}
CODE = ROOT / "engine" / "pattern_reliability.py"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_ram(min_gb=2.5, poll=60, give_up=1200):
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available / 1e9 < min_gb:
        if time.time() - t0 > give_up:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "ABORTED.txt").write_text(f"free RAM stayed under {min_gb} GB for {give_up}s")
            raise SystemExit("RAM never freed up")
        log(f"waiting for RAM ({psutil.virtual_memory().available / 1e9:.1f} GB free)")
        time.sleep(poll)


def code_hash():
    return hashlib.sha256(CODE.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- build
def select_candidates(fr, top, min_t=1.5):
    """Same rule B26 uses: discovery and confirmation halves agree in sign, confirmation |t| >= min_t, strongest first."""
    if fr is None or len(fr) == 0 or "t_conf" not in fr:
        return pd.DataFrame(columns=["key_named", "effect"])
    fr = fr[fr["status"].ne("duplicate")] if "status" in fr else fr
    ok = (np.sign(fr["t_disc"]) == np.sign(fr["t_conf"])) & (fr["t_conf"].abs() >= min_t) & fr["effect"].notna()
    fr = fr[ok].assign(_a=lambda d: d["t_conf"].abs()).sort_values(["_a", "key_named"], ascending=[False, True]).head(top)
    return fr[["key_named", "effect", "t_conf"]]


def registry(per_block, max_patterns):
    """Patterns in discovery order, deduplicated by name: (name, sign, block end T)."""
    reg, seen = [], set()
    for csv in sorted(BLOCKS.glob("miner_*.csv")):
        T = pd.Timestamp(csv.stem.split("_")[1])
        fr = pd.read_csv(csv) if csv.stat().st_size > 5 else pd.DataFrame()
        for _, r in select_candidates(fr, per_block).iterrows():
            if r["key_named"] not in seen:
                seen.add(r["key_named"])
                reg.append((r["key_named"], 1.0 if r["effect"] >= 0 else -1.0, T))
    if len(reg) > max_patterns:                                          # keep the spread over time: every k-th
        keep = np.unique(np.linspace(0, len(reg) - 1, max_patterns).astype(int))
        reg = [reg[i] for i in keep]
    return reg


def macro_and_market(X, dates):
    """Raw context frame (date-indexed) and the builder that turns it into point-in-time weekly context."""
    m = pd.read_parquet(K.CACHE / "macro.parquet")
    m = m.loc[m.index >= dates.min() - pd.Timedelta(days=400), list(MACRO)].rename(columns={k: v[0] for k, v in MACRO.items()})
    g = X.groupby(level=0)
    mk = pd.DataFrame({"m_liq": g["log_dv"].median(), "m_atr": g["atr_pct"].median(), "m_vol20": g["vol20"].median()})
    mc = [c for c in X.columns if c.startswith("m_")]
    mk = mk.join(g[mc].first())
    raw = mk.join(m, how="outer")
    lags = {c: 0 for c in mk.columns}
    lags.update({v[0]: v[1] for v in MACRO.values()})
    builder = lambda r, d: PR.build_context(r, d, lags=lags, min_hist=52)
    return raw, builder


def build(a):
    import run_pattern_bank as rpb
    from engine.pattern_lifecycle import Panel, parse_key_named
    OUT.mkdir(parents=True, exist_ok=True)
    reg = registry(a.per_block, a.max_patterns)
    if not reg:
        raise SystemExit(f"no mined blocks found under {BLOCKS}")
    log(f"{len(reg)} patterns registered across {len({r[2] for r in reg})} blocks")
    wait_for_ram()
    X, y, pick = rpb.load(a.tickers, a.seed)
    log(f"loaded {len(X):,} rows, {len(pick)} tickers")
    last = X.index.get_level_values(0).max()
    panel = Panel.build(X, y, last + pd.Timedelta(days=30), {"min_obs": 8})
    dates = panel.dates
    raw, builder = macro_and_market(X, dates)
    ctx = builder(raw, dates)
    cuts = PR.audit_context_builder(builder, raw, dates, n_cuts=12, seed=a.seed)
    log(f"context builder passed the truncation audit at rows {cuts}")
    ctx = ctx.loc[:, ctx.notna().mean() > 0.6]
    nd, names = len(dates), [r[0] for r in reg]
    R = np.full((nd, len(names)), np.nan)
    share = np.full_like(R, np.nan)
    masks = {}
    first = {}
    for j, (nm, sgn, T) in enumerate(reg):
        key = parse_key_named(nm)
        mean = panel.means(key)
        m = panel.mask(key)
        if mean is None or m is None:
            continue
        fp = int(dates.searchsorted(T, side="right"))                    # first week strictly after the block that found it
        first[nm] = fp
        R[fp:, j] = sgn * mean[fp:]
        share[:, j] = np.bincount(panel.dcode[m], minlength=nd) / np.maximum(np.bincount(panel.dcode, minlength=nd), 1)
        masks[j] = m
    crowd = np.full_like(R, np.nan)
    rows_ = np.arange(nd)
    for j, mj in masks.items():
        nj = np.bincount(panel.dcode[mj], minlength=nd).astype(float)
        acc, cnt = np.zeros(nd), np.zeros(nd)
        for k, mk_ in masks.items():
            if k == j:
                continue
            live = rows_ >= first[names[k]]                                  # only patterns already registered that week
            inter = np.bincount(panel.dcode[mj & mk_], minlength=nd)
            acc += np.where(live & (nj > 0), inter / np.maximum(nj, 1), 0.0)
            cnt += live
        crowd[:, j] = acc / np.maximum(cnt, 1)
    keep = [j for j in range(len(names)) if np.isfinite(R[:, j]).sum() >= 60]
    idx = pd.DatetimeIndex(dates)
    cols = [names[j] for j in keep]
    rets = pd.DataFrame(R[:, keep], index=idx, columns=cols)
    ex = {"share": pd.DataFrame(share[:, keep], index=idx, columns=cols), "crowd": pd.DataFrame(crowd[:, keep], index=idx, columns=cols)}
    for k in ex:
        ex[k] = ex[k].where(rets.notna() | (np.arange(nd)[:, None] >= np.array([first.get(c, nd) for c in cols])[None, :]))
    rets.to_parquet(OUT / "timelines_rets.parquet")
    ctx.to_parquet(OUT / "timelines_ctx.parquet")
    ex["share"].to_parquet(OUT / "timelines_share.parquet")
    ex["crowd"].to_parquet(OUT / "timelines_crowd.parquet")
    (OUT / "timelines_meta.json").write_text(json.dumps({
        "patterns": cols, "first_row": {c: first[c] for c in cols}, "tickers": len(pick), "seed": a.seed,
        "weeks": nd, "first_date": str(idx[0].date()), "last_date": str(idx[-1].date()),
        "block_ends": sorted({str(r[2].date()) for r in reg}), "ctx_columns": list(ctx.columns)}, indent=1))
    log(f"built {rets.shape[1]} pattern timelines x {nd} weeks, {ctx.shape[1]} context columns")


def load_timelines():
    meta = json.loads((OUT / "timelines_meta.json").read_text())
    rets = pd.read_parquet(OUT / "timelines_rets.parquet")
    ctx = pd.read_parquet(OUT / "timelines_ctx.parquet")
    ex = {k: pd.read_parquet(OUT / f"timelines_{k}.parquet") for k in ("share", "crowd")}
    fp = pd.Series({c: meta["first_row"][c] for c in rets.columns})
    return PR.Timelines(rets, ctx, ex, fp), meta


# ---------------------------------------------------------------- analyse
def flat(summary):
    return {f"{p}|{c}": (None if pd.isna(v) else float(v)) for p, r in summary.iterrows() for c, v in r.items() if isinstance(v, (int, float, np.floating))}


def analyse(a):
    from engine import provenance
    tl, meta = load_timelines()
    h0 = code_hash()
    cfg = {"n_perm": a.n_perm, "boot": a.boot, "seed": a.seed}
    kinds = ("pooled", "pooled_ctx", "own_history", "per_pattern", "base_own")
    log(f"main run: {len(tl.patterns)} patterns x {tl.n_weeks} weeks, code {h0}")
    t0 = time.time()
    res = PR.run_reliability(tl, cfg, kinds=kinds, guards=True, audit_rows=[int(tl.n_weeks * f) for f in (0.5, 0.7, 0.9)])
    log(f"main run done in {time.time() - t0:.0f}s")
    rep = PR.render_report(res, "Pattern reliability on the real weekly panel")
    inv, st = res["investigations"], res["status"]
    inv.to_csv(OUT / "investigations.csv", index=False)
    st.to_csv(OUT / "status_last_week.csv", index=False)
    res["calibration"].to_csv(OUT / "calibration.csv", index=False)
    res["evaluation"]["weekly"].to_csv(OUT / "policy_weekly_returns.csv")
    res["evaluation"]["summary"].to_csv(OUT / "policy_summary.csv")
    res["evaluation"]["eras"].to_csv(OUT / "policy_by_era.csv", index=False)
    res["explain"]["table"].to_csv(OUT / "driver_table.csv", index=False)
    (OUT / "explanations.json").write_text(json.dumps([e.as_dict() for e in res["explain"]["explanations"]], indent=1, default=float))
    res["health"].events.to_csv(OUT / "health_events.csv", index=False)
    res["health"].ledger(tl.n_weeks - 1).to_csv(OUT / "health_ledger_last_week.csv", index=False)
    pd.DataFrame({k: v for k, v in res["pred"]["pooled"].importance.items()}, index=[0]).T.to_csv(OUT / "feature_importance.csv", header=["share"])
    tab = pd.DataFrame([{**{"model": k}, **{c: v for c, v in s.items() if not isinstance(v, str)}} for k, s in res["scores"].items()])
    tab.to_csv(OUT / "model_scores.csv", index=False)
    sens = []
    for arl0 in (() if a.no_sens else (100, 250, 500)):
        for mode in ("hard", "soft"):
            for model in ("lgbm", "logit"):
                if (arl0, mode, model) == (cfg.get("arl0", 500), "hard", "lgbm"):
                    continue
                t1 = time.time()
                r2 = PR.run_reliability(tl, {**cfg, "arl0": arl0, "gate_mode": mode, "model": model}, kinds=("pooled",), guards=False, explain=False)
                s = r2["evaluation"]["summary"]
                row = {"arl0": arl0, "gate_mode": mode, "model": model, "seconds": round(time.time() - t1)}
                for pol in ("always_on", "discard_revive", "gated_raw", "gated_c60", "gated_c61"):
                    row[f"{pol}_mean"] = float(s.loc[pol, "mean"])
                    if pol != "always_on":
                        row[f"{pol}_gain"] = float(s.loc[pol, "gain_vs_always_on"])
                        row[f"{pol}_gain_lo"] = float(s.loc[pol, "gain_vs_always_on_lo"])
                        row[f"{pol}_gain_hi"] = float(s.loc[pol, "gain_vs_always_on_hi"])
                row["unknown_share"] = r2["unknown_cause"]["share"]
                row["breaks"] = int(len(r2["health"].events))
                sens.append(row)
                pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)
                log(f"sensitivity arl0={arl0} {mode} {model}: gated_raw gain {row['gated_raw_gain']:+.4%}")
    stale = code_hash() != h0
    out = {"code_hash": h0, "stale_code": stale, "seed": a.seed, "provenance": provenance.stamp(cfg, a.seed),
           "weeks": tl.n_weeks, "patterns": len(tl.patterns), "meta": {k: meta[k] for k in ("tickers", "first_date", "last_date")},
           "scores": {k: {c: v for c, v in s.items() if not isinstance(v, str)} for k, s in res["scores"].items()},
           "policy_summary": flat(res["evaluation"]["summary"]), "on_off": res["evaluation"]["on_off"],
           "unknown_cause": res["unknown_cause"], "invariants": res["invariants"],
           "verdicts": inv["verdict"].value_counts().to_dict() if len(inv) else {},
           "n_explanations": len(res["explain"]["explanations"]), "status_counts": st["status"].value_counts().to_dict(),
           "causality_audit": res.get("causality"), "peek_flagged": [],
           "unproven": ["learning-harness (B22) repeated-run curve: not wired yet (see INTEGRATION)",
                        "survivor-only price panel: every level here is survivorship-biased",
                        "patterns come from a 500-ticker seeded sample"]}
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o)))
    (OUT / "report.md").write_text(rep + ("\n**STALE: engine/pattern_reliability.py changed while this ran; rerun.**\n" if stale else ""), encoding="utf-8")
    try:
        from engine.improve import log_experiment
        log_experiment({"name": "pattern_reliability_study", "kind": "reliability_gating", "code_hash": h0, "stale": stale,
                        "gated_raw_gain": out["policy_summary"].get("gated_raw|gain_vs_always_on"),
                        "unknown_share": out["unknown_cause"]["share"]}, cfg, a.seed)
    except Exception as e:                                                # the registry is a courtesy, never a gate
        log(f"log_experiment skipped: {e}")
    log("analyse done")


# ---------------------------------------------------------------- memory gate skill on B26's store
def memory(a):
    from engine.pattern_memory import PatternMemory
    if not (MEM_STORE / "chain.jsonl").exists():
        log("no real pattern-memory store: memory stage skipped")
        return
    mem = PatternMemory(MEM_STORE)
    rows = []
    for T in pd.date_range("2019-01-01", "2026-07-01", freq="6MS"):
        g = PR.MemoryGate.fit(mem, T)
        rows.append({"real_now": str(T.date()), **{k: v for k, v in g.skill.items()}})
        log(f"memory gate {T.date()}: {g.skill}")
    pd.DataFrame(rows).to_csv(OUT / "memory_gate_skill.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["build", "analyse", "memory", "all"])
    ap.add_argument("--tickers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-block", type=int, default=4)
    ap.add_argument("--max-patterns", type=int, default=60)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sens", action="store_true", help="skip the sensitivity grid")
    ap.add_argument("--out-dir", default=None, help="write here instead of state/research/pattern_reliability (smoke runs)")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("build", "all") and (a.force or not (OUT / "timelines_meta.json").exists()):
        build(a)
    if a.stage in ("analyse", "all"):
        analyse(a)
    if a.stage in ("memory", "all"):
        memory(a)


if __name__ == "__main__":
    main()
