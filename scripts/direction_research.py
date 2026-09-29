"""Real-cache runner: is there ANY direction signal on predicted movers? (B25; Bible V2/V5; canons C23, C24, C56).

Pipeline (all point-in-time; see engine/direction_features.py for the protocol):
  1. seeded ticker sample from data/cache/stocks_*.parquet (column subset), weekly decisions at each ISO week's last close,
     entry at the next open, +-10% first-touch labels over the next 5 sessions;
  2. the weekly feature panel (data/cache/panel.parquet, decision dates only) + derived PEAD / news-reversal inputs;
  3. a walk-forward LightGBM mover model (each year scored by a model fitted on earlier years) -> the week's top-N picks
     are the PREDICTED movers, the top-`pool` train the direction models;
  4. per target (up_first among realised movers, up_sign of the week's return over all picks) and per pick size:
     yearly walk-forward of every input family x {linear, gbm}, the pattern miner, a blend, controls (random, shuffled
     labels), a planted-signal positive control and a look-ahead canary; frontier by coverage with week-bootstrap CIs,
     per segment (event type, regime), per era, calibration, the 80% question, the point-in-time audit;
  5. results and provenance in state/research/direction2/<tag>/, a checkpoint after every stage (rerun resumes).

usage: direction_research.py [--tickers 2600] [--seed 7] [--tag run1] [--picks 30 10] [--pool 100] [--first-test 2018]
                             [--no-miner] [--quick] [--wait-ram 20]
Run detached (PowerShell Start-Process); kill only the PID it prints."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from engine import config as K
from engine import direction_features as D
from engine.provenance import stamp

OUT = K.STATE / "research" / "direction2"
NEED_GB = 2.5
TARGETS = ("up_first", "up_sign")


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 1e9


def wait_for_ram(minutes):
    t0 = time.time()
    while free_gb() < NEED_GB:
        if time.time() - t0 > minutes * 60:
            print(f"gave up: only {free_gb():.2f} GB free after {minutes} min", flush=True)
            sys.exit(3)
        print(f"waiting for RAM: {free_gb():.2f} GB free", flush=True)
        time.sleep(60)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- stage 1-3: universe, labels, panel, movers --------------------------------------------------------------------
def prepare(a, out: Path) -> pd.DataFrame:
    """Build (or load) the pool table: one row per (decision week, ticker) among the mover model's top-`pool`, with labels,
    every panel feature, derived inputs, segments, and the mover score. Cached in out/prep.parquet."""
    f = out / "prep.parquet"
    if f.exists():
        log(f"prep cached: {f}")
        return pd.read_parquet(f)
    import pyarrow.parquet as pq
    tick = pd.read_parquet(K.CACHE / "panel.parquet", columns=["r1"]).index.get_level_values(1).unique().sort_values()
    rng = np.random.default_rng(a.seed)
    tick = tick[np.sort(rng.choice(len(tick), min(a.tickers, len(tick)), replace=False))]
    log(f"{len(tick)} tickers sampled (seed {a.seed})")
    names = set(pq.ParquetFile(K.CACHE / "stocks_close.parquet").schema_arrow.names)
    want = [t for t in tick if t in names]
    px = {}
    for k in ("open", "high", "low", "close"):
        d = pd.read_parquet(K.CACHE / f"stocks_{k}.parquet", columns=want)
        px[k] = d.loc["2012-06-01":].astype("float32")
        del d
    cols = px["close"].columns
    sess = px["close"].index
    dec = D.week_end_positions(sess)
    lab = D.weekly_labels(px["open"].reindex(columns=cols).to_numpy(), px["high"].reindex(columns=cols).to_numpy(),
                          px["low"].reindex(columns=cols).to_numpy(), px["close"].to_numpy(), dec)
    LF = D.labels_frame(sess, cols, lab)
    del px, lab
    log(f"labels: {len(LF):,} rows, mover rate {LF['mover'].mean():.3f}, ambiguous {LF['amb'].sum():,}")
    wk = pd.DatetimeIndex(LF.index.get_level_values(0).unique())
    X = pd.read_parquet(K.CACHE / "panel.parquet", filters=[("date", "in", list(wk))])
    X = X[X.index.get_level_values(1).isin(cols)]
    X = X.astype("float32")
    log(f"panel weekly rows: {len(X):,} ({X.shape[1]} features)")
    J = X.join(LF, how="inner")
    del X
    J = D.add_derived(J.astype({c: "float32" for c in J.columns if J[c].dtype == "float64"}))
    J["seg"], J["reg"] = D.event_type(J).to_numpy(), D.regime_type(J).to_numpy()
    feat = [c for c in J.columns if c not in ("entry_date", "end_date", "fwd", "mover", "amb", "up_first", "up_sign", "seg", "reg")]
    R = D.xs_rank(J, feat)
    years = range(a.first_test - 3, int(J.index.get_level_values(0).year.max()) + 1)
    score, mlog = D.mover_walk_forward(R, J["mover"].astype(float), years, seed=a.seed)
    log("mover model:\n" + mlog.to_string())
    J["mover_score"] = score.to_numpy()
    J = J[J["mover_score"].notna()]
    pk = D.select_picks(J["mover_score"], n_pick=max(a.picks), n_pool=a.pool)
    J = J.join(pk[["pool", "rank"]])
    J = J[J["pool"]].drop(columns="pool")
    J.to_parquet(f)
    mlog.to_csv(out / "mover_model_log.csv", index=False)
    return J


# ---- stage 4: direction ----------------------------------------------------------------------------------------------
def build_F(P: pd.DataFrame, target: str, seed: int, planted_frac=0.05):
    """Family matrices aligned with the pool table, plus the positive control and the look-ahead canary."""
    F = {k: P[[c for c in v if c in P]].to_numpy("float32") for k, v in D.FAMILIES.items()}
    up = P[target].to_numpy(float)
    rng = np.random.default_rng(seed + 99)
    F["planted"] = D.plant_signal(up, rng.uniform(size=len(P)) < planted_frac, acc=0.9, seed=seed + 5)[:, None]
    F["leak"] = np.where(np.isfinite(up), up * 2 - 1, 0.0).astype("float32")[:, None]     # the future itself
    return F


def run_target(P, target, n_pick, a, out: Path):
    tag = f"pick{n_pick}_{target}"
    f = out / f"pred_{tag}.parquet"
    if f.exists():
        log(f"{tag} cached")
        return pd.read_parquet(f), pd.read_csv(out / f"blocks_{tag}.csv")
    pool = pd.DataFrame({"date": P.index.get_level_values(0), "year": P.index.get_level_values(0).year,
                         "pick": (P["rank"] <= n_pick).to_numpy(), target: P[target].to_numpy(),
                         "seg": P["seg"].to_numpy(), "reg": P["reg"].to_numpy()})
    F = build_F(P, target, a.seed)
    years = list(range(a.first_test, int(pool["year"].max()) + 1))
    extra = None
    if not a.no_miner:
        ydir = np.where(np.isfinite(P[target].to_numpy(float)), P[target].to_numpy(float) * 2 - 1, np.nan)
        feats = [c for c in D.FAMILIES["combined"] if c in P and c not in ("evt_earn", "evt_filing")]
        extra = {"miner": D.miner_hook(P[feats], ydir, pool["date"].to_numpy(),
                                       {"max_rows": 150_000, "null_reps": 1, "horizon": D.HORIZON, "seed": a.seed})}
    t0 = time.time()
    pred, blocks = D.walk_forward(pool, F, target, years, extra=extra, seed=a.seed)
    if extra:
        pd.DataFrame(extra["miner"].info).to_csv(out / f"miner_{tag}.csv", index=False)
    log(f"{tag}: walk-forward {time.time() - t0:.0f}s, {len(pred):,} prediction rows")
    pred.to_parquet(f)
    blocks.to_csv(out / f"blocks_{tag}.csv", index=False)
    return pred, blocks


def analyse(pred, blocks, tag, out: Path, B):
    """All tables for one (pick size, target)."""
    ft = D.frontier_table(pred, coverages=D.GRID, B=B, seed=1)
    ft.to_csv(out / f"frontier_{tag}.csv", index=False)
    cal = D.calibration_table(pred)
    cal.to_csv(out / f"calibration_{tag}.csv", index=False)
    era = D.era_table(pred)
    era.to_csv(out / f"era_{tag}.csv", index=False)
    q = D.eighty_question(ft)
    aud = D.audit_point_in_time(blocks)
    rel = {m: D.reliability(pred, m).round(4).to_dict("records") for m in ("combined:gbm", "blend") if m in set(pred["model"])}
    (out / f"summary_{tag}.json").write_text(json.dumps(dict(eighty=q, audit=aud, reliability=rel), indent=1, default=str))
    return ft, cal, era, q, aud


# ---- report ---------------------------------------------------------------------------------------------------------
def fmt_cell(r):
    if r is None or r["n"] == 0:
        return "-"
    return f"{r['acc']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] n={int(r['n'])} ({r['cov_real']:.1%})"


def frontier_md(ft, segcol=None, seg=None):
    d = ft[(ft["segcol"].isna() if segcol is None else (ft["segcol"] == segcol) & (ft["seg"] == seg))]
    rows = ["| model | " + " | ".join(f"c={c:.0%}" for c in D.COVERAGES) + " |", "|---|" + "---|" * len(D.COVERAGES)]
    for m in sorted(d["model"].unique()):
        cells = []
        for c in D.COVERAGES:
            s = d[(d["model"] == m) & np.isclose(d["cov_target"], c)]
            cells.append(fmt_cell(s.iloc[0] if len(s) else None))
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return chr(10).join(rows)


def best_md(ft, k=8):
    real = ft[~ft["model"].str.startswith(D.CONTROLS) & (ft["n"] >= 30)]
    top = real.sort_values("lo", ascending=False).head(k)
    rows = ["| model | segment | c | n | acc | 95% CI | adj lower | up-rate of bets |", "|---|---|---|---|---|---|---|---|"]
    for r in top.itertuples():
        seg = "all" if r.segcol is None or pd.isna(r.segcol) else f"{r.segcol}={r.seg}"
        rows.append(f"| {r.model} | {seg} | {r.cov_target:.1%} | {int(r.n)} | {r.acc:.3f} | [{r.lo:.3f},{r.hi:.3f}] | {r.lo_adj:.3f} | {r.up_rate:.3f} |")
    return "\n".join(rows)


def report(sections, a, prov, out: Path):
    L = [f"# Direction on predicted movers (B25) - run {a.tag}", "",
         f"tickers sampled {a.tickers} (seed {a.seed}), pool {a.pool}, picks {a.picks}, first test year {a.first_test}. "
         "Survivor-only price panel: every accuracy below is optimistic if anything (delisted names are missing).", ""]
    for tag, (ft, cal, era, q, aud, info) in sections.items():
        L += [f"## {tag}", "", info, "", "### Frontier: accuracy at coverage c (95% CI over weeks; thresholds from the previous year)", "",
              frontier_md(ft), ""]
        for sc in ("seg", "reg"):
            for s in sorted(ft[ft["segcol"] == sc]["seg"].dropna().unique()):
                L += [f"#### {sc} = {s}", "", frontier_md(ft, sc, s), ""]
        L += ["### Best cells by CI lower bound (each is one of "
              f"{q['cells']} looked at; 'adj lower' pays for that)", "", best_md(ft), "",
              "### Calibration (Brier vs train base rate; skill > 0 = probabilities carry information)", "",
              md_table(cal) if len(cal) else "(none)", "",
              "### The 80% question", "",
              f"- cells with >=30 bets: {q['cells_with_min_bets']}; accuracy >= 0.80 in {q['reached_acc']}; CI lower bound >= 0.80 in "
              f"{q['reached_lo']}; after the multiple-comparison correction in {q['reached_lo_adj']}; control models with lower bound >= 0.80: "
              f"{q['control_hits']}. Positive controls (planted signal / look-ahead canary) recovered: {q['positive_controls_found']}.",
              f"- best cell: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in (q['best'] or {}).items()})}", "",
              "### Point-in-time audit (C56)", "", f"- ok: {aud['ok']} {aud['violations']}", ""]
        e = era[era["model"].isin(["combined:gbm", "blend", "ctl_random"]) & np.isclose(era["cov_target"], 0.25)]
        if len(e):
            L += ["### Per era, coverage 25%", "", md_table(e), ""]
    L += ["## Provenance", "", "```", json.dumps(prov, indent=1, default=str)[:2500], "```"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--picks", type=int, nargs="+", default=[30, 10, 100])
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--first-test", type=int, default=2018)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-miner", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--wait-ram", type=int, default=20)
    a = ap.parse_args()
    if a.quick:
        a.tickers, a.boot, a.picks = 500, 200, [30]
    print(f"PID {__import__('os').getpid()}", flush=True)
    wait_for_ram(a.wait_ram)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    prov = stamp({k: v for k, v in vars(a).items()}, a.seed)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    P = prepare(a, out)
    log(f"pool rows {len(P):,}; mover hit rate among rank<=30: {P.loc[P['rank'] <= 30, 'mover'].mean():.3f} vs universe base "
        f"{P['mover'].mean():.3f} (pool)")
    sections = {}
    for n_pick in a.picks:
        for target in TARGETS:
            tag = f"pick{n_pick}_{target}"
            pred, blocks = run_target(P, target, n_pick, a, out)
            ft, cal, era, q, aud = analyse(pred, blocks, tag, out, a.boot)
            pk = P[P["rank"] <= n_pick]
            base = pk[target].mean()
            info = (f"{len(pk):,} picks over {pk.index.get_level_values(0).nunique()} weeks; realised-mover rate among picks "
                    f"{pk['mover'].mean():.3f}; base rate of `{target}` = {base:.3f} (always-up scores {max(base, 1 - base):.3f}); "
                    f"test years {a.first_test}-{P.index.get_level_values(0).year.max()}, {int(len(pred) / max(pred['model'].nunique(), 1)):,} scored rows per model.")
            sections[tag] = (ft, cal, era, q, aud, info)
            report(sections, a, prov, out)
            log(f"{tag}: 80% question -> {q['reached_lo']} cells lower>=0.80 ({q['reached_lo_adj']} adjusted), controls {q['control_hits']}")
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("done")


if __name__ == "__main__":
    main()
