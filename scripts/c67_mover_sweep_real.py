"""C67 / C75 1C mover-episode research on the REAL daily caches (F29). SURVIVOR_ONLY: the price panel has no delisted names.

Streams the market year by year and universe slice by slice (mapping rule 27: only episode rows and sufficient statistics are kept),
through the research brain's own mover-episode lab (engine.research.precursors.step - the same entry the research loop schedules):
  sweep      every (year, slice) unit, all four lenses (c2c, rng, o2c, gap), every built-in precursor family plus decoy features, labelled
             outcomes (engine.research.episode_paths), matched-control and path-class contrasts with shuffled-label copies, ONE cumulative
             multiple-testing ledger; committed atomically about once a minute, so a kill costs at most a minute and resumes where it stopped
  report     per-label and per-year outcome counts, the 'hundreds a day' ledger, tracked precursor candidates with walk-forward and era checks,
             a held-out-years out-of-sample check, question objects raised through engine.research.questions, the MaturedRecord release
             gate, an episode-level truncation audit on real bars, and the list of UNMEASURED fields
  intraday   intraday-bar path labels for the recent sessions where 5-minute bars were collected (data/intraday/5m)
Usage: python scripts/c67_mover_sweep_real.py [--phase all|sweep|report|intraday] [--years 2000-2025] [--slices 4] [--out DIR]
Results: state/research/c67_movers/<run>/ (lab/ checkpoint, progress.jsonl, report.md, report.json, questions.json, holdout.csv ...)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine import config as K
from engine.research import episode_paths as EP
from engine.research import episodes as EPI
from engine.research import precursors as PC

LABEL = "SURVIVOR_ONLY"
SURVIVOR_NOTE = ("SURVIVOR_ONLY: the daily price panel contains only names that still trade (77 of ~9,250 exits are priced); episodes of names "
                 "that later delisted are missing, so collapse and loss rates are UNDERSTATED and every rate here is conditional on survival.")


def log(msg: str) -> None:
    print(f"[c67 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_memory(min_gb: float = 2.5, poll_s: float = 60.0, give_up_s: float = 1200.0) -> float:
    """CONTEXT rule 10: poll free RAM every minute, give up after twenty minutes."""
    import psutil
    t0 = time.monotonic()
    while True:
        free = psutil.virtual_memory().available / 1e9
        if free >= min_gb:
            return free
        if time.monotonic() - t0 > give_up_s:
            raise SystemExit(f"only {free:.1f} GB free after {give_up_s / 60:.0f} min (< {min_gb} GB): not starting")
        log(f"{free:.2f} GB free < {min_gb} GB; waiting")
        time.sleep(poll_s)


def rss_gb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def parse_years(s: str) -> list[int]:
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def sector_context(cache_dir: Path):
    """2-digit SIC codes per ticker (a CURRENT mapping: a name's sector as known today, the one mild look-ahead of the sector family; it
    groups names, it never carries a return)."""
    p = cache_dir / "sic.parquet"
    if not p.exists():
        return None
    sic = pd.read_parquet(p)
    m = dict(zip(sic["ticker"].astype(str), sic["sic"].astype(str).str[:2]))

    def fn(unit, grid):
        codes = pd.factorize(pd.Series([m.get(str(t), "na") for t in grid.tickers]))[0]
        return np.asarray(codes, int), {}
    return fn


def lab_config(a) -> PC.LabConfig:
    return PC.LabConfig(n_slices=a.slices, lenses=tuple(a.lenses.split(",")), n_perm=a.n_perm, cluster_months=a.cluster_months,
                        max_controls=a.max_controls, seed=a.seed)


def open_lab(a, out: Path):
    cfg = lab_config(a)
    reg = PC.registry_with_decoys(a.decoys, seed=a.seed)
    st, store = PC.open_state(out / "lab", parse_years(a.years), cfg, registry=reg, widen=True, on_drift="archive")
    if not store.exists():
        st.next_eval_at = a.first_eval
    return st, store


# ------------------------------------------------------------------------------------------------------------------ sweep
def run_sweep(a, out: Path) -> dict:
    wait_for_memory()
    st, store = open_lab(a, out)
    loader = PC.cache_loader(a.slices, 0)
    ctx_fn = sector_context(Path(K.CACHE))
    total = len(st.book.all_units())
    log(f"sweep: {st.units_done()}/{total} units already done; years {st.book.years[0]}-{st.book.years[-1]}, {a.slices} slices, "
        f"lenses {st.cfg.lenses}, {len(st.registry.columns())} precursor columns")
    t0, last_commit, n_steps = time.monotonic(), time.monotonic(), 0
    prog = open(out / "progress.jsonl", "a", encoding="utf-8")
    while True:
        if a.budget_min and time.monotonic() - t0 > a.budget_min * 60:
            log("budget reached; committing and stopping (resume by running again)")
            break
        ts = time.monotonic()
        rep = PC.step(st, a.now, loader, max_units=len(st.cfg.lenses), store=None, context_fn=ctx_fn, grouped=True)
        n_steps += 1
        eps = sum(st.book.records[u].n_episodes for u in rep.done)
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "done": rep.done, "seconds": round(time.monotonic() - ts, 2), "rss_gb": round(rss_gb(), 3),
               "episodes": eps, "units_done": st.units_done(), "units_total": total, "evaluated": rep.evaluation is not None,
               "candidates": rep.candidates, "exhausted": rep.exhausted, "widened": rep.widened, "ledger_looks": st.ledger.total_trials}
        prog.write(json.dumps(row) + "\n")
        prog.flush()
        log(f"{rep.done} {row['seconds']}s rss {row['rss_gb']} GB, {st.units_done()}/{total} units"
            + (f", EVALUATED: {rep.evaluation.by_status}" if rep.evaluation else ""))
        if time.monotonic() - last_commit >= a.commit_every_s or rep.exhausted or not rep.done:
            store.commit(st)
            last_commit = time.monotonic()
        if rep.exhausted or not rep.done:
            break
    rep = PC.evaluate_state(st, a.now, force=True)
    store.commit(st)
    prog.close()
    return {"steps": n_steps, "seconds": round(time.monotonic() - t0, 1), "units_done": st.units_done(), "units_total": total,
            "final_eval": rep.to_dict() if rep else None}


# ------------------------------------------------------------------------------------------------------------------ report
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = True) -> str:
    if df is None or len(df) == 0:
        return "(none)\n"
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        cells = []
        for v in r.to_numpy():
            if isinstance(v, (float, np.floating)):
                cells.append(("{:.0f}".format(v) if float(v).is_integer() and abs(v) >= 1 else floatfmt.format(v)) if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def truncation_on_real(a, st) -> list[dict]:
    """The F09 truncation method on real bars: one slice of each audit year, cuts on days with episodes."""
    out = []
    names = EPI.cache_tickers(None)
    for year, sl in ((int(y), i % a.slices) for i, y in enumerate(a.audit_years.split(","))):
        wait_for_memory()
        tick = EPI.slice_tickers(names, sl, a.slices, 0)[: a.audit_names]
        s, e = EPI.year_window(year, st.ecfg, st.registry.extra_warm(), 0)
        bars = EPI.load_bars(s, e, tick)
        ctx_fn = sector_context(Path(K.CACHE))
        g = EPI.build_grid(bars, st.ecfg)
        sec = ctx_fn(None, g)[0] if ctx_fn else None
        res = PC.truncation_audit(bars, st.registry, st.ecfg, st.pcfg, n_cuts=a.audit_cuts, seed=a.seed + year, sector_codes=sec)
        res.update({"year": year, "slice": sl, "names": len(tick)})
        out.append(res)
        log(f"truncation audit {year} slice {sl}: clean={res['clean']} episodes {res['episodes_checked']} cells {res['cells_compared']}")
    return out


def build_report(a, out: Path, sweep_info: dict | None) -> dict:
    t0 = time.monotonic()
    st, store = open_lab(a, out)
    if not store.exists():
        raise SystemExit("no lab state: run --phase sweep first")
    now = a.now
    lenses = list(st.cfg.lenses)
    rep: dict = {"label": LABEL, "survivor_note": SURVIVOR_NOTE, "now": now, "config": {"lab": st.cfg.digest(), "episodes": st.ecfg.digest(),
                 "paths": st.pcfg.digest(), "rules": st.rules.digest()}, "sweep": sweep_info}
    from engine.provenance import stamp
    rep["provenance"] = stamp({"lab": vars(a)}, a.seed)
    cov = st.book.report(st.registry.columns())
    rep["coverage"] = {k: cov[k] for k in ("units_total", "units_done", "fraction_done", "years_untouched")}
    hb = PC.hundreds_by_year(st, minimum=100)
    rep["hundreds_by_year"] = hb.to_dict("records")
    # labels per year and per label
    lab_year = {lens: {h: PC.label_year_table(st, lens, h, band=None) for h in (1, 5)} for lens in lenses}
    all_counts = {lens: EP.merge_label_counts(list(PC.labels_by_year(st, lens).values())) for lens in lenses}
    rep["labels_per_year"] = {lens: {h: t.reset_index().to_dict("records") for h, t in d.items()} for lens, d in lab_year.items()}
    rep["label_shares"] = {lens: {h: EP.label_table(all_counts[lens], h).reset_index().rename(columns={"index": "type"}).to_dict("records")
                                  for h in (1, 2, 3, 5)} for lens in lenses}
    rep["categories"] = {lens: {c: EP.category_table(all_counts[lens], c).reset_index().rename(columns={"index": "type"}).to_dict("records")
                                for c in ("gap_type", "intraday_path", "nd_close_loc", "day_shape_1", "move_close_loc")} for lens in lenses}
    rep["means"] = {lens: EP.mean_table(all_counts[lens]).reset_index().rename(columns={"index": "type"}).to_dict("records") for lens in lenses}
    # candidates, holdout, questions, release
    ct = PC.candidates_table(st)
    rep["tracked_by_status"] = {str(k): int(v) for k, v in ct["status"].astype(str).value_counts().items()} if len(ct) else {}
    rep["decoys"] = PC.decoy_report(st)
    rep["ledger"] = {"looks": st.ledger.total_trials, "distinct": st.ledger.distinct_trials, "expected_best_null_t": st.ledger.expected_best_null_t(),
                     "evaluations": st.eval_seq, "evidence_cells": len(st.evidence.cells), "feature_tests": st.evidence.n_tests()}
    um = PC.unknown_map(st)
    rep["unknown_map"] = {"cells": int(len(um)), "no_precursor_found": int((um["verdict"] == "UNKNOWN: no precursor found").sum()) if len(um) else 0,
                          "with_candidates": int((um["candidates"] > 0).sum()) if len(um) else 0}
    hold, hsum = PC.holdout_check(st, a.holdout, now)
    hold.to_csv(out / "holdout.csv", index=False)
    rep["holdout"] = hsum
    qrep, qled = PC.raise_questions(st, now, min_abs_t=a.question_t, max_n=a.max_questions)
    qs = [q.to_dict() for q in qrep.questions]
    (out / "questions.json").write_text(json.dumps({"label": LABEL, "questions": qs, "refused": [list(r) for r in qrep.refused],
                                                    "merged": qrep.merged, "ledger": qled.rows}, indent=1, default=str), encoding="utf-8")
    rep["questions"] = {"created": len(qs), "refused": len(qrep.refused), "merged": qrep.merged,
                        "sample": [{"qid": q["qid"], "text": q["text"], "priority": q["priority"], "problem": q["problem"]} for q in qs[:10]]}
    recs, held = PC.release(st, now)
    rep["release"] = {"released_records": len(recs), "held": {k: v for k, v in list(held.items())[:5]}, "held_total": len(held)}
    if len(ct):
        top = ct[~ct["feature"].str.startswith(PC.DECOY_PREFIX)].head(a.top)
        keep = ["candidate_id", "status", "key", "feature", "family", "effect", "t", "q", "p_perm", "n1", "n0", "clusters", "wf_tested", "wf_passed",
                "eras", "text"]
        rep["top_candidates"] = top[keep].to_dict("records")
        hmap = hold.set_index("candidate_id") if len(hold) else pd.DataFrame()
        for r in rep["top_candidates"]:
            if len(hmap) and r["candidate_id"] in hmap.index:
                h = hmap.loc[r["candidate_id"]]
                r["holdout"] = {"t_disc": float(h["t_disc"]), "effect_hold": float(h["effect_hold"]), "t_hold": float(h["t_hold"]),
                                "confirmed": bool(h["confirmed"])}
    else:
        rep["top_candidates"] = []
    rep["market_state_tests"] = sum(len(v) for v in st.market.values())
    hashes: dict[str, int] = {}
    for r in st.book.records.values():
        hashes[r.code_hash] = hashes.get(r.code_hash, 0) + 1
    rep["unit_code_hashes"] = hashes                                         # other builders edit engine/ too: more than one hash is expected
    rep["notes_tail"] = st.notes[-10:]
    rep["unmeasured"] = EP.UNMEASURED_FIELDS
    rep["truncation_audit"] = truncation_on_real(a, st) if a.audit_years else []
    rep["report_seconds"] = round(time.monotonic() - t0, 1)
    store.commit(st)                                                         # the holdout looks are ledger entries: persist them
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    write_markdown(out, rep, lab_year, all_counts, hb, hold)
    return rep


def write_markdown(out: Path, rep: dict, lab_year: dict, all_counts: dict, hb: pd.DataFrame, hold: pd.DataFrame) -> None:
    L = [f"# C67 mover-episode research on real bars - {LABEL}", "", SURVIVOR_NOTE, "",
         f"now = {rep['now']}; code {rep['provenance'].get('code_hash', '')}; coverage {rep['coverage']['units_done']}/{rep['coverage']['units_total']} units.", ""]
    if rep.get("sweep"):
        L += [f"Sweep: {rep['sweep']['steps']} steps, {rep['sweep']['seconds']} s.", ""]
    L += ["## Hundreds of 5-10% movers a day? (close-to-close, whole swept universe)", "", _md_table(hb, index=False)]
    for lens, d in lab_year.items():
        for h, t in d.items():
            L += [f"## Outcome classes per year - lens {lens}, horizon {h} (counts, all bands and sides)", "", _md_table(t, "{:.0f}")]
    for lens in all_counts:
        for h in (1, 5):
            L += [f"## Outcome shares by episode type - lens {lens}, horizon {h}", "", _md_table(EP.label_table(all_counts[lens], h))]
        for c in ("gap_type", "nd_close_loc", "day_shape_1", "move_close_loc", "intraday_path"):
            L += [f"### {c} - lens {lens}", "", _md_table(EP.category_table(all_counts[lens], c))]
        L += [f"### mean raw outcomes - lens {lens} (signed by the move's side; r1 = next close-to-close)", "", _md_table(EP.mean_table(all_counts[lens]), "{:.4f}")]
    L += ["## Multiple testing", "", "```", json.dumps(rep["ledger"], indent=1), json.dumps(rep["decoys"], indent=1), "```",
          f"Tracked by status: {rep['tracked_by_status']}", f"Unknown map: {rep['unknown_map']}", ""]
    L += ["## Top precursor candidates (not promotions: research questions at most)", ""]
    for r in rep["top_candidates"]:
        h = r.get("holdout")
        ht = f"holdout t {h['t_hold']:.2f} confirmed={h['confirmed']}" if h else "not re-found in the pre-holdout discovery"
        L.append(f"- [{r['status']}] {r['feature']} {r['key']} shift {r['effect']:+.4f} t {r['t']:.2f} q {r['q']:.3g} wf {r['wf_passed']}/{r['wf_tested']}; {ht}")
    L += ["", f"## Held-out years (discovery < {rep['holdout']['split_year']}, measured on the rest)", "", "```", json.dumps(rep["holdout"], indent=1, default=str), "```", "",
          _md_table(hold.drop(columns=["text"]).head(25) if len(hold) else hold, index=False)]
    L += ["## Questions raised in the research brain (engine.research.questions.generate)", "", f"created {rep['questions']['created']}, refused {rep['questions']['refused']}", ""]
    for q in rep["questions"]["sample"]:
        L.append(f"- ({q['problem']}, priority {q['priority']:.3g}) {q['text']}")
    L += ["", f"Release gate: {rep['release']['released_records']} MaturedRecords released, {rep['release']['held_total']} held.", "",
          "## Truncation audit on real bars (F09 method)", "", "```", json.dumps(rep["truncation_audit"], indent=1, default=str), "```", "",
          "## UNMEASURED", ""] + [f"- {k}: {v}" for k, v in rep["unmeasured"].items()]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------ intraday
def run_intraday(a, out: Path) -> dict:
    """Intraday-bar path labels where 5-minute bars exist (recent sessions, liquid names only)."""
    wait_for_memory()
    root = Path(K.DATA) / "intraday" / "5m"
    files = sorted(root.glob("*.parquet"))
    if not files:
        return {"episodes": 0, "note": "no intraday bars collected"}
    first, last = pd.Timestamp(files[0].stem), pd.Timestamp(files[-1].stem)
    ib = EP.load_intraday(root)
    tick = sorted(ib["ticker"].unique())
    ecfg = EPI.EpisodeConfig()
    bars = EPI.load_bars(str((first - pd.Timedelta(days=200)).date()), str((last + pd.Timedelta(days=10)).date()), tick)
    g = EPI.build_grid(bars, ecfg)
    lo = int(g.dates.searchsorted(first))
    eps = EPI.episode_frame(g, lo, g.shape[0])
    epp = EP.with_paths(eps, EP.label_paths(g, eps))
    ibl = EP.intraday_bar_labels(epp, ib, g.dates)
    d = pd.concat([epp.reset_index(drop=True), ibl], axis=1)
    band, side = EPI.lens_band_side(d, "c2c")
    d = d[band > 0].reset_index(drop=True)
    res = {"label": LABEL, "sessions": [str(first.date()), str(last.date())], "names_with_bars": len(tick), "c2c_episodes": int(len(d)),
           "measured_move_day": int(d["ib_measured"].sum()), "measured_next_day": int((d["ib_d1_shape"] != "UNMEASURED").sum()),
           "move_day_shapes": d["ib_d0_shape"].value_counts().to_dict(), "next_day_shapes": d["ib_d1_shape"].value_counts().to_dict()}
    m = d[d["ib_measured"]]
    if len(m):
        res["move_day_median_minutes_to_favourable_extreme"] = float(m["ib_d0_t_fav"].median())
        res["move_day_share_extreme_in_first_hour"] = float((m["ib_d0_t_fav"] < 60).mean())
        res["next_day_mean_signed_close"] = float(np.nanmean(m["ib_d1_close"])) if m["ib_d1_close"].notna().any() else float("nan")
        res["cross_shape_next_day"] = pd.crosstab(m["ib_d0_shape"], m["ib_d1_shape"]).to_dict()
    cols = ["date", "ticker", "c2c", "side", "cls_1", "gap_type", "nd_close_loc", "day_shape_1"] + list(EP.INTRADAY_COLUMNS)
    d[cols].to_csv(out / "intraday_episodes.csv", index=False)
    (out / "intraday.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    log(f"intraday: {res['c2c_episodes']} episodes, {res['measured_move_day']} with move-day bars")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="all", choices=["all", "sweep", "report", "intraday"])
    ap.add_argument("--years", default="2000-2025")
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--lenses", default="c2c,rng,o2c,gap")
    ap.add_argument("--n-perm", type=int, default=4)
    ap.add_argument("--cluster-months", type=int, default=6)
    ap.add_argument("--max-controls", type=int, default=30)
    ap.add_argument("--decoys", type=int, default=3)
    ap.add_argument("--first-eval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--now", default="2026-09-30")
    ap.add_argument("--holdout", type=int, default=2021)
    ap.add_argument("--question-t", type=float, default=3.0)
    ap.add_argument("--max-questions", type=int, default=25)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--audit-years", default="2008,2019")
    ap.add_argument("--audit-names", type=int, default=400)
    ap.add_argument("--audit-cuts", type=int, default=4)
    ap.add_argument("--budget-min", type=float, default=0.0)
    ap.add_argument("--commit-every-s", type=float, default=60.0)
    ap.add_argument("--out", default=str(Path(K.STATE) / "research" / "c67_movers" / "real_v1"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    info = None
    if a.phase in ("all", "sweep"):
        info = run_sweep(a, out)
        (out / "sweep.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    elif (out / "sweep.json").exists():
        info = json.loads((out / "sweep.json").read_text(encoding="utf-8"))
    if a.phase in ("all", "report"):
        build_report(a, out, info)
    if a.phase in ("all", "intraday"):
        run_intraday(a, out)
    log("finished")


if __name__ == "__main__":
    main()
