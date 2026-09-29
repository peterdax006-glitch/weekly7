"""Bible Phase 46: the final report, assembled from the real artefacts - never typed by hand. Eleven sections:
implementation, validation (every gate), research (what improved), failures, remaining uncertainty, champion,
challengers, performance DISTRIBUTION, leakage audit, reproducibility audit, checklist. Fail closed: a section
without evidence says NO EVIDENCE, and an audit is PASS only when every one of its checks has evidence and passed.

usage: final_report.py [--no-replay]      (--no-replay skips recomputing the Test windows' weekly returns)
writes state/reports/FINAL_REPORT.md and final_report.json"""
import glob, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

NO = "NO EVIDENCE"


def jload(rel):
    p = ROOT / rel
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def sh(*cmd):
    try:
        return subprocess.run(list(cmd), cwd=ROOT, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------- helpers
def distribution(weekly, max_dd=None):
    """Phase 46 section 8: the whole distribution, not just an average."""
    w = np.asarray([x for x in weekly if x is not None and np.isfinite(x)], dtype=float)
    if not len(w):
        return None
    eq = np.cumprod(1 + w)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    q = np.percentile(w, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {"weeks": int(len(w)), "mean": float(w.mean()), "median": float(np.median(w)), "sd": float(w.std(ddof=1)) if len(w) > 1 else 0.0,
            "percentiles": dict(zip(["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"], map(float, q))),
            "worst": float(w.min()), "best": float(w.max()), "drawdown_chained": dd,
            "share_in_band_5_10": float(((np.abs(w) >= 0.05) & (np.abs(w) <= 0.10)).mean()),
            "share_positive": float((w > 0).mean())}


def audit(checks):
    """checks: {name: True/False/None}. PASS only if every check has evidence (not None) and passed."""
    if not checks:
        return {"verdict": NO, "checks": {}}
    if any(v is None for v in checks.values()):
        verdict = "INCOMPLETE (missing evidence: " + ", ".join(k for k, v in checks.items() if v is None) + ")"
    elif all(checks.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL (" + ", ".join(k for k, v in checks.items() if not v) + ")"
    return {"verdict": verdict, "checks": {k: ("pass" if v else ("fail" if v is not None else NO)) for k, v in checks.items()}}


def checklist_counts(text):
    states = re.findall(r"^\s*- \[(.| )\]", text, re.M)
    names = {" ": "not started", "~": "implemented/testing", "?": "unproven", "x": "validated", "!": "failed"}
    return {names.get(s, s): states.count(s) for s in sorted(set(states))}


def registry():
    p = ROOT / "state" / "experiments.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


# ---------------------------------------------------------------------------------------------------------- sections
def s_implementation():
    mods = []
    for f in sorted(glob.glob(str(ROOT / "engine" / "*.py"))):
        t = Path(f).read_text(encoding="utf-8", errors="replace")
        doc = re.match(r'\s*"""(.*?)(\n|""")', t, re.S)
        mods.append({"module": Path(f).name, "lines": len(t.splitlines()), "summary": (doc.group(1).strip()[:140] if doc else "")})
    log = sh("git", "log", "--name-only", "--format=@@%h", "--", "engine")
    touched = {}
    for block in log.split("@@")[1:]:
        for f in block.splitlines()[1:]:
            if f.strip():
                touched[Path(f).name] = touched.get(Path(f).name, 0) + 1
    for m in mods:
        m["commits"] = touched.get(m["module"], 0)
    return {"engine_modules": mods, "total_engine_lines": sum(m["lines"] for m in mods)}


def s_validation():
    q = jload("state/quality/quality_gate.json")
    planted = jload("state/research/algorithm/planted/summary.json")
    parity = jload("state/research/parity/latest.json")
    blind = jload("state/research/blind_gates/results.json")
    return {
        "quality_gate": {"exit_code": q.get("exit_code"), "failing": len(q.get("failing", []))} if q else NO,
        "planted_calibration": {"validated": planted["verdict"]["validated"],
                                "failed": [k for k, c in planted["verdict"]["criteria"].items() if not c["pass"]]} if planted else NO,
        "parity": {"passed": parity.get("passed"), "failed": parity.get("failed")} if parity else NO,
        "blind_gates": {k: blind[k] for k in ("seals", "disguise", "probe", "retest", "health") if k in blind} if blind else NO,
        "registry_gates": [{"event": r.get("event"), "gates": r["gates"]} for r in registry() if isinstance(r.get("gates"), dict)][-30:],
    }


def s_research_and_failures():
    R = registry()
    good = [{"event": r.get("event"), "reason": r.get("reason"), "t": r.get("t")} for r in R if r.get("outcome") == "adopt"]
    bad = [{"event": r.get("event"), "reason": r.get("reason"), "t": r.get("t")} for r in R if r.get("outcome") == "reject"]
    planted = jload("state/research/algorithm/planted/summary.json")
    if planted and not planted["verdict"]["validated"]:
        bad.append({"event": "planted_calibration", "reason": "Phase 25 NOT VALIDATED: " + ", ".join(
            k for k, c in planted["verdict"]["criteria"].items() if not c["pass"])})
    ck = (ROOT / "state" / "CHECKLIST.md").read_text(encoding="utf-8") if (ROOT / "state" / "CHECKLIST.md").exists() else ""
    failed_items = [l.strip()[:220] for l in ck.splitlines() if l.strip().startswith("- [!]")]
    return {"improved": good or NO, "failed": bad + [{"checklist": x} for x in failed_items] or NO}


def s_uncertainty():
    ck = (ROOT / "state" / "CHECKLIST.md").read_text(encoding="utf-8") if (ROOT / "state" / "CHECKLIST.md").exists() else ""
    items = [l.strip()[:220] for l in ck.splitlines() if re.match(r"\s*- \[(~|\?| )\]", l)]
    trace = jload("state/build/bible_trace.json")
    return {"checklist_open": items, "bible_trace": (trace.get("counts") if isinstance(trace, dict) else NO) if trace else NO,
            "standing_caveats": ["price panel is survivor-only (Phase 1 audit): historical returns are upper bounds"]}


def s_champion_and_challengers():
    meta = jload("state/models/meta.json")
    loop = jload("state/livesim/loop2.json")
    champ = {"live_champion": meta, "live_champion_hash": cfg_hash(meta) if meta else None}
    if loop:
        champ["test_basis"] = {"version": loop.get("version"), "cfg": loop.get("cfg"), "meta": loop.get("meta"),
                               "hash": cfg_hash({"cfg": loop.get("cfg"), "meta": loop.get("meta")})}
    ch = [{"event": r.get("event"), "id": r.get("id"), "t": r.get("t")} for r in registry()
          if str(r.get("event", "")).startswith(("challenger", "promoted", "rejected", "rolled_back"))]
    return champ, (ch or NO)


def s_performance(replay=True):
    """Replay every REVEALED Test window (cycles.json lists them) through the real Session with the current basis."""
    cyc = jload("state/livesim/cycles.json")
    loop = jload("state/livesim/loop2.json")
    if not cyc or not loop:
        return NO
    revealed = {c["run_id"]: c.get("revealed_year") for c in cyc.get("cycles", []) if c.get("revealed_year")}
    if not replay:
        return {"windows_revealed": len(revealed), "note": "replay skipped"}
    import importlib.util
    spec = importlib.util.spec_from_file_location("loop2", ROOT / "scripts" / "livesim_loop2.py")
    L2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(L2)
    rows, allw = [], []
    for rid, year in sorted(revealed.items()):
        d = ROOT / "state" / "livesim" / rid
        if not list(d.glob("wsnap_*.parquet")):
            continue
        try:
            r = L2.run_window(L2.load_window(d), loop["cfg"], loop["meta"])
        except Exception as e:                            # a window that cannot replay is reported, not skipped silently
            rows.append({"window": rid, "year": year, "error": str(e)[:160]}); continue
        wk = r.get("weekly_returns") or []
        allw += wk
        rows.append({"window": rid, "year": year, "year_return": r["year_return"], "max_dd": r["max_dd"],
                     "mean_week": r["mean_week"], "in_band": r.get("in_band")})
    df = pd.DataFrame([x for x in rows if "error" not in x])
    eras = {}
    if len(df):
        df["era"] = (df["year"] // 10 * 10).astype(int).astype(str) + "s"
        eras = df.groupby("era").agg(windows=("window", "size"), mean_week=("mean_week", "mean"),
                                     median_year=("year_return", "median"), worst_dd=("max_dd", "min")).round(4).to_dict("index")
        y = df["year_return"].values
        per_window = {"windows": int(len(y)), "mean": float(y.mean()), "median": float(np.median(y)),
                      "percentiles": dict(zip(["p5", "p25", "p50", "p75", "p95"], map(float, np.percentile(y, [5, 25, 50, 75, 95])))),
                      "worst": float(y.min()), "best": float(y.max()), "share_negative_years": float((y < 0).mean()),
                      "worst_max_drawdown": float(df["max_dd"].min())}
    else:
        per_window = None
    return {"windows_replayed": int(len(df)), "errors": [x for x in rows if "error" in x],
            "weekly": distribution(allw) or NO, "window_year_returns": per_window or NO, "eras": eras or NO,
            "basis_hash": cfg_hash({"cfg": loop["cfg"], "meta": loop["meta"]}),
            "caveat": "survivor-biased universe; upper bound"}


def get(d, *path):
    """Exact path lookup; None (= no evidence) when any step is missing."""
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def s_leakage():
    parity = jload("state/research/parity/latest.json")
    blind = jload("state/research/blind_gates/results.json")
    pit = jload("state/research/pit/pit_audit.json")
    probe = get(blind, "probe")
    feats = get(pit, "sections", "features")
    return audit({
        # research-vs-live and batch-vs-incremental feature parity on real caches
        "feature_parity": (not parity.get("failed") and bool(parity.get("passed"))) if parity else None,
        # honest ranking unchanged when the future is scrambled AND a peeking twin is caught, at every probe date
        "future_scramble_probe": (all(x.get("honest_passes") and x.get("peeker_caught") for x in probe)
                                  if isinstance(probe, list) and probe else None),
        # engine.features.build invariant to appended future rows, and a planted shift(-5) canary detected
        "features_future_invariant": (bool(feats.get("real_features_passed")) and bool(feats.get("canary_shift_minus5_detected")))
                                     if isinstance(feats, dict) else None,
    })


def s_reproducibility():
    ao = get(jload("state/research/antioverfit/real_v1/results.json"), "repro")
    rt = get(jload("state/research/blind_gates/results.json"), "retest")
    planted = {k: v for k, v in rt.items() if k != "faithful"} if isinstance(rt, dict) else {}
    return audit({
        "reference_evaluator_reproducible": get(ao, "reference_evaluator", "reproducible"),
        "fresh_process_hash_seeds": get(ao, "fresh_process_hashseeds", "reproducible"),
        "pattern_miner_reproducible": get(ao, "pattern_miner", "reproducible"),
        "retester_faithful_replay": (get(rt, "faithful", "status") == "PASS") if get(rt, "faithful", "status") else None,
        # the re-tester must also FAIL every planted divergence, or its PASS means nothing
        # (a run from different code must be classed STALE_CODE, never compared; every other planted arm must FAIL)
        "retester_catches_planted_divergence": (all(get(v, "status") == ("STALE_CODE" if k == "other_code" else "FAIL")
                                                    for k, v in planted.items()) if planted else None),
    })


def render(R):
    L = ["# WEEKLY7 FINAL REPORT (Bible Phase 46)", "",
         f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from artefacts by scripts/final_report.py. Nothing here is typed by hand._", ""]
    for i, (title, key) in enumerate([("Implementation", "implementation"), ("Validation (every gate)", "validation"),
                                      ("Research: what genuinely improved", "improved"), ("Failures", "failed"),
                                      ("Remaining uncertainty", "uncertainty"), ("Champion configuration", "champion"),
                                      ("Challenger configurations", "challengers"), ("Performance distribution", "performance"),
                                      ("Leakage audit", "leakage"), ("Reproducibility audit", "reproducibility"),
                                      ("Checklist", "checklist")], 1):
        L += [f"## {i}. {title}", "", "```json", json.dumps(R[key], indent=1, default=str)[:12000], "```", ""]
    return "\n".join(L)


def main():
    replay = "--no-replay" not in sys.argv
    ck = (ROOT / "state" / "CHECKLIST.md").read_text(encoding="utf-8") if (ROOT / "state" / "CHECKLIST.md").exists() else ""
    rf = s_research_and_failures()
    champ, chall = s_champion_and_challengers()
    R = {"implementation": s_implementation(), "validation": s_validation(), "improved": rf["improved"],
         "failed": rf["failed"], "uncertainty": s_uncertainty(), "champion": champ, "challengers": chall,
         "performance": s_performance(replay), "leakage": s_leakage(), "reproducibility": s_reproducibility(),
         "checklist": {"counts": checklist_counts(ck), "file": "state/CHECKLIST.md"}}
    out = ROOT / "state" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_report.json").write_text(json.dumps(R, indent=1, default=str))
    (out / "FINAL_REPORT.md").write_text(render(R), encoding="utf-8")
    print(f"leakage: {R['leakage']['verdict']} | reproducibility: {R['reproducibility']['verdict']} | "
          f"checklist: {R['checklist']['counts']}")


if __name__ == "__main__":
    main()
