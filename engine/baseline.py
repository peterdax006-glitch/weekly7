"""Bible Phase 0.4: the immutable baseline snapshot, and the comparison of any later run against it.

`freeze` measures nothing itself. It collects the champion configuration, the code hash (engine.provenance.code_hash),
the data snapshot and the key metrics already recorded in state/experiments.jsonl, and writes them as a checkpoint
bundle (engine.checkpoint: write-once, read-only, sha256 manifest) under state/baseline/<id>/. Phase 0.4 items that the
log cannot supply (mover accuracy, pattern results, ...) are listed under `unmeasured` with the reason; they are never
filled with a guess, and `diff_vs_baseline` reports them as having no baseline. `verify` re-hashes the bundle and also
reports whether the code on disk still matches the frozen code hash."""
import json
from pathlib import Path

import numpy as np

from . import checkpoint as C, config as K, provenance as P
from .registry import Registry

ROOT = K.STATE / "baseline"

# Phase 0.4 list -> where the numbers come from in the log (None = not derivable from experiments.jsonl)
PHASE04 = {"blind_window_results": "livesim_cycle", "weekly_distribution": "historical_test",
           "max_drawdown": "historical_test", "worst_weeks": "historical_test", "turnover": "historical_test",
           "costs": "historical_test", "model_results": "walk_forward", "mover_accuracy": None, "pattern_results": None,
           "analog_results": None, "missed_winners": None, "direction_accuracy": None}

# metric -> True when a larger value is better (used to label a delta better / worse)
POLARITY = {"mean_week": True, "median_week": True, "pct_ge_7": True, "pct_le_m7": False, "win_weeks": True,
            "worst_week": True, "best_week": True, "max_dd": True, "max_drawdown": True, "turnover": False,
            "cost_total": False, "share_in_band": True, "p05_week": True, "positive_week_pct": True,
            "blind_mean_week": True, "blind_weeks_ge_7": True, "blind_year_return": True}
# a difference smaller than this fraction of the baseline magnitude (or 1e-4 absolute) is 'unchanged'
REL_TOL = 0.02


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def extract_metrics(records):
    """Pull the Phase 0.4 measurements out of experiment records. Returns (metrics, sources, unmeasured)."""
    m, src = {}, {}
    ht = [r for r in records if r.get("event") == "historical_test"]
    if ht:
        r = ht[-1]
        for k in ("mean_week", "median_week", "pct_ge_7", "pct_le_m7", "win_weeks", "worst_week", "best_week", "max_dd",
                  "turnover", "cost_total", "weeks"):
            if _num(r.get(k)) is not None:
                m[k] = _num(r[k])
        src["historical_test"] = {"t": r.get("t"), "variant": r.get("variant"), "experiment_id": r.get("experiment_id")}
    ls = [r for r in records if r.get("event") == "livesim_cycle" and _num(r.get("mean_week")) is not None]
    if ls:
        m["blind_mean_week"] = float(np.mean([_num(r["mean_week"]) for r in ls]))
        m["blind_weeks_ge_7"] = float(np.mean([_num(r.get("weeks_ge_7")) or 0 for r in ls]))
        yr = [_num(r.get("year_return")) for r in ls if _num(r.get("year_return")) is not None]
        if yr:
            m["blind_year_return"] = float(np.mean(yr))
        m["blind_cycles"] = float(len(ls))
        src["livesim_cycle"] = {"n": len(ls), "runs": sorted({str(r.get("run_id")) for r in ls})}
    wf = [r for r in records if r.get("event") == "walk_forward"]
    if wf:
        for k in ("blend", "evidence", "model", "blend_t"):
            if _num(wf[-1].get(k)) is not None:
                m[f"walk_forward_{k}"] = _num(wf[-1][k])
        src["walk_forward"] = {"t": wf[-1].get("t")}
    have = {e for e in src}
    unmeasured = {k: ("no source in experiments.jsonl" if ev is None else f"no '{ev}' record")
                  for k, ev in PHASE04.items() if ev is None or ev not in have}
    return m, src, unmeasured


def engine_files():
    """A baseline stands for the whole engine, not for whichever modules happen to be imported when it is frozen."""
    return sorted(q.relative_to(P.ROOT).as_posix() for q in (P.ROOT / "engine").glob("*.py"))


def freeze(champion_cfg, now, root=None, registry=None, baseline_id=None, seeds=None, note=""):
    """Write state/baseline/<id>/. Refuses an empty champion config and refuses to overwrite an existing baseline id.
    Returns (path, summary)."""
    if not champion_cfg:
        raise ValueError("a baseline needs the champion configuration it is the baseline of")
    reg = registry if registry is not None else Registry()  # an empty Registry is falsy
    metrics, sources, unmeasured = extract_metrics(reg.records)
    if not metrics:
        raise ValueError("no baseline metrics found in the registry; run the existing system first (Phase 0.4)")
    files = engine_files()
    ch = P.code_hash(files)
    bid = baseline_id or f"B_{str(now)[:10]}_{ch[:8]}"
    prov = {"code_hash": ch, "code_files": files, "code_mixed": [], "data_snapshot": P.data_snapshot(), "git_commit": P.git_commit(),
            "blueprint_version": P.BLUEPRINT_VERSION, "config_hash": P.config_hash(champion_cfg),
            "n_registry_records": len(reg)}
    summary = {"baseline_id": bid, "sources": sources, "unmeasured": unmeasured, "note": note,
               "metric_names": sorted(metrics)}
    d = C.write_checkpoint(root or ROOT, bid, champion_cfg, metrics, seeds or {"registry_replay": 0}, summary, now,
                           provenance=prov)
    return d, summary


def load(path):
    b = C.load(path)
    return {"config": b["config"], "metrics": b["metrics"], "summary": b["summary"],
            "provenance": b["manifest"]["provenance"], "created": b["manifest"]["created"]}


def verify(path):
    """Bundle integrity plus code drift: `code_drift` is True when engine code on disk is not what was frozen
    (informational - the baseline stays valid, but a comparison should say the code has moved)."""
    v = C.verify(path)
    out = {"ok": v["ok"], "problems": v["problems"], "code_drift": None}
    if v["ok"]:
        out["code_drift"] = P.stale(load(path)["provenance"])       # same file list re-hashed; legacy records count as drift
    return out


def latest(root=None):
    r = Path(root or ROOT)
    ids = C.list_checkpoints(r)
    return r / ids[-1] if ids else None


def diff_vs_baseline(run_metrics, path=None):
    """Compare a run's metrics with the frozen baseline. Fails closed: a missing or corrupt baseline raises.
    Returns {'baseline_id', 'code_drift', 'rows': {metric: {base, run, delta, verdict}}, 'no_baseline': [...],
    'unmeasured': {...}, 'better': n, 'worse': n}. verdict is better / worse / unchanged / unknown-polarity."""
    p = Path(path) if path else latest()
    if p is None:
        raise FileNotFoundError("no baseline frozen")
    v = verify(p)
    if not v["ok"]:
        raise C.CheckpointError(f"baseline failed verification: {v['problems']}")
    b = load(p)
    rows, nob = {}, []
    for k, rv in run_metrics.items():
        rv = _num(rv)
        bv = _num(b["metrics"].get(k))
        if rv is None:
            continue
        if bv is None:
            nob.append(k)
            continue
        delta = rv - bv
        if abs(delta) <= max(1e-4, REL_TOL * abs(bv)):
            verdict = "unchanged"
        elif k not in POLARITY:
            verdict = "unknown-polarity"
        else:
            verdict = "better" if (delta > 0) == POLARITY[k] else "worse"
        rows[k] = {"base": bv, "run": rv, "delta": delta, "verdict": verdict}
    return {"baseline_id": b["summary"]["baseline_id"], "code_drift": v["code_drift"], "rows": rows, "no_baseline": nob,
            "unmeasured": b["summary"]["unmeasured"],
            "better": sum(r["verdict"] == "better" for r in rows.values()),
            "worse": sum(r["verdict"] == "worse" for r in rows.values())}
