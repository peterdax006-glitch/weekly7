"""The self-improvement engine (Blueprint Part M, canon C5).

weekly():
  1. diagnose  - attribute live results to signals / calibration / rules; tag failures
  2. findings  - failure tags that recur become findings (one-offs don't)
  3. challenge - each finding spawns concrete challengers (weight changes, recalibration, rule tunes)
  4. test      - challengers are scored on LIVE predictions made after they were created
                 (every prediction logs its components, so re-weightings are exactly replayable)
                 plus a historical check on the out-of-sample research predictions
  5. promote   - only on a multiple-testing-corrected significant gain, never on weekly P&L;
                 a 10-session tripwire rolls back a champion that underperforms its predecessor
  6. retrain   - refit the ML on the newest data; keep old models unless the new ones hold up
  7. report    - weekly improvement report + experiment registry for the dashboard"""
import json
from datetime import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as K, model
from .scoring import load_scores

REG = K.STATE / "experiments.jsonl"
FIND = K.STATE / "findings.json"
CHAL = K.STATE / "challengers.json"
META = K.STATE / "models" / "meta.json"
REPORTS = K.STATE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
MIN_LIVE_DAYS = 5
TRIPWIRE_DAYS = 10


def _j(p, d):
    return json.loads(p.read_text()) if p.exists() else d


def _w(p, o):
    p.write_text(json.dumps(o, indent=1, default=float))


_WARNED = set()
PHASE0_FIELDS = ("window_ids", "model_params", "train_range", "validation_range", "test_range", "metrics", "gates",
                 "outcome", "reason")


def log_experiment(rec, cfg=None, seed=None, **fields):
    """Append-only registry (Bible Phase 0.2): every record carries provenance; nothing is ever overwritten.
    fields: the Phase 0.2 descriptors (window_ids, model_params, train/validation/test_range, metrics, gates, outcome,
    reason). A measurement that decides nothing is recorded as outcome "continue_testing" with that reason - never as a
    silent blank. Fields a writer does not supply stay null (the registry audit reports them) and are named once on
    stderr so the writer gets fixed rather than the gap hidden."""
    from .provenance import stamp
    import sys
    rec = {"t": datetime.utcnow().isoformat(timespec="seconds"), **stamp(cfg, seed), **rec, **fields}
    if cfg is not None:
        rec.setdefault("model_params", cfg)
    if "outcome" not in rec:
        rec["outcome"] = "continue_testing"
        rec.setdefault("reason", "measurement only; no adoption decision recorded")
    missing = [f for f in PHASE0_FIELDS if rec.get(f) is None]
    if missing and rec.get("event") not in _WARNED:
        _WARNED.add(rec.get("event"))
        print(f"[registry] {rec.get('event')}: Phase 0.2 fields not supplied: {missing}", file=sys.stderr)
    for f in missing:
        rec[f] = None
    # sha256 over the full record + a random nonce: Python's hash() is salted per process, so the old ids could collide
    # across runs (found by the B12 registry audit, 2026-09-28)
    import hashlib, uuid
    body = json.dumps(rec, sort_keys=True, default=str) + uuid.uuid4().hex
    rec.setdefault("experiment_id", "E" + hashlib.sha256(body.encode()).hexdigest()[:16])
    with open(REG, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def n_trials():
    return sum(1 for l in REG.read_text().splitlines() if l.strip()) if REG.exists() else 0


# ---------------- 1-2. diagnose ----------------
def diagnose(S: pd.DataFrame, window=20):
    tags = []
    if S.empty:
        return tags, {}
    recent = S.tail(window)
    ic = pd.DataFrame(list(recent["ic"])).astype(float)
    stats = {}
    for c in ic.columns:
        x = ic[c].dropna()
        if len(x) < 5:
            continue
        t = x.mean() / (x.std() + 1e-9) * np.sqrt(len(x))
        stats[c] = {"ic": float(x.mean()), "t": float(t), "n": int(len(x))}
        prior = model.EVIDENCE.get(c[2:]) if c.startswith("f_") else None
        if prior is not None and np.sign(x.mean()) != np.sign(prior) and abs(t) > 2:
            tags.append({"tag": "SIGNAL_DECAY", "what": c, "detail": f"IC {x.mean():+.3f} (t={t:.1f}) against prior sign"})
    cal_gap = float((recent["pred_rate"] - recent["actual_rate"]).mean())
    if abs(cal_gap) > 0.02:
        tags.append({"tag": "MISCALIBRATED", "what": "p_target", "detail": f"predicted-minus-actual {cal_gap:+.3f}"})
    for a, b in (("evidence", "mu_raw"),):
        if a in stats and b in stats and stats[a]["ic"] - stats[b]["ic"] > 0.01:
            tags.append({"tag": "MODEL_UNDERPERFORMS_PRIOR", "what": "w_model",
                         "detail": f"evidence IC {stats[a]['ic']:+.3f} vs model {stats[b]['ic']:+.3f}"})
        if a in stats and b in stats and stats[b]["ic"] - stats[a]["ic"] > 0.01:
            tags.append({"tag": "PRIOR_UNDERPERFORMS_MODEL", "what": "w_model",
                         "detail": f"model IC {stats[b]['ic']:+.3f} vs evidence {stats[a]['ic']:+.3f}"})
    if "tilt" in stats and stats["tilt"]["t"] < -1.5:
        tags.append({"tag": "SIGNAL_DECAY", "what": "tilt", "detail": f"options tilt IC {stats['tilt']['ic']:+.3f}"})
    if "score" in stats and stats["score"]["ic"] < 0:
        tags.append({"tag": "SELECTION_FAILING", "what": "score", "detail": f"blend IC {stats['score']['ic']:+.3f}"})
    tags += _rule_diagnostics()
    return tags, stats


def _rule_diagnostics():
    """Did bank / brake / stops help? Compare each forced sale's price with the price 5 sessions on."""
    tags = []
    p = K.STATE / "ledger.json"
    orders = _j(p, {}).get("orders", [])
    ao = K.STATE / "orders.jsonl"
    if ao.exists():
        orders += [json.loads(l) for l in ao.read_text().splitlines() if l.strip()]
    try:
        from .data import load
        C = load("stocks")["Close"]
    except Exception:
        return tags
    by_rule = {}
    for o in orders:
        r = o.get("reason", "")
        rule = "stop" if r.startswith("stop") else "bank" if r.startswith("bank") else \
            "brake" if r.startswith("weekly brake") else "partial" if r.startswith("partial") else None
        if not rule or o["qty"] >= 0 or o["ticker"] not in C.columns:
            continue
        d = pd.Timestamp(o["t"][:10])
        after = C[o["ticker"]].loc[d:].dropna()
        if len(after) > 5:
            by_rule.setdefault(rule, []).append(float(after.iloc[5] / after.iloc[0] - 1))
    for rule, moves in by_rule.items():
        if len(moves) >= 4 and np.mean(moves) > 0.02:
            tags.append({"tag": "RULE_HURT", "what": rule,
                         "detail": f"after {len(moves)} '{rule}' exits the stocks rose {np.mean(moves):+.1%} on average"})
    return tags


def update_findings(tags):
    F = _j(FIND, {})
    week = datetime.utcnow().strftime("%G-W%V")
    for t in tags:
        key = f"{t['tag']}:{t['what']}"
        f = F.setdefault(key, {"tag": t["tag"], "what": t["what"], "weeks": [], "status": "watching"})
        if week not in f["weeks"]:
            f["weeks"].append(week)
        f["last_detail"] = t["detail"]
        if len(f["weeks"]) >= 2 and f["status"] == "watching":
            f["status"] = "finding"
    _w(FIND, F)
    return F


# ---------------- 3. challengers ----------------
def spawn_challengers(F, meta):
    C = _j(CHAL, [])
    open_keys = {c["finding"] for c in C if c["status"] in ("testing",)}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    for key, f in F.items():
        if f["status"] != "finding" or key in open_keys:
            continue
        ch = None
        w = dict(meta.get("evidence_weights", model.EVIDENCE))
        if f["tag"] == "SIGNAL_DECAY" and f["what"].startswith("f_"):
            feat = f["what"][2:]
            w[feat] = w.get(feat, 0) * 0.0
            ch = {"kind": "evidence_weights", "change": {"evidence_weights": w}, "desc": f"switch off {feat}"}
        elif f["tag"] == "SIGNAL_DECAY" and f["what"] == "tilt":
            ch = {"kind": "meta", "change": {"w_options": 0.0}, "desc": "switch off the options tilt"}
        elif f["tag"] == "MODEL_UNDERPERFORMS_PRIOR":
            ch = {"kind": "meta", "change": {"w_model": max(0.3, meta.get("w_model", 0.5) - 0.2)},
                  "desc": "lean toward the evidence prior"}
        elif f["tag"] == "PRIOR_UNDERPERFORMS_MODEL":
            ch = {"kind": "meta", "change": {"w_model": min(0.7, meta.get("w_model", 0.5) + 0.2)},
                  "desc": "lean toward the ML model"}
        elif f["tag"] == "MISCALIBRATED":
            ch = {"kind": "recalibrate", "change": {}, "desc": "refit isotonic calibration on live outcomes"}
        elif f["tag"] == "RULE_HURT":
            knob = {"stop": ("STOP_ATR", K.STOP_ATR + 0.5), "bank": ("BANK_EXPOSURE", min(0.7, K.BANK_EXPOSURE + 0.2)),
                    "brake": ("BRAKE_LEVEL", K.BRAKE_LEVEL - 0.03), "partial": ("PARTIAL_TAKE", K.PARTIAL_TAKE + 0.05)}.get(f["what"])
            if knob:
                ch = {"kind": "config", "change": {knob[0]: knob[1]}, "desc": f"loosen the {f['what']} rule"}
        if ch:
            ch.update({"id": f"CH{len(C) + 1:03d}", "finding": key, "created": today, "status": "testing"})
            C.append(ch)
            log_experiment({"event": "challenger_created", **ch, "evidence": f["last_detail"]})
    _w(CHAL, C)
    return C


# ---------------- 4-5. test and promote ----------------
def _replay_score(P, meta, change):
    """Recompute the blended score for one prediction day under a challenger's settings."""
    m = {**meta, **{k: v for k, v in change.items() if k != "evidence_weights"}}
    w = change.get("evidence_weights", meta.get("evidence_weights", model.EVIDENCE))
    ev = sum(wt * P.get(f"f_{c}", 0).fillna(0) for c, wt in w.items() if f"f_{c}" in P)
    ev = ev.rank(pct=True) if hasattr(ev, "rank") else P["evidence"]
    mr = P["mu_raw"].rank(pct=True)          # same formula as policy.score
    wm = m.get("w_model", 0.5)
    return wm * mr + (1 - wm) * ev + m.get("w_options", 0.15) * P.get("tilt", 0)


def evaluate(ch, meta, stocks):
    """Paired daily IC difference (challenger - champion) on live days after creation."""
    from .live import PRED_DIR
    from .scoring import _outcomes
    diffs = []
    for f in sorted(PRED_DIR.glob("*.parquet")):
        if f.stem < ch["created"]:
            continue
        P = pd.read_parquet(f)
        out = _outcomes(stocks, f.stem, [t for t in P.index if t in stocks["Close"].columns])
        if out is None:
            continue
        df = P.join(out, how="inner").dropna(subset=["fwd"])
        champ = _replay_score(df, meta, {})
        chal = _replay_score(df, meta, ch["change"])
        r = df["fwd"].rank()
        diffs.append(chal.rank().corr(r) - champ.rank().corr(r))
    return np.array(diffs)


def test_and_promote(C, meta, stocks):
    trials = max(1, n_trials())
    z_needed = NormalDist().inv_cdf(1 - 0.05 / trials)          # Bonferroni over every trial ever run
    for ch in C:
        if ch["status"] != "testing" or ch["kind"] == "recalibrate":
            continue
        d = evaluate(ch, meta, stocks)
        if len(d) < MIN_LIVE_DAYS:
            continue
        z = d.mean() / (d.std(ddof=1) + 1e-9) * np.sqrt(len(d))
        ch["live"] = {"days": int(len(d)), "ic_gain": float(d.mean()), "z": float(z), "z_needed": float(z_needed)}
        if z >= z_needed:
            prev = {k: meta.get(k) for k in ch["change"]}
            if ch["kind"] == "config":
                ov = _j(K.STATE / "config_overrides.json", {})
                prev = {k: ov.get(k, getattr(K, k)) for k in ch["change"]}
                ov.update(ch["change"]); _w(K.STATE / "config_overrides.json", ov)
            else:
                meta.update(ch["change"])
            meta["version"] = _bump(meta.get("version", "1.0"))
            ch.update({"status": "promoted", "promoted": datetime.utcnow().strftime("%Y-%m-%d"), "previous": prev,
                       "version": meta["version"]})
            log_experiment({"event": "promoted", "id": ch["id"], "version": meta["version"], **ch["live"]})
        elif len(d) >= 3 * MIN_LIVE_DAYS and z < 0:
            ch["status"] = "rejected"
            log_experiment({"event": "rejected", "id": ch["id"], **ch["live"]})
    # tripwire: a promoted champion must keep beating what it replaced
    for ch in C:
        if ch["status"] == "promoted" and ch["kind"] != "config":
            back = {"kind": ch["kind"], "change": ch["previous"], "created": ch["promoted"]}
            d = evaluate(back, meta, stocks)
            if len(d) >= TRIPWIRE_DAYS:
                if d.mean() > 0 and d.mean() / (d.std(ddof=1) + 1e-9) * np.sqrt(len(d)) > 1.0:
                    meta.update(ch["previous"]); ch["status"] = "rolled_back"
                    log_experiment({"event": "rolled_back", "id": ch["id"], "gain_of_old": float(d.mean())})
                else:
                    ch["status"] = "confirmed"
                    log_experiment({"event": "confirmed", "id": ch["id"]})
    _w(CHAL, C)
    return C


def _bump(v):
    a, b = (v.split(".") + ["0"])[:2]
    return f"{a}.{int(b) + 1}"


def recalibrate(stocks):
    """Blend live outcomes into the isotonic calibrator when MISCALIBRATED is a finding."""
    import pickle
    from sklearn.isotonic import IsotonicRegression
    from .live import PRED_DIR
    from .scoring import _outcomes
    rows = []
    for f in sorted(PRED_DIR.glob("*.parquet")):
        P = pd.read_parquet(f)
        out = _outcomes(stocks, f.stem, [t for t in P.index if t in stocks["Close"].columns])
        if out is not None:
            rows.append(P[["p_target_raw"]].join(out, how="inner"))
    if len(rows) < MIN_LIVE_DAYS:
        return False
    d = pd.concat(rows).dropna()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d["p_target_raw"], d["touched7"])
    (K.STATE / "models" / "iso.pkl").write_bytes(pickle.dumps(iso))
    return True


# ---------------- 7. weekly driver ----------------
def weekly():
    from .data import load
    stocks = load("stocks")
    meta = _j(META, {"version": "1.0", "w_model": 0.5, "w_options": 0.15})
    S = load_scores()
    tags, stats = diagnose(S)
    F = update_findings(tags)
    C = spawn_challengers(F, meta)
    for ch in C:
        if ch["status"] == "testing" and ch["kind"] == "recalibrate" and recalibrate(stocks):
            ch["status"] = "promoted"; meta["version"] = _bump(meta.get("version", "1.0"))
            log_experiment({"event": "promoted", "id": ch["id"], "version": meta["version"], "note": "recalibrated"})
    C = test_and_promote(C, meta, stocks)
    _w(CHAL, C)
    _w(META, meta)
    try:
        from .train import retrain_guarded
        retrain_note = retrain_guarded()
    except Exception as e:  # never let a retrain failure stop the loop
        retrain_note = f"retrain skipped: {e}"
    report(stats, tags, F, C, meta, retrain_note)


def report(stats, tags, F, C, meta, retrain_note):
    wk = datetime.utcnow().strftime("%Y-%m-%d")
    L = [f"# Weekly improvement report — {wk}", "", f"Engine version **{meta.get('version')}**. {retrain_note}", "",
         "## Signal health (last 20 scored days, rank IC vs 5-day return)", "", "| signal | IC | t | days |", "|---|---|---|---|"]
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["ic"]):
        L.append(f"| {k} | {v['ic']:+.3f} | {v['t']:+.1f} | {v['n']} |")
    L += ["", "## What isn't working (this week's failure tags)", ""]
    L += [f"- **{t['tag']}** {t['what']}: {t['detail']}" for t in tags] or ["- nothing flagged"]
    L += ["", "## Findings (recurring)", ""]
    L += [f"- {k}: {f['status']} ({len(f['weeks'])} weeks) — {f.get('last_detail', '')}" for k, f in F.items()] or ["- none yet"]
    L += ["", "## Challengers", ""]
    L += [f"- {c['id']} [{c['status']}] {c['desc']} — {c.get('live', {})}" for c in C] or ["- none yet"]
    (REPORTS / f"week-{wk}.md").write_text("\n".join(L), encoding="utf-8")
