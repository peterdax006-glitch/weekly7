"""Bible Phase 36: the required machine-readable and human-readable report after each major run.

`build_report` computes every Tier 1-3 figure from the weekly-return series itself (nothing is typed in), takes the
algorithm / analog / memory / adaptation / gate sections from what the run produced, and reports a field it was not
given as null and lists it under `missing` - a blank is never rendered as zero. `decide` applies the fail-closed
decision rule (Phase 35 spirit): a failed gate or a train/validation/test overlap is REJECT, a missing gate or too little
evidence is CONTINUE TESTING, and ADOPT is reached only when nothing is missing or failed."""
import html as _html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline as B, config as K

BAND = (0.05, 0.10)
CATASTROPHE = -0.15
GATES = ("parity", "retester", "future_scramble", "time_fence", "worker_health", "reproducibility")
HEADER = ("run_id", "git_commit", "canon_hash", "blueprint_version", "config_hash", "data_snapshot", "seed")
STATUSES = ("active", "rescoped", "no_gain", "duplicate", "discarded", "rejected")
DECISIONS = ("ADOPT", "REJECT", "CONTINUE TESTING")


def _f(x):
    """JSON-safe scalar: numpy -> python, NaN/inf -> None."""
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return _f(o)


# ---------------- tiers ----------------
def tier1(weekly):
    w = np.asarray(weekly, float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return {"n_weeks": 0, "mean_week": None, "median_week": None, "share_in_band": None,
                "share_gt_10": None, "share_lt_5": None}
    lo, hi = BAND
    return {"n_weeks": len(w), "mean_week": w.mean(), "median_week": np.median(w),
            "share_in_band": np.mean((w >= lo) & (w <= hi)), "share_gt_10": np.mean(w > hi), "share_lt_5": np.mean(w < lo)}


def max_drawdown(weekly):
    w = np.asarray(weekly, float)
    if len(w) == 0:
        return None
    eq = np.cumprod(1 + np.clip(w, -1, None))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.min(eq / peak - 1))


def tier2(weekly, turnover=None, costs=None):
    w = np.asarray(weekly, float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return {"max_drawdown": None, "worst_week": None, "p05_week": None, "catastrophic_losses": None,
                "turnover": turnover, "costs": costs}
    return {"max_drawdown": max_drawdown(w), "worst_week": w.min(), "p05_week": np.percentile(w, 5),
            "catastrophic_losses": int((w <= CATASTROPHE).sum()), "turnover": turnover, "costs": costs}


def calibration(prob, outcome, bins=10):
    """Brier score and expected calibration error of forecast probabilities against 0/1 outcomes."""
    p, y = np.asarray(prob, float), np.asarray(outcome, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"brier": None, "ece": None, "n": 0}
    if p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities must lie in [0,1]")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    return {"brier": np.mean((p - y) ** 2), "ece": ece, "n": len(p)}


def eight_of_ten(weekly, k=8, n=10):
    """Share of rolling 10-week windows with at least 8 positive weeks (the '8/10 rate')."""
    pos = (np.asarray(weekly, float) > 0).astype(int)
    if len(pos) < n:
        return None
    return float(np.mean(np.convolve(pos, np.ones(n, int), "valid") >= k))


def tier3(weekly, pred_dir=None, actual_ret=None, prob=None, outcome=None):
    w = np.asarray(weekly, float)
    w = w[np.isfinite(w)]
    out = {"positive_week_pct": None, "positive_in_band_pct": None, "direction_accuracy": None,
           "calibration": {"brier": None, "ece": None, "n": 0}, "rate_8_of_10": eight_of_ten(w)}
    if len(w):
        pos = w > 0
        out["positive_week_pct"] = pos.mean()
        # of the weeks that made money, how many landed in the target band rather than under or over it
        out["positive_in_band_pct"] = (np.mean((w[pos] >= BAND[0]) & (w[pos] <= BAND[1])) if pos.any() else 0.0)
    if pred_dir is not None and actual_ret is not None:
        d, r = np.asarray(pred_dir, float), np.asarray(actual_ret, float)
        if len(d) != len(r):
            raise ValueError("pred_dir and actual_ret differ in length")
        ok = np.isfinite(d) & np.isfinite(r) & (r != 0) & (d != 0)
        out["direction_accuracy"] = float(np.mean(np.sign(d[ok]) == np.sign(r[ok]))) if ok.any() else None
    if prob is not None and outcome is not None:
        out["calibration"] = calibration(prob, outcome)
    return out


# ---------------- other sections ----------------
def algorithm_section(patterns):
    """From a PatternMiner.patterns DataFrame. False-discovery diagnostics: the expected number of false actives is the
    sum of (p_hallucinated + p_coincidence) over active patterns; the P(real) distribution is reported by quantile."""
    empty = {"pattern_count": 0, "active": 0, "failed": 0, "rescoped": 0, "discarded": 0,
             "p_real": None, "false_discovery": None}
    if patterns is None or len(patterns) == 0:
        return empty
    st = patterns["status"].value_counts().to_dict() if "status" in patterns else {}
    act = patterns[patterns["status"] == "active"] if "status" in patterns else patterns.iloc[0:0]
    out = {"pattern_count": len(patterns), "active": st.get("active", 0), "failed": st.get("rejected", 0) + st.get("no_gain", 0),
           "rescoped": st.get("rescoped", 0), "discarded": st.get("discarded", 0), "by_status": st}
    if "p_real" in patterns:
        pr = patterns["p_real"].astype(float).dropna()
        out["p_real"] = ({"min": pr.min(), "p25": pr.quantile(.25), "median": pr.median(), "p75": pr.quantile(.75), "max": pr.max()}
                         if len(pr) else None)
    if {"p_hallucinated", "p_coincidence"} <= set(patterns.columns):
        fa = (act["p_hallucinated"].astype(float) + act["p_coincidence"].astype(float)).sum() if len(act) else 0.0
        out["false_discovery"] = {"expected_false_active": fa, "expected_false_share": fa / len(act) if len(act) else None}
    else:
        out["false_discovery"] = None
    return out


def analogs_section(distances=None, forecast=None, realised=None):
    out = {"count": 0, "distance": None, "forecast_quality": None}
    if distances is not None and len(distances):
        d = np.asarray(distances, float)
        d = d[np.isfinite(d)]
        out["count"] = len(d)
        if len(d):
            out["distance"] = {"min": d.min(), "median": np.median(d), "p90": np.percentile(d, 90), "max": d.max()}
    if forecast is not None and realised is not None and len(forecast):
        f, r = np.asarray(forecast, float), np.asarray(realised, float)
        ok = np.isfinite(f) & np.isfinite(r)
        if ok.sum() >= 3:
            out["forecast_quality"] = {"n": int(ok.sum()), "corr": np.corrcoef(f[ok], r[ok])[0, 1] if f[ok].std() and r[ok].std() else None,
                                       "mae": np.mean(np.abs(f[ok] - r[ok])), "direction_hit": np.mean(np.sign(f[ok]) == np.sign(r[ok]))}
    return out


def windows_check(win):
    """train < validation < test, non-overlapping. Returns (violations, normalised). A window is {'start','end'}."""
    bad = []
    parts = [(n, win.get(n)) for n in ("train", "validation", "test")]
    rng = {}
    for n, w in parts:
        if not w or not w.get("start") or not w.get("end"):
            continue
        s, e = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        if e < s:
            bad.append(f"{n} ends before it starts")
        rng[n] = (s, e)
    order = [n for n, _ in parts if n in rng]
    for a, b in zip(order, order[1:]):
        if rng[b][0] <= rng[a][1]:
            bad.append(f"{b} starts on or before {a} ends")
    return bad


# ---------------- regime / era breakdown ----------------
def era_labels(dates, edges=(2000, 2010, 2020)):
    """Decade-style era label per week date: '<2000', '2000-2009', '2010-2019', '2020+'."""
    out = []
    for d in pd.to_datetime(list(dates)):
        y = d.year
        i = sum(y >= e for e in edges)
        out.append(f"<{edges[0]}" if i == 0 else f"{edges[i - 1]}+" if i == len(edges) else f"{edges[i - 1]}-{edges[i] - 1}")
    return out


def breakdown(weekly, labels, min_n=20):
    """Tier figures per regime/era label. A group with fewer than `min_n` weeks is marked thin: its numbers are shown
    but never counted as evidence. `dominance` is the largest share of the total summed return supplied by one
    group (only defined when the total is positive) - a high value means the result rests on one regime."""
    w = np.asarray(weekly, float)
    lab = np.asarray(list(labels), object)
    if len(w) != len(lab):
        raise ValueError("labels must align with weekly returns")
    ok = np.isfinite(w)
    w, lab = w[ok], lab[ok]
    groups = {}
    for g in sorted(set(lab.tolist()), key=str):
        x = w[lab == g]
        t1 = tier1(x)
        groups[str(g)] = {"n": len(x), "thin": len(x) < min_n, "mean_week": t1["mean_week"], "share_in_band": t1["share_in_band"],
                          "positive_week_pct": float(np.mean(x > 0)), "max_drawdown": max_drawdown(x), "worst_week": float(x.min()),
                          "sum_return": float(x.sum())}
    tot = float(w.sum()) if len(w) else 0.0
    dom = max((g["sum_return"] / tot for g in groups.values()), default=None) if tot > 0 else None
    solid = [g for g in groups.values() if not g["thin"]]
    return {"groups": groups, "dominance": dom, "n_thin": len(groups) - len(solid),
            "mean_week_range": (max(g["mean_week"] for g in solid) - min(g["mean_week"] for g in solid)) if len(solid) > 1 else None}


# ---------------- comparison with the frozen baseline ----------------
def baseline_metrics(weekly, t2, t3):
    """Run figures on the same definitions as engine.baseline.extract_metrics (turnover is left out on purpose: the
    logged baseline turnover is a period total and the run's is per week, so the two are not comparable)."""
    w = np.asarray(weekly, float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return {}
    return {"mean_week": w.mean(), "median_week": np.median(w), "worst_week": w.min(), "max_dd": t2["max_drawdown"],
            "pct_ge_7": np.mean(w >= 0.07), "pct_le_m7": np.mean(w <= -0.07), "win_weeks": t3["positive_week_pct"]}


def baseline_section(run, weekly, t2, t3):
    path = run.get("baseline_dir")
    if not path:
        return None
    try:
        return B.diff_vs_baseline(baseline_metrics(weekly, t2, t3), path)
    except Exception as e:  # missing / corrupt baseline: reported, and the decision will not ADOPT
        return {"error": f"{type(e).__name__}: {e}"}


def unproven_list(rep, extra=()):
    """Everything the report cannot claim. Rendered in its own block so a reader sees the gaps beside the good news."""
    u = [f"not provided: {m}" for m in rep["missing"]]
    for name in ("regimes", "eras"):
        b = rep.get(name)
        if not b:
            u.append(f"no {name} breakdown supplied: robustness across {name} is unknown")
            continue
        thin = [g for g, v in b["groups"].items() if v["thin"]]
        if thin:
            u.append(f"{name} with too few weeks to count as evidence: {thin}")
        if b["dominance"] is not None and b["dominance"] > 0.6:
            u.append(f"{b['dominance']:.0%} of the total return came from one {name[:-1]}: not shown to generalise")
    bl = rep.get("baseline")
    if bl is None:
        u.append("no comparison with the frozen baseline (Phase 0.4)")
    elif "error" in bl:
        u.append(f"baseline comparison unavailable: {bl['error']}")
    else:
        if bl["code_drift"]:
            u.append("engine code has changed since the baseline was frozen; the comparison mixes code and idea")
        u += [f"baseline never measured: {k} ({why})" for k, why in bl["unmeasured"].items()]
    return u + [str(x) for x in extra]


# ---------------- assembly ----------------
def build_report(run, now):
    """run: dict with header fields (HEADER), 'windows' {train,validation,test:{start,end}}, 'window_ids',
    'weekly' (test-period weekly returns), optional turnover/costs, pred_dir/actual_ret, prob/outcome, patterns,
    analog_*, memory, adaptation, gates {name: bool}. Returns the full report dict incl. decision."""
    rep = {"generated_at": str(now)}
    for h in HEADER:
        rep[h] = run.get(h)
    win = run.get("windows") or {}
    rep["windows"] = {"ids": run.get("window_ids"), "train": win.get("train"), "validation": win.get("validation"),
                      "test": win.get("test"), "violations": windows_check(win)}
    weekly = run.get("weekly")
    weekly = np.asarray([] if weekly is None else weekly, float)
    rep["tier1"] = tier1(weekly)
    rep["tier2"] = tier2(weekly, run.get("turnover"), run.get("costs"))
    rep["tier3"] = tier3(weekly, run.get("pred_dir"), run.get("actual_ret"), run.get("prob"), run.get("outcome"))
    rep["algorithm"] = algorithm_section(run.get("patterns"))
    rep["analogs"] = analogs_section(run.get("analog_distances"), run.get("analog_forecast"), run.get("analog_realised"))
    mem = run.get("memory") or {}
    rep["memory"] = {k: mem.get(k) for k in ("lesson_count", "accepted_lessons", "rejected_lessons", "shock_events", "memory_switches")}
    ad = run.get("adaptation") or {}
    rep["adaptation"] = {k: ad.get(k) for k in ("parameter_changes", "reverts", "evidence", "confidence")}
    g = run.get("gates") or {}
    rep["gates"] = {n: g.get(n) for n in GATES}
    dates = run.get("week_dates")
    rep["regimes"] = breakdown(weekly, run["regimes"]) if run.get("regimes") is not None else None
    rep["eras"] = breakdown(weekly, era_labels(dates)) if dates is not None and len(dates) == len(weekly) else None
    rep["baseline"] = baseline_section(run, weekly, rep["tier2"], rep["tier3"])
    rep["missing"] = _missing(rep)
    rep["unproven"] = unproven_list(rep, run.get("unproven") or ())
    rep["decision"], rep["reason"] = decide(rep, run.get("requested_decision"), run.get("min_weeks", 100))
    return clean(rep)


def _missing(rep):
    miss = [h for h in HEADER if rep.get(h) in (None, "")]
    for name in ("train", "validation", "test"):
        if not rep["windows"].get(name):
            miss.append(f"windows.{name}")
    for t in ("tier1", "tier2", "tier3"):
        miss += [f"{t}.{k}" for k, v in rep[t].items() if v is None]
    miss += [f"gates.{n}" for n, v in rep["gates"].items() if v is None]
    for sec in ("memory", "adaptation"):
        miss += [f"{sec}.{k}" for k, v in rep[sec].items() if v is None]
    if rep["algorithm"]["pattern_count"] == 0:
        miss.append("algorithm.patterns")
    if rep["analogs"]["count"] == 0:
        miss.append("analogs.distances")
    return miss


def decide(rep, requested=None, min_weeks=100):
    """Fail-closed decision. Returns (decision, reason). A requested ADOPT that the evidence does not support is
    downgraded, and the downgrade is stated in the reason."""
    reasons = []
    failed = [n for n, v in rep["gates"].items() if v is False]
    if rep["windows"]["violations"]:
        decision, reasons = "REJECT", [f"window leakage: {rep['windows']['violations']}"]
    elif failed:
        decision, reasons = "REJECT", [f"gates failed: {failed}"]
    else:
        t1, t2 = rep["tier1"], rep["tier2"]
        gates_missing = [n for n, v in rep["gates"].items() if v is None]
        core_missing = [k for k in ("mean_week", "share_in_band") if t1.get(k) is None] + \
                       [k for k in ("max_drawdown", "worst_week") if t2.get(k) is None]
        if (t1.get("n_weeks") or 0) < min_weeks:
            decision, reasons = "CONTINUE TESTING", [f"only {t1.get('n_weeks') or 0} weeks (< {min_weeks})"]
        elif gates_missing or core_missing:
            decision, reasons = "CONTINUE TESTING", [f"not evidenced: {gates_missing + core_missing}"]
        elif not (BAND[0] <= t1["mean_week"] <= BAND[1]):
            decision, reasons = "REJECT", [f"mean weekly move {t1['mean_week']:.3%} outside the {BAND[0]:.0%}-{BAND[1]:.0%} band"]
        elif t2["catastrophic_losses"]:
            decision, reasons = "REJECT", [f"{t2['catastrophic_losses']} catastrophic week(s) <= {CATASTROPHE:.0%}"]
        elif (rep.get("baseline") or {}).get("error"):
            decision, reasons = "CONTINUE TESTING", [f"baseline comparison failed: {rep['baseline']['error']}"]
        elif ((rep.get("baseline") or {}).get("rows", {}).get("mean_week") or {}).get("verdict") == "worse":
            decision, reasons = "REJECT", ["mean weekly move is worse than the frozen baseline"]
        else:
            decision, reasons = "ADOPT", [f"all gates pass; mean {t1['mean_week']:.2%}/week, {t1['share_in_band']:.0%} of weeks in band, "
                                         f"max drawdown {t2['max_drawdown']:.1%}"]
    if requested and requested != decision:
        reasons.append(f"requested {requested} overruled by the evidence")
    return decision, "; ".join(reasons)


# ---------------- rendering ----------------
def _p(v, pct=True, nd=1):
    if v is None:
        return "n/a"
    return f"{v * 100:.{nd}f}%" if pct else f"{v:.{nd}f}"


def render_text(rep):
    t1, t2, t3 = rep["tier1"], rep["tier2"], rep["tier3"]
    al, an, w = rep["algorithm"], rep["analogs"], rep["windows"]
    rng = lambda x: "n/a" if not x else f"{x.get('start')} to {x.get('end')}"
    L = [f"{'RANDOM SEED' if h == 'seed' else h.upper().replace('_', ' ')}: {rep.get(h) if rep.get(h) is not None else 'n/a'}" for h in HEADER]
    L += ["", "WINDOWS", f"  ids: {w['ids'] if w['ids'] is not None else 'n/a'}", f"  TRAIN PERIOD: {rng(w['train'])}",
          f"  VALIDATION PERIOD: {rng(w['validation'])}", f"  TEST PERIOD: {rng(w['test'])}"]
    if w["violations"]:
        L.append(f"  VIOLATIONS: {w['violations']}")
    L += ["", "TIER 1", f"  weekly average move: {_p(t1['mean_week'], nd=2)}", f"  weekly median move: {_p(t1['median_week'], nd=2)}",
          f"  share 5-10%: {_p(t1['share_in_band'])}", f"  share >10%: {_p(t1['share_gt_10'])}", f"  share <5%: {_p(t1['share_lt_5'])}",
          f"  weeks: {t1['n_weeks']}", "", "TIER 2", f"  max drawdown: {_p(t2['max_drawdown'])}", f"  worst week: {_p(t2['worst_week'])}",
          f"  5th percentile week: {_p(t2['p05_week'])}", f"  catastrophic losses: {t2['catastrophic_losses'] if t2['catastrophic_losses'] is not None else 'n/a'}",
          f"  turnover: {_p(t2['turnover'], pct=False, nd=2)}", f"  costs: {_p(t2['costs'], pct=False, nd=4)}", "",
          "TIER 3", f"  positive in-band percentage: {_p(t3['positive_in_band_pct'])}", f"  positive weeks: {_p(t3['positive_week_pct'])}",
          f"  direction accuracy: {_p(t3['direction_accuracy'])}",
          f"  calibration: brier {_p(t3['calibration']['brier'], pct=False, nd=4)}, ECE {_p(t3['calibration']['ece'], pct=False, nd=4)}",
          f"  8/10 rate: {_p(t3['rate_8_of_10'])}", "", "ALGORITHM", f"  pattern count: {al['pattern_count']}", f"  active pattern count: {al['active']}",
          f"  failed patterns: {al['failed']}", f"  rescoped patterns: {al['rescoped']}", f"  discarded patterns: {al['discarded']}"]
    pr = al.get("p_real")
    L.append("  P(real) distribution: " + ("n/a" if not pr else ", ".join(f"{k} {v:.2f}" for k, v in pr.items())))
    fd = al.get("false_discovery")
    L.append("  false discovery diagnostics: " + ("n/a" if not fd else f"expected false active {fd['expected_false_active']:.2f}"
                                                 + (f" ({fd['expected_false_share']:.1%} of active)" if fd["expected_false_share"] is not None else "")))
    d = an["distance"]
    fq = an["forecast_quality"]
    L += ["", "ANALOGS", f"  analog count: {an['count']}",
          "  distance distribution: " + ("n/a" if not d else ", ".join(f"{k} {v:.3f}" for k, v in d.items())),
          "  forecast quality: " + ("n/a" if not fq else f"n {fq['n']}, corr {_p(fq['corr'], pct=False, nd=2)}, direction hit {_p(fq['direction_hit'])}"),
          "", "MEMORY"] + [f"  {k.replace('_', ' ')}: {v if v is not None else 'n/a'}" for k, v in rep["memory"].items()]
    L += ["", "ADAPTATION"] + [f"  {k.replace('_', ' ')}: {v if v is not None else 'n/a'}" for k, v in rep["adaptation"].items()]
    L += ["", "GATES"] + [f"  {n.replace('_', ' ')}: " + {True: "PASS", False: "FAIL", None: "NOT RUN"}[v] for n, v in rep["gates"].items()]
    L += ["", "DECISION", f"  {rep['decision']}", "", "REASON", f"  {rep['reason']}"]
    for name in ("regimes", "eras"):
        b = rep.get(name)
        if b:
            L += ["", name.upper()] + [f"  {g}: {v['n']} weeks{' (THIN)' if v['thin'] else ''}, mean {_p(v['mean_week'], nd=2)}, in band "
                                       f"{_p(v['share_in_band'])}, positive {_p(v['positive_week_pct'])}, max drawdown {_p(v['max_drawdown'])}"
                                       for g, v in b["groups"].items()]
    bl = rep.get("baseline")
    if bl:
        L += ["", "VS BASELINE"]
        if "error" in bl:
            L.append(f"  unavailable: {bl['error']}")
        else:
            L += [f"  baseline {bl['baseline_id']}: {bl['better']} better, {bl['worse']} worse"] + \
                 [f"  {k}: {v['base']:.4f} -> {v['run']:.4f} ({v['verdict']})" for k, v in bl["rows"].items()]
    L += ["", f"UNPROVEN ({len(rep['unproven'])})"] + [f"  - {u}" for u in rep["unproven"]]
    return "\n".join(L) + "\n"


def render_html(rep):
    """Self-contained HTML (no external assets). Every value is escaped; the UNPROVEN block sits directly under the
    decision so it cannot be scrolled past."""
    e = _html.escape
    txt = render_text(rep).split("\n")
    blocks, cur, title = [], [], None
    for line in txt:
        if line and not line.startswith(" ") and ": " not in line:
            if title is not None or cur:
                blocks.append((title, cur))
            title, cur = line, []
        elif line.strip():
            cur.append(line)
    blocks.append((title, cur))
    colour = {"ADOPT": "#1a7f37", "REJECT": "#b42318", "CONTINUE TESTING": "#9a6700"}[rep["decision"]]
    out = [f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Run {e(str(rep.get('run_id')))}</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:1rem auto;"
           f"padding:0 1rem}}h2{{font-size:1rem;margin:1.2rem 0 .2rem;border-bottom:1px solid #8884}}pre{{margin:0;white-space:pre-wrap}}"
           f".d{{color:{colour};font-weight:700}}.u{{border-left:4px solid #9a6700;padding-left:.7rem}}</style>",
           f"<h1>Run {e(str(rep.get('run_id')))} <span class=d>{e(rep['decision'])}</span></h1>"]
    for title, lines in blocks:
        cls = " class=u" if title and title.startswith("UNPROVEN") else ""
        out.append(f"<section{cls}>" + (f"<h2>{e(title)}</h2>" if title else "") + f"<pre>{e(chr(10).join(lines))}</pre></section>")
    return "\n".join(out) + "\n"


def write_report(rep, out_dir=None, html_dir=None):
    """Write <run_id>.json and <run_id>.txt (and <run_id>.html into html_dir, e.g. docs/reports); never overwrites."""
    rid = rep.get("run_id")
    if not rid or Path(str(rid)).name != str(rid):
        raise ValueError(f"unusable run_id {rid!r}")
    d = Path(out_dir) if out_dir else K.STATE / "reports" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    j, t = d / f"{rid}.json", d / f"{rid}.txt"
    h = Path(html_dir) / f"{rid}.html" if html_dir else None
    if j.exists() or t.exists() or (h is not None and h.exists()):
        raise FileExistsError(f"report for {rid} exists; reports are never overwritten")
    if h is not None:
        h.parent.mkdir(parents=True, exist_ok=True)
        h.write_text(render_html(rep), encoding="utf-8")
    j.write_text(json.dumps(rep, indent=1, sort_keys=True, allow_nan=False), encoding="utf-8")
    t.write_text(render_text(rep), encoding="utf-8")
    return j, t
