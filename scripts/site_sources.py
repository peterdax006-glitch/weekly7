"""Bible Phase 29 - readers that turn state/ artefacts into plain, JSON-safe structures for the public pages.

Nothing here writes to state/. Every loader tolerates a missing or damaged source (returns an empty structure plus a
`problems` note that the page prints), so one bad file never blanks the site and never hides that it is missing.
Reads go through site_safety.guard_path: the sealed simulation tree is unreachable except cycles.json, revealed only."""
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import site_safety as S

ROOT = Path(__file__).resolve().parent.parent
CTX_NAMES = ["m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion"]       # engine.patterns.CTX order
CTX_PLAIN = {"m_vix": "fear index (VIX)", "m_vix_term": "VIX term structure", "m_spy_ma200": "market trend vs 200-day average",
             "m_breadth": "market breadth", "m_dispersion": "stock dispersion"}
STATUS_ORDER = ["active", "rescoped", "watch", "discarded", "no_gain", "duplicate", "rejected", "candidate", "failed", "cause_search"]
LIVE_STATES = {"active", "rescoped"}


def num(x):
    """float or None: NaN, inf, strings and numpy scalars become plain floats or None so json.dumps(allow_nan=False) works."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- patterns
def parse_scope(scope):
    """Miner scope is (context_index, 'low'|'mid'|'high', lo_cut, hi_cut); the bank stores a similar tuple/list or None.
    Returns (readable text, structured dict) or (None, None) for 'applies everywhere'."""
    if scope is None or (isinstance(scope, float) and not math.isfinite(scope)):
        return None, None
    if isinstance(scope, str):
        if not scope.strip() or scope.strip().lower() in ("none", "nan"):
            return None, None
        try:
            scope = ast.literal_eval(scope)
        except (ValueError, SyntaxError):
            return scope[:80], {"raw": scope[:80]}
    if isinstance(scope, dict):
        return json.dumps(scope, sort_keys=True)[:80], scope
    try:
        ci, lab, lo, hi = scope[:4]
        name = CTX_NAMES[int(ci)] if isinstance(ci, (int, float, np.integer)) and 0 <= int(ci) < len(CTX_NAMES) else str(ci)
    except (TypeError, ValueError, IndexError):
        return str(scope)[:80], {"raw": str(scope)[:80]}
    cut = {"low": f"below {num(lo):.3g}" if num(lo) is not None else "low third",
           "mid": f"between {num(lo):.3g} and {num(hi):.3g}" if num(lo) is not None and num(hi) is not None else "middle third",
           "high": f"above {num(hi):.3g}" if num(hi) is not None else "high third"}.get(str(lab), str(lab))
    return f"only when {CTX_PLAIN.get(name, name)} is {cut}", {"context": name, "band": str(lab), "lo": num(lo), "hi": num(hi)}


def plain_expr(key_named, plain=None):
    """'ev_offering q4 & overnight20 q4 unless x q0' -> readable words. Quintile q0..q4 becomes 'bottom fifth'...'top fifth'."""
    qn = {"0": "bottom fifth", "1": "second fifth", "2": "middle fifth", "3": "fourth fifth", "4": "top fifth"}
    out = []
    for part in re.split(r"( & | unless )", str(key_named)):
        if part in (" & ", " unless "):
            out.append(" and " if part == " & " else " unless ")
            continue
        f, sep, q = part.strip().rpartition(" q")
        if sep and q in qn:
            name = plain(f) if plain else f.replace("_", " ")
            out.append(f"{name} in the {qn[q]}")
        else:
            out.append(part)
    return "".join(out)


def _pattern_from_row(r, run, plain):
    scope_txt, scope_obj = parse_scope(r.get("scope"))
    dup = r.get("duplicate_of")
    return {
        "run": run, "expr": str(r.get("key_named")), "words": plain_expr(r.get("key_named"), plain),
        "status": str(r.get("status")), "effect": num(r.get("effect")),
        "m_all": num(r.get("m_all")), "t_all": num(r.get("t_all")), "n_eff": num(r.get("n_eff")),
        "m_disc": num(r.get("m_disc")), "t_disc": num(r.get("t_disc")), "m_conf": num(r.get("m_conf")), "t_conf": num(r.get("t_conf")),
        "m_recent": num(r.get("m_recent")), "t_recent": num(r.get("t_recent")), "n_recent": num(r.get("n_recent")),
        "p_real": num(r.get("p_real")), "p_halluc": num(r.get("p_hallucinated")), "p_coinc": num(r.get("p_coincidence")),
        "fdr_pass": bool(r.get("fdr_pass")) if r.get("fdr_pass") is not None else None,
        "confirmed": bool(r.get("confirmed")) if r.get("confirmed") is not None else None,
        "scope": scope_txt, "scope_obj": scope_obj, "duplicate_of": None if dup is None or (isinstance(dup, float)) else str(dup),
        "history": [], "trust": None}


LIST_CAPS = {"active": 50, "rescoped": 50, "watch": 50, "discarded": 60, "no_gain": 25, "duplicate": 15, "rejected": 40}


def _strongest(df, cap):
    """The `cap` most convincing rows of one status: highest P(real), then largest |t| over the whole history."""
    if len(df) <= cap:
        return df
    key = pd.DataFrame({"p": df["p_real"].fillna(0) if "p_real" in df else 0.0, "t": df["t_all"].abs().fillna(0) if "t_all" in df else 0.0}, index=df.index)
    return df.loc[key.sort_values(["p", "t"], ascending=False).index[:cap]]


def load_pattern_runs(algo_dir, plain=None, problems=None, caps=None):
    """One entry per state/research/algorithm/patterns_<cut>[_tag].parquet miner output. `test_<cut>[_tag].json` beside it
    supplies the run's IC and funnel. Each run keeps ALL counts, but only the `caps` strongest patterns per status are listed
    (a 4,000-row table is unreadable on a phone); the page states how many were left out. Negative statuses are capped too,
    never dropped: rejected and discarded patterns appear next to the live ones."""
    problems = problems if problems is not None else []
    caps = {**LIST_CAPS, **(caps or {})}
    runs = []
    algo_dir = Path(algo_dir)
    for f in sorted(algo_dir.glob("patterns_*.parquet")):
        S.guard_path(f)
        stem = f.stem[len("patterns_"):]
        m = re.match(r"(\d{4}-\d{2}-\d{2})(?:_(.+))?$", stem)
        cut, tag = (m.group(1), m.group(2) or "") if m else (stem, "")
        try:
            df = pd.read_parquet(f)
        except Exception as e:                              # damaged parquet: say so, keep going
            problems.append(f"{f.name}: unreadable ({type(e).__name__})")
            continue
        if "status" not in df:
            problems.append(f"{f.name}: no status column")
            continue
        test = S.safe_read_json(algo_dir / f"test_{stem}.json", {}) or {}
        counts = df["status"].value_counts().to_dict()
        keep = pd.concat([_strongest(g, caps.get(st, 25)) for st, g in df.groupby("status")]) if len(df) else df
        pats = [_pattern_from_row(r, stem, plain) for r in keep.to_dict("records")]
        runs.append({"run": stem, "cut": cut, "tag": tag, "counts": {k: int(v) for k, v in counts.items()}, "tested": int(len(df)),
                     "test": {k: (num(v) if not isinstance(v, (dict, list)) else None) for k, v in test.items()
                              if k in ("ic_mean", "ic_t", "top_decile_excess_week", "weeks", "tested", "fdr_pass", "gate_corr_confirm",
                                       "null_patterns", "null_t_95pct", "real_t_95pct")},
                     "listed": len(pats), "patterns": pats})
    return runs


def load_bank(root, plain=None, problems=None):
    """Long-term bank (engine.pattern_bank) if one exists at `root`: adds the dated lifecycle history and trust per pattern.
    Read-only: the directory is never created, and a corrupt head is reported, not repaired."""
    problems = problems if problems is not None else []
    root = Path(root)
    if not root.is_dir() or not any(root.glob("bank_v*.json")):
        return {"available": False, "version": 0, "patterns": [], "verify": None}
    from engine.pattern_bank import PatternBank, summarize
    bank = PatternBank(root)
    ver, payload, digest = bank.head()
    vr = bank.verify()
    if bank.corrupt:
        problems.append(f"pattern bank: {len(bank.corrupt)} newer version(s) failed verification and were skipped")
    out = []
    for pid, rec in (payload.get("records") or {}).items():
        wins = rec.get("windows") or []
        last = max((w["window_end"] for w in wins), default=rec.get("changed") or rec.get("first_seen"))
        try:
            summ = summarize(rec, last)
        except Exception:
            summ = {}
        scope_txt, scope_obj = parse_scope(rec.get("scope"))
        hist = [{"as_of": h.get("as_of"), "kind": h.get("kind"), "frm": h.get("frm"), "to": h.get("to"),
                 "reason": str(h.get("reason"))[:240]} for h in rec.get("history", [])]
        disc = rec.get("discovery") or {}
        out.append({"id": pid, "run": "bank", "expr": str(rec.get("name")), "words": plain_expr(rec.get("name"), plain),
                    "status": str(rec.get("state")), "effect": num(rec.get("effect")), "scope": scope_txt, "scope_obj": scope_obj,
                    "p_real": num(disc.get("p_real")), "t_disc": num(disc.get("t_disc")), "t_conf": num(disc.get("t_conf")),
                    "n_eff": num(disc.get("n_eff")), "m_all": num(summ.get("pooled_effect")), "t_all": num(summ.get("pooled_t")),
                    "m_recent": num((rec.get("last_validation") or {}).get("m_recent")),
                    "t_recent": num((rec.get("last_validation") or {}).get("t_recent")),
                    "n_recent": num((rec.get("last_validation") or {}).get("n_recent")),
                    "trust": num(summ.get("trust")), "n_windows": summ.get("n_windows"), "last_validation": summ.get("last_validation"),
                    "discard_reason": rec.get("discard_reason"), "history": hist, "windows": [
                        {"end": w.get("window_end"), "t": num(w.get("t")), "m": num(w.get("m")), "verdict": w.get("verdict")} for w in wins][-40:]})
    return {"available": True, "version": ver, "digest": digest, "patterns": out, "verify": {k: vr[k] for k in ("ok", "versions")}}


# ---------------------------------------------------------------- sensitivity
def _mean_ci(x):
    x = np.array([v for v in x if num(v) is not None], float)
    if len(x) < 3:
        return None, None, None, len(x)
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se, len(x)


def _shown(v):
    """A tested value of None means the setting was switched off; say so instead of printing 'None'."""
    return "off" if v is None else v


def build_sensitivity(sens, problems=None):
    """Per knob: every tested value with its effect (change in mean weekly return versus the champion), a 95% interval from
    the per-year effects, stability (share of years the effect had the same sign), the selected value and the reason.
    A Bonferroni line marks how large |t| must be before a single tested value means anything given how many were tried."""
    problems = problems if problems is not None else []
    if not sens or "variants" not in sens:
        problems.append("sensitivity results not found")
        return {"knobs": [], "n_variants": 0, "bonferroni_t": None, "champion": {}, "champion_mean_week": None, "years": 0}
    champ = sens.get("champion") or {}
    ew = champ.get("ew") or {}
    variants = sens["variants"]
    n = len(variants)
    # two-sided normal critical value for alpha = 0.05 / n via bisection on the erfc tail (no scipy needed)
    alpha = 0.05 / max(n, 1)
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    crit = hi
    by = {}
    for v in variants:
        by.setdefault((v.get("group"), v.get("knob")), []).append(v)
    ranking = {(k.get("group"), k.get("knob")): k for k in sens.get("by_knob", [])}
    knobs = []
    for (group, knob), vs in by.items():
        selected = champ.get(knob) if group == "knob" else (ew.get(knob) if group == "indicator" else "traded as champion")
        vals = []
        for v in vs:
            m, lo_, hi_, ny = _mean_ci(v.get("per_year") or [])
            d = num(v.get("d_mean_week"))
            vals.append({"value": _shown(v.get("value")), "raw": v.get("value"), "effect": d, "lo": lo_, "hi": hi_, "n_years": ny, "t": num(v.get("t")),
                         "same_way": num(v.get("share_years_same_way")), "class": v.get("size"), "consistency": v.get("consistency"),
                         "ge7_per_year": num(v.get("d_weeks_ge7_per_year")), "d_worst_dd": num(v.get("d_worst_dd")),
                         "per_year": [num(x) for x in (v.get("per_year") or [])],
                         "is_selected": v.get("value") == selected and group == "knob",
                         "beats_multiple_testing": bool(num(v.get("t")) is not None and abs(v["t"]) >= crit)})
        best = max((x for x in vals if x["effect"] is not None), key=lambda x: x["effect"], default=None)
        rk = ranking.get((group, knob), {})
        real = [x for x in vals if x["beats_multiple_testing"] and (x["effect"] or 0) > 0]
        if best is None:
            reason = "no usable results"
        elif real:
            reason = (f"a tested value beats the champion after multiple-testing correction "
                      f"(t {real[0]['t']:+.2f} vs needed {crit:.2f}); a candidate for the next proposal, not yet adopted")
        elif best["effect"] <= 0:
            reason = "every tested alternative did the same or worse than the champion; the champion value stays"
        else:
            reason = (f"the best alternative gained {best['effect']:+.4f}/week (t {best['t'] if best['t'] is not None else float('nan'):+.2f}) "
                      f"but that is below the multiple-testing bar of {crit:.2f}, so it is not distinguishable from luck; the champion value stays")
        knobs.append({"group": group, "knob": str(knob), "selected": _shown(selected), "values": vals, "range": num(rk.get("range_of_effect")),
                      "class": rk.get("class"), "best_value": "—" if best is None else best["value"], "reason": reason,
                      "negative": sum(1 for x in vals if (x["effect"] or 0) < 0), "positive": sum(1 for x in vals if (x["effect"] or 0) > 0)})
    knobs.sort(key=lambda k: -(k["range"] or 0))
    return {"knobs": knobs, "n_variants": n, "bonferroni_t": crit, "champion": champ, "champion_mean_week": num(sens.get("champion_mean_week")),
            "years": int(sens.get("hidden_years") or 0)}


# ---------------------------------------------------------------- checklist
ITEM = re.compile(r"^- \[(?P<s>[ ~?x!])\]\s+(?P<id>[A-Z]+\d+[a-z]?(?:\.\d+)?)\s+(?P<rest>.*)$")
STATE_NAME = {" ": "not started", "~": "implemented / testing", "?": "unproven", "x": "validated", "!": "failed"}


def parse_checklist(text):
    """Parse state/CHECKLIST.md: '## Phase' headings, '- [s] ID title (details). Evidence: ...' items. Lines that do not
    parse are counted in `unparsed` so a format change cannot silently drop items."""
    phases, cur, unparsed = [], None, 0
    for line in (text or "").splitlines():
        if line.startswith("## "):
            cur = {"name": line[3:].strip(), "items": []}
            phases.append(cur)
            continue
        m = ITEM.match(line.strip())
        if m:
            if cur is None:
                cur = {"name": "Unsorted", "items": []}
                phases.append(cur)
            rest = m.group("rest")
            ev = ""
            if "Evidence:" in rest:
                rest, ev = rest.split("Evidence:", 1)
            title, _, detail = rest.strip().partition(" (")
            cur["items"].append({"id": m.group("id"), "state": m.group("s"), "state_name": STATE_NAME[m.group("s")],
                                 "title": title.strip().rstrip("."), "detail": detail.rstrip(")").strip(), "evidence": ev.strip(),
                                 "raw": rest.strip()})
        elif line.strip().startswith("- ["):
            unparsed += 1
    counts = {}
    for p in phases:
        for it in p["items"]:
            counts[it["state"]] = counts.get(it["state"], 0) + 1
    return {"phases": phases, "counts": counts, "total": sum(counts.values()), "unparsed": unparsed}


def build_unproven(checklist, registry_events):
    """Everything not yet validated, worst first, plus the honest negatives from the registry (rollbacks, voided rounds)."""
    rank = {"!": 0, "?": 1, "~": 2, " ": 3}
    items = []
    for p in checklist["phases"]:
        for it in p["items"]:
            if it["state"] != "x":
                items.append({**it, "phase": p["name"].split(":")[0]})
    items.sort(key=lambda i: (rank.get(i["state"], 4), i["id"]))
    negatives = [{"t": e.get("t"), "event": e.get("event"), "text": str(e.get("reason") or e.get("note") or "")[:300], "id": e.get("id")}
                 for e in registry_events if e.get("event") in ("rolled_back", "voided_adjustment")]
    return {"items": items, "negatives": negatives}


# ---------------------------------------------------------------- registry
BIG_KEYS = {"rows", "calibration", "configs", "champion", "best", "recovery", "per_year", "diagnosis"}
PROV_KEYS = ("git_commit", "canon_sha", "bible_sha", "blueprint_version", "config_hash", "data_snapshot", "seed")


def load_registry(path, problems=None):
    """state/experiments.jsonl -> list of records. A line that is not JSON is counted and reported, never dropped silently."""
    problems = problems if problems is not None else []
    p = Path(path)
    if not p.exists():
        problems.append("experiment registry not found")
        return []
    out, bad = [], 0
    for i, line in enumerate(S.safe_read_text(p).splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(rec, dict):
            rec["_line"] = i + 1
            out.append(S.scrub(rec))
    if bad:
        problems.append(f"experiment registry: {bad} line(s) are not valid JSON")
    return out


def registry_view(rec):
    """Scalar headline fields for the table plus a size-capped JSON for the expandable detail."""
    head = {k: v for k, v in rec.items() if not isinstance(v, (dict, list)) and k not in ("t", "event", "experiment_id", "_line") and k not in PROV_KEYS}
    nested = {k: (f"{len(v)} entries" if isinstance(v, (dict, list)) else v) for k, v in rec.items() if k in BIG_KEYS}
    detail = json.dumps({k: v for k, v in rec.items() if k != "_line"}, default=str, indent=1)
    if len(detail) > 2500:
        detail = detail[:2500] + "\n... (truncated; full record in state/experiments.jsonl)"
    prov = {k: rec.get(k) for k in PROV_KEYS if rec.get(k) not in (None, "None")}
    return {"headline": head, "nested": nested, "detail": detail, "prov": prov}


def provenance_coverage(records):
    """Share of registry records carrying each provenance field. Early records predate provenance; the page shows that."""
    n = len(records) or 1
    return {k: sum(1 for r in records if r.get(k) not in (None, "None", "")) / n for k in PROV_KEYS}


# ---------------------------------------------------------------- runs
def load_cycles(path, problems=None):
    problems = problems if problems is not None else []
    data = S.safe_read_json(path, None)
    if not data:
        problems.append("simulation cycle list not found")
        return {"cycles": [], "rounds": [], "hidden": 0}
    allc = data.get("cycles") or []
    pub = S.public_cycles(allc)
    rounds = []
    for r in data.get("rounds") or []:
        rounds.append({"round": r.get("round"), "adopted": r.get("adopted"), "voided": r.get("voided"),
                       "before": num(r.get("avg_week_all_years_before")), "after": num(r.get("avg_week_all_years_after"))})
    return {"cycles": pub, "rounds": rounds, "hidden": len(allc) - len(pub)}


def era_of(year):
    m = re.search(r"(\d{4})", str(year))
    return f"{int(m.group(1)) // 10 * 10}s" if m else "unknown"


def era_breakdown(cycles):
    """Per decade: number of windows, mean weekly return, share of windows that beat the market, weeks >= +7% per window."""
    eras = {}
    for c in cycles:
        eras.setdefault(era_of(c["year"]), []).append(c)
    out = []
    for e, cs in sorted(eras.items()):
        mw = [num(c.get("mean_week")) for c in cs if num(c.get("mean_week")) is not None]
        beat = [1.0 if (num(c.get("year_return")) or 0) > (num(c.get("market_year_return")) or 0) else 0.0 for c in cs
                if num(c.get("year_return")) is not None and num(c.get("market_year_return")) is not None]
        out.append({"era": e, "windows": len(cs), "mean_week": float(np.mean(mw)) if mw else None,
                    "beat_market": float(np.mean(beat)) if beat else None,
                    "ge7": float(np.mean([num(c.get("weeks_ge_7")) or 0 for c in cs])), "worst_dd": min((num(c.get("max_dd")) or 0 for c in cs), default=None)})
    return out


def list_reports(dirs):
    """File-level index of report artefacts: name, kind, size, modified time, and headline scalars for small JSON files."""
    out = []
    for label, d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_dir() or f.suffix not in (".json", ".csv", ".md", ".txt", ".parquet"):
                continue
            try:
                S.guard_path(f)
            except S.Leak:
                continue
            head = {}
            if f.suffix == ".json" and f.stat().st_size < 400_000:
                js = S.safe_read_json(f, {})
                if isinstance(js, dict):
                    head = {k: v for k, v in js.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and num(v) is not None}
                    head = dict(list(head.items())[:6])
            out.append({"area": label, "name": f.name, "kind": f.suffix[1:], "bytes": f.stat().st_size, "mtime": f.stat().st_mtime, "head": head})
    return out
