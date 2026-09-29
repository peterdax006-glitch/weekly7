"""Survivor-only gate (C69 ledger W-03; C69 sections 23 and 39; C66 RG04). IMPLEMENTED - NOT VALIDATED.

The price panel holds only names that still trade today, so every real-data backtest is survivorship-biased (memory:
weekly7-price-panel-is-survivor-only). C69 section 23 forbids such a result silently becoming final evidence. This gate makes
that a machine check instead of a promise:

  1. CHECKLIST ROWS. Every VALIDATED row in the four checklists whose evidence rests on the survivor-only panel (an evidence
     path under a real-panel result directory, or notes/validation_result that say the result comes from the real panel) must
     carry the literal label SURVIVOR_ONLY in its notes (or validation_result). A row re-measured on a delisted-inclusive panel
     may carry SURVIVOR_CORRECTED instead. Neither label present -> violation.
  2. REPORTS. Every report file under a real-panel result directory of state/research that presents itself as final (words such
     as FINAL, VALIDATED, PROMOTED, ACCEPTED, VERDICT) must itself mention survivor-only / survivorship, unless a labelled
     checklist row (any status) cites it as evidence (the row's label then covers it).
  3. EVIDENCE. A VALIDATED row that cites no artefact existing on disk is reported as a warning (the checklist tool refuses to
     create one, but older rows predate that rule); --strict-evidence turns the warning into a violation.

Exit codes: 0 clean, 1 violations found, 2 nothing was scanned (fail closed: an empty scan proves nothing).

  python scripts/survivorship_gate.py [--root DIR] [--no-reports] [--strict-evidence] [--json OUT]
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKLISTS = ("SELF_LEARNING_MASTER_CHECKLIST.json", "RESEARCH_BRAIN_CHECKLIST.json",
              "PREDICTION_ERROR_CHECKLIST.json", "TEN_HOUR_CHECKLIST.json")
LABELS = ("SURVIVOR_ONLY", "SURVIVOR_CORRECTED")
# result directories (under state/research) produced from the real, survivor-only price panel. Planted/synthetic-world
# directories (algorithm/planted, acceptance_mini, ...) and the delisted recovery itself are deliberately absent.
REAL_PANEL_DIRS = ("fv", "direction", "direction2", "exits_stops", "gaprisk", "heavy_algo", "learning_delta", "learners",
                   "pattern_movers", "pattern_reliability", "pattern_memory", "pattern_bank", "pattern_fail", "timeline_basis",
                   "analogs_ext", "tuning", "voltarget", "memory_adapter", "lessons", "miner_coverage", "leak_audit")
REAL_PANEL_WORDS = re.compile(r"real[- ](?:panel|data|cache|windows?)|walk-forward|price panel|real[- ]panel", re.I)
FINAL_CLAIM = re.compile(r"\b(FINAL|VALIDATED|PROMOTED|ACCEPTED|VERDICT)\b")
TEXT_LABEL = re.compile(r"survivor[- _]?only|survivorship|SURVIVOR_CORRECTED", re.I)
REPORT_SUFFIXES = (".md", ".txt", ".json")
MAX_REPORT_BYTES = 2_000_000


def _norm(p):
    return str(p).replace("\\", "/").strip()


def _real_panel_evidence(path):
    parts = _norm(path).split("/")
    return len(parts) >= 3 and parts[0] == "state" and parts[1] == "research" and parts[2] in REAL_PANEL_DIRS


def _labelled(text):
    return any(lab in (text or "") for lab in LABELS)


def row_dependence(row):
    """Why this row rests on the survivor-only panel ('' when it does not)."""
    ev = [_norm(e) for e in row.get("evidence") or []]
    hits = [e for e in ev if _real_panel_evidence(e)]
    if hits:
        return "evidence under real-panel dir: " + hits[0]
    text = " ".join(str(row.get(k) or "") for k in ("notes", "validation_result"))
    if REAL_PANEL_WORDS.search(text):
        return "notes say the result comes from the real panel"
    return ""


def scan_checklists(root, strict_evidence=False):
    """-> (violations, warnings, covered_paths, rows_scanned, validated_seen)."""
    viol, warn, covered, scanned, validated = [], [], set(), 0, 0
    for name in CHECKLISTS:
        f = Path(root) / "state" / "build" / name
        if not f.exists():
            continue
        try:
            items = json.loads(f.read_text(encoding="utf-8"))["items"]
        except (ValueError, KeyError) as e:
            viol.append({"where": name, "id": "-", "why": f"checklist unreadable: {e}"})
            continue
        for it in items:
            scanned += 1
            if _labelled(" ".join(str(it.get(k) or "") for k in ("notes", "validation_result"))):
                covered.update(_norm(e) for e in it.get("evidence") or [])   # a label on any row covers the reports it cites
            if it.get("status") != "VALIDATED":
                continue
            validated += 1
            rid = f"{name.split('_')[0]}:{it.get('id')}"
            text = " ".join(str(it.get(k) or "") for k in ("notes", "validation_result"))
            dep = row_dependence(it)
            if dep and not _labelled(text):
                viol.append({"where": name, "id": it.get("id"), "why": "VALIDATED on the survivor-only panel without a "
                             f"SURVIVOR_ONLY label ({dep})"})
            if not [e for e in it.get("evidence") or [] if (Path(root) / e).exists()]:
                (viol if strict_evidence else warn).append({"where": name, "id": it.get("id"),
                                                              "why": "VALIDATED with no evidence artefact on disk"})
    return viol, warn, covered, scanned, validated


def scan_reports(root, covered):
    """Reports in real-panel dirs that claim finality yet never mention survivorship, and no labelled row covers them."""
    viol, seen = [], 0
    base = Path(root) / "state" / "research"
    for d in REAL_PANEL_DIRS:
        for f in sorted((base / d).rglob("*")) if (base / d).is_dir() else []:
            if not f.is_file() or f.suffix.lower() not in REPORT_SUFFIXES or f.stat().st_size > MAX_REPORT_BYTES:
                continue
            seen += 1
            rel = _norm(f.relative_to(root))
            if rel in covered:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            if FINAL_CLAIM.search(text) and not TEXT_LABEL.search(text):
                viol.append({"where": rel, "id": "-", "why": "report presents a survivor-only result as final "
                             "(FINAL/VALIDATED/PROMOTED/ACCEPTED/VERDICT) with no SURVIVOR_ONLY label"})
    return viol, seen


def run(root=ROOT, reports=True, strict_evidence=False):
    root = Path(root)
    viol, warn, covered, scanned, validated = scan_checklists(root, strict_evidence)
    seen = 0
    if reports:
        rv, seen = scan_reports(root, covered)
        viol += rv
    code = 1 if viol else (2 if scanned == 0 and seen == 0 else 0)
    return {"exit_code": code, "rows_scanned": scanned, "validated_rows": validated, "reports_scanned": seen,
            "violations": viol, "warnings": warn}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--no-reports", action="store_true")
    ap.add_argument("--strict-evidence", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = run(a.root, not a.no_reports, a.strict_evidence)
    print(f"survivorship gate: {res['rows_scanned']} rows ({res['validated_rows']} VALIDATED), "
          f"{res['reports_scanned']} reports, {len(res['violations'])} violations, {len(res['warnings'])} warnings")
    for v in res["violations"]:
        print(f"  VIOLATION {v['where']} {v['id']}: {v['why']}")
    for w in res["warnings"]:
        print(f"  warning   {w['where']} {w['id']}: {w['why']}")
    if res["exit_code"] == 2:
        print("  nothing scanned: refusing to pass an empty scan")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
