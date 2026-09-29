"""Bible Phase 29/39 (dashboard) + Phase 42 (claim standard): build docs/dashboard.html, the one-page status board.

    python scripts/dashboard_build.py                 # write docs/dashboard.html
    python scripts/dashboard_build.py --dry-run       # build and audit in memory, write nothing
    python scripts/dashboard_build.py --check         # exit 1 when docs/dashboard.html differs from what the sources give
    python scripts/dashboard_build.py --built-at 2026-09-28T12:00:00Z    # fixed stamp: byte-identical output

What it shows, each from a real artefact and each labelled with the file it came from: foundation status, Bible trace
counts and the biggest gaps, the checklist, the final-report audits, the latest research numbers, the experiment
memory's coverage, and the running jobs with their age and health. A source that is missing or unreadable is shown as
"not available" - never as a zero or a pass.

Safety (canon C19/C11, reusing scripts/site_safety.py): every file is read through the path guard (sealed windows and the
data cache are unreachable), livesim rows never enter the page except as a count, and the finished HTML goes through
the secret / sealed-date audit BEFORE anything is written: a leak raises, exit code 2, nothing on disk changes.
Deterministic given the sources, `--built-at` and `--now`; the only clock read is the default for those two."""
import argparse
import hashlib
import html as _html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import site_safety as S
from engine import resources as R
from engine import experiment_memory as EM

OUT_REL = Path("docs") / "dashboard.html"
MANIFEST_REL = Path("state") / "research" / "dashboard" / "manifest.json"
STATE_NAMES = {"[x]": "validated", "[~]": "implemented / testing", "[?]": "unproven", "[!]": "failed", "[ ]": "not started"}
STATE_ORDER = ["[x]", "[~]", "[?]", "[!]", "[ ]"]
CHECK_STATES = {"x": "validated", "~": "implemented / testing", "?": "unproven", "!": "failed", " ": "not started"}
NAV = [("./", "Dashboard"), ("explorer.html", "Pattern Explorer"), ("sensitivity2.html", "Sensitivity"), ("runs.html", "Runs and registry"),
       ("checklist.html", "Checklist"), ("dashboard.html", "Status board")]
CAVEAT = "The price panel is survivor-only, so every historical return here is an upper bound. Paper trading only; not investment advice."


# ------------------------------------------------------------------------------------------------ source loading
class Sources:
    """Reads state files through the path guard, remembers what it read (for the manifest) and what it could not."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.used, self.problems = {}, []

    def _p(self, rel):
        return self.root / rel

    def text(self, rel):
        p = self._p(rel)
        try:
            t = S.safe_read_text(p)
        except S.SealedAccess:
            raise
        except OSError as e:
            self.problems.append(f"{rel}: not available ({type(e).__name__})")
            return None
        self.used[str(rel).replace("\\", "/")] = hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()[:16]
        return t

    def json(self, rel):
        t = self.text(rel)
        if t is None:
            return None
        try:
            return json.loads(t)
        except ValueError:
            self.problems.append(f"{rel}: not valid JSON")
            return None


def parse_foundation(md):
    """Rows of the FOUNDATION.md table: unit, phases, range, lines, status, tests."""
    rows = []
    for line in (md or "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0] not in ("unit", "---") and not set(cells[0]) <= {"-", " "}:
            rows.append(dict(zip(("unit", "phases", "range", "lines", "status", "tests"), cells)))
    return rows


def parse_checklist(md):
    """{'counts': {state: n}, 'items': [{id, state, text, section}]} from state/CHECKLIST.md. Nested sub-bullets are not
    items; only top-level '- [x] ID text' lines are."""
    items, section = [], ""
    for line in (md or "").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = re.match(r"^- \[([x~?! ])\] (\S+)\s*(.*)$", line)
        if m:
            items.append({"state": CHECK_STATES[m.group(1)], "id": m.group(2), "text": m.group(3).strip(), "section": section})
    counts = {}
    for it in items:
        counts[it["state"]] = counts.get(it["state"], 0) + 1
    return {"counts": counts, "items": items}


def collect(src, now, info=R.process_info, free_fn=None):
    """Every dashboard input as plain data. `now` (epoch seconds) is used for job ages only."""
    d = {"problems": src.problems}
    d["foundation"] = parse_foundation(src.text("state/build/FOUNDATION.md"))
    d["trace"] = src.json("state/build/bible_trace.json")
    d["checklist"] = parse_checklist(src.text("state/CHECKLIST.md"))
    d["final"] = src.json("state/reports/final_report.json")
    d["backtest"] = src.json("state/research/backtest_summary.json")
    rows, bad = EM.load_experiments(src._p("state/experiments.jsonl")) if (src._p("state/experiments.jsonl")).exists() else ([], 0)
    if rows:
        src.used["state/experiments.jsonl"] = hashlib.sha256(json.dumps(len(rows)).encode()).hexdigest()[:16]
    else:
        src.problems.append("state/experiments.jsonl: not available")
    d["experiments"] = {"rows": rows, "unparseable": bad}
    reg = R.ProcRegistry(src._p("state/procs.json"))
    d["procs_file_exists"] = reg.path.exists()
    d["jobs"] = R.worker_health(reg, now, free_fn=free_fn, info=info)
    d["heartbeat"] = src.json("state/heartbeat.json")
    return d


# ------------------------------------------------------------------------------------------------ formatting helpers
e = _html.escape


def pct(v, nd=1):
    return "not available" if v is None else f"{v * 100:.{nd}f}%"


def num(v, nd=2):
    return "not available" if v is None else f"{v:,.{nd}f}"


def age(s):
    if s is None:
        return "n/a"
    s = max(0, int(s))
    return f"{s // 86400}d {s % 86400 // 3600}h" if s >= 86400 else f"{s // 3600}h {s % 3600 // 60}m" if s >= 3600 else f"{s // 60}m {s % 60}s"


def badge(text, kind=""):
    return f'<span class="badge {kind}">{e(str(text))}</span>'


def verdict_badge(v):
    v = str(v)
    up = v.upper()
    return badge(v, "b-good" if up in ("PASS", "TRUE", "VALIDATED", "IN RANGE", "OK") else "b-bad" if up in ("FAIL", "FALSE", "FAILED", "BELOW RANGE") else "b-warn")


def table(headers, rows, cls=""):
    """rows: lists of already-escaped cell strings."""
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def tile(k, v, note=""):
    return f'<div class="tile"><div class="k">{e(k)}</div><div class="v">{e(str(v))}</div><div class="note">{e(note)}</div></div>'


def src_note(*paths):
    return '<p class="note">Source: ' + ", ".join(f'<span class="mono">{e(p)}</span>' for p in paths) + "</p>"


def stack_bar(counts, order=STATE_ORDER, names=STATE_NAMES):
    tot = sum(counts.get(k, 0) for k in order) or 1
    col = {"[x]": "var(--good)", "[~]": "var(--s1)", "[?]": "var(--warn)", "[!]": "var(--bad)", "[ ]": "var(--muted)"}
    segs = "".join(f'<i style="width:{counts.get(k, 0) / tot * 100:.2f}%;background:{col[k]}" title="{e(names[k])}: {counts.get(k, 0)}"></i>'
                   for k in order if counts.get(k, 0))
    return f'<div class="bar" role="img" aria-label="{e(", ".join(f"{names[k]} {counts.get(k, 0)}" for k in order))}">{segs}</div>'


def section(title, sub, body, anchor):
    return f'<section class="card" id="{anchor}"><h2>{e(title)}</h2><div class="sub">{e(sub)}</div>{body}</section>'


def na(msg):
    return f'<p class="muted">Not available: {e(msg)}</p>'


# ------------------------------------------------------------------------------------------------ sections
def s_foundation(d):
    rows = d["foundation"]
    if not rows:
        return section("Foundation status", "Line counts per work unit against the Bible ranges.", na("state/build/FOUNDATION.md has no table"), "foundation")
    inr = sum(1 for r in rows if r["status"] == "IN RANGE")
    below = [r for r in rows if r["status"] == "BELOW RANGE"]
    body = f'<p>{inr} of {len(rows)} units are in their Bible line range' + (f"; below range: {e(', '.join(r['unit'] for r in below))}." if below else ".") + "</p>"
    body += table(["Unit", "Phases", "Range", "Lines", "Status", "Tests"],
                  [[e(r["unit"]), e(r["phases"]), e(r["range"]), e(r["lines"]), verdict_badge(r["status"]), e(r["tests"])] for r in rows])
    return section("Foundation status", "One row per work unit: lines against the sum of the Bible ranges it covers. A range is a requirement, not a target to pad to.",
                   body + src_note("state/build/FOUNDATION.md"), "foundation")


def s_trace(d):
    t = d["trace"]
    if not t:
        return section("Bible trace", "Every Bible requirement linked to code, tests and evidence.", na("state/build/bible_trace.json"), "trace")
    counts = t["summary"]["counts"]
    total = sum(counts.values())
    body = stack_bar(counts)
    body += '<div class="tiles">' + "".join(tile(STATE_NAMES[k], counts.get(k, 0), pct(counts.get(k, 0) / total if total else None)) for k in STATE_ORDER) + "</div>"
    idx = t.get("index") or {}
    body += (f'<p class="note">{total} requirements traced across {idx.get("modules", "?")} modules, {idx.get("test_files", "?")} test files and '
             f'{idx.get("evidence_files", "?")} evidence files. "Validated" means traceable to code, a test that names it and passing evidence; '
             f"the linking is a term-overlap heuristic, not a proof of meaning.</p>")
    pp = t["summary"].get("per_phase") or {}
    rows = []
    for ph, c in pp.items():
        cc = c.get("counts", c) if isinstance(c, dict) else {}
        n = sum(v for v in cc.values() if isinstance(v, int))
        if n:
            rows.append((int(ph) if str(ph).isdigit() else 999, str(ph), cc, n))
    rows.sort(key=lambda r: (-(r[2].get("[!]", 0) + r[2].get("[ ]", 0)) / r[3], r[0]))
    if rows:
        body += "<h3>Phases with the largest failed or unstarted share</h3>" + table(
            ["Phase", "Requirements", "Validated", "Failed", "Not started", "Share validated"],
            [[e(r[1]), str(r[3]), str(r[2].get("[x]", 0)), str(r[2].get("[!]", 0)), str(r[2].get("[ ]", 0)), pct(r[2].get("[x]", 0) / r[3])] for r in rows[:10]])
    tm = t.get("top_missing") or []
    if tm:
        body += "<h3>Highest-importance requirements still missing</h3>" + table(
            ["Id", "Phase", "Requirement"], [[e(str(x.get("id"))), str(x.get("phase")), e(str(x.get("text")))] for x in tm[:10]])
    fails = t.get("failing_tests_cache") or {}
    if fails:
        body += f"<h3>Failing tests on record ({sum(len(v) for v in fails.values())})</h3>" + table(
            ["File", "Failing tests"], [[e(f), e(", ".join(v))] for f, v in sorted(fails.items())])
    pend = t.get("integration_pending") or {}
    if pend:
        body += f'<p class="note">{sum(len(v) for v in pend.values())} integration hooks are still pending across {len(pend)} modules.</p>'
    return section("Bible trace", f"State of all {total} requirements, generated {t.get('generated', 'unknown')}.", body + src_note("state/build/bible_trace.json"), "trace")


def s_checklist(d):
    ck = d["checklist"]
    if not ck["items"]:
        return section("Master checklist", "Phase 37/38 checklist.", na("state/CHECKLIST.md has no items"), "checklist")
    c = ck["counts"]
    order = ["validated", "implemented / testing", "unproven", "failed", "not started"]
    body = '<div class="tiles">' + "".join(tile(k.capitalize(), c.get(k, 0)) for k in order) + "</div>"
    hot = [i for i in ck["items"] if i["state"] in ("failed", "not started")]
    body += "<h3>Failed or not started</h3>" + table(["Id", "State", "Section", "Item"], [
        [e(i["id"]), verdict_badge("FAILED") if i["state"] == "failed" else badge("not started", "b-warn"),
         e(i["section"]), e(i["text"][:220])] for i in hot])
    unp = [i for i in ck["items"] if i["state"] == "unproven"]
    if unp:
        body += "<h3>Unproven</h3>" + table(["Id", "Item"], [[e(i["id"]), e(i["text"][:220])] for i in unp])
    return section("Master checklist", f"{len(ck['items'])} top-level items. Nothing is validated because code exists.", body + src_note("state/CHECKLIST.md"), "checklist")


def s_audits(d):
    f = d["final"]
    if not f:
        return section("Final-report audits", "Leakage and reproducibility, assembled from artefacts.", na("state/reports/final_report.json"), "audits")
    body = ""
    for key, title in (("leakage", "Leakage audit"), ("reproducibility", "Reproducibility audit")):
        a = f.get(key) or {}
        body += f"<h3>{e(title)} {verdict_badge(a.get('verdict', 'NO EVIDENCE'))}</h3>"
        body += table(["Check", "Result"], [[e(k.replace("_", " ")), verdict_badge(v)] for k, v in (a.get("checks") or {}).items()]) if a.get("checks") else na("no checks recorded")
    v = f.get("validation") or {}
    rows = []
    q = v.get("quality_gate") or {}
    if q:
        rows.append(["Quality gate", verdict_badge("PASS" if q.get("exit_code") == 0 else "FAIL"), f"{q.get('failing', '?')} failing"])
    pc = v.get("planted_calibration") or {}
    if pc:
        rows.append(["Planted-pattern calibration", verdict_badge("PASS" if pc.get("validated") else "FAIL"),
                     "failed: " + ", ".join(pc.get("failed") or []) if pc.get("failed") else "all scenarios pass"])
    pa = v.get("parity") or {}
    if pa:
        rows.append(["Parity", verdict_badge("PASS" if pa.get("passed") else "FAIL"), ", ".join(pa.get("failed") or []) or "no failures"])
    if rows:
        body += "<h3>Validation gates</h3>" + table(["Gate", "Result", "Detail"], [[e(a), b, e(c)] for a, b, c in rows])
    cc = (f.get("checklist") or {}).get("counts")
    if cc:
        body += f'<p class="note">Checklist at report time: {e(", ".join(f"{k} {v}" for k, v in cc.items()))}.</p>'
    return section("Final-report audits", "An audit passes only when every one of its checks has evidence and passed.", body + src_note("state/reports/final_report.json"), "audits")


def s_research(d):
    body, srcs = "", []
    f, b = d["final"], d["backtest"]
    perf = ((f or {}).get("performance") or {}).get("weekly")
    if perf:
        body += "<h3>Weekly performance over the 39 replayed windows</h3>"
        body += '<div class="tiles">' + tile("Mean week", pct(perf.get("mean"), 2), "target 7%") + tile("Median week", pct(perf.get("median"), 2)) \
            + tile("Weeks in 5-10% band", pct(perf.get("share_in_band_5_10"))) + tile("Positive weeks", pct(perf.get("share_positive"))) \
            + tile("Worst week", pct(perf.get("worst"))) + tile("Weeks", num(perf.get("weeks"), 0)) + "</div>"
        body += '<p class="note">Definition: simple weekly return of the champion configuration, all revealed windows pooled, gross of nothing not stated in the source file. The 7% target is reported against, never redefined.</p>'
        srcs.append("state/reports/final_report.json")
    if b and b.get("weekly7"):
        w = b["weekly7"]
        body += "<h3>Backtest summary</h3>" + table(["Variant", "Mean week", "Weeks >= 7%", "Weeks <= -7%", "Positive weeks", "Worst week", "Max drawdown"],
                                                   [[e(k), pct(v.get("mean_week"), 2), pct(v.get("pct_ge_7")), pct(v.get("pct_le_m7")), pct(v.get("win_weeks")),
                                                     pct(v.get("worst_week")), pct(v.get("max_dd"))] for k, v in b.items() if isinstance(v, dict)])
        srcs.append("state/research/backtest_summary.json")
    rows = [r for r in d["experiments"]["rows"] if r.get("event") != "livesim_cycle"]
    n_live = len(d["experiments"]["rows"]) - len(rows)
    if rows:
        last = sorted(rows, key=lambda r: str(r.get("t") or r.get("timestamp") or ""))[-10:][::-1]
        body += "<h3>Latest experiments</h3>" + table(["When", "Event", "Outcome", "Reason / headline"], [
            [e(str(r.get("t") or r.get("timestamp") or "")[:19]), e(str(r.get("event"))), verdict_badge(r["outcome"].upper()) if r.get("outcome") else badge("no outcome", "b-warn"),
             e(str(r.get("reason") or _headline(r))[:200])] for r in last])
        srcs.append("state/experiments.jsonl")
    if n_live:
        body += f'<p class="note">{n_live} blind-window cycle records exist in the registry; they are counted here and never shown row by row, because their windows may still be sealed.</p>'
    if not body:
        body = na("no research artefacts found")
    return section("Latest research numbers", "Each figure names the file it came from. " + CAVEAT, body + src_note(*srcs) if srcs else body, "research")


def _headline(r):
    m = r.get("metrics")
    if isinstance(m, dict) and m:
        k, v = next(iter(m.items()))
        return f"{k} {v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k} {v}"
    return ""


def s_memory(d):
    rows = d["experiments"]["rows"]
    if not rows:
        return section("Experiment memory", "How much of the registry can answer the Phase 30 questions.", na("state/experiments.jsonl"), "memory")
    cov = EM.answer_coverage(rows)
    n = cov["n"]
    rej = [r for r in rows if r.get("outcome") == "reject"]
    body = '<div class="tiles">' + tile("Experiments logged", n) + tile("With train/test ranges", cov["with_ranges"], pct(cov["with_ranges"] / n)) \
        + tile("Unseen data verifiably clean", cov["with_clean_unseen"], pct(cov["with_clean_unseen"] / n)) \
        + tile("With metrics", cov["with_metrics"], pct(cov["with_metrics"] / n)) + tile("With an outcome", cov["with_outcome"], pct(cov["with_outcome"] / n)) \
        + tile("Without an id", cov["without_experiment_id"], "orphans") + "</div>"
    body += f'<p class="note">{len(rej)} rejected experiments are kept and never hidden. {d["experiments"]["unparseable"]} unreadable lines in the log.</p>'
    if rej:
        body += table(["Experiment", "Event", "Reason"], [[e(str(r.get("experiment_id") or "-")), e(str(r.get("event"))), e(str(r.get("reason") or "no reason recorded"))] for r in rej[:15]])
    return section("Experiment memory", "Phase 30: an experiment that cannot answer what was unseen, what worsened and whether it survived is an orphan.", body + src_note("state/experiments.jsonl"), "memory")


def s_jobs(d, now):
    j = d["jobs"]
    free, rec = j["free_gb"], j["recommended"]
    body = '<div class="tiles">' + tile("Free memory", "not available" if free is None else f"{free:.1f} GB", f"of {j['total_gb']:.1f} GB" if j.get("total_gb") else "") \
        + tile("Recommended workers", rec["workers"], rec["limit"]) + tile("Registered running jobs", j["n_running"]) \
        + tile("Stale findings", len(j["stale"]["findings"]), "healthy" if j["stale"]["healthy"] else "needs attention") + "</div>"
    if j["running"]:
        body += table(["Job", "PID", "Age", "Last heartbeat", "RSS", "Estimate", "Alive"], [
            [e(r["job"]), str(r["pid"]), age(r["age_s"]), age(r["heartbeat_age_s"]) + " ago", "n/a" if r["rss_gb"] is None else f"{r['rss_gb']:.2f} GB",
             "n/a" if r["est_gb"] is None else f"{r['est_gb']:.1f} GB", verdict_badge("OK" if r["alive"] else "FAIL")] for r in j["running"]])
    else:
        body += '<p class="muted">No jobs are registered as running in state/procs.json.</p>' if d["procs_file_exists"] else \
            '<p class="muted">state/procs.json does not exist yet: no job has been started through the resource manager.</p>'
    if j["stale"]["findings"]:
        body += "<h3>Stale or duplicate processes</h3>" + table(["Job", "Kind", "Detail"], [[e(f["job"]), badge(f["kind"], "b-bad"), e(f["detail"])] for f in j["stale"]["findings"]])
    hb = d.get("heartbeat") or {}
    hj = hb.get("jobs") if isinstance(hb, dict) else None
    if hj:
        body += "<h3>Heartbeat file jobs</h3>" + table(["Entry"], [[e(json.dumps(x, sort_keys=True)[:200])] for x in hj[:20]])
    elif isinstance(hb, dict) and "jobs" in hb:
        body += f'<p class="note">Heartbeat file (last written {e(str(hb.get("t")))}) lists no jobs.</p>'
    body += '<p class="note">Jobs are stopped only by their own recorded PID (never by image name). Memory decides how many run: 3-7 workers, fewer when memory says so.</p>'
    return section("Running jobs", "Phase 44: age, heartbeat and memory of every registered job.", body + src_note("state/procs.json", "state/heartbeat.json"), "jobs")


def s_problems(d):
    if not d["problems"]:
        return ""
    return section("Sources that were not available", "Shown so a gap is never mistaken for good news.",
                   "<ul>" + "".join(f"<li>{e(p)}</li>" for p in d["problems"]) + "</ul>", "problems")


# ------------------------------------------------------------------------------------------------ page
CSS = """:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--good:#006300;--bad:#d03b3b;--warn:#a66300;--wash:#e9f2fd;--goodw:#e6f3e6;--badw:#fbeaea;--warnw:#fbf1dc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--s1:#3987e5;--good:#0ca30c;--bad:#e66767;--warn:#e0a030;--wash:#16263a;--goodw:#12261a;--badw:#2e1717;--warnw:#2b2312}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--s1:#3987e5;--good:#0ca30c;--bad:#e66767;--warn:#e0a030;--wash:#16263a;--goodw:#12261a;--badw:#2e1717;--warnw:#2b2312}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:20px 16px 64px}a{color:var(--s1)}nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin-bottom:14px}nav a.on{font-weight:650;color:var(--ink);text-decoration:none}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}h2{font-size:17px;margin:0 0 4px}h3{font-size:14px;margin:16px 0 4px}.sub{color:var(--ink-2);font-size:13px}.note{font-size:12px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px;margin:14px 0}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--page);border:1px solid var(--ring);border-radius:12px;padding:10px 12px}.tile .k{color:var(--ink-2);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.tile .v{font-size:21px;font-weight:650}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}td{padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}.badge{display:inline-block;border-radius:6px;padding:1px 7px;font-size:12px;background:var(--wash);white-space:nowrap}.b-good{background:var(--goodw);color:var(--good)}.b-bad{background:var(--badw);color:var(--bad)}.b-warn{background:var(--warnw);color:var(--warn)}
.muted{color:var(--muted)}.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}.bar{height:12px;border-radius:6px;background:var(--grid);overflow:hidden;display:flex;margin:8px 0}.bar i{display:block;height:100%}"""


def headline_tiles(d):
    t = d["trace"]
    fo = d["foundation"]
    ck = d["checklist"]["counts"]
    f = d["final"] or {}
    audits = [(f.get(k) or {}).get("verdict") for k in ("leakage", "reproducibility")]
    tiles = [tile("Foundation units in range", f"{sum(1 for r in fo if r['status'] == 'IN RANGE')}/{len(fo)}" if fo else "n/a"),
             tile("Requirements validated", f"{t['summary']['counts'].get('[x]', 0)}/{sum(t['summary']['counts'].values())}" if t else "n/a"),
             tile("Checklist validated", ck.get("validated", 0) if d["checklist"]["items"] else "n/a", f"{ck.get('failed', 0)} failed" if d["checklist"]["items"] else ""),
             tile("Audits passing", f"{sum(1 for a in audits if a == 'PASS')}/2" if f else "n/a", "leakage, reproducibility"),
             tile("Running jobs", d["jobs"]["n_running"], "stale findings: %d" % len(d["jobs"]["stale"]["findings"]))]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def render(d, built_at, now=None):
    now = now if now is not None else 0
    nav = "".join(f'<a href="{h}"{" class=on" if h == "dashboard.html" else ""}>{e(t)}</a>' for h, t in NAV)
    parts = [s_foundation(d), s_trace(d), s_checklist(d), s_audits(d), s_research(d), s_memory(d), s_jobs(d, now), s_problems(d)]
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>Status board</title><meta name="description" content="Foundation status, Bible trace, checklist, audits, research numbers and running jobs.">\n'
            f"<style>{CSS}</style></head><body><main>\n<nav>{nav}</nav>\n<h1>Status board</h1>"
            f'<div class="sub">Generated from the artefacts in state/. Unavailable sources are stated, never shown as zero.</div>\n'
            f"{headline_tiles(d)}\n" + "\n".join(p for p in parts if p) +
            f'\n<p class="note">Built {e(built_at)}. Generated by scripts/dashboard_build.py; sealed test windows and credentials are never read. {e(CAVEAT)}</p>\n</main></body></html>\n')


# ------------------------------------------------------------------------------------------------ build
def build(root=ROOT, built_at=None, now=None, dry_run=False, info=R.process_info, free_fn=None, out_rel=OUT_REL):
    """Collect, render, AUDIT, then write atomically (LF). Raises site_safety.Leak before writing when the page contains a
    secret or a sealed mark. Returns {'path','sha256','bytes','problems','sources'}."""
    root = Path(root)
    now = time.time() if now is None else now
    built_at = built_at or datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    src = Sources(root)
    d = collect(src, now, info=info, free_fn=free_fn)
    page = render(d, built_at, now)
    S.audit_text("dashboard.html", page)
    data = page.encode("utf-8")
    out = root / out_rel
    sha = hashlib.sha256(data).hexdigest()
    res = {"path": str(out), "sha256": sha, "bytes": len(data), "problems": list(src.problems), "sources": dict(src.used)}
    if dry_run:
        return res
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    man = root / MANIFEST_REL
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"page": str(out_rel).replace("\\", "/"), "sha256": sha, "built_at": built_at, "sources": src.used}, indent=1, sort_keys=True), encoding="utf-8")
    return res


def check(root=ROOT, out_rel=OUT_REL):
    """Compare the file on disk with the sources it was built from (the manifest's source hashes) and re-audit it.
    Returns {'ok', 'reasons'}; ok is False for a missing page, a tampered page, a leak, or a source that changed."""
    root = Path(root)
    page, man = root / out_rel, root / MANIFEST_REL
    reasons = []
    if not page.exists():
        return {"ok": False, "reasons": ["docs/dashboard.html does not exist"]}
    text = page.read_text(encoding="utf-8")
    try:
        S.audit_text("dashboard.html", text)
    except S.Leak as ex:
        reasons.append(f"leak: {ex}")
    if not man.exists():
        return {"ok": False, "reasons": reasons + ["no manifest: cannot tell what the page was built from"]}
    m = json.loads(man.read_text(encoding="utf-8"))
    if hashlib.sha256(page.read_bytes()).hexdigest() != m.get("sha256"):
        reasons.append("page differs from the manifest hash (edited by hand or rebuilt without a manifest)")
    src = Sources(root)
    for rel, h in sorted((m.get("sources") or {}).items()):
        if rel == "state/experiments.jsonl":
            continue
        t = src.text(rel)
        if t is None:
            reasons.append(f"source {rel} is gone")
        elif hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()[:16] != h:
            reasons.append(f"source {rel} changed since the page was built")
    return {"ok": not reasons, "reasons": reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--built-at")
    ap.add_argument("--now", type=float, help="epoch seconds used for job ages (default: the clock)")
    a = ap.parse_args(argv)
    if a.check:
        r = check()
        print("dashboard is current" if r["ok"] else "dashboard is stale: " + "; ".join(r["reasons"]))
        return 0 if r["ok"] else 1
    try:
        r = build(built_at=a.built_at, now=a.now, dry_run=a.dry_run)
    except S.Leak as ex:
        print(f"LEAK - nothing written: {ex}")
        return 2
    print(("would write " if a.dry_run else "wrote ") + f"{r['path']} ({r['bytes']} bytes, sha256 {r['sha256'][:12]}); {len(r['problems'])} source problem(s)")
    for p in r["problems"]:
        print("  -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
