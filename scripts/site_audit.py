"""Bible Phase 29 - structural audit of the generated pages (complements site_safety, which audits content for leaks).

Checks, per page: mobile viewport meta; exactly one h1 and a title; no external script except cdnjs and no external
stylesheet; no 'nan'/'None'/'undefined' leaking out of a formatter; unique element ids; every filter box points at a table
that exists; every relative link resolves to a file in the site folder; page under a size budget; and the promises the
Bible makes for the page (Explorer lists a non-live status, Sensitivity lists at least one worse alternative, Checklist
has an UNPROVEN section). Findings are returned as dicts {page, severity, check, detail}; nothing raises, so one run
reports everything. `python scripts/site_audit.py` exits 1 when any 'error' is found and writes state/research/site/audit.json."""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_BYTES = 3_000_000
ALLOWED_SCRIPT_HOSTS = ("cdnjs.cloudflare.com",)
LEAK_WORDS = re.compile(r">\s*(nan|None|undefined|NaN|inf|-inf)\s*<|\b(nan%|None%|\$nan)\b")
PROMISES = {
    "explorer.html": [("non-live patterns are shown", re.compile(r'data-status="(rejected|discarded|no_gain|duplicate)"')),
                      ("P(real) column present", re.compile(r"P\(real\)")), ("lifecycle shown", re.compile(r"lifecycle"))],
    "sensitivity2.html": [("negative results section", re.compile(r"Negative results")), ("selected value shown", re.compile(r"selected")),
                          ("multiple-testing note", re.compile(r"Bonferroni"))],
    "checklist.html": [("UNPROVEN section", re.compile(r"UNPROVEN")), ("evidence column", re.compile(r"Evidence"))],
    "runs.html": [("registry browser", re.compile(r'id="registry"')), ("sealed windows not listed", re.compile(r"still sealed"))],
}


class _Scan(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.dupe_ids, self.h1, self.links, self.scripts, self.sheets = set(), [], 0, [], [], []
        self.viewport, self.title, self.control_tables, self._in_title = False, "", [], False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            (self.dupe_ids.append(a["id"]) if a["id"] in self.ids else self.ids.add(a["id"]))
        if tag == "h1":
            self.h1 += 1
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "script" and a.get("src"):
            self.scripts.append(a["src"])
        elif tag == "link" and a.get("rel") == "stylesheet":
            self.sheets.append(a.get("href", ""))
        elif tag == "meta" and a.get("name") == "viewport" and "width=device-width" in (a.get("content") or ""):
            self.viewport = True
        elif tag == "title":
            self._in_title = True
        if a.get("data-table"):
            self.control_tables.append(a["data-table"])

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def audit_page(name, text, site_dir):
    """Return the findings for one page. `site_dir` resolves relative links."""
    f = []
    add = lambda sev, chk, det: f.append({"page": name, "severity": sev, "check": chk, "detail": det})
    sc = _Scan()
    sc.feed(text)
    if not sc.viewport:
        add("error", "viewport", "no mobile viewport meta")
    if sc.h1 != 1:
        add("error", "h1", f"{sc.h1} h1 elements (need exactly 1)")
    if not sc.title.strip():
        add("error", "title", "empty title")
    for s in sc.scripts:
        if s.startswith(("http://", "https://", "//")) and not any(h in s for h in ALLOWED_SCRIPT_HOSTS):
            add("error", "external script", s)
    for s in sc.sheets:
        if s.startswith(("http", "//")):
            add("error", "external stylesheet", s)
    m = LEAK_WORDS.search(text)
    if m:
        add("error", "formatter leak", f"raw {m.group(0)!r} in the page")
    if sc.dupe_ids:
        add("error", "duplicate ids", ", ".join(sorted(set(sc.dupe_ids))[:5]))
    for t in sc.control_tables:
        if t not in sc.ids:
            add("error", "dangling filter", f"controls point at missing table '{t}'")
    for href in sc.links:
        if href.startswith(("http", "mailto:", "#", "//")):
            continue
        target = href.split("#")[0].split("?")[0]
        if target in ("", "./", "."):
            continue
        if not (Path(site_dir) / target).exists():
            add("warning", "broken link", f"{href} does not exist in {Path(site_dir).name}/ (page may be published later)")
    n = len(text.encode("utf-8"))
    if n > MAX_BYTES:
        add("error", "size", f"{n:,} bytes exceeds {MAX_BYTES:,}")
    for label, rx in PROMISES.get(name, []):
        if not rx.search(text):
            add("error", "promise", f"missing: {label}")
    return f


def audit_site(site_dir, names=None):
    site_dir = Path(site_dir)
    names = names or sorted(PROMISES)
    out = []
    for n in names:
        p = site_dir / n
        if not p.exists():
            out.append({"page": n, "severity": "error", "check": "missing", "detail": "page not built"})
            continue
        out += audit_page(n, p.read_text(encoding="utf-8", errors="replace"), site_dir)
    return out


def main():
    from scripts import site_build as B
    paths = B.default_paths()
    findings = audit_site(paths["out"])
    dest = ROOT / "state" / "research" / "site" / "audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(json.dumps({"findings": findings}, indent=1).encode("utf-8"))
    errs = [x for x in findings if x["severity"] == "error"]
    for x in findings:
        print(f"[{x['severity']}] {x['page']}: {x['check']} - {x['detail']}")
    print(f"{len(errs)} error(s), {len(findings) - len(errs)} warning(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
