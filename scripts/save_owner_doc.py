"""Save an owner document VERBATIM from the session transcript (never retyped): repo copy + sha256 lock + canon entry + memory
file + a machine checklist of its '* [ ]' boxes grouped by '# N. TITLE' sections. Reusable for every future owner document.

  python scripts/save_owner_doc.py --marker "<title line>" --must "<text near the end>" --file NAME.md --lock canon/x.lock.json
         --canon C69 --title "..." --said "owner's own words" --memory name --checklist state/build/X.json --prefix EX"""
import argparse, hashlib, json, pathlib, re

CLAUDE_PROJECT = pathlib.Path.home() / ".claude" / "projects" / ("C--Users-" + pathlib.Path.home().name)
T = CLAUDE_PROJECT / "e1467aef-8afc-4d51-a6e4-927d5141a284.jsonl"      # that session transcript; pass another by editing T
MEM = CLAUDE_PROJECT / "memory"


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def find(marker, must):
    found = None
    for line in T.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") == "assistant":
            continue
        for t in strings(rec):
            for m in re.finditer(r"<pasted_content[^>]*>\n?(.*?)\n?</pasted_content", t, re.S):
                body = m.group(1).rstrip("\n")
                if body.startswith("# " + marker) and must in body:
                    found = body
    if not found:
        raise SystemExit("document not found in transcript")
    return found


def boxes(text, prefix, any_heading=False):
    """'* [ ]' boxes grouped by '# N. TITLE' sections; with any_heading (30 Sep, the C75 prompt uses '# PHASE N — ...' and
    '## 1A — ...' headings) every '#'/'##'/'###' heading starts a group, nested under the last top-level heading."""
    items, sec, n, top = [], "PREAMBLE", 0, ""
    for line in text.splitlines():
        s = re.match(r"^# (\d+)\. (.+)$", line)
        if s and not any_heading:
            sec = f"{s.group(1)}. {s.group(2).strip()}"
            continue
        h = re.match(r"^(#{1,3}) (.+)$", line) if any_heading else None
        if h:
            title = h.group(2).strip()
            if len(h.group(1)) == 1:
                top, sec = title, title
            else:
                sec = f"{top} / {title}" if top else title
            continue
        b = re.match(r"^\s*(?:\d+\. )?\[ \] (.+)$", line.replace("* [ ]", "[ ]", 1)) or re.match(r"^\s*\* \[ \] (.+)$", line)
        if b:
            n += 1
            items.append({"id": f"{prefix}{n:03d}", "group": sec, "description": b.group(1).strip(), "owner": "Claude",
                          "status": "NOT_STARTED", "code_paths": [], "tests": [], "evidence": [], "notes": ""})
    return items


def main():
    ap = argparse.ArgumentParser()
    for a in ("marker", "must", "file", "lock", "canon", "prev_canon", "title", "said", "memory", "checklist", "prefix", "summary"):
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--date", default="2026-09-29")
    ap.add_argument("--any-heading", action="store_true")
    a = ap.parse_args()
    text = find(a.marker, a.must)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    pathlib.Path(a.file).write_text(text + "\n", encoding="utf-8", newline="\n")
    pathlib.Path(a.lock).write_text(json.dumps({"sha256": h, "chars": len(text), "saved": a.date, "file": a.file}, indent=1),
                                    encoding="utf-8", newline="\n")
    p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
    if f'("{a.canon}"' not in s:
        end = s.index("\n]\n", s.index(f'("{a.prev_canon}"'))
        title = f"{a.title} (verbatim in {a.file}, sha256 {h[:16]})"
        body = f"{a.said} [{a.marker}, stored verbatim in {a.file}]"
        s = s[:end + 1] + f'    ("{a.canon}", "{a.date}", {title!r},\n     {body!r}),\n' + s[end + 1:]
        compile(s, "build_canon.py", "exec"); p.write_text(s, encoding="utf-8", newline="\n")
    items = boxes(text, a.prefix, a.any_heading)
    out = pathlib.Path(a.checklist)
    if not out.exists():
        out.write_text(json.dumps({"contract_sha256": h, "created": a.date, "allowed_status":
            ["NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTING", "VALIDATED", "FAILED", "BLOCKED", "SCIENTIFIC_LIMITATION"],
            "items": items, "line_budget": []}, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    (MEM / f"{a.memory}.md").write_text(f"""---
name: {a.memory}
description: {a.canon} ({a.date}) - {a.summary}, VERBATIM (sha256 {h[:12]}); {len(items)} checkboxes
metadata:
  type: project
---

The owner sent this on {a.date} ({a.said!r}). Canon {a.canon}. Repo copy weekly7/{a.file} (sha256 {h}; {a.lock}).
Machine checklist: weekly7/{a.checklist} ({len(items)} boxes, grouped by section).
Related: [[weekly7-research-brain-contract-verbatim]], [[weekly7-prediction-error-addition-verbatim]], [[weekly7-wired-means-reachable]].

---- VERBATIM TEXT ----

{text}
""", encoding="utf-8", newline="\n")
    print("saved:", len(text), "chars sha256", h[:16], "| boxes", len(items))


if __name__ == "__main__":
    main()
