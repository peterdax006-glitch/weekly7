"""Save the owner's MASTER CHECKLIST ADDITION (prediction error -> market change -> self-calibration -> adaptive exit
intelligence, 29 Sep 2026) VERBATIM from the session transcript: PREDICTION_ERROR_ADDITION.md + sha256 lock, canon C68, memory
file, and a machine checklist (checklists A-Z, the Z adversarial tests, the 18 completion requirements)."""
import hashlib, json, pathlib, re

T = pathlib.Path(r"C:\Users\Peter\.claude\projects\C--Users-Peter\e1467aef-8afc-4d51-a6e4-927d5141a284.jsonl")
MARK = "WEEKLY7 — MASTER CHECKLIST ADDITION"
def strings(o):
    """Every string anywhere in a transcript record (mid-turn messages are stored in several record shapes)."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


found = None
for line in T.read_text(encoding="utf-8", errors="replace").splitlines():
    if MARK not in line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    if rec.get("type") == "assistant":                             # never take my own quotes of it
        continue
    for t in strings(rec):
        for m in re.finditer(r"<pasted_content[^>]*>\n?(.*?)\n?</pasted_content", t, re.S):
            body = m.group(1).rstrip("\n")
            if body.startswith("# " + MARK) and "# FINAL PRINCIPLE" in body:   # the real paste, not code that mentions it
                found = body
assert found, "addition not found in transcript"
text = found
assert text.startswith("# " + MARK) and "without ever having seen the future" in text, repr(text[:80])
h = hashlib.sha256(text.encode("utf-8")).hexdigest()
pathlib.Path("PREDICTION_ERROR_ADDITION.md").write_text(text + "\n", encoding="utf-8", newline="\n")
pathlib.Path("canon/prediction_error_addition.lock.json").write_text(json.dumps(
    {"sha256": h, "chars": len(text), "saved": "2026-09-29", "file": "PREDICTION_ERROR_ADDITION.md"}, indent=1), encoding="utf-8", newline="\n")

title = (f"Master checklist ADDITION: prediction error -> market change -> self-calibration -> adaptive exit intelligence "
         f"(verbatim in PREDICTION_ERROR_ADDITION.md, sha256 {h[:16]}); extends C66, replaces nothing")
body = "i have an addition to the master checklist: [WEEKLY7 MASTER CHECKLIST ADDITION, stored verbatim in PREDICTION_ERROR_ADDITION.md]"
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
if '("C68"' not in s:
    end = s.index("\n]\n", s.index('("C67"'))
    s = s[:end + 1] + f'    ("C68", "2026-09-29", {title!r},\n     {body!r}),\n' + s[end + 1:]
    compile(s, "build_canon.py", "exec"); p.write_text(s, encoding="utf-8", newline="\n")

items = []
for mm in re.finditer(r"^# CHECKLIST ([A-Z]) — (.+)$", text, re.M):
    items.append({"id": f"PE{mm.group(1)}", "group": "CHECKLIST", "description": f"Checklist {mm.group(1)}: {mm.group(2).strip()}",
                  "owner": "Claude", "status": "NOT_STARTED", "code_paths": [], "tests": [], "evidence": [], "notes": ""})
z = text[text.index("# CHECKLIST Z"):text.index("# COMPLETION REQUIREMENT")]
for i, b in enumerate(re.findall(r"^\* (.+)$", z, re.M), 1):
    items.append({"id": f"PZ{i:02d}", "group": "ADVERSARIAL TEST", "description": b.strip(), "owner": "Claude",
                  "status": "NOT_STARTED", "code_paths": [], "tests": [], "evidence": [], "notes": ""})
c = text[text.index("# COMPLETION REQUIREMENT"):text.index("# FINAL PRINCIPLE")]
for n, b in re.findall(r"^(\d+)\. (.+)$", c, re.M):
    items.append({"id": f"PC{int(n):02d}", "group": "COMPLETION", "description": b.strip(), "owner": "Claude",
                  "status": "NOT_STARTED", "code_paths": [], "tests": [], "evidence": [], "notes": ""})
out = pathlib.Path("state/build/PREDICTION_ERROR_CHECKLIST.json")
if not out.exists():
    out.write_text(json.dumps({"contract_sha256": h, "created": "2026-09-29", "allowed_status":
        ["NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTING", "VALIDATED", "FAILED", "BLOCKED"],
        "items": items, "line_budget": []}, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")

mem = pathlib.Path(r"C:/Users/Peter/.claude/projects/C--Users-Peter/memory")
(mem / "weekly7-prediction-error-addition-verbatim.md").write_text(f"""---
name: weekly7-prediction-error-addition-verbatim
description: C68 master-checklist ADDITION (29 Sep 2026) - prediction error -> market change -> self-calibration -> adaptive exit intelligence, VERBATIM (sha256 {h[:12]}); {len(items)} items; extends C66, replaces nothing
metadata:
  type: project
---

The owner said on 2026-09-29, verbatim: "i have an addition to the master checklist:" followed by the text below.
It is an ADDITION to the C66 research-brain checklist: integrate, never replace, weaken or duplicate. Existing surprise, calibration,
experiment memory, research priority, meta-learning, pattern health/break, observer, winner/loser, compute, firewall, disguised-year,
research loop, volatility and direction labs are EXTENDED and SPECIALISED for prediction-error-driven market discovery.
Repo copy: weekly7/PREDICTION_ERROR_ADDITION.md (sha256 {h}; canon/prediction_error_addition.lock.json; canon C68).
Machine checklist: weekly7/state/build/PREDICTION_ERROR_CHECKLIST.json ({len(items)} items: PEA-PEZ checklists, PZ adversarial tests, PC completion requirements).

**Key rules:**
- Pre-prediction expectations are immutable.
- Exits are learned independently. The ±1 pp target is an evaluation target, NEVER an exit rule.
- Enforce the 5-10% selection constraint and never game it.
- Unknowable stays unknowable.
- Regime detection is forward-time only.
- Research depth scales with error x confidence x repeatability x value x significance, and feeds the EXISTING priority/compute system, not a second scheduler.
- C63 foundation first applies: all code with unit tests, then testing.

Related: [[weekly7-research-brain-contract-verbatim]], [[weekly7-wired-means-reachable]].

---- VERBATIM TEXT ----

{text}
""", encoding="utf-8", newline="\n")
print("saved:", len(text), "chars sha256", h[:16], "| items", len(items))
