"""Save the owner's Self-Learning Engine contract (29 Sep 2026) VERBATIM: repo copy + sha256 lock, canon C62, memory file,
and the machine-readable master checklist it demands (section 68). Text is pulled from the session transcript, never retyped."""
import hashlib, json, pathlib, re, sys

SRC = pathlib.Path(sys.argv[1])            # extracted verbatim body (utf-8)
text = SRC.read_bytes().decode("utf-8").rstrip("\n")
assert text.startswith("# WEEKLY7 — MASTER SELF-LEARNING ENGINE") and "# 88. FINAL COMMAND" in text, repr(text[:60])
h = hashlib.sha256(text.encode("utf-8")).hexdigest()

pathlib.Path("SELF_LEARNING_CONTRACT.md").write_text(text + "\n", encoding="utf-8", newline="\n")
pathlib.Path("canon/contract.lock.json").write_text(
    json.dumps({"sha256": h, "chars": len(text), "saved": "2026-09-29", "file": "SELF_LEARNING_CONTRACT.md"}, indent=1),
    encoding="utf-8", newline="\n")

# canon C62: the owner's own sentence, verbatim; the contract body is locked by hash
OWNER = "Save this to memory verbatum as your checkoff list you cant stop until its done: "
title = f"Master Self-Learning Engine contract: the checkoff list; no stopping until it is done (verbatim in SELF_LEARNING_CONTRACT.md, sha256 {h[:16]})"
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
if '("C62"' not in s:
    end = s.index("\n]\n", s.index('("C61"'))
    s = s[:end + 1] + f'    ("C62", "2026-09-29", {title!r},\n     {(OWNER + "[WEEKLY7 MASTER SELF-LEARNING ENGINE contract, stored verbatim in SELF_LEARNING_CONTRACT.md]")!r}),\n' + s[end + 1:]
    compile(s, "build_canon.py", "exec")
    p.write_text(s, encoding="utf-8", newline="\n")

# section 68: machine-readable checklist from the verbatim items of sections 69-80
items, phase = [], None
for line in text.splitlines():
    m = re.match(r"^# \d+\. MASTER CHECKLIST — PHASE ([A-L]): (.+)$", line)
    if m:
        phase = (m.group(1), m.group(2).strip()); continue
    m = re.match(r"^\* \[ \] ([A-L]\d\d) (.+)$", line)
    if m and phase:
        items.append({"id": m.group(1), "phase": f"{phase[0]} {phase[1]}", "description": m.group(2).strip(), "owner": "Claude",
                      "status": "NOT_STARTED", "code_paths": [], "tests": [], "evidence": [], "minimum_lines": None,
                      "actual_lines": None, "validation_result": None, "last_run": None, "code_hash": None, "notes": ""})
budget = []
for line in text.splitlines():
    m = re.match(r"^\| ([^|]+?)\s+\|\s+([\d,]+) \|$", line)
    if m and not m.group(1).startswith("Section") and not m.group(1).startswith("-"):
        budget.append({"section": m.group(1).strip(), "minimum_lines": int(m.group(2).replace(",", "")), "actual_lines": None,
                       "code_paths": [], "status": "NOT_STARTED"})
assert len(items) == 191, len(items)
out = pathlib.Path("state/build/SELF_LEARNING_MASTER_CHECKLIST.json")
if not out.exists():
    out.write_text(json.dumps({"contract_sha256": h, "created": "2026-09-29", "allowed_status":
                               ["NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTING", "VALIDATED", "FAILED", "BLOCKED"],
                               "items": items, "line_budget": budget}, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")

mem = pathlib.Path(r"C:/Users/Peter/.claude/projects/C--Users-Peter/memory")
(mem / "weekly7-self-learning-contract-verbatim.md").write_text(f"""---
name: weekly7-self-learning-contract-verbatim
description: THE CHECKOFF LIST - owner's Weekly7 Master Self-Learning Engine contract, VERBATIM (sha256 {h[:12]}); 191 items A01-L26; cannot stop until done
metadata:
  type: project
---

The owner said on 2026-09-29, verbatim: "{OWNER.strip()}" followed by the contract below.
It is THE checkoff list. I may not stop until every item is genuinely complete and independently validated (its sections 58, 59, 81).
Repo copy: weekly7/SELF_LEARNING_CONTRACT.md (sha256 {h}; locked by canon/contract.lock.json; canon C62).
Machine checklist (its section 68): weekly7/state/build/SELF_LEARNING_MASTER_CHECKLIST.json (191 items, line budget per section).
Statuses: NOT_STARTED / IN_PROGRESS / IMPLEMENTED / TESTING / VALIDATED / FAILED / BLOCKED; never DONE without evidence;
"IMPLEMENTED — NOT VALIDATED" / "— FAILED VALIDATION" / "— INSUFFICIENT EVIDENCE" are honest states.

**How to apply:** follow its section 84 loop at every checkpoint; priority order is its section 85 (future-info/provenance first,
dashboards last). Never pad lines, never lower a test. Related: [[weekly7-bible-verbatim]], [[weekly7-timeline-pattern-memory]],
[[weekly7-judge-the-learning-by-blind-reruns]].

---- VERBATIM TEXT ----

{text}
""", encoding="utf-8", newline="\n")
idx = mem / "MEMORY.md"
t = idx.read_text(encoding="utf-8")
if "weekly7-self-learning-contract-verbatim" not in t:
    lines = t.rstrip("\n").split("\n")
    lines.insert(1, "- [Weekly7 SELF-LEARNING CONTRACT = THE CHECKOFF LIST](weekly7-self-learning-contract-verbatim.md) — VERBATIM; 191 items; never stop until done")
    idx.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("saved contract:", len(text), "chars, sha256", h[:16], "| checklist items", len(items), "| budget rows", len(budget))
