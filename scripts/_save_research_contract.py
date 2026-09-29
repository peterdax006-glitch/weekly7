"""Save the owner's AUTONOMOUS DEEP-RESEARCH SELF-LEARNING ENGINE build prompt v1.0 (29 Sep 2026) VERBATIM: pulled from the
session transcript (never retyped) -> RESEARCH_BRAIN_CONTRACT.md + sha256 lock, canon C66, memory file, and the machine checklist
(section 52 items + section 45 line budget)."""
import hashlib, json, pathlib, re, sys

T = pathlib.Path(r"C:\Users\Peter\.claude\projects\C--Users-Peter\e1467aef-8afc-4d51-a6e4-927d5141a284.jsonl")
MARK = "WEEKLY7 — AUTONOMOUS DEEP-RESEARCH SELF-LEARNING ENGINE"
found = None
for line in T.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        rec = json.loads(line)
    except Exception:
        continue
    if rec.get("type") != "user":
        continue
    c = rec.get("message", {}).get("content")
    for t in ([c] if isinstance(c, str) else [b.get("text", "") for b in (c or []) if isinstance(b, dict)]):
        if MARK in t and "# 53. EXECUTION RULE" in t:
            found = t
assert found, "prompt not found in transcript"
m = re.search(r"<pasted_content[^>]*>\n?(.*?)\n?</pasted_content", found, re.S)
text = (m.group(1) if m else found).rstrip("\n")
assert text.startswith("# " + MARK) and text.rstrip().endswith("was made."), repr(text[-80:])
h = hashlib.sha256(text.encode("utf-8")).hexdigest()
pathlib.Path("RESEARCH_BRAIN_CONTRACT.md").write_text(text + "\n", encoding="utf-8", newline="\n")
pathlib.Path("canon/research_contract.lock.json").write_text(json.dumps(
    {"sha256": h, "chars": len(text), "saved": "2026-09-29", "file": "RESEARCH_BRAIN_CONTRACT.md"}, indent=1), encoding="utf-8", newline="\n")

# canon C66: the owner sent this as the new master checklist he promised ("i will create another master checklist for you")
title = f"Autonomous Deep-Research Self-Learning Engine build prompt v1.0: the new master checklist (verbatim in RESEARCH_BRAIN_CONTRACT.md, sha256 {h[:16]})"
body = "[WEEKLY7 AUTONOMOUS DEEP-RESEARCH SELF-LEARNING ENGINE, MASTER BUILD PROMPT v1.0, stored verbatim in RESEARCH_BRAIN_CONTRACT.md; sent as the new master checklist after the owner said 'let me know when they finish that and i will create another master checklist for you']"
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
if '("C66"' not in s:
    end = s.index("\n]\n", s.index('("C65"'))
    s = s[:end + 1] + f'    ("C66", "2026-09-29", {title!r},\n     {body!r}),\n' + s[end + 1:]
    compile(s, "build_canon.py", "exec"); p.write_text(s, encoding="utf-8", newline="\n")

# machine checklist: section 52 items (groups) + section 45 budget rows
items, group, n = [], None, {}
for line in text.splitlines():
    g = re.match(r"^## (FOUNDATION|TESTING|SCIENTIFIC VALIDATION|AUTONOMOUS OPERATION|FINAL GATE)\s*$", line)
    if g:
        group = g.group(1); continue
    it = re.match(r"^\* \[ \] (.+)$", line)
    if it and group:
        pre = {"FOUNDATION": "RF", "TESTING": "RT", "SCIENTIFIC VALIDATION": "RS", "AUTONOMOUS OPERATION": "RA", "FINAL GATE": "RG"}[group]
        n[pre] = n.get(pre, 0) + 1
        items.append({"id": f"{pre}{n[pre]:02d}", "group": group, "description": it.group(1).strip(), "owner": "Claude",
                      "status": "NOT_STARTED", "code_paths": [], "tests": [], "evidence": [], "notes": ""})
budget = []
for line in text.splitlines():
    b = re.match(r"^\| ([^|]+?)\s+\|\s+([\d,]+)[–-]([\d,]+) \|$", line)
    if b:
        budget.append({"section": b.group(1).strip(), "minimum_lines": int(b.group(2).replace(",", "")),
                       "upper_lines": int(b.group(3).replace(",", "")), "actual_lines": None, "code_paths": [],
                       "status": "NOT_STARTED", "notes": ""})
out = pathlib.Path("state/build/RESEARCH_BRAIN_CHECKLIST.json")
if not out.exists():
    out.write_text(json.dumps({"contract_sha256": h, "created": "2026-09-29", "allowed_status":
        ["NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTING", "VALIDATED", "FAILED", "BLOCKED"],
        "items": items, "line_budget": budget}, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")

mem = pathlib.Path(r"C:/Users/Peter/.claude/projects/C--Users-Peter/memory")
(mem / "weekly7-research-brain-contract-verbatim.md").write_text(f"""---
name: weekly7-research-brain-contract-verbatim
description: THE NEW MASTER CHECKLIST (29 Sep 2026, canon C66) - Autonomous Deep-Research Self-Learning Engine v1.0, VERBATIM (sha256 {h[:12]}); {len(items)} checklist items, {len(budget)} budget rows; do not stop until complete
metadata:
  type: project
---

The owner sent this on 2026-09-29 as the new master checklist he had promised after the C62 foundation code was finished.
It EXTENDS C62 (it does not replace it): volatility first, then direction among predicted movers, an autonomous research brain,
losses studied as hard as winners, the "could I have known?" test, research/trader firewall, and an honest 80% question.
Repo copy: weekly7/RESEARCH_BRAIN_CONTRACT.md (sha256 {h}; locked by canon/research_contract.lock.json; canon C66).
Machine checklist: weekly7/state/build/RESEARCH_BRAIN_CHECKLIST.json ({len(items)} items RF/RT/RS/RA/RG + {len(budget)} line-budget rows,
~58-90k meaningful lines expected, integrate rather than duplicate).

**How to apply:** it is governed like C62 - same ruler (scripts/contract_lines.py), same honest statuses, C63 foundation first
(all code, then testing), its section 53 execution rule (never stop early, never lower a test), and it must reuse the existing
engine/learning modules (research_priority, research_policy, meta_learning, failed_learners, experiment_memory, compute,
knowledge_graph, curator/trader_view, break_detection, ...). Related: [[weekly7-self-learning-contract-verbatim]],
[[weekly7-blind-trader-knowing-curator]], [[weekly7-wired-means-reachable]].

---- VERBATIM TEXT ----

{text}
""", encoding="utf-8", newline="\n")
print("saved:", len(text), "chars sha256", h[:16], "| items", len(items), "| budget rows", len(budget))
