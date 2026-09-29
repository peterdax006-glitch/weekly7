import json, hashlib, pathlib
B = chr(92)
src = pathlib.Path(r"C:/Users/Peter/.claude/projects/C--Users-Peter/e1467aef-8afc-4d51-a6e4-927d5141a284.jsonl")
S = '<pasted_content id=' + B + '"4600' + B + '">' + B + 'n'
E = B + 'n</pasted_content id=' + B + '"4600' + B + '">'
best = ""
for line in src.read_text(encoding="utf-8").splitlines():
    a = line.find(S)
    while a >= 0:
        b = line.find(E, a + 1)
        if b > a:
            try:
                t = json.loads('"' + line[a + len(S): b] + '"')
                if len(t) > len(best):
                    best = t
            except Exception:
                pass
        a = line.find(S, a + 1)
text = best
assert text.startswith("# WEEKLY7") and "BEGIN NOW" in text, repr(text[:60])
h = hashlib.sha256(text.encode("utf-8")).hexdigest()
pathlib.Path("BIBLE.md").write_text(text + "\n", encoding="utf-8", newline="\n")
pathlib.Path("canon/bible.lock.json").write_text(json.dumps({"sha256": h, "chars": len(text), "saved": "2026-09-28"}, indent=1))
mem = pathlib.Path(r"C:/Users/Peter/.claude/projects/C--Users-Peter/memory")
rule1 = "im also creating a rule where if you do not have the number of code lines in the expected range its not detailed enough so correct it to be more detailed"
rule2 = "check off the master list in whatever order you see best just make sure as you do it you are running tests to collect data, and that all the checkboxes get checked before you call it"
(mem / "weekly7-bible-verbatim.md").write_text(f"""---
name: weekly7-bible-verbatim
description: THE BIBLE - the owner's Weekly7 Master Autonomous Build & Validation Prompt v1.0, VERBATIM, sha256 {h[:12]}
metadata:
  type: project
---

The owner said on 2026-09-28: "save this as your master prompt task to follow ... or just call it your bible".
Hierarchy (its own section 0): CANON.md > master blueprint > code > test results > my decisions.
Verbatim copy: weekly7/BIBLE.md (sha256 {h}; checked by canon/bible.lock.json).

Owner rules added the same day, VERBATIM:
- "{rule1}" (apply with meaningful code and tests, never padding)
- "{rule2}"

**How to apply:** work the Bible's master checklist in the best order, always with tests running to collect data; don't call it done until every box is checked (validated) or honestly marked UNPROVEN with the reason. Implement, test, attack, validate, record, improve, repeat. Related: [[weekly7-algorithm-directive-verbatim]], [[weekly7-stock-picker]].

---- VERBATIM TEXT ----

{text}
""", encoding="utf-8", newline="\n")
idx = mem / "MEMORY.md"
t = idx.read_text(encoding="utf-8")
if "weekly7-bible-verbatim" not in t:
    idx.write_text(t.rstrip() + "\n- [Weekly7 BIBLE (master prompt) VERBATIM](weekly7-bible-verbatim.md) — governs all Weekly7 work; phases 0-46; line-count and check-every-box rules\n", encoding="utf-8", newline="\n")
print("saved bible:", len(text), "chars, sha256", h[:16])
