import pathlib
D = [("C45", "The Bible: Master Autonomous Build & Validation Prompt v1.0 (verbatim in BIBLE.md, sha256 in canon/bible.lock.json)",
      "save this as your master prompt task to follow: [WEEKLY7 MASTER AUTONOMOUS BUILD & VALIDATION PROMPT v1.0, stored verbatim in BIBLE.md] or just call it your bible"),
     ("C46", "Line-count rule",
      "im also creating a rule where if you do not have the number of code lines in the expected range its not detailed enough so correct it to be more detailed"),
     ("C47", "Check every box, in the best order, with tests running",
      "check off the master list in whatever order you see best just make sure as you do it you are running tests to collect data, and that all the checkboxes get checked before you call it")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
end = s.index("\n]\n", s.index('("C44"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-28", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
p.write_text(s, encoding="utf-8", newline="\n")
