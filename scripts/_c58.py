"""Append canon C58 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C58", "Pattern memory always improves and keeps a timeline of when each pattern was seen; nothing from a date that has not yet occurred",
      "the memory for patterns should always be improved but it should organize timelines for when it noticed different patternsso then it knows how releavant a pattern is to the year its working with but you also need to enforce no cheating so there is not info it can gather from a date that has yet to occur")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C58"' not in s, "already added"
end = s.index("\n]\n", s.index('("C57"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C58")
