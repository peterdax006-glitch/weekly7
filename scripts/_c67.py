"""Append canon C67 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C67", "Mover-episode deep research: every day study hundreds of 5-10% movers and what they did next (consolidated, expanded, stopped, spiked, reversed) - 24/7 deep research on tons of stocks for even the smallest predictable patterns",
      "it should be designed so it looks at hundereds of stocks in a day that were 5-10% volatile, it should also see the ones that were volatile then consolidated or were volatile durring the day and got way more volatile or stopped or spiked the next day or if it reversed the next day, it should look at hundereds of stocks olike that and do ai deep research 24/7 on tons and tons of stocks to find even the smallest patterns it couldve predicted")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C67"' not in s, "already added"
end = s.index("\n]\n", s.index('("C66"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C67")
