"""Append canon C56 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C56", "Only information available live at that moment: audit every way the system might know the future",
      "the system should be fully oblivious to any details that would have not been available at the time its operating live so it cant predict the future in any way, double check any way it even might be able to know the future if events or speeches or anything slipped by where the self training could research it that ruins the whole point so ensure it only has live info when testing")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C56"' not in s, "already added"
end = s.index("\n]\n", s.index('("C55"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C56")
