"""Append canon C63 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C63", "Self-Learning contract: foundation first - write ALL the code before moving on to testing and perfecting",
      "start working on it, remember foundation first right all the code before you move onto testing and perfecting")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C63"' not in s, "already added"
end = s.index("\n]\n", s.index('("C62"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C63")
