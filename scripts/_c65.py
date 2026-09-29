"""Append canon C65 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C65", "Live is parked: no work on the Live account until the Test self-learning system is perfected",
      "we arent worrying about the live one until the test self learning one is perfected")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C65"' not in s, "already added"
end = s.index("\n]\n", s.index('("C64"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C65")
