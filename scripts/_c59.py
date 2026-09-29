"""Append canon C59 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C59", "Consistent memory is relevant to every year; memory that does not stay consistent is disregarded for that year",
      "some memory should be releavant for every year if it stays consisistent, if it doesnt stay consistenet it should be disguarderded for that year")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C59"' not in s, "already added"
end = s.index("\n]\n", s.index('("C58"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C59")
