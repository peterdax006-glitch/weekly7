"""Append canon C54 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C54", "The learning system is the product: judge a test by how results CHANGE when the same year is run again",
      "remember you arent as focused on the system as you are the learning system meaning when a test comes back its less about the results you see that time, but the changes in the results the second time you run the same year")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C54"' not in s, "already added"
end = s.index("\n]\n", s.index('("C53"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C54")
