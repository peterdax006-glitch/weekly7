"""Append canon C61 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C61", "Constant pattern health checks: no phantom patterns; every break is investigated - predict it or discard; admit unknown causes",
      "also we need to ensure that there is a system in place to ensure all patterns are working so it constantly checks that so we never have patterns that dont exist because the system is forced to find out the why for a pattern stopped to learn how to predict the pattern or disguard it if it truly is just unpredictable,  it should also be careful to know when indicators and the things the system has in not the factor at hand but rather its something no one knows the answer to until it happens")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C61"' not in s, "already added"
end = s.index("\n]\n", s.index('("C60"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C61")
