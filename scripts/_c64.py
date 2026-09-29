"""Append canon C64 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C64", "Blind trader, knowing curator: the running system never knows the year and gets info day by day as it became available; memory is filed under the real year it operated in; relative memory is a separate hidden system that knows the year and prioritises memory automatically where the system cannot see",
      "how i want it is while its running the system should not know the year and it should only have access to the live data at the time and as the year progresses it gains access to more info so it gets info on a daily basis based on when info was available, and when it all gets saved to memory it saves in the file as the year it operated in, now the relative memory is still important but that should be a seperate system where the system doesnt know the year, but the workings behind the system know the year so it prioritzies different memory but the system itself doesnt know that, that all happens in a automatic spot it cant see")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C64"' not in s, "already added"
end = s.index("\n]\n", s.index('("C63"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C64")
