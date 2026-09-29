"""Append canon C57 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C57", "Memory persists across runs and eras; the same year rerun many times must keep improving on its patterns (never on stock identities)",
      "well its memory should remain no matter what era it is so if you run the same year twice it should not remember stocks from that era but it should remember the same patterns it delt with last time and the second time it should deal with those patterns better than it did previously, meaning you can run the same year 500 times and it should still be learning from that year because its constantly learning new patterns and memory, the memory its learned in those 500 times should also stay releavent because you are still working with the same year")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C57"' not in s, "already added"
end = s.index("\n]\n", s.index('("C56"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C57")
