"""Append canon C60 (owner directive of 29 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C60", "Patterns die within a year: learn WHY they flip, predict when a pattern will be unreliable, and gate it instead of discarding it",
      "also some memory should be diguarded throughout the year as it finds patterns at the begining  of the year that have no relevance at the end of the year, but it should try to self learn to find what changed to make the pattern flip or stop working and become unpredictable, that should be a massive part where it should develop patterns, but it should also develop explanations for how to know what patterns will be unpredictable, because the stock market is always changning and the same problems wont always be predictable so it should be constantly learning to find how to know when and how some pattern it created will be consisistent, and if you can determine when a pattern will be unpredictable, and knowing when not to rely on a pattern then the pattern does not have to be disguarded")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C60"' not in s, "already added"
end = s.index("\n]\n", s.index('("C59"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-29", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added C60")
