import pathlib
D = [
 ("C40", "Master blueprint, in Downloads, very long", "make me a master blueprint\nput it in file downloads\nit should be very long"),
 ("C41", "Rare-analog memory: notice when today resembles the one other time something happened; every data point knows its date and relevance",
  "the algorithm should be so advanced that if there was only one other time in history that a certain stock market problem had ever occured like a bubble pop or a sector disapear or a sector arise, or even something very small most humans would never find it but a small thing only happened one other time in history, the self learning should notice the simalarity, the self learning should also know what year all data points occured to know what coordinates with what, and how relavent data points are to today"),
 ("C42", "7% average over a year; timeline-aware caution/aggression without cheating; learn from mistakes so a rerun would not repeat them (by pattern, not by memorised answers)",
  "it should be so advanced that it aims for the average of 7% a week over the course of a year, but it also notices the timelines to be more cautious or more aggressive without cheating by investigating what timeline it is, and if it messes up on a test, then it should self learn so if we run the same test again it wouldn't mess up the second time, in fact you should give that a try where it doesnt remember oh yeah now i need to pick this stock, but oh yeah, i noticed a time where this pattern caused this to happen so im going to try something else instead"),
 ("C43", "Never hold a failed pattern: improve it until consistent up to that point, or discard it",
  "we should never hold onto a pattern that failed, the pattern either needs to be improved to something that would be consistent up to that point or disguarded"),
]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
end = s.index("\n]\n", s.index('("C39"'))
ins = "".join(f'    ("{c}", "2026-09-28", {t!r},\n     {x!r}),\n' for c, t, x in D)
s = s[:end + 1] + ins + s[end + 1:]
p.write_text(s, encoding="utf-8", newline="\n")
mem = pathlib.Path(r"C:/Users/Peter/.claude/projects/C--Users-Peter/memory/weekly7-algorithm-directive-verbatim.md")
m = mem.read_text(encoding="utf-8").rstrip() + "\n\n**More directives, VERBATIM (canon C40-C43):**\n" + "".join(f"\n- **{c}:** > {x}\n" for c, _, x in D)
mem.write_text(m + "\n", encoding="utf-8", newline="\n")
print("ok")
