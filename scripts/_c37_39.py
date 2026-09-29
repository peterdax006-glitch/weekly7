import pathlib
T37 = "make the bluprint first and structure it aropund the microscopic effects only a computer would notice the calculate the chances of statistical likelyness of the noticed pattern being real, hallucinated, or coincidence, and structure it around being able to learn thousands of patterns and it never stops trying to learn, but also it never learns for no reason"
T38 = "also make sure that the self learning keeps in mind 7% weekly is the goal so anything below that should be seen as to low and anything above it should be seen as to risky, and you should spend time just giving it more data while you build it so the self learning has more to work with"
T39 = "it should minimize risk as much as possible but 7 % weekly is top priority, try to minimize risk after you achieve that, that should be a fundamental built into the system where as long as most weeks are 7% volatility then the top priority has been met, after that you can increase the percentage of the 7% being positive rather than negative"
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
end = s.index("\n]\n", s.index('("C36"'))
ins = "".join(f'    ("{c}", "2026-09-28", {t!r},\n     {x!r}),\n' for c, t, x in [
    ("C37", "Algorithm blueprint first: microscopic effects; P(real / hallucinated / coincidence); thousands of patterns; never stops, never learns for no reason", T37),
    ("C38", "7% weekly is the goal: below is too low, above is too risky; keep feeding it more data", T38),
    ("C39", "Priority: 7% weekly volatility first, then minimise risk, then raise the share of positive 7% weeks", T39)])
s = s[:end + 1] + ins + s[end + 1:]
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
