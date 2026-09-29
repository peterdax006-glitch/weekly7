import pathlib
p=pathlib.Path("canon/build_canon.py"); s=p.read_text(encoding="utf-8")
anchor = '''"we need a algorithm for advanced self learning system, but we also need the algorithm to have advanced memory knowing how prevelant to make memory based on a variety of factors"),
    ("C35",'''
assert anchor in s
i = s.rindex("]")
s = s[:i] + '''    ("C36", "2026-09-28", "Priority: Algorithm, then Find volatility, then minimal direct Test changes",
     "highest priority is algorithm, find volatility and direct changes to test happen below that but we priortize the learning sytem above direct changes to test, only change what is necessarry for task test because algorithm should be able to do most of it"),
''' + s[i:]
p.write_text(s,encoding="utf-8",newline="\n")
