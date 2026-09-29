import pathlib
p=pathlib.Path("canon/build_canon.py"); s=p.read_text(encoding="utf-8")
ins = '''    ("C36", "2026-09-28", "Priority: Algorithm, then Find volatility, then minimal direct Test changes",
     "highest priority is algorithm, find volatility and direct changes to test happen below that but we priortize the learning sytem above direct changes to test, only change what is necessarry for task test because algorithm should be able to do most of it"),
'''
bad = '''    missing = [c for c in lock if c not in {d[0] for d in DIRECTIVES}''' + ins + ''']'''
assert bad in s
s = s.replace(bad, '''    missing = [c for c in lock if c not in {d[0] for d in DIRECTIVES}]''')
end = s.index("\n]\n", s.index('("C35"'))
s = s[:end + 1] + ins + s[end + 1:]
p.write_text(s, encoding="utf-8", newline="\n")
