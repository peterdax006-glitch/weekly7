import pathlib
p=pathlib.Path("canon/build_canon.py"); s=p.read_text(encoding="utf-8")
old='''"ensure there is both no  cheating and never any after market trading or weekend trading"),
]'''
new='''"ensure there is both no  cheating and never any after market trading or weekend trading"),
    ("C34", "2026-09-28", "Advanced self-learning with advanced, factor-weighted memory",
     "we need a algorithm for advanced self learning system, but we also need the algorithm to have advanced memory knowing how prevelant to make memory based on a variety of factors"),
]'''
assert old in s; p.write_text(s.replace(old,new),encoding="utf-8",newline="\n")
