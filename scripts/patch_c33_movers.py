import pathlib
p = pathlib.Path("scripts/movers.py"); s = p.read_text(encoding="utf-8")
old = '''def labels(stocks):
    C, Hh, L = stocks["Close"], stocks["High"], stocks["Low"]
    hi = pd.concat([Hh.shift(-k) for k in range(1, H + 1)]).groupby(level=0).max()
    lo = pd.concat([L.shift(-k) for k in range(1, H + 1)]).groupby(level=0).min()
    up, dn = hi / C - 1, lo / C - 1
    close = C.shift(-H) / C - 1
    return up, dn, close'''
new = '''def labels(stocks):
    """Canon C33: a pick decided after day t's close can only be bought at day t+1's OPEN. Moves are measured
    from that entry over sessions t+1..t+5 (regular hours only)."""
    C, Hh, L, O = stocks["Close"], stocks["High"], stocks["Low"], stocks["Open"]
    entry = O.shift(-1)
    hi = pd.concat([Hh.shift(-k) for k in range(1, H + 1)]).groupby(level=0).max()
    lo = pd.concat([L.shift(-k) for k in range(1, H + 1)]).groupby(level=0).min()
    up, dn = hi / entry - 1, lo / entry - 1
    close = C.shift(-H) / entry - 1
    return up, dn, close'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
p = pathlib.Path("engine/livesim.py"); s = p.read_text(encoding="utf-8")
old = '''        C, Hh, L = stocks["Close"], stocks["High"], stocks["Low"]
        hi = pd.concat([Hh.shift(-k) for k in range(1, 6)]).groupby(level=0).max()
        lo = pd.concat([L.shift(-k) for k in range(1, 6)]).groupby(level=0).min()
        touch = ((hi / C - 1 >= 0.10) | (lo / C - 1 <= -0.10)).astype(float)'''
new = '''        C, Hh, L = stocks["Close"], stocks["High"], stocks["Low"]
        entry = stocks["Open"].shift(-1)                                   # C33: bought at the next open
        hi = pd.concat([Hh.shift(-k) for k in range(1, 6)]).groupby(level=0).max()
        lo = pd.concat([L.shift(-k) for k in range(1, 6)]).groupby(level=0).min()
        touch = ((hi / entry - 1 >= 0.10) | (lo / entry - 1 <= -0.10)).astype(float)'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("movers patched")
