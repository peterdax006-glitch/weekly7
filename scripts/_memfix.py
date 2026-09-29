import pathlib
p = pathlib.Path("engine/memory.py"); s = p.read_text(encoding="utf-8")
old = '''            for r in long_term.itertuples(index=False):
                self.ep.append((r.arm, -1e6, np.asarray(r.ctx, dtype=float), float(r.outcome), 1))'''
new = '''            import ast
            for r in long_term.itertuples(index=False):
                arm = ast.literal_eval(r.arm) if isinstance(r.arm, str) else r.arm      # stored as text
                self.ep.append((arm, -1e6, np.asarray(r.ctx, dtype=float), float(r.outcome), 1))'''
assert old in s; s = s.replace(old, new)
old = '''        return pd.DataFrame([{"arm": a, "ctx": list(c), "outcome": o} for a, _, c, o, s in self.ep if s == 0])'''
new = '''        return pd.DataFrame([{"arm": repr(a), "ctx": [float(x) for x in c], "outcome": float(o)}
                             for a, _, c, o, s in self.ep if s == 0])'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
p = pathlib.Path("scripts/livesim_loop2.py"); s = p.read_text(encoding="utf-8")
old = '''        bank = DIR / "memory_bank.parquet"
        (pd.concat([pd.read_parquet(bank), ep]) if bank.exists() else ep).to_parquet(bank)'''
new = '''        bank = DIR / "memory_bank.parquet"
        import os
        lock = str(bank) + ".lock"
        for _ in range(600):                          # three workers finish together: one writer at a time
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL); os.close(fd); break
            except FileExistsError:
                time.sleep(0.5)
        try:
            (pd.concat([pd.read_parquet(bank), ep]) if bank.exists() else ep).to_parquet(bank)
        finally:
            os.remove(lock)'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
p = pathlib.Path("scripts/fetch_macro.py"); s = p.read_text(encoding="utf-8")
s = s.replace('''            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60,
                             headers={"User-Agent": "Weekly7 research"})''', '''            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60,
                             headers={"User-Agent": "curl/8.9.1", "Accept": "*/*"})''')
p.write_text(s, encoding="utf-8", newline="\n")
print("fixed")
