"""A4: long-term pattern memory bank. Keys are stored by FEATURE NAME (portable across runs); a window may only load
patterns from windows that ended before it began (the feed knows real dates; the trader never does)."""
import pathlib
p = pathlib.Path("engine/patterns.py"); s = p.read_text(encoding="utf-8")
old = '''    def export(self):'''
new = '''    def key_names(self, key):
        f = lambda j: self.feats[j]
        if key[0] == "s":
            return ("s", f(key[1]), key[2])
        if key[0] == "p":
            return ("p", f(key[1]), key[2], f(key[3]), key[4])
        return ("u", f(key[1]), key[2], f(key[3]), key[4], f(key[5]), key[6])

    def key_from_names(self, nk, feats):
        ix = {c: i for i, c in enumerate(feats)}
        try:
            if nk[0] == "s":
                return ("s", ix[nk[1]], int(nk[2]))
            if nk[0] == "p":
                return ("p", ix[nk[1]], int(nk[2]), ix[nk[3]], int(nk[4]))
            return ("u", ix[nk[1]], int(nk[2]), ix[nk[3]], int(nk[4]), ix[nk[5]], int(nk[6]))
        except KeyError:
            return None

    def export_bank(self):
        P = self.patterns
        if not len(P):
            return P
        P = P[P["status"].isin(["active", "rescoped"])]
        return pd.DataFrame({"names": [list(map(str, self.key_names(k))) for k in P["key"]],
                             "effect": P["effect"].values, "p_real": P["p_real"].values})

    def export(self):'''
assert old in s; s = s.replace(old, new)
old = '''        if prior is not None and len(prior):
            for key in prior["key"]:
                key = tuple(key)
                try:
                    mask = self._mask_from_key(key, Q, [prior_feat for prior_feat in feats])
                except (ValueError, IndexError):
                    continue'''
new = '''        if prior is not None and len(prior):
            for nk in prior["names"]:
                key = self.key_from_names(tuple(nk), feats)
                if key is None:
                    continue
                mask = self._mask_from_key(key, Q, feats)'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")

p = pathlib.Path("scripts/movers.py"); s = p.read_text(encoding="utf-8")
old = '''        miner = PatternMiner({"max_rows": 400_000, "null_reps": 1}).fit(Xm, swing, now=first)'''
new = '''        bank_f = OUT / "pattern_bank_move.parquet"
        real_start = first - feed._shift
        prior = None
        if bank_f.exists():
            b = pd.read_parquet(bank_f)
            b = b[pd.to_datetime(b["real_end"]) < real_start]      # only windows that ENDED before this one began
            prior = b if len(b) else None
        miner = PatternMiner({"max_rows": 400_000, "null_reps": 1}).fit(Xm, swing, now=first, prior=prior)
        ex = miner.export_bank()
        if len(ex):
            ex["real_end"] = str((sessions[-1] - feed._shift).date())
            ex["window"] = rid
            import filelock_free as _fl if False else None
            (pd.concat([pd.read_parquet(bank_f), ex]) if bank_f.exists() else ex).to_parquet(bank_f)'''
assert old in s; s = s.replace(old, new)
s = s.replace("            import filelock_free as _fl if False else None\n", "")
p.write_text(s, encoding="utf-8", newline="\n")
print("bank wired")
