import pathlib
p=pathlib.Path("engine/patterns.py"); s=p.read_text(encoding="utf-8")
old='''        Q = self._quintiles(Xday.assign(_d=0).set_index("_d", append=True).swaplevel(0, 1)[self.feats]) \
            if not isinstance(Xday.index, pd.MultiIndex) else self._quintiles(Xday[self.feats])'''
new='''        Xf = Xday[self.feats]
        Q = np.empty(Xf.shape, dtype=np.int8)
        for j, c in enumerate(self.feats):                      # one day: rank across that day's stocks
            r = Xf[c].rank(pct=True).fillna(0.5).values
            Q[:, j] = np.minimum((r * 5).astype(np.int8), 4)'''
assert old in s; p.write_text(s.replace(old,new),encoding="utf-8",newline="\n")
