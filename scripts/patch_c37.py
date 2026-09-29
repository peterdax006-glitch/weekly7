"""Blueprint section 4 in the miner: week-clustered statistics, permutation null (P(hallucinated)), P(real),
redundancy pruning, and the validation-gain gate (never learns for no reason)."""
import pathlib

p = pathlib.Path("engine/patterns.py"); s = p.read_text(encoding="utf-8")

# 1) week-clustered test replaces the stock-level one
old = '''        def test(mask):
            md, mc = mask & disc, mask & conf
            if md.sum() < self.p["min_n"] or mc.sum() < self.p["min_n"] // 3:
                return None
            m_d, t_d, n_d = _wstats(w[md], yv[md])
            m_c, t_c, n_c = _wstats(w[mc], yv[mc])
            m_a, t_a, n_a = _wstats(w[mask], yv[mask])
            m_r, t_r, n_r = _wstats(w[mask & rec], yv[mask & rec]) if (mask & rec).sum() >= 30 else (np.nan, 0.0, 0.0)
            return m_d, t_d, m_c, t_c, m_a, t_a, n_a, m_r, t_r, n_r'''
new = '''        dcode = pd.factorize(dates)[0]
        nd = int(dcode.max()) + 1

        def ctest(mask, yy):
            """Week-clustered: one observation per week (stocks in a week move together)."""
            if mask.sum() == 0:
                return 0.0, 0.0, 0.0
            wm = w[mask]
            sw = np.bincount(dcode[mask], weights=wm, minlength=nd)
            sy = np.bincount(dcode[mask], weights=wm * yy[mask], minlength=nd)
            ok = sw > 0
            if ok.sum() < 5:
                return 0.0, 0.0, float(ok.sum())
            wk_mean = sy[ok] / sw[ok]
            ww = sw[ok]                                   # a week counts by the relevance weight of its rows
            m = float((ww * wk_mean).sum() / ww.sum())
            v = float((ww * (wk_mean - m) ** 2).sum() / ww.sum())
            n_eff = ww.sum() ** 2 / (ww ** 2).sum()
            return m, m / np.sqrt(max(v, 1e-12) / max(n_eff, 1.0)), float(n_eff)

        self._ctest = ctest

        def test(mask, yy=None):
            yy = yv if yy is None else yy
            md, mc = mask & disc, mask & conf
            if md.sum() < self.p["min_n"] or mc.sum() < self.p["min_n"] // 3:
                return None
            m_d, t_d, n_d = ctest(md, yy)
            m_c, t_c, n_c = ctest(mc, yy)
            m_a, t_a, n_a = ctest(mask, yy)
            m_r, t_r, n_r = ctest(mask & rec, yy) if (mask & rec).sum() >= 30 else (np.nan, 0.0, 0.0)
            return m_d, t_d, m_c, t_c, m_a, t_a, n_a, m_r, t_r, n_r'''
assert old in s; s = s.replace(old, new)

# 2) permutation null + P(hallucinated), P(real), redundancy pruning, validation-gain gate
old = '''        R["p_coincidence"] = pv
        R["fdr_pass"] = bh'''
new = '''        R["p_coincidence"] = pv
        R["fdr_pass"] = bh
        # P(hallucinated): rerun the same search on outcomes shuffled WITHIN each week (keeps each week's spread,
        # destroys any link between features and outcomes) - how often does the search invent a pattern this strong?
        null_t = []
        for rep in range(self.p.get("null_reps", 2)):
            yperm = yv.copy()
            order_d = np.argsort(dcode, kind="stable")
            bounds = np.flatnonzero(np.diff(dcode[order_d])) + 1
            for grp in np.split(order_d, bounds):
                yperm[grp] = yperm[rng.permutation(grp)]
            for key in R["key"].sample(min(len(R), 1500), random_state=rep).values:
                mask = self._mask_from_key(key, Q, feats)
                r0 = test(mask, yperm) if mask is not None else None
                if r0:
                    null_t.append(abs(r0[1]))
        null_t = np.sort(np.array(null_t)) if null_t else np.array([0.0])
        real_t = np.sort(R["t_disc"].abs().values)
        def local_fdr(t):
            frac_null = 1 - np.searchsorted(null_t, t) / len(null_t)
            frac_real = max(1 - np.searchsorted(real_t, t) / len(real_t), 1e-9)
            return float(min(1.0, frac_null / frac_real))
        R["p_hallucinated"] = [local_fdr(abs(t)) for t in R["t_disc"]]
        from math import erf, sqrt
        conf_factor = [(0.5 * (1 + erf(tc / sqrt(2))) if np.sign(mc) == np.sign(md) else 0.0)
                       for tc, mc, md in zip(R["t_conf"].abs(), R["m_conf"], R["m_disc"])]
        R["p_real"] = (1 - np.maximum(R["p_hallucinated"], np.minimum(1, R["p_coincidence"] * m))) * np.array(conf_factor)
        self.null_summary = {"null_patterns": int(len(null_t)), "null_t_95pct": float(np.quantile(null_t, 0.95)),
                             "real_t_95pct": float(np.quantile(real_t, 0.95))}'''
assert old in s; s = s.replace(old, new)

old = '''        R["confirmed"] = bh & (np.sign(R["m_conf"]) == np.sign(R["m_disc"])) & (R["t_conf"].abs() >= 1.0)'''
new = '''        R["confirmed"] = (R["p_real"] >= self.p.get("p_real_min", 0.8))'''
assert old in s; s = s.replace(old, new)

old = '''        self.patterns = R.reset_index(drop=True)'''
new = '''        # redundancy: patterns firing on nearly the same rows are one idea - keep the strongest
        R = R.reindex(R["effect"].abs().sort_values(ascending=False).index)
        R["duplicate_of"] = None
        sub = rng.choice(len(yv), min(len(yv), 200_000), replace=False)
        kept = []
        for i in R.index[R["status"].isin(["active", "rescoped"])]:
            mk = self._mask_from_key(R.at[i, "key"], Q[sub], feats)
            dup = None
            for j, mj in kept:
                inter = (mk & mj).sum(); uni = (mk | mj).sum()
                if uni and inter / uni > 0.8:
                    dup = R.at[j, "key_named"]; break
            if dup:
                R.at[i, "status"], R.at[i, "duplicate_of"] = "duplicate", dup
            else:
                kept.append((i, mk))
        # never learns for no reason: add patterns greedily, keep one only if it improves out-of-sample prediction
        cand = [i for i in R.index if R.at[i, "status"] in ("active", "rescoped")]
        score = np.zeros(len(yv)); best = 0.0; gate_min = self.p.get("gate_min_gain", 0.0005)
        yc = yv[conf]
        for i in cand:
            mk = self._mask_from_key(R.at[i, "key"], Q, feats)
            trial = score + R.at[i, "effect"] * mk
            sc = trial[conf]
            corr = float(np.corrcoef(sc, yc)[0, 1]) if sc.std() > 0 else 0.0
            if corr - best > gate_min:
                score, best = trial, corr
            else:
                R.at[i, "status"] = "no_gain"                 # recorded, not used
        self.gate_corr = best
        self.patterns = R.reset_index(drop=True)'''
assert old in s; s = s.replace(old, new)

old = '''        self.report = {"tested": int(m), "fdr_pass": int(bh.sum()), "active": int((R["status"] == "active").sum()),
                       "benched": int((R["status"] == "benched").sum()), "rescoped": int((R["status"] == "rescoped").sum())}'''
new = '''        st = self.patterns["status"].value_counts().to_dict()
        self.report = {"tested": int(m), "fdr_pass": int(bh.sum()), **{k: int(v) for k, v in st.items()},
                       "gate_corr_confirm": round(self.gate_corr, 4), **getattr(self, "null_summary", {})}'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("miner upgraded")
