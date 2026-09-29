"""Self-learning pattern miner with relevance-weighted memory (canon C35).

The system, not a person, discovers patterns:
  * candidates: single conditions (feature in quintile q), pairs, and "A and B UNLESS C" exceptions;
  * every observation is weighted by relevance to the moment the model is used - recency (half-life in years),
    market-context similarity (fear, trend, breadth, dispersion) - so old eras count rarely, never zero;
  * discovery on the earlier 70% of dates, confirmation on the later 30% (walk-forward);
  * coincidence: Benjamini-Hochberg false-discovery control across EVERY candidate tried, plus confirmation;
  * effect size shrunk by evidence (n_eff / (n_eff + k));
  * pattern death: if the most recent stretch contradicts the long-run effect, the pattern is benched. The miner then
    searches for the cause - a context split under which the pattern still holds - and re-admits it only in that
    narrowed form. With no cause found it stays benched;
  * long-term memory: patterns found in earlier windows are always re-tested as candidates here.
Deterministic; uses only the rows it is given (the caller guarantees they end before the decision date)."""
import numpy as np
import pandas as pd

MINER_DEFAULT = {"half_life_years": 4.0, "ctx_bandwidth": 1.5, "fdr_q": 0.05, "min_n": 300, "shrink_k": 400,
                 "max_pairs": 4000, "max_unless": 600, "recent_frac": 0.2, "max_rows": 1_200_000, "seed": 7}
CTX = ["m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion"]


def _t_to_p(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def _wstats(w, y):
    sw = w.sum()
    if sw <= 0:
        return 0.0, 0.0, 0.0
    m = float((w * y).sum() / sw)
    v = float((w * (y - m) ** 2).sum() / sw)
    n_eff = sw ** 2 / (w ** 2).sum()
    t = m / np.sqrt(max(v, 1e-12) / max(n_eff, 1.0))
    return m, t, float(n_eff)


class PatternMiner:
    def __init__(self, params=None):
        self.p = {**MINER_DEFAULT, **(params or {})}
        self.patterns = pd.DataFrame()
        self.feats = []
        self.cuts = {}
        self.report = {}

    # ---------- data preparation ----------
    def _quintiles(self, X):
        Q = np.empty(X.shape, dtype=np.int8)
        for j, c in enumerate(X.columns):
            r = X[c].groupby(level=0).rank(pct=True).fillna(0.5).values
            Q[:, j] = np.minimum((r * 5).astype(np.int8), 4)
        return Q

    def _weights(self, dates, ctx, now, ctx_now):
        age_y = (pd.Timestamp(now) - pd.DatetimeIndex(dates)).days.values / 365.25
        w = 0.5 ** (np.maximum(age_y, 0) / self.p["half_life_years"])
        if ctx is not None and ctx_now is not None:
            C = np.nan_to_num(ctx.astype(float))
            sd = C.std(axis=0); sd[sd < 1e-9] = 1.0
            d = (C - np.nan_to_num(ctx_now)) / sd
            w = w * np.exp(-(d ** 2).sum(axis=1) / (2 * self.p["ctx_bandwidth"] ** 2))
        return w

    # ---------- learning ----------
    def fit(self, X, y, now, ctx_cols=CTX, prior=None):
        """X: rows (date, ticker) of features; y: forward excess return; now: the moment the miner will be used."""
        rng = np.random.default_rng(self.p["seed"])
        ok = y.notna().values
        X, y = X[ok], y[ok]
        if len(X) > self.p["max_rows"]:
            keep = np.sort(rng.choice(len(X), self.p["max_rows"], replace=False))
            X, y = X.iloc[keep], y.iloc[keep]
        dates = X.index.get_level_values(0)
        ctx = X[[c for c in ctx_cols if c in X]].values if any(c in X for c in ctx_cols) else None
        last_day = dates.max()
        ctx_now = X.xs(last_day, level=0)[[c for c in ctx_cols if c in X]].iloc[0].values if ctx is not None else None
        feats = [c for c in X.columns if not c.startswith("m_")]
        self.feats = feats
        Q = self._quintiles(X[feats])
        yv = y.values.astype(float)
        w = self._weights(dates, ctx, now, ctx_now)
        ud = np.sort(dates.unique())
        split = ud[int(len(ud) * 0.7)]
        recent = ud[int(len(ud) * (1 - self.p["recent_frac"]))]
        disc, conf, rec = (dates < split), (dates >= split), (dates >= recent)
        F = len(feats)

        dcode = pd.factorize(dates)[0]
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
            return m_d, t_d, m_c, t_c, m_a, t_a, n_a, m_r, t_r, n_r

        rows = []
        singles = []
        for j in range(F):
            for q in range(5):
                mask = Q[:, j] == q
                r = test(mask)
                if r:
                    rows.append((("s", j, q), r)); singles.append((abs(r[1]), j, q))
        singles.sort(reverse=True)
        top = [(j, q) for _, j, q in singles[:60]]
        pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:] if a[0] != b[0]]
        # plus random pairs so small, unexpected interactions get a chance too
        for _ in range(self.p["max_pairs"] - len(pairs)):
            j1, j2 = rng.choice(F, 2, replace=False)
            pairs.append(((int(j1), int(rng.integers(5))), (int(j2), int(rng.integers(5)))))
        pair_res = []
        for (j1, q1), (j2, q2) in pairs[: self.p["max_pairs"]]:
            mask = (Q[:, j1] == q1) & (Q[:, j2] == q2)
            r = test(mask)
            if r:
                rows.append((("p", j1, q1, j2, q2), r)); pair_res.append((abs(r[1]), j1, q1, j2, q2))
        # "A and B UNLESS C": the pair's effect reverses when C is in quintile q3
        pair_res.sort(reverse=True)
        n_un = 0
        for _, j1, q1, j2, q2 in pair_res[:40]:
            base = (Q[:, j1] == q1) & (Q[:, j2] == q2)
            for j3 in rng.permutation(F)[:15]:
                if j3 in (j1, j2) or n_un >= self.p["max_unless"]:
                    continue
                for q3 in (0, 4):
                    r = test(base & (Q[:, j3] != q3))
                    if r:
                        rows.append((("u", j1, q1, j2, q2, int(j3), q3), r)); n_un += 1
        # long-term memory: previously found patterns are always re-tested here
        if prior is not None and len(prior):
            for nk in prior["names"]:
                key = self.key_from_names(tuple(nk), feats)
                if key is None:
                    continue
                mask = self._mask_from_key(key, Q, feats)
                r = test(mask) if mask is not None else None
                if r:
                    rows.append((key, r))
        if not rows:
            self.patterns = pd.DataFrame(); return self
        keys = [k for k, _ in rows]
        R = pd.DataFrame([r for _, r in rows], columns=["m_disc", "t_disc", "m_conf", "t_conf", "m_all", "t_all", "n_eff",
                                                        "m_recent", "t_recent", "n_recent"])
        R["key"] = keys
        R["key_named"] = [self._name(k) for k in keys]
        R = R.drop_duplicates("key_named")
        # coincidence: BH false-discovery control over EVERY candidate tried (discovery p-values)
        pv = np.array([_t_to_p(t) for t in R["t_disc"]])
        order = np.argsort(pv); m = len(pv)
        bh = np.zeros(m, bool)
        thresh = self.p["fdr_q"] * (np.arange(1, m + 1) / m)
        passed = pv[order] <= thresh
        if passed.any():
            bh[order[: np.max(np.where(passed)[0]) + 1]] = True
        R["p_coincidence"] = pv
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
                             "real_t_95pct": float(np.quantile(real_t, 0.95))}
        R["confirmed"] = (R["p_real"] >= self.p.get("p_real_min", 0.8))
        R["effect"] = R["m_all"] * R["n_eff"] / (R["n_eff"] + self.p["shrink_k"])        # shrunk by evidence
        # death: the latest stretch contradicts the long-run effect
        dying = R["confirmed"] & R["m_recent"].notna() & (np.sign(R["m_recent"]) != np.sign(R["m_all"])) & (R["t_recent"].abs() >= 1.5)
        R["status"] = np.where(~R["confirmed"], "rejected", np.where(dying, "failed", "active"))   # C43
        # look for WHY a pattern died: a context tercile where it still holds both long-run and recently
        R["scope"] = None
        if ctx is not None:
            for i in R.index[R["status"] == "failed"][:100]:
                mask = self._mask_from_key(R.at[i, "key"], Q, feats)
                sgn = np.sign(R.at[i, "m_all"])
                for ci in range(ctx.shape[1]):
                    cv = np.nan_to_num(ctx[:, ci].astype(float))
                    lo_, hi_ = np.quantile(cv, [1 / 3, 2 / 3])
                    for lab, cm in (("low", cv <= lo_), ("mid", (cv > lo_) & (cv <= hi_)), ("high", cv > hi_)):
                        a = _wstats(w[mask & cm], yv[mask & cm]) if (mask & cm).sum() >= self.p["min_n"] // 2 else (0, 0, 0)
                        rr = _wstats(w[mask & cm & rec], yv[mask & cm & rec]) if (mask & cm & rec).sum() >= 30 else (0, 0, 0)
                        # improved form must be consistent through the WHOLE history up to now: long-run,
                        # discovery half, confirmation half and the recent stretch all agree (C43)
                        dsc = _wstats(w[mask & cm & disc], yv[mask & cm & disc]) if (mask & cm & disc).sum() >= 30 else (0, 0, 0)
                        cnf = _wstats(w[mask & cm & conf], yv[mask & cm & conf]) if (mask & cm & conf).sum() >= 30 else (0, 0, 0)
                        if (np.sign(a[0]) == sgn and abs(a[1]) >= 2 and np.sign(rr[0]) == sgn and abs(rr[1]) >= 1
                                and np.sign(dsc[0]) == sgn and np.sign(cnf[0]) == sgn):
                            R.at[i, "status"] = "rescoped"
                            R.at[i, "scope"] = (ci, lab, float(lo_), float(hi_))
                            break
                    if R.at[i, "status"] == "rescoped":
                        break
        R.loc[R["status"] == "failed", "status"] = "discarded"      # C43: never hold a failed pattern
        # redundancy: patterns firing on nearly the same rows are one idea - keep the strongest
        # order by EVIDENCE, not raw effect: a small noisy child ("f0 q4 & f7 q2") has a bigger effect by chance than
        # its parent single and would otherwise be kept first and block the parent (planted calibration, 2026-09-28).
        # Simpler first (single < pair < unless), then combined |t| across discovery and confirmation.
        nm = R["key_named"].astype(str)
        R["_order_k"] = nm.str.count(" & ") + 2 * nm.str.contains(" unless ", regex=False).astype(int)
        R["_order_t"] = -(R["t_disc"].abs().fillna(0) + R["t_conf"].abs().fillna(0))
        R = R.sort_values(["_order_k", "_order_t"], kind="mergesort").drop(columns=["_order_k", "_order_t"])
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
        self.patterns = R.reset_index(drop=True)
        st = self.patterns["status"].value_counts().to_dict()
        self.report = {"tested": int(m), "fdr_pass": int(bh.sum()), **{k: int(v) for k, v in st.items()},
                       "gate_corr_confirm": round(self.gate_corr, 4), **getattr(self, "null_summary", {})}
        return self

    # ---------- use ----------
    def _mask_from_key(self, key, Q, feats):
        if key[0] == "s":
            return Q[:, key[1]] == key[2]
        if key[0] == "p":
            return (Q[:, key[1]] == key[2]) & (Q[:, key[3]] == key[4])
        if key[0] == "u":
            return (Q[:, key[1]] == key[2]) & (Q[:, key[3]] == key[4]) & (Q[:, key[5]] != key[6])
        return None

    def _name(self, key):
        f = lambda j: self.feats[j] if j < len(self.feats) else f"f{j}"
        if key[0] == "s":
            return f"{f(key[1])} q{key[2]}"
        if key[0] == "p":
            return f"{f(key[1])} q{key[2]} & {f(key[3])} q{key[4]}"
        return f"{f(key[1])} q{key[2]} & {f(key[3])} q{key[4]} unless {f(key[5])} q{key[6]}"

    def score(self, Xday):
        """Pattern score for one day's stocks: sum of shrunk effects of every active (or in-scope rescoped) pattern."""
        if self.patterns.empty:
            return pd.Series(0.0, index=Xday.index)
        Xf = Xday[self.feats]
        Q = np.empty(Xf.shape, dtype=np.int8)
        for j, c in enumerate(self.feats):                      # one day: rank across that day's stocks
            r = Xf[c].rank(pct=True).fillna(0.5).values
            Q[:, j] = np.minimum((r * 5).astype(np.int8), 4)
        s = np.zeros(len(Xday))
        ctx_today = np.array([float(Xday[c].iloc[0]) if c in Xday else np.nan for c in CTX])
        for r in self.patterns.itertuples():
            if r.status == "active" or (r.status == "rescoped" and r.scope is not None and self._in_scope(r.scope, ctx_today)):
                s += r.effect * self._mask_from_key(r.key, Q, self.feats)
        return pd.Series(s, index=Xday.index)

    @staticmethod
    def _in_scope(scope, ctx_today):
        ci, lab, lo_, hi_ = scope
        v = ctx_today[ci]
        if not np.isfinite(v):
            return False
        return (v <= lo_) if lab == "low" else (v > hi_) if lab == "high" else (lo_ < v <= hi_)

    def key_names(self, key):
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

    def export(self):
        P = self.patterns
        return P[P["status"].isin(["active", "rescoped"])][["key", "key_named", "effect", "status"]].copy() if len(P) else P
