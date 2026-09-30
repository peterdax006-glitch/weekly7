"""Self-learning pattern miner with relevance-weighted memory (canon C35), on the Phase 3 hardening libraries.

The system, not a person, discovers patterns:
  * candidates (engine.candidates): every single (feature in quintile q), pairs (strongest singles crossed plus SEEDED
    random pairs drawn without replacement), "A and B UNLESS C" exceptions, bank re-tests of earlier windows, candle/micro
    signals when the caller merges them into X, and - opt-in `context_candidates` - market-context columns through an
    expanding (point-in-time) time-series quantile, since a cross-sectional rank of a per-date value is constant;
  * every observation is weighted by relevance to the moment the model is used (engine.pattern_stats.relevance):
    recency (half-life in years) x market-context similarity x era weight, point-in-time - rows dated after `now` are dropped;
  * discovery on the earlier 70% of dates, confirmation on the later 30% (walk-forward);
  * statistics (engine.pattern_stats, blueprint section 30): week-clustered t; when rows are denser than the label horizon
    (consecutive dates share forward-return days) the standard error is overlap-robust (Newey-West on the cluster series),
    otherwise the plain cluster test; Benjamini-Hochberg over EVERY candidate tried; permutation null within date ->
    local false-discovery rate; P(real) = (1 - max(P(hallucinated), BH-corrected P(coincidence))) x Phi(t_confirm);
    effect = mean x n_eff / (n_eff + k) (or empirical-Bayes shrinkage, `effect_method="eb"`);
  * pattern death: if the most recent stretch contradicts the long-run effect, the pattern is benched; the miner searches
    for the cause - a context tercile in which the CLUSTER-tested pattern still holds long-run, in both halves and recently -
    and re-admits it only in that narrowed form. With no cause found it is discarded (nothing stays benched);
  * redundancy (overlap > 80% with a stronger, simpler pattern) and the validation-gain gate (delta correlation > 0.0005
    on unseen dates) decide what is actually used; everything else is recorded and unused;
  * every pattern gets an identity (engine.pattern_identity): canonical expression, sha id, windows, scope, stats, state;
    `records` / `book` hold them and `prior=` accepts a PatternBook or the export_bank frame.
Deterministic; uses only the rows it is given (rows after `now` are ignored)."""
import numpy as np
import pandas as pd

from . import candidates as _C
from . import pattern_identity as _I
from . import pattern_stats as _S

MINER_DEFAULT = {"half_life_years": 4.0, "ctx_bandwidth": 1.5, "fdr_q": 0.05, "min_n": 300, "shrink_k": 400,
                 "max_pairs": 4000, "max_unless": 600, "recent_frac": 0.2, "max_rows": 1_200_000, "seed": 7,
                 # Phase 3 additions
                 "horizon": 5,                  # label horizon in sessions (forward 5-session return)
                 "hac_lags": None,              # None = detect overlap from date spacing; int overrides; 0 disables
                 "hac_kernel": "bartlett",
                 "p_method": "bh",              # blueprint section 30; "bonferroni" reproduces the pre-Phase-3 miner
                 # "eb" empirical Bayes (default since 2026-09-28): with week-clustered n_eff the blueprint k=400 rule
                 # shrank every planted effect to ~25% of truth; eb recovers 0.8-1.0 on strong/negative/pair plants
                 "effect_method": "eb",
                 "context_candidates": False,   # opt-in until pattern_movers builds its quantiles with quantile_matrix()
                 "era_weights": None,           # learned later (blueprint 29); None = neutral
                 "ts_min_history": 60, "top_singles": 60, "unless_top_pairs": 40, "unless_thirds": 15,
                 "conf_frac": 0.3, "p_real_min": 0.8, "gate_min_gain": 0.0005, "null_reps": 2, "null_search": "full",       # "full" repeats the staged search on shuffled outcomes; "masks" reuses the real masks
                 "null_max_patterns": 1500, "redundancy_overlap": 0.8}
CTX = ["m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion"]


def _t_to_p(t):
    """Kept for callers; the exact tail-safe implementation lives in engine.pattern_stats."""
    return float(_S.t_to_p(t))


def _wstats(w, y):
    """Row-level (mean, t, n_eff). Over-confident (rows of one date are not independent); kept for callers and tests."""
    return _S.weighted_stats(w, y)


def with_candles(X, stocks):
    """Merge engine.candles signals into a (date, ticker) panel: the hook every X-builder should call so the 25 candle /
    micro signals reach the miner (the cached panel.parquet has none of them)."""
    from . import candles
    X = X.copy()
    for k, v in candles.build(stocks).items():
        X[k] = v.stack(future_stack=True).reindex(X.index).astype("float32").values
    return X


class _Masks:
    """Row masks computed on demand from expressions: thousands of candidates x ~10^6 rows cannot all be held."""

    def __init__(self, exprs, Q, feats, cache=False):
        self.exprs, self.Q, self.feats = list(exprs), Q, list(feats)
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.exprs)

    def __getitem__(self, i):
        i = int(i)
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        m = self.exprs[i].mask(self.Q, self.feats)
        if self._cache is not None:
            self._cache[i] = m
        return m


class PatternMiner:
    def __init__(self, params=None):
        self.p = {**MINER_DEFAULT, **(params or {})}
        self.patterns = pd.DataFrame()
        self.feats = []               # per-stock features (cross-sectional quintiles); key indices < len(feats) refer here
        self.ctx_feats = []           # market-context candidates (time-series quintiles); their key indices follow feats
        self.cuts = {}
        self.report = {}
        self.records = []
        self.book = _I.PatternBook()
        self.overlap = {}
        self.ctx_names = []           # context columns present at fit time; a rescoped pattern's scope indexes into this
        self._ctx_series = {}

    @property
    def all_feats(self):
        return list(self.feats) + list(self.ctx_feats)

    # ---------- data preparation ----------
    def _quintiles(self, X):
        return _C.xs_quintile_codes(X, list(X.columns))

    def _weights(self, dates, ctx, now, ctx_now):
        """Relevance = recency x context similarity x era (era is neutral unless `era_weights` is set)."""
        return _S.relevance(dates, now, ctx, ctx_now, {"half_life_years": self.p["half_life_years"],
                                                       "ctx_bandwidth": self.p["ctx_bandwidth"]})

    def _overlap(self, ud):
        """Do consecutive dates share forward-return days? -> Newey-West lags for the cluster standard error."""
        p = self.p["hac_lags"]
        if len(ud) < 3:
            return {"spacing": None, "hac_lags": 0}
        spacing = float(np.median(np.busday_count(ud[:-1].astype("datetime64[D]"), ud[1:].astype("datetime64[D]"))))
        if p is not None:
            return {"spacing": spacing, "hac_lags": int(p)}
        h = int(self.p["horizon"])
        if spacing >= h or spacing <= 0:
            return {"spacing": spacing, "hac_lags": 0}
        # 2 x (overlapping neighbours): the Bartlett kernel under-corrects an exact MA(h-1) at lags = h-1
        return {"spacing": spacing, "hac_lags": 2 * (int(np.ceil(h / spacing)) - 1)}

    # ---------- learning ----------
    def fit(self, X, y, now, ctx_cols=CTX, prior=None):
        """X: rows (date, ticker) of features; y: forward excess return; now: the moment the miner will be used."""
        P = self.p
        rng = np.random.default_rng(P["seed"])
        ok = y.notna().values
        X, y = X[ok], y[ok]
        # name invariance (Bible T16 / blind disguise): every seeded draw below works on row POSITIONS (row
        # subsample, permutation null, redundancy subsample). Rows sorted by ticker made those draws depend on what
        # the tickers are called (a disguised market picked different stocks). Order by date, then by row CONTENT.
        if len(X):
            content = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).values
            order = np.lexsort((content, X.index.get_level_values(0).values))
            X, y = X.iloc[order], y.iloc[order]
        d_all = X.index.get_level_values(0)
        past = np.asarray(d_all <= pd.Timestamp(now))
        dropped_future = int((~past).sum())
        if dropped_future:
            X, y = X[past], y[past]                              # point-in-time: nothing after `now` is evidence
        if len(X) > P["max_rows"]:
            keep = np.sort(rng.choice(len(X), P["max_rows"], replace=False))
            X, y = X.iloc[keep], y.iloc[keep]
        dates = X.index.get_level_values(0)
        ud = np.sort(dates.unique().values)
        self.report = {}
        if len(ud) < 8 or len(X) == 0:
            self.patterns = pd.DataFrame()
            self.report = {"tested": 0, "fdr_pass": 0, "rows_after_now_dropped": dropped_future}
            return self
        ctx_names = [c for c in ctx_cols if c in X]
        ctx = X[ctx_names].values if ctx_names else None
        ctx_now = _S.context_now(ctx, dates, now) if ctx is not None else None
        self.ctx_names = ctx_names

        # ---- candidate universe and quantiles (per-stock cross-sectional; market context expanding time-series)
        uni = _C.universe_from_columns([c for c in X.columns if c not in ("date", "ticker")], _C.CANDLE_SIGNALS)
        if not P["context_candidates"]:
            uni = _C.Universe(uni.engine, uni.candle, ())
        min_hist = max(10, min(int(P["ts_min_history"]), len(ud) // 4))
        qm = _C.quantise(X, uni, min_history=min_hist)
        n_stock = len(uni.stock_features)
        self.feats = list(qm.features[:n_stock])
        self.ctx_feats = list(qm.features[n_stock:])
        allf = self.all_feats
        Q = qm.Q
        self._ctx_series = {c: X[c].groupby(level=0).first().sort_index() for c in self.ctx_feats}
        yv = y.values.astype(float)
        w = self._weights(dates, ctx, now, ctx_now)
        w_era = None
        if P["era_weights"]:
            w_era = w * _S.era_weight(dates, P["era_weights"], True)
        codes = np.searchsorted(ud, dates.values)
        nd = len(ud)
        split, recent = _S.split_dates(dates, P["conf_frac"], P["recent_frac"])
        disc, conf, rec = np.asarray(dates < split), np.asarray(dates >= split), np.asarray(dates >= recent)
        self.overlap = self._overlap(ud)
        lags = self.overlap["hac_lags"]

        def cstat(mask, yy, wv=None):
            """Week-clustered (one observation per date); overlap-robust when dates overlap the label horizon."""
            wv = w if wv is None else wv
            if lags:
                return _S.hac_cluster_test(codes, wv, yy, mask, nd, lags, kernel=P["hac_kernel"])
            return _S.cluster_test(codes, wv, yy, mask, nd)

        def ctest(mask, yy, wv=None):
            return cstat(mask, yy, wv).as_tuple()

        self._ctest = ctest
        sens = {f for f in allf if f in _S.MICROSTRUCTURE_FEATURES}

        def test(mask, yy=None, expr=None):
            yy = yv if yy is None else yy
            wv = w_era if (w_era is not None and expr is not None and sens & set(expr.features)) else w
            md, mc = mask & disc, mask & conf
            if md.sum() < P["min_n"] or mc.sum() < P["min_n"] // 3:
                return None
            m_d, t_d, _ = ctest(md, yy, wv)
            m_c, t_c, _ = ctest(mc, yy, wv)
            m_a, t_a, n_a = ctest(mask, yy, wv)
            se_a = abs(m_a / t_a) if t_a else float("nan")
            if lags and t_a:                                    # rows overlap: n_eff must shrink with the standard error
                iid = _S.cluster_test(codes, wv, yy, mask, nd)
                if iid.se > 0 and np.isfinite(iid.se) and se_a > 0:
                    n_a = n_a * min(1.0, (iid.se / se_a) ** 2)
            if (mask & rec).sum() >= 30:
                rs = cstat(mask & rec, yy, wv)
                m_r, t_r, n_r, se_r = rs.mean, rs.t, rs.n_eff, rs.se
            else:
                m_r, t_r, n_r, se_r = np.nan, 0.0, 0.0, np.nan
            return m_d, t_d, m_c, t_c, m_a, t_a, n_a, m_r, t_r, n_r, se_a, se_r

        gen_params = {"max_pairs": P["max_pairs"], "max_unless": P["max_unless"], "top_singles": P["top_singles"],
                      "unless_top_pairs": P["unless_top_pairs"], "unless_thirds": P["unless_thirds"],
                      "min_history": min_hist}
        empty_lv = _C.empty_levels(qm, P["min_n"])

        def testable(expr):
            return not any((t.feature, t.level) in empty_lv for t in expr.base + expr.unless)

        def search(gen, yy, tfun, sink, seen, bank=None):
            """The staged candidate search (singles -> pairs from the strongest singles -> exceptions from the strongest
            pairs -> bank re-tests). Each stage is steered by |t_disc| of the previous, so the SAME function runs on real
            and on shuffled outcomes: the null then reproduces the selection the real search performs."""
            def run(cands):
                scores = {}
                for c in cands:
                    if c.id in seen or not testable(c.expression):
                        continue
                    seen.add(c.id)
                    r = tfun(qm.mask(c.expression), yy, c.expression)
                    if r:
                        sink.append((c, r))
                        scores[c.text] = abs(r[1])
                return scores
            sc = run(gen.stage_singles())
            pc = run(gen.stage_pairs(sc))
            run(gen.stage_unless(pc))
            if bank is not None:
                run(gen.stage_bank(bank))

        gen = _C.CandidateGenerator(qm.universe, {**gen_params, "seed": P["seed"]}, empty=empty_lv)
        rows = []
        search(gen, yv, test, rows, set(), _prior_frame(prior))
        if not rows:
            self.patterns = pd.DataFrame()
            self.report = {"tested": 0, "fdr_pass": 0, "rows_after_now_dropped": dropped_future}
            return self

        R = pd.DataFrame([r for _, r in rows], columns=["m_disc", "t_disc", "m_conf", "t_conf", "m_all", "t_all", "n_eff",
                                                        "m_recent", "t_recent", "n_recent", "se_all", "se_recent"])
        exprs = [c.expression for c, _ in rows]
        R["key"] = [e.to_key(allf) for e in exprs]
        R["key_named"] = [e.text for e in exprs]
        R["pattern_id"] = [c.id for c, _ in rows]
        R["origin"] = [c.origin for c, _ in rows]
        m = len(R)
        # coincidence: BH false-discovery control over EVERY candidate tried (discovery p-values)
        pv = np.asarray(_S.t_to_p(R["t_disc"].values))
        R["p_coincidence"] = pv
        R["q_value"] = _S.bh_qvalues(pv)
        R["fdr_pass"] = _S.bh_reject(pv, P["fdr_q"])
        # P(hallucinated): the same search on outcomes shuffled WITHIN each date, then the local false-discovery rate
        lazy = _Masks(exprs, Q, allf)

        def t_disc_only(mask, yy, expr=None):
            md = mask & disc
            if md.sum() < P["min_n"] or (mask & conf).sum() < P["min_n"] // 3:
                return None
            return (None, ctest(md, yy)[1])

        null_rng = np.random.default_rng(P["seed"] + 1)
        if P["null_search"] == "full":
            nulls = []
            for rep in range(P["null_reps"]):
                yperm = _S.permute_within_clusters(yv, codes, null_rng)
                sink = []
                search(_C.CandidateGenerator(qm.universe, {**gen_params, "seed": P["seed"] + 101 + rep},
                                             empty=empty_lv), yperm,
                       t_disc_only, sink, set())
                ts = np.array([abs(r[1]) for _, r in sink])
                if len(ts) > P["null_max_patterns"]:
                    ts = null_rng.choice(ts, P["null_max_patterns"], replace=False)
                nulls.append(ts)
            null_t = np.sort(np.concatenate(nulls)) if nulls and sum(map(len, nulls)) else np.array([0.0])
        else:
            def t_only(mk, yy):
                r = t_disc_only(mk, yy)
                return None if r is None else r[1]
            null_t = _S.null_t_distribution(lazy, t_only, yv, codes, null_rng, P["null_reps"], P["null_max_patterns"])
        real_t = np.sort(R["t_disc"].abs().values)
        R["p_hallucinated"] = [_S.local_fdr(abs(t), null_t, real_t) for t in R["t_disc"]]
        R["conf_factor"] = _S.confirmation_factor(R["t_conf"].abs().values, R["m_conf"].values, R["m_disc"].values)
        R["p_real"] = _S.p_real(R["p_hallucinated"].values, pv, R["conf_factor"].values, P["p_method"])
        self.null_summary = {"null_patterns": int(len(null_t)), "null_t_95pct": float(np.quantile(null_t, 0.95)),
                             "real_t_95pct": float(np.quantile(real_t, 0.95))}
        R["confirmed"] = (R["p_real"] >= P["p_real_min"])
        R["effect_k"] = _S.shrink_effect(R["m_all"].values, R["n_eff"].values, P["shrink_k"])     # shrunk by evidence
        R["effect_eb"] = _S.eb_shrink(R["m_all"].values, R["se_all"].values, center=0.0)["post_mean"]
        R["effect"] = R["effect_eb"] if P["effect_method"] == "eb" else R["effect_k"]
        # death: the latest stretch contradicts the long-run effect
        # death: the latest stretch contradicts the long-run effect (a "vanished" effect - recent mean near zero - is NOT
        # detectable on the planted data because per-date demeaning leaves every non-plant cell with a nonzero baseline;
        # a shortfall rule was tried and changed nothing there, so it is not shipped unvalidated)
        dying = R["confirmed"] & R["m_recent"].notna() & (np.sign(R["m_recent"]) != np.sign(R["m_all"])) & (R["t_recent"].abs() >= 1.5)
        # decay (C43, 2026-09-29): same sign but the recent effect has shrunk SIGNIFICANTLY below the long-run effect the
        # pattern would be scored at. Planted decaying pattern: long-run -0.7%, recent -0.25%, shortfall ~3 SE - it was
        # being held at 6x its true recent size. Measured against the recent SE, not "near zero", so the demeaning
        # baseline (every non-plant cell carries a small offset) does not hide it.
        shortfall = (R["m_all"].abs() - R["m_recent"].abs()) / R["se_recent"].where(R["se_recent"] > 0)
        decayed = R["confirmed"] & R["m_recent"].notna() & (np.sign(R["m_recent"]) == np.sign(R["m_all"])) \
            & (shortfall >= P.get("decay_z", 2.5))
        dying = dying | decayed.fillna(False)
        R["status"] = np.where(~R["confirmed"], "rejected", np.where(dying, "failed", "active"))   # C43
        # look for WHY a pattern died: a context tercile where it still holds (cluster tests, never row-level t)
        R["scope"] = None
        if ctx is not None:
            # the search tries 3 terciles of every context column; the recent-evidence bar is Bonferroni-corrected for
            # that many tries (with |t| >= 2 and 9 tries a dead pattern was "rescued" by chance, 2026-09-29)
            from scipy.stats import norm as _norm
            n_tries = 3 * ctx.shape[1]
            rescue_t = max(P.get("rescue_min_t", 2.0), float(_norm.ppf(1 - 0.025 / max(n_tries, 1))))
            for i in R.index[R["status"] == "failed"][:100]:
                mask = lazy[i]
                sgn = np.sign(R.at[i, "m_all"])
                for ci in range(ctx.shape[1]):
                    cv = np.nan_to_num(ctx[:, ci].astype(float))
                    lo_, hi_ = np.quantile(cv, [1 / 3, 2 / 3])
                    for lab, cm in (("low", cv <= lo_), ("mid", (cv > lo_) & (cv <= hi_)), ("high", cv > hi_)):
                        sub = mask & cm
                        a = ctest(sub, yv) if sub.sum() >= P["min_n"] // 2 else (0, 0, 0)
                        rr = ctest(sub & rec, yv) if (sub & rec).sum() >= 30 else (0, 0, 0)
                        # improved form must be consistent through the WHOLE history up to now: long-run,
                        # discovery half, confirmation half and the recent stretch all agree (C43)
                        dsc = ctest(sub & disc, yv) if (sub & disc).sum() >= 30 else (0, 0, 0)
                        cnf = ctest(sub & conf, yv) if (sub & conf).sum() >= 30 else (0, 0, 0)
                        # ...and it must not itself have DECAYED: the recent in-scope effect may not fall significantly
                        # short of the in-scope long-run effect (else a persistent context - a fear tercile that lines
                        # up with the pre-decay years - "rescues" a dead pattern; planted calibration, 2026-09-29)
                        # (burden on the rescued form: in a tercile the recent SE is ~1.7x larger, so "not significantly
                        # smaller" alone passes a dead pattern; it must also keep at least half its in-scope size now)
                        se_rr = abs(rr[0] / rr[1]) if rr[1] else np.inf
                        not_decayed = ((abs(a[0]) - abs(rr[0])) / se_rr < P.get("decay_z", 2.5)
                                       and abs(rr[0]) >= P.get("rescue_min_ratio", 0.5) * abs(a[0]))
                        # recent in-scope evidence must be REAL, |t| >= 2 (was 1): with |t| >= 1 two dead patterns
                        # (exact true in-scope recent effect -0.12% and +0.01%) were rescued and held at -0.7%
                        if (np.sign(a[0]) == sgn and abs(a[1]) >= 2 and np.sign(rr[0]) == sgn
                                and abs(rr[1]) >= rescue_t
                                and np.sign(dsc[0]) == sgn and np.sign(cnf[0]) == sgn and not_decayed):
                            R.at[i, "status"] = "rescoped"
                            R.at[i, "scope"] = (ci, lab, float(lo_), float(hi_))
                            # score the rescued form at its IN-SCOPE size (with the pattern's own shrinkage ratio), not
                            # at the unscoped long-run effect it failed with
                            ratio = R.at[i, "effect"] / R.at[i, "m_all"] if R.at[i, "m_all"] else 0.0
                            R.at[i, "effect"] = float(a[0]) * float(np.clip(ratio, 0.0, 1.0))
                            break
                    if R.at[i, "status"] == "rescoped":
                        break
        R.loc[R["status"] == "failed", "status"] = "discarded"      # C43: never hold a failed pattern
        # redundancy: patterns firing on nearly the same rows are one idea - keep the simplest, strongest by EVIDENCE
        # (a small noisy child has a bigger effect by chance than its parent; planted calibration, 2026-09-28)
        order = _S.evidence_order(R["key_named"].astype(str).tolist(), R["t_disc"].values, R["t_conf"].values)
        R = R.iloc[order].reset_index(drop=True)
        exprs = [_I.Expression.parse(t) for t in R["key_named"]]
        R["duplicate_of"] = None
        sub_rows = np.sort(rng.choice(len(yv), min(len(yv), 200_000), replace=False))
        live = [i for i in R.index if R.at[i, "status"] in ("active", "rescoped")]
        _redundancy_masks = _Masks(exprs, Q[sub_rows], allf, cache=True)
        kept, dup = _S.prune_redundant(live, _redundancy_masks, P["redundancy_overlap"])
        for i, j in dup.items():
            R.at[i, "status"], R.at[i, "duplicate_of"] = "duplicate", R.at[j, "key_named"]
        from .learning import wiring                # S17a: keep what pruning throws away as REDUNDANT_WITH edges (sink)
        wiring.on_redundancy(R["key_named"].astype(str).tolist(), dup, _redundancy_masks, now, P["redundancy_overlap"])
        # never learns for no reason: add patterns greedily, keep one only if it improves out-of-sample prediction
        cand = [i for i in R.index if R.at[i, "status"] in ("active", "rescoped")]
        full = _Masks(exprs, Q, allf)
        kept, best, _ = _S.validation_gain_gate(cand, full, R["effect"].values, yv[conf], conf, P["gate_min_gain"])
        keep_set = set(kept)
        for i in cand:
            if i not in keep_set:
                R.at[i, "status"] = "no_gain"                     # recorded, not used
        self.gate_corr = best
        self.patterns = R
        st = R["status"].value_counts().to_dict()
        self.report = {"tested": int(m), "fdr_pass": int(R["fdr_pass"].sum()), **{k: int(v) for k, v in st.items()},
                       "gate_corr_confirm": round(self.gate_corr, 4), **self.null_summary,
                       "hac_lags": int(lags), "date_spacing": self.overlap["spacing"],
                       "rows_after_now_dropped": dropped_future, "bank_skipped": len(gen.skipped_bank)}
        self.records = _I.records_from_miner(R, allf, ctx_names, dates, target=P.get("target", "excess_5d"),
                                             disc_frac=1 - P["conf_frac"], transform_of=qm.universe.transform_of)
        self.book = _I.PatternBook(self.records)
        return self

    # ---------- identity and persistence ----------
    def export_records(self):
        """Every pattern found (any status) as identity records, ready for a bank or a JSON file."""
        return list(self.records)

    def save_book(self, path):
        from pathlib import Path
        Path(path).write_text(self.book.to_json(), encoding="utf-8")

    @staticmethod
    def prior_from_book(book):
        """The `prior` frame fit() re-tests: usable patterns of an earlier window, from a PatternBook or file text."""
        if isinstance(book, str):
            book = _I.PatternBook.from_json(book)
        return _I.records_to_prior(list(book))

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
        return _I.Expression.from_key(key, self.all_feats).text

    def _ts_level(self, name, value):
        """Expanding time-series quintile of today's context value against the fit history (point-in-time)."""
        h = self._ctx_series.get(name)
        if h is None or not np.isfinite(value) or len(h) < 10:
            return -1
        H = np.sort(h.values)
        lo, hi = np.searchsorted(H, value, "left"), np.searchsorted(H, value, "right") + 1
        return min(int((lo + hi) / 2.0 / (len(H) + 1) * 5), 4)

    def quantile_matrix(self, X):
        """Quantile codes for every miner feature (columns = all_feats) of a multi-date panel: stock features
        cross-sectionally per date, market context by the expanding time-series rank over fit history plus the panel's
        own earlier dates. Consumers that rebuild masks from `key` (pattern_movers) should use this, not per-date ranks."""
        stock = _C.xs_quintile_codes(X, self.feats) if self.feats else np.empty((len(X), 0), np.int8)
        cols = [stock]
        dates = X.index.get_level_values(0)
        min_hist = max(10, min(int(self.p["ts_min_history"]), len(self._ctx_series.get(self.ctx_feats[0], [])) // 4)) \
            if self.ctx_feats else 10
        for c in self.ctx_feats:
            per = X[c].groupby(level=0).first().sort_index()
            hist = self._ctx_series[c]
            new = per[per.index > hist.index.max()]
            lv = _C.ts_quintile_series(pd.concat([hist, new]), min_hist).reindex(dates).fillna(-1).values.astype(np.int8)
            cols.append(lv[:, None])
        return np.hstack(cols)

    def score(self, Xday):
        """Pattern score for one day's stocks: sum of shrunk effects of every active (or in-scope rescoped) pattern."""
        if self.patterns.empty:
            return pd.Series(0.0, index=Xday.index)
        Xf = Xday[self.feats]
        Q = np.empty((len(Xday), len(self.feats) + len(self.ctx_feats)), dtype=np.int8)
        for j, c in enumerate(self.feats):                      # one day: rank across that day's stocks
            r = Xf[c].rank(pct=True).fillna(0.5).values
            Q[:, j] = np.minimum((r * 5).astype(np.int8), 4)
        for k, c in enumerate(self.ctx_feats):                  # market context: one value, ranked against its own history
            v = float(Xday[c].iloc[0]) if c in Xday else float("nan")
            Q[:, len(self.feats) + k] = self._ts_level(c, v)
        s = np.zeros(len(Xday))
        # scope indices refer to the context columns PRESENT at fit time (self.ctx_names), not to the full CTX list
        ctx_today = np.array([float(Xday[c].iloc[0]) if c in Xday else np.nan for c in self.ctx_names])
        for r in self.patterns.itertuples():
            if r.status == "active" or (r.status == "rescoped" and r.scope is not None and self._in_scope(r.scope, ctx_today)):
                s += r.effect * self._mask_from_key_safe(r.key, Q)
        return pd.Series(s, index=Xday.index)

    def _mask_from_key_safe(self, key, Q):
        """Scoring mask that treats an unknown context level (-1) as neither a match nor an exception."""
        m = self._mask_from_key(key, Q, self.feats)
        if key[0] == "u":
            m = m & (Q[:, key[5]] >= 0)
        return m

    @staticmethod
    def _in_scope(scope, ctx_today):
        ci, lab, lo_, hi_ = scope
        v = ctx_today[ci]
        if not np.isfinite(v):
            return False
        return (v <= lo_) if lab == "low" else (v > hi_) if lab == "high" else (lo_ < v <= hi_)

    def key_names(self, key):
        f = lambda j: self.all_feats[j]
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


def _prior_frame(prior):
    """The bank to re-test, from the export_bank frame, a PatternBook, a list of PatternRecords, or None."""
    if prior is None:
        return None
    if isinstance(prior, pd.DataFrame):
        return prior if len(prior) else None
    if isinstance(prior, _I.PatternBook):
        prior = list(prior)
    return _I.records_to_prior(prior)
