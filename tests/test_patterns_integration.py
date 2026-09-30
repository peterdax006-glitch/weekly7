"""Phase 3 integration: engine.patterns.PatternMiner running on the hardening libraries (identity, statistics,
candidates). Public API compatibility, market-context candidates, sparse flags, candles, persistence and priors.
Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from engine import candidates as C
from engine import pattern_identity as I
from engine.patterns import CTX, PatternMiner, with_candles

FAST = {"max_pairs": 260, "max_unless": 40, "null_reps": 1, "min_n": 150, "half_life_years": 50}


def panel(weeks=260, stocks=120, seed=0, regime_effect=0.02, n_feat=8):
    """Weekly rows. y: week shock + noise + an effect for top-quintile f0 that exists ONLY while m_vix is high."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=weeks * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(stocks)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), n_feat)), index=idx, columns=[f"f{i}" for i in range(n_feat)])
    reg = np.zeros(weeks)
    for t in range(1, weeks):
        reg[t] = 0.85 * reg[t - 1] + np.sqrt(1 - 0.85 ** 2) * rng.standard_normal()
    d = idx.get_level_values(0)
    X["m_vix"] = pd.Series(reg, index=dates).reindex(d).values
    X["m_vix_term"] = pd.Series(rng.standard_normal(weeks), index=dates).reindex(d).values
    top = X["f0"].groupby(level=0).rank(pct=True).values >= 0.8
    hi = X["m_vix"].values > np.quantile(reg, 0.8)
    y = pd.Series(rng.normal(0, 0.05, len(idx)) + pd.Series(rng.normal(0, 0.02, weeks), index=dates).reindex(d).values
                  + regime_effect * (top & hi), index=idx)
    y = y - y.groupby(level=0).transform("mean")
    return X, y


@pytest.fixture(scope="module")
def with_ctx():
    X, y = panel()
    m = PatternMiner({**FAST, "context_candidates": True}).fit(X, y, X.index.get_level_values(0).max())
    return X, y, m


def active(m):
    P = m.patterns
    return P[P["status"].isin(["active", "rescoped"])]


# ------------------------------------------------------------------ public API is unchanged
def test_patterns_frame_keeps_its_columns_statuses_and_keys(with_ctx):
    X, y, m = with_ctx
    P = m.patterns
    for c in ("key", "key_named", "status", "effect", "p_real", "p_coincidence", "p_hallucinated", "fdr_pass", "scope",
              "duplicate_of", "t_disc", "t_conf", "m_all", "n_eff", "confirmed"):
        assert c in P.columns
    assert set(P["status"]) <= {"active", "rescoped", "no_gain", "duplicate", "discarded", "rejected"}
    assert P["key"].map(lambda k: k[0] in "spu").all() and P["key_named"].is_unique and P["pattern_id"].is_unique
    for k in ("tested", "fdr_pass", "gate_corr_confirm", "null_t_95pct", "real_t_95pct"):
        assert k in m.report
    assert 0.0 <= m.gate_corr <= 1.0 and m.null_summary["null_patterns"] > 0


def test_export_and_export_bank_round_trip_through_key_from_names(with_ctx):
    X, y, m = with_ctx
    ex, bank = m.export(), m.export_bank()
    assert len(ex) == len(bank) == len(active(m)) and set(bank.columns) == {"names", "effect", "p_real"}
    for names, key in zip(bank["names"], ex["key"]):
        assert m.key_from_names(tuple(names), m.all_feats) == key
    assert m.key_from_names(("s", "nonexistent", 1), m.all_feats) is None


def test_key_named_is_the_canonical_expression_and_pattern_id_its_hash(with_ctx):
    X, y, m = with_ctx
    for r in m.patterns.head(80).itertuples():
        e = I.Expression.parse(r.key_named)
        assert e.text == r.key_named and I.pattern_id(e, m.book.get(r.pattern_id).transform, m.book.get(r.pattern_id).target) == r.pattern_id
        assert m._name(r.key) == r.key_named


def test_fit_is_deterministic_and_seeded():
    X, y = panel(weeks=120, stocks=60, seed=1)
    now = X.index.get_level_values(0).max()
    a = PatternMiner(FAST).fit(X, y, now)
    b = PatternMiner(FAST).fit(X, y, now)
    pd.testing.assert_frame_equal(a.patterns.drop(columns=["scope"]), b.patterns.drop(columns=["scope"]))
    c = PatternMiner({**FAST, "seed": 8}).fit(X, y, now)
    assert set(c.patterns["key_named"]) != set(a.patterns["key_named"])


def test_degenerate_inputs_return_an_empty_miner_not_an_error():
    X, y = panel(weeks=4, stocks=10, seed=2)
    m = PatternMiner(FAST).fit(X, y, X.index.get_level_values(0).max())
    assert m.patterns.empty and m.report["tested"] == 0
    assert (m.score(X.xs(X.index.get_level_values(0)[0], level=0)) == 0).all()
    X2, y2 = panel(weeks=60, stocks=40, seed=2)
    none = PatternMiner(FAST).fit(X2, y2 * np.nan, X2.index.get_level_values(0).max())
    assert none.patterns.empty
    late = PatternMiner(FAST).fit(X2, y2, pd.Timestamp("1990-01-01"))       # every row postdates `now`: nothing is evidence
    assert late.patterns.empty and late.report["rows_after_now_dropped"] == len(X2)


# ------------------------------------------------------------------ market-context candidates
def test_context_candidates_find_a_regime_pattern_that_cross_sectional_ranks_cannot(with_ctx):
    X, y, m = with_ctx
    A = active(m)
    hit = A[A["key_named"].str.contains("f0 q4") & A["key_named"].str.contains("m_vix q")]
    assert len(hit) and (hit["effect"] > 0).all(), m.report
    assert "m_vix" in m.ctx_feats and all(f not in m.feats for f in m.ctx_feats)
    off = PatternMiner(FAST).fit(X, y, X.index.get_level_values(0).max())          # default: context candidates off
    assert off.ctx_feats == [] and not off.patterns["key_named"].str.contains("m_").any()


def test_score_uses_the_time_series_level_of_todays_context(with_ctx):
    X, y, m = with_ctx
    d = X.index.get_level_values(0)
    per = X["m_vix"].groupby(level=0).first()
    hi_day, lo_day = per.idxmax(), per.idxmin()
    gaps = []
    for day in (hi_day, lo_day):
        Xd = X.xs(day, level=0)
        s = m.score(Xd)
        top = Xd["f0"].rank(pct=True) >= 0.8
        gaps.append(s[top].mean() - s[~top].mean())
    assert gaps[0] > gaps[1] + 0.001                                   # the regime pattern fires in the high-vix week only


def test_quantile_matrix_reproduces_the_fit_quantiles_and_extends_point_in_time(with_ctx):
    X, y, m = with_ctx
    Q = m.quantile_matrix(X)
    assert Q.shape == (len(X), len(m.all_feats))
    uni = C.Universe(tuple(m.feats), (), tuple(m.ctx_feats))
    ref = C.quantise(X, uni, min_history=max(10, min(60, X.index.get_level_values(0).nunique() // 4)))
    assert (Q == ref.Q).all()
    # a later panel is ranked against fit history plus its own earlier dates only
    later = X.copy()
    later.index = later.index.set_levels(later.index.levels[0] + pd.Timedelta(days=4000), level=0)
    Q2 = m.quantile_matrix(later)
    j = m.all_feats.index("m_vix")
    assert Q2[:, j].min() >= 0 and set(np.unique(Q2[:, j])) <= set(range(5))


def test_unknown_context_level_never_fires_a_pattern(with_ctx):
    X, y, m = with_ctx
    day = X.index.get_level_values(0)[0]
    Xd = X.xs(day, level=0).copy()
    Xd["m_vix"] = np.nan
    s = m.score(Xd)
    assert np.isfinite(s.values).all()


# ------------------------------------------------------------------ sparse flags and candles
def test_sparse_flags_do_not_generate_untestable_candidates():
    X, y = panel(weeks=120, stocks=60, seed=3)
    X["flag"] = (np.random.default_rng(3).random(len(X)) < 0.02).astype(float)
    m = PatternMiner({**FAST, "min_n": 100}).fit(X, y, X.index.get_level_values(0).max())
    flag_rows = m.patterns[m.patterns["key_named"].str.contains(r"\bflag q")]
    assert set(flag_rows["key_named"].str.extract(r"flag q(\d)")[0]) <= {"2", "4"}      # only the two occupied levels


def test_candle_signals_reach_the_miner_when_merged_and_are_classified():
    rng = np.random.default_rng(4)
    n, k = 160, 30
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = [f"T{i:02d}" for i in range(k)]
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
    o = c * np.exp(rng.normal(0, 0.01, (n, k)))
    h, l = np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99
    stocks = {nm: pd.DataFrame(v, index=idx, columns=cols) for nm, v in (("Open", o), ("High", h), ("Low", l), ("Close", c))}
    base = pd.DataFrame({"r5": rng.normal(size=n * k)},
                        index=pd.MultiIndex.from_product([idx, cols], names=["date", "ticker"]))
    X = with_candles(base, stocks)
    assert set(C.CANDLE_SIGNALS) <= set(X.columns) and len(X.columns) == 1 + len(C.CANDLE_SIGNALS)
    y = pd.Series(rng.normal(0, 0.05, len(X)), index=X.index)
    m = PatternMiner({**FAST, "min_n": 100, "hac_lags": 0}).fit(X.iloc[k * 30:], y.iloc[k * 30:], idx[-1])
    assert set(C.CANDLE_SIGNALS) & set(m.feats) == set(C.CANDLE_SIGNALS)              # all 25 are miner features


# ------------------------------------------------------------------ persistence and priors
def test_book_holds_every_pattern_and_survives_json(with_ctx, tmp_path):
    X, y, m = with_ctx
    assert len(m.book) == len(m.patterns) and len(m.export_records()) == len(m.patterns)
    f = tmp_path / "book.json"
    m.save_book(f)
    again = I.PatternBook.from_json(f.read_text(encoding="utf-8"))
    assert again.digest() == m.book.digest()
    assert m.book.counts() == m.patterns["status"].value_counts().sort_index().to_dict()


def test_prior_from_book_is_retested_by_the_next_window(with_ctx):
    X, y, m = with_ctx
    prior = PatternMiner.prior_from_book(m.book.to_json())
    assert len(prior) == len(active(m)) >= 1
    later = PatternMiner({**FAST, "max_pairs": 10, "max_unless": 0, "context_candidates": True}).fit(
        X, y, X.index.get_level_values(0).max(), prior=m.book)
    have = set(later.patterns["key_named"])
    for r in active(m).itertuples():
        assert r.key_named in have                                      # a banked pattern is always re-tested
    assert (later.patterns.loc[later.patterns["origin"] == "bank", "key_named"].map(lambda t: I.Expression.parse(t).order)).min() >= 1


def test_bank_row_with_a_vanished_feature_is_skipped_not_fatal():
    X, y = panel(weeks=100, stocks=50, seed=5)
    prior = pd.DataFrame({"names": [["s", "f0", 4], ["s", "feature_that_left", 2]], "effect": [0.01, 0.01], "p_real": [0.9, 0.9]})
    m = PatternMiner(FAST).fit(X, y, X.index.get_level_values(0).max(), prior=prior)
    assert m.report["bank_skipped"] == 1 and "f0 q4" in set(m.patterns["key_named"])


# ------------------------------------------------------------------ the null replicates the search's selection
def test_full_search_null_is_no_more_permissive_than_the_masks_only_null_on_pure_noise():
    """PLANTED DEFECT guard: pairs and exceptions are picked from the strongest singles, so a null that reuses the real
    masks understates how strong a *selected* pattern gets by chance. The full-search null must admit no more."""
    admitted = _noise_admissions()
    assert admitted["full"] <= admitted["masks"]


def _noise_admissions(reps=2):
    admitted = {"full": 0, "masks": 0}
    for seed in range(6, 12):
        X, y = panel(weeks=200, stocks=80, seed=seed, regime_effect=0.0)
        now = X.index.get_level_values(0).max()
        for mode in admitted:
            m = PatternMiner({**FAST, "null_reps": reps, "null_search": mode, "p_method": "bh"}).fit(X, y, now)
            admitted[mode] += len(active(m))
    return admitted


def test_noise_false_admissions_within_budget():
    """Was a strict xfail (2026-09-28: 4 false admissions over these 6 panels). F25 fix: the FDR budget is spent on the
    confirmation block (engine.patterns._confirmation_fdr). 240 fresh noise panels: 31 -> 1 false admission."""
    assert _noise_admissions()["full"] <= 2


# ------------------------------------------------------------------ F25: confirmation-stage false-discovery control
def _frame(t_disc, t_conf, ph, same_sign=None):
    t_disc, t_conf = np.asarray(t_disc, float), np.asarray(t_conf, float)
    same = np.ones(len(t_disc), bool) if same_sign is None else np.asarray(same_sign, bool)
    from engine import pattern_stats as S
    return pd.DataFrame({"t_disc": t_disc, "t_conf": t_conf, "m_disc": np.sign(t_disc) * 0.01,
                         "m_conf": np.where(same, np.sign(t_disc), -np.sign(t_disc)) * 0.01,
                         "p_coincidence": S.t_to_p(t_disc), "p_hallucinated": np.asarray(ph, float)})


def test_confirmation_fdr_refuses_a_screened_candidate_that_confirms_only_weakly():
    """PLANTED DEFECT (the pre-F25 admission rule): discovery t = 6 with P(hallucinated) = 0 and a confirmation t of 1.05
    gave P(real) = (1 - q) x Phi(1.05) ~ 0.85 >= 0.8 -> admitted on a one-sided confirmation p of 0.15."""
    from engine.patterns import MINER_DEFAULT, _confirmation_fdr
    from engine import pattern_stats as S
    R = _frame([6.0] + [0.3] * 50, [1.05] + [0.2] * 50, [0.0] + [1.0] * 50)
    p = {**MINER_DEFAULT}
    real = S.p_real(R["p_hallucinated"].values, R["p_coincidence"].values,
                    S.confirmation_factor(R["t_conf"].abs().values, R["m_conf"].values, R["m_disc"].values), "bh")
    assert real[0] >= p["p_real_min"]                                   # the old rule would admit it
    m = PatternMiner()
    ok = _confirmation_fdr(R, p, m)
    assert not ok[0] and R["screened"].iloc[0] and m.confirm_summary == {"screened": 1, "confirm_q": 0.05,
                                                                         "confirm_pass": 0, "confirm_fdr": True}
    R2 = _frame([6.0] + [0.3] * 50, [3.5] + [0.2] * 50, [0.0] + [1.0] * 50)
    assert _confirmation_fdr(R2, p, m)[0]                               # a real confirmation still passes


def test_confirmation_fdr_counts_the_whole_screened_family_and_needs_the_sign_to_repeat():
    from engine.patterns import MINER_DEFAULT, _confirmation_fdr
    p = {**MINER_DEFAULT}
    # 20 screened candidates, one confirms at t = 2.2 (one-sided p 0.014): alone it passes, among 20 nulls it does not
    alone = _frame([7.0], [2.2], [0.0])
    assert _confirmation_fdr(alone, p, PatternMiner())[0]
    fam = _frame([7.0] * 20, [2.2] + [0.1] * 19, [0.0] * 20)
    assert not _confirmation_fdr(fam, p, PatternMiner()).any()
    flipped = _frame([7.0], [5.0], [0.0], same_sign=[False])           # strong, but the sign reversed out of sample
    assert not _confirmation_fdr(flipped, p, PatternMiner())[0] and flipped["p_confirm"].iloc[0] == 1.0
    unscreened = _frame([1.0], [9.0], [0.0])                             # never screened on discovery: never admitted
    assert not _confirmation_fdr(unscreened, p, PatternMiner())[0]
    assert _confirmation_fdr(_frame([7.0], [2.2], [0.0]), {**p, "confirm_q": 0.001}, PatternMiner()).sum() == 0


def test_confirmation_fdr_on_an_empty_frame_and_the_legacy_switch():
    from engine.patterns import MINER_DEFAULT, _confirmation_fdr
    m = PatternMiner()
    empty = _frame([], [], [])
    assert len(_confirmation_fdr(empty, {**MINER_DEFAULT}, m)) == 0 and m.confirm_summary["screened"] == 0
    weak = _frame([6.0], [1.05], [0.0])
    assert _confirmation_fdr(weak, {**MINER_DEFAULT, "confirm_fdr": False}, m)[0]      # legacy: measurement only


def test_fit_reports_the_confirmation_family_and_every_admitted_pattern_passed_it(with_ctx):
    X, y, m = with_ctx
    for k in ("screened", "confirm_q", "confirm_pass", "confirm_fdr"):
        assert k in m.report
    A = active(m)
    assert len(A) and A["confirm_fdr_pass"].all() and A["screened"].all()
    assert m.report["confirm_pass"] <= m.report["screened"] <= m.report["tested"]


def test_legacy_admission_on_the_known_bad_noise_panels_is_what_the_fix_removes():
    """The six panels the xfail was recorded on: the legacy rule still over-admits there (so this test can fail if the
    fix is ever bypassed), the fixed rule stays inside the budget."""
    legacy = 0
    for seed in range(6, 12):
        X, y = panel(weeks=200, stocks=80, seed=seed, regime_effect=0.0)
        mm = PatternMiner({**FAST, "null_reps": 2, "p_method": "bh", "confirm_fdr": False}).fit(
            X, y, X.index.get_level_values(0).max())
        legacy += len(active(mm))
    assert legacy > 2


# ------------------------------------------------------------------ scope indices and missing context columns
def test_scope_is_evaluated_against_the_context_columns_present_at_fit_time():
    """PLANTED DEFECT: with m_vix_term missing at fit time, scope index 1 means m_breadth, not CTX[1]=m_vix_term."""
    m = PatternMiner()
    m.feats, m.ctx_feats, m.ctx_names = ["f0", "f1"], [], ["m_vix", "m_breadth"]
    m.patterns = pd.DataFrame([{"key": ("s", 0, 4), "key_named": "f0 q4", "status": "rescoped", "effect": 1.0,
                                "scope": (1, "high", 1.0, 2.0)}])
    Xd = pd.DataFrame({"f0": np.arange(10.0), "f1": np.zeros(10), "m_vix": 0.0, "m_vix_term": -100.0, "m_breadth": 5.0},
                      index=[f"T{i}" for i in range(10)])
    assert m.score(Xd).sum() == 3.0                                   # the three top-quintile stocks fire: breadth 5 > 2
    Xd["m_breadth"] = 0.0
    Xd["m_vix_term"] = 100.0                                          # CTX[1] would now (wrongly) be in scope
    assert m.score(Xd).sum() == 0.0
    del Xd["m_breadth"]
    assert m.score(Xd).sum() == 0.0                                   # a scope column absent today never fires (fail closed)


def test_fit_with_a_missing_context_column_records_the_columns_it_used():
    X, y = panel(weeks=120, stocks=60, seed=7)
    assert "m_spy_ma200" not in X and "m_vix_term" in X
    m = PatternMiner(FAST).fit(X, y, X.index.get_level_values(0).max())
    assert m.ctx_names == ["m_vix", "m_vix_term"] and [c for c in CTX if c in X] == m.ctx_names
    s = m.score(X.xs(X.index.get_level_values(0)[-1], level=0))
    assert np.isfinite(s.values).all()


def test_tuning_parameters_are_honoured():
    """A7 (engine/miner_tuning) drives these five; each must change the fit."""
    X, y = panel(weeks=140, stocks=60, seed=8)
    now = X.index.get_level_values(0).max()
    for name, value in (("half_life_years", 1.0), ("ctx_bandwidth", 0.3), ("shrink_k", 4000), ("min_n", 900), ("fdr_q", 0.5)):
        # shrink_k only drives the blueprint "k" shrinkage; under the default empirical-Bayes rule it is inert by design
        extra = {"effect_method": "k"} if name == "shrink_k" else {}
        base = PatternMiner({**FAST, **extra}).fit(X, y, now)
        alt = PatternMiner({**FAST, **extra, name: value}).fit(X, y, now)
        cols = ["effect", "n_eff", "t_disc", "fdr_pass"] if name in ("shrink_k", "fdr_q", "half_life_years", "ctx_bandwidth") else ["key_named"]
        same = len(alt.patterns) == len(base.patterns) and all(
            alt.patterns[c].reset_index(drop=True).equals(base.patterns[c].reset_index(drop=True)) for c in cols)
        assert not same, f"{name} had no effect"
