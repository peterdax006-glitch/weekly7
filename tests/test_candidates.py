"""Bible Phase 3.1 (canons C35, C37): candidate generation and the coverage audit.

The audit is only worth having if it can fail, so several tests plant a hole (a feature with no singles, a wrong
universe size, a candle name the builder does not produce, a flag that fills two quintiles) and require it be named."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import candidates as C
from engine import candles
from engine import pattern_identity as I
from engine.patterns import PatternMiner

ROOT = Path(__file__).resolve().parent.parent
TINY = C.Universe(("e1", "e2", "e3", "e4"), ("c1", "c2"), ("m_a", "m_b"))


def E(t):
    return I.Expression.parse(t)


# ------------------------------------------------------------------ the universe
def test_universe_matches_the_blueprint_counts_and_has_no_overlaps():
    u = C.Universe()
    assert len(u.engine) == 46 and len(u.context) == 8 and len(u.engine) + len(u.context) == 54
    assert len(u.candle) == 25 and len(u.all) == 79 and u.all == tuple(sorted(u.all))
    assert set(u.engine).isdisjoint(u.candle) and set(u.engine).isdisjoint(u.context)
    assert all(f.startswith("m_") for f in u.context) and not any(f.startswith("m_") for f in u.engine + u.candle)


def test_candle_names_equal_what_candles_build_really_produces():
    rng = np.random.default_rng(0)
    n, k = 120, 3
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
    o, h, l = c * 1.001, c * 1.02, c * 0.98
    cols = list("ABC")
    bars = {nm: pd.DataFrame(v, index=idx, columns=cols) for nm, v in (("Open", o), ("High", h), ("Low", l), ("Close", c))}
    assert set(candles.build(bars)) == set(C.CANDLE_SIGNALS)


def test_universe_rejects_duplicates_and_unwritable_names():
    with pytest.raises(I.IdentityError, match="two groups"):
        C.Universe(("a",), ("a",), ())
    for bad in ("has space", "a&b", ""):
        with pytest.raises(I.IdentityError):
            C.Universe((bad,), (), ())
    assert TINY.group_of("c2") == "candle" and TINY.is_context("m_a") and not TINY.is_context("e1")
    with pytest.raises(KeyError):
        TINY.group_of("zzz")


def test_universe_from_columns_classifies_and_restrict_drops_absent():
    u = C.universe_from_columns(["date", "ticker", "r5", "gap", "m_vix", "streak", "vol20"], C.CANDLE_SIGNALS)
    assert u.engine == ("r5", "vol20") and u.candle == ("gap", "streak") and u.context == ("m_vix",)
    assert TINY.restrict(["e1", "c1", "m_b", "other"]).all == ("c1", "e1", "m_b")


def test_transform_is_time_series_when_any_term_is_market_context():
    assert TINY.transform_of(E("e1 q1 & e2 q2")) == "xs_quintile"
    assert TINY.transform_of(E("e1 q1 & m_a q2")) == "ts_quintile"
    assert TINY.transform_of(E("e1 q1 & e2 q0 unless m_b q4")) == "ts_quintile"


# ------------------------------------------------------------------ singles, pairs, unless, bank
def test_singles_cover_every_feature_at_every_level_exactly_once():
    s = C.single_candidates(C.Universe())
    assert len(s) == 79 * 5 and len({c.id for c in s}) == 395
    assert {c.origin for c in s} == {"all_single"} and {c.family for c in s} == {"single"}
    assert {c.transform for c in s if c.text.startswith("m_")} == {"ts_quintile"}
    assert {c.transform for c in s if not c.text.startswith("m_")} == {"xs_quintile"}


def test_random_pairs_are_distinct_cross_feature_and_exactly_the_requested_count():
    rng = np.random.default_rng(1)
    got = C.random_pair_candidates(C.Universe(), 3000, rng)
    assert len(got) == 3000 and len({c.id for c in got}) == 3000
    assert all(len(c.expression.base) == 2 and not c.expression.unless for c in got)
    assert all(len(c.expression.features) == 2 for c in got)                    # never two levels of one feature
    excl = {c.id for c in got[:500]}
    more = C.random_pair_candidates(C.Universe(), 500, np.random.default_rng(1), exclude=excl)
    assert excl.isdisjoint(c.id for c in more) and len(more) == 500


def test_random_pairs_exhaust_a_small_space_instead_of_looping_or_short_changing_silently():
    space = len(TINY.all) * (len(TINY.all) - 1) // 2 * 25
    got = C.random_pair_candidates(TINY, 10 ** 6, np.random.default_rng(2))
    assert len(got) == space == 28 * 25
    assert len({c.id for c in got}) == space
    assert C.random_pair_candidates(TINY, 0, np.random.default_rng(2)) == []
    assert C.random_pair_candidates(C.Universe((), (), ()), 5, np.random.default_rng(2)) == []


def test_random_pairs_are_seeded():
    a = [c.id for c in C.random_pair_candidates(C.Universe(), 200, np.random.default_rng(5))]
    b = [c.id for c in C.random_pair_candidates(C.Universe(), 200, np.random.default_rng(5))]
    c = [c.id for c in C.random_pair_candidates(C.Universe(), 200, np.random.default_rng(6))]
    assert a == b and a != c


def test_top_terms_break_ties_by_text_and_pairs_skip_same_feature():
    scores = {"e2 q1": 3.0, "e1 q4": 3.0, "e1 q0": 2.0, "e3 q2": -5.0, "e4 q3": 0.1}
    top = C.top_terms(scores, 4)
    assert [str(t) for t in top] == ["e3 q2", "e1 q4", "e2 q1", "e1 q0"]
    pairs = C.top_pair_candidates(top, TINY)
    texts = {p.text for p in pairs}
    assert len(pairs) == 5 and "e1 q0 & e1 q4" not in texts and "e1 q4 & e3 q2" in texts     # 6 combos, one same-feature
    assert {p.origin for p in pairs} == {"top_pair"}


def test_unless_third_feature_is_new_and_the_cap_is_honoured():
    pairs = [E("e1 q4 & e2 q0"), E("e3 q1 & e4 q1"), E("e1 q1 & c1 q2 unless e3 q0")]      # the last is not a plain pair
    got = C.unless_candidates(pairs, TINY, np.random.default_rng(3), top_pairs=10, thirds=6, levels=(0, 4), max_total=100)
    assert got and all(len(c.expression.unless) == 1 and len(c.expression.base) == 2 for c in got)
    for c in got:
        third = c.expression.unless[0].feature
        assert third not in {t.feature for t in c.expression.base} and c.expression.unless[0].level in (0, 4)
    assert not any("c1 q2" in c.text for c in got)                                       # non-pair rows are skipped
    capped = C.unless_candidates(pairs[:2], TINY, np.random.default_rng(3), thirds=6, max_total=3)
    assert len(capped) == 3


def test_bank_candidates_parse_all_shapes_and_report_what_they_skip():
    prior = pd.DataFrame({"names": [["s", "e1", 2], ["p", "e1", 1, "c1", 3], ["u", "e1", 1, "e2", 3, "m_a", 0],
                                    ["s", "gone", 1], ["p", "e1", 9, "e2", 1], ["x", "e1", 1], ["s"]]})
    cands, skipped = C.bank_candidates(prior, TINY)
    assert [c.text for c in cands] == ["e1 q2", "c1 q3 & e1 q1", "e1 q1 & e2 q3 unless m_a q0"]
    assert {c.origin for c in cands} == {"bank"} and cands[2].transform == "ts_quintile"
    assert len(skipped) == 4 and any("not in universe" in why for _, why in skipped)
    assert C.bank_candidates(None, TINY) == ([], []) and C.bank_candidates(pd.DataFrame(), TINY) == ([], [])


# ------------------------------------------------------------------ ids and sets
def test_candidate_id_is_its_identity_hash():
    c = C.single_candidates(TINY)[0]
    assert c.id == I.pattern_id(c.expression, c.transform, c.target) and c.text == c.expression.text
    m = [x for x in C.single_candidates(TINY) if x.text.startswith("m_a")][0]
    assert m.id == I.pattern_id(m.expression, "ts_quintile", "excess_5d") and m.id != I.pattern_id(m.expression)


def test_set_dedupes_by_id_and_records_who_collided():
    s = C.CandidateSet()
    single = C.single_candidates(TINY)[0]
    twin = C.Candidate(single.expression, "single", "bank", single.transform)
    assert s.add(single) and not s.add(twin) and len(s) == 1
    assert s.collisions == {("all_single", "bank"): 1} and single in s and single.id in s
    assert s.counts() == {"all_single": 1} and s.features_used() == {single.expression.features[0]}


def test_set_digest_ignores_generation_order():
    cs = C.single_candidates(TINY)
    a, b = C.CandidateSet(), C.CandidateSet()
    a.extend(cs)
    b.extend(reversed(cs))
    assert a.digest() == b.digest() and a.ids() != b.ids() and len(a) == len(TINY.all) * 5
    frame = a.frame()
    assert list(frame.columns) == ["id", "text", "family", "origin", "transform", "order"] and len(frame) == len(a)


def test_generation_is_reproducible_and_seed_dependent_only_where_it_should_be():
    g1, g2 = C.enumerate_static(C.Universe(), seed=7), C.enumerate_static(C.Universe(), seed=7)
    g3 = C.enumerate_static(C.Universe(), seed=8)
    assert g1.set.digest() == g2.set.digest() and g1.set.ids() == g2.set.ids()
    assert g1.set.digest() != g3.set.digest()
    singles = lambda g: {c.id for c in g.set if c.origin == "all_single"}
    assert singles(g1) == singles(g3)                                     # singles do not depend on the seed


def test_candidate_ids_do_not_depend_on_the_interpreter_hash_seed():
    """PLANTED DEFECT guard: python's builtin hash() is salted per process; ids must not use it."""
    code = ("import sys; sys.path.insert(0, %r);"
            "from engine import candidates as C;"
            "g = C.enumerate_static(C.Universe(), seed=3, params={'max_pairs': 300, 'max_unless': 50});"
            "print(g.set.digest())" % str(ROOT))
    out = []
    for hs in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        out.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
                                  cwd=str(ROOT)).stdout.strip())
    assert len(out[0]) == 64 and out[0] == out[1]


def test_staged_generator_uses_scores_to_pick_pairs_and_bank_rows_collide_with_singles():
    g = C.CandidateGenerator(TINY, {"seed": 1, "max_pairs": 60, "top_singles": 4, "max_unless": 10,
                                    "unless_top_pairs": 3, "unless_thirds": 3})
    singles = g.stage_singles()
    scores = {c.text: (10.0 if c.text in ("e1 q4", "c1 q0", "e2 q2") else 0.5) for c in singles}
    pairs = g.stage_pairs(scores)
    top = [p for p in pairs if p.origin == "top_pair"]
    assert {"c1 q0 & e1 q4", "e1 q4 & e2 q2", "c1 q0 & e2 q2"} <= {p.text for p in top}
    assert len(pairs) == 60 and len({p.id for p in pairs}) == 60
    un = g.stage_unless({p.text: float(i) for i, p in enumerate(pairs)})
    assert 0 < len(un) <= 10
    bank = g.stage_bank(pd.DataFrame({"names": [["s", "e1", 4], ["s", "nope", 1]]}))
    assert len(bank) == 1 and g.set.collisions[("all_single", "bank")] == 1 and len(g.skipped_bank) == 1
    assert len(g.set) == len(singles) + len(pairs) + len(un)


def test_generator_with_no_scores_falls_back_to_random_pairs_only():
    g = C.CandidateGenerator(TINY, {"max_pairs": 50})
    g.stage_singles()
    pairs = g.stage_pairs({})
    assert len(pairs) == 50 and {p.origin for p in pairs} == {"random_pair"}
    assert g.stage_unless({}) == [] and g.stage_bank(None) == []


def test_plan_size_arithmetic_and_pair_space_cap():
    p = C.plan_size(C.Universe())
    assert p["singles"] == 395 and p["pairs"] == 4000 and p["unless"] == 600 and p["total"] == 4995
    assert p["pair_space"] == (79 * 78 // 2) * 25
    small = C.plan_size(TINY, n_bank=7)
    assert small["pairs"] == small["pair_space"] < 4000 and small["bank"] == 7
    assert C.plan_size(TINY, {"max_unless": 10})["unless"] == 10


# ------------------------------------------------------------------ quantile transforms
def panel(n_dates=90, n_tick=40, seed=0, cols=("e1", "e2", "m_a")):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"T{i:02d}" for i in range(n_tick)]], names=["date", "ticker"])
    X = pd.DataFrame({c: rng.normal(size=len(idx)) for c in cols if not c.startswith("m_")}, index=idx)
    for c in cols:
        if c.startswith("m_"):
            X[c] = pd.Series(rng.normal(size=n_dates), index=dates).repeat(n_tick).values
    return X


def test_xs_quintiles_equal_the_miners_on_data_with_gaps():
    X = panel(cols=("e1", "e2"))
    X.iloc[::17, 0] = np.nan
    ours = C.xs_quintile_codes(X, ["e1", "e2"])
    theirs = PatternMiner()._quintiles(X[["e1", "e2"]])
    assert (ours == theirs).all() and set(np.unique(ours)) == {0, 1, 2, 3, 4}


def test_ts_quintiles_are_point_in_time_and_warm_up_is_unknown():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=300), index=pd.bdate_range("2020-01-01", periods=300))
    q = C.ts_quintile_series(s, min_history=60)
    assert (q.iloc[:59] == -1).all() and (q.iloc[59:] >= 0).all()
    q_short = C.ts_quintile_series(s.iloc[:200], min_history=60)
    assert (q_short.values == q.iloc[:200].values).all()                       # later data cannot move an earlier level
    tampered = s.copy()
    tampered.iloc[250:] += 100.0
    assert (C.ts_quintile_series(tampered, 60).iloc[:250].values == q.iloc[:250].values).all()


def test_ts_quintiles_on_known_shapes():
    up = pd.Series(np.arange(200.0), index=pd.bdate_range("2020-01-01", periods=200))
    assert (C.ts_quintile_series(up, 30).iloc[30:] == 4).all()                  # always the highest so far
    down = pd.Series(-np.arange(200.0), index=up.index)
    assert (C.ts_quintile_series(down, 30).iloc[30:] == 0).all()
    const = pd.Series(1.0, index=up.index)
    assert (C.ts_quintile_series(const, 30).iloc[30:] == 2).all()               # ties take the mid-rank, not the top
    holes = up.copy()
    holes.iloc[70:75] = np.nan
    assert (C.ts_quintile_series(holes, 30).iloc[70:75] == -1).all()
    assert (C.ts_quintile_series(up.iloc[:0], 30)).empty


def test_quantise_builds_masks_that_match_the_miner_and_reports_missing_features():
    X = panel(cols=("e1", "e2", "c1", "m_a"))
    uni = C.Universe(("e1", "e2", "ghost"), ("c1",), ("m_a",))
    qm = C.quantise(X, uni, min_history=20)
    assert qm.missing == ("ghost",) and set(qm.features) == {"e1", "e2", "c1", "m_a"}
    miner = PatternMiner()
    Qm = miner._quintiles(X[["c1", "e1", "e2"]])                                   # the miner's own matrix, its column order
    for text, key in (("e1 q4", ("s", 1, 4)), ("c1 q0 & e2 q3", ("p", 0, 0, 2, 3)), ("e1 q1 & e2 q2 unless c1 q4", ("u", 1, 1, 2, 2, 0, 4))):
        assert (qm.mask(E(text)) == miner._mask_from_key(key, Qm, ["c1", "e1", "e2"])).all()
    dates = X.index.get_level_values(0)
    unknown = qm.mask(E("m_a q0 unless e1 q0")) | qm.mask(E("m_a q1"))
    early = dates < dates.unique()[19]
    assert not unknown[early].any()                                                # warm-up rows match no context level
    with pytest.raises(I.IdentityError):
        C.quantise(X.reset_index(), uni)


def test_context_quantile_is_constant_across_stocks_within_a_date():
    X = panel(cols=("e1", "m_a"))
    qm = C.quantise(X, C.Universe(("e1",), (), ("m_a",)), min_history=10)
    j = qm.features.index("m_a")
    per_date = pd.Series(qm.Q[:, j]).groupby(X.index.get_level_values(0).values).nunique()
    assert (per_date == 1).all()
    assert len(set(qm.Q[:, j]) - {-1}) == 5                                       # a cross-sectional rank would give one level


def test_occupancy_flags_a_flag_that_fills_only_two_quintiles():
    """PLANTED DEFECT: an event flag that is 1 for 3% of rows cannot host five single candidates."""
    X = panel(cols=("e1",))
    X["flag"] = (np.random.default_rng(4).random(len(X)) < 0.03).astype(float)
    qm = C.quantise(X, C.Universe(("e1", "flag"), (), ()))
    occ = qm.occupancy().set_index("feature")
    assert occ.loc["e1", "levels_occupied"] == 5 and occ.loc["flag", "levels_occupied"] == 2
    empty = C.empty_levels(qm, min_rows=1)
    assert {(f, q) for f, q in empty if f == "flag"} == {("flag", 0), ("flag", 1), ("flag", 3)} and not any(f == "e1" for f, _ in empty)
    cset = C.CandidateSet()
    cset.extend(C.single_candidates(C.Universe(("e1", "flag"), (), ())))
    kept, dropped = C.prune_unoccupied(cset, empty)
    assert len(dropped) == 3 and len(kept) == 7 and all("flag" in d.text for d in dropped)


# ------------------------------------------------------------------ the coverage audit
def audit_of(uni, **kw):
    g = C.enumerate_static(uni, seed=7, params={"max_pairs": 1500, "max_unless": 60})
    return g, C.coverage_audit(uni, g.set, **kw)


def test_audit_passes_on_the_full_universe_and_every_feature_is_reachable():
    g, rep = audit_of(C.Universe(), expected={"engine": 46, "context": 8, "candle": 25})
    assert rep["ok"] and rep["problems"] == [] and rep["total_features"] == 79
    T = rep["table"]
    assert (T["single_levels"] == 5).all() and (T["in_pairs"] > 0).all() and rep["min_pairs_per_feature"] >= 1
    assert rep["sizes"] == {"engine": 46, "context": 8, "candle": 25} and rep["n_candidates"] == len(g.set)
    assert '"ok": true' in C.audit_json(rep)


def test_audit_names_a_feature_whose_singles_were_removed():
    uni = C.Universe()
    g = C.enumerate_static(uni, seed=7, params={"max_pairs": 400})
    broken = C.CandidateSet()
    broken.extend(c for c in g.set if not (c.family == "single" and c.expression.features == ("r5",) and c.expression.base[0].level > 2))
    rep = C.coverage_audit(uni, broken)
    assert not rep["ok"] and any(p.startswith("r5: only 3 of 5") for p in rep["problems"])


def test_audit_names_a_feature_that_never_reaches_a_pair():
    uni = C.Universe()
    g = C.enumerate_static(uni, seed=7, params={"max_pairs": 400})
    broken = C.CandidateSet()
    broken.extend(c for c in g.set if not (c.family in ("pair", "unless") and "m_vix" in c.expression.features))
    rep = C.coverage_audit(uni, broken)
    assert any(p.startswith("m_vix: never appears in a pair") for p in rep["problems"])


def test_audit_catches_wrong_sizes_and_disagreement_with_the_real_sources():
    uni = C.Universe(C.ENGINE_FEATURES[:-1], C.CANDLE_SIGNALS, C.MARKET_CONTEXT)
    _, rep = audit_of(uni, expected={"engine": 46, "context": 8, "candle": 25})
    assert any("45 engine features, expected 46" in p for p in rep["problems"])
    panel_cols = list(C.ENGINE_FEATURES) + list(C.MARKET_CONTEXT) + ["date", "ticker", "surprise_col"]
    _, rep = audit_of(C.Universe(), panel_columns=panel_cols[:-2] + ["surprise_col"], candle_names=C.CANDLE_SIGNALS[:-1] + ("mystery",))
    text = " | ".join(rep["problems"])
    assert "surprise_col" in text and "mystery" in text and "day_vs_week_body" in text and not rep["ok"]
    _, rep = audit_of(C.Universe(), panel_columns=[c for c in panel_cols if c != "r5" and c != "surprise_col"])
    assert any(p.startswith("r5: in the universe but not a column") for p in rep["problems"])


def test_audit_reports_what_the_miner_reaches_today():
    miner_feats = [f for f in C.ENGINE_FEATURES]                                   # non-m_ panel columns, no candles merged
    _, rep = audit_of(C.Universe(), miner_features=miner_feats)
    assert set(rep["unreached_by_miner_today"]) == set(C.CANDLE_SIGNALS) | set(C.MARKET_CONTEXT)
    _, rep2 = audit_of(C.Universe(), miner_features=miner_feats + list(C.CANDLE_SIGNALS))
    assert set(rep2["unreached_by_miner_today"]) == set(C.MARKET_CONTEXT)
    _, none = audit_of(C.Universe())
    assert none["unreached_by_miner_today"] == []


def test_a_starved_pair_budget_is_caught_by_the_audit_not_hidden():
    """PLANTED DEFECT: with 100 random pairs some features are never paired; the audit must say so."""
    g = C.enumerate_static(C.Universe(), seed=7, params={"max_pairs": 400, "max_unless": 60})
    rep = C.coverage_audit(C.Universe(), g.set)
    assert not rep["ok"] and rep["min_pairs_per_feature"] == 0 and any("never appears in a pair" in p for p in rep["problems"])


def test_top_pairs_cannot_crowd_out_the_random_draw():
    g = C.CandidateGenerator(C.Universe(), {"max_pairs": 800, "top_singles": 60})
    singles = g.stage_singles()
    pairs = g.stage_pairs({c.text: float(i % 97) for i, c in enumerate(singles)})
    rnd = [p for p in pairs if p.origin == "random_pair"]
    assert len(pairs) == 800 and len(rnd) >= 200


def test_every_seed_reaches_every_feature_in_a_pair():
    uni = C.Universe()
    for seed in range(6):
        g = C.enumerate_static(uni, seed=seed, params={"max_pairs": 1500, "max_unless": 100})
        rep = C.coverage_audit(uni, g.set)
        assert rep["ok"], (seed, rep["problems"][:3])
        assert rep["min_pairs_per_feature"] >= 1


def test_empty_candidate_set_fails_the_audit_loudly():
    rep = C.coverage_audit(TINY, C.CandidateSet())
    assert not rep["ok"] and len(rep["problems"]) == 2 * len(TINY.all) and rep["n_candidates"] == 0
    assert rep["min_pairs_per_feature"] == 0


# ------------------------------------------------------------------ context interactions
def test_context_pairs_cross_the_strongest_stock_terms_with_every_context_level():
    terms = [I.Term("e1", 4), I.Term("m_a", 2), I.Term("c1", 0)]
    got = C.context_pair_candidates(terms, TINY)
    assert len(got) == 2 * 2 * 5                                     # two stock terms x two context columns x five levels
    assert {c.origin for c in got} == {"context_pair"} and {c.transform for c in got} == {"ts_quintile"}
    assert all(sum(TINY.is_context(f) for f in c.expression.features) == 1 for c in got)
    assert C.context_pair_candidates(terms, C.Universe(("e1",), ("c1",), ())) == []


def test_generator_reserves_budget_for_context_pairs_and_random_pairs():
    u = C.Universe(tuple(f"e{i}" for i in range(20)), (), ("m_a", "m_b"))
    g = C.CandidateGenerator(u, {"max_pairs": 400, "top_singles": 30, "context_top": 10})
    singles = g.stage_singles()
    pairs = g.stage_pairs({c.text: float(i % 50) for i, c in enumerate(singles) if not c.text.startswith("m_")})
    by = {}
    for p in pairs:
        by[p.origin] = by.get(p.origin, 0) + 1
    assert len(pairs) == 400 and by["context_pair"] == 90 and by["random_pair"] >= 100 and by["top_pair"] > 0


def test_a_context_single_against_a_date_demeaned_outcome_has_exactly_zero_effect():
    """WHY the context-pair family exists: a per-date value cannot predict a per-date-demeaned return on its own."""
    X = panel(cols=("e1", "m_a"), n_dates=100, n_tick=30, seed=9)
    y = pd.Series(np.random.default_rng(9).normal(size=len(X)), index=X.index)
    y = y - y.groupby(level=0).transform("mean")
    qm = C.quantise(X, C.Universe(("e1",), (), ("m_a",)), min_history=20)
    for q in range(5):
        m = qm.mask(E(f"m_a q{q}"))
        assert abs(y[m].groupby(level=0).mean().mean()) < 1e-12
