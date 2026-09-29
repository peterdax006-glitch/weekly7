"""Bible Phase 3.2 (canons C35, C37, C43): pattern identity.

An id must mean one idea forever: order-independent, statistics-independent, tamper-evident. The bridge to
engine.patterns.PatternMiner is checked on a real fit, not on hand-written keys."""
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from engine import pattern_identity as I
from engine.patterns import PatternMiner

FEATS = ["f0", "f1", "f2", "f3", "f4", "f5"]
CTX = ["m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion"]


def E(text):
    return I.Expression.parse(text)


# ------------------------------------------------------------------ terms and canonical expressions
def test_term_validates_feature_and_level():
    assert str(I.Term("r5", 3)) == "r5 q3"
    assert I.Term.parse("  ev_shelf   q0 ") == I.Term("ev_shelf", 0)
    for bad in ("", "r5", "r5 q", "r5 q9", "q3", "r5 3", "r5 q-1"):
        with pytest.raises(I.IdentityError):
            I.Term.parse(bad)
    with pytest.raises(I.IdentityError):
        I.Term("", 1)
    with pytest.raises(I.IdentityError):
        I.Term("x", 5)
    with pytest.raises(I.IdentityError):
        I.Term("x", True)                                   # a bool is not a quantile level
    assert I.Term("x", np.int64(2)).level == 2 and type(I.Term("x", np.int64(2)).level) is int


def test_expression_is_order_independent_for_every_permutation():
    base = ["a q1", "b q2", "c q0"]
    texts = {" & ".join(p) for p in itertools.permutations(base)}
    assert len(texts) == 6
    canon = {E(t).text for t in texts}
    assert canon == {"a q1 & b q2 & c q0"}
    assert E("A q1 & B q2") == E("B q2 & A q1") and hash(E("A q1 & B q2")) == hash(E("B q2 & A q1"))


def test_unless_terms_are_canonical_too_and_distinct_from_the_base():
    a = E("x q4 & y q1 unless z q0 unless w q3")
    b = E("y q1 & x q4 unless w q3 unless z q0")
    assert a == b and a.text == "x q4 & y q1 unless w q3 unless z q0"
    assert E("x q4 & y q1 unless z q0") != E("x q4 & y q1 & z q0")             # 'not z0' is not 'z0'
    assert a.family == "unless" and E("x q1").family == "single" and E("x q1 & y q2").family == "pair"
    assert a.order == 4 and a.features == ("w", "x", "y", "z")


def test_duplicate_terms_collapse_and_contradictions_are_refused():
    assert E("a q1 & a q1 & b q2").text == "a q1 & b q2"
    with pytest.raises(I.IdentityError, match="contradiction"):
        E("a q1 & a q3")
    with pytest.raises(I.IdentityError, match="removes the whole"):
        E("a q1 & b q2 unless a q1")
    assert E("a q1 & b q2 unless a q4").text == "a q1 & b q2"                  # already implied: exception is redundant
    with pytest.raises(I.IdentityError):
        I.Expression.make([])
    with pytest.raises(I.IdentityError):
        E("")
    with pytest.raises(I.IdentityError):
        E("a q1 & ")


def test_parents_drop_one_term_at_a_time():
    p = E("a q1 & b q2 unless c q0").parents()
    assert {x.text for x in p} == {"b q2 unless c q0", "a q1 unless c q0", "a q1 & b q2"}
    assert E("a q1").parents() == []


# ------------------------------------------------------------------ hashing and ids
def test_id_ignores_order_but_depends_on_expression_transform_and_target():
    a = I.pattern_id(E("a q1 & b q2"))
    assert a == I.pattern_id(E("b q2 & a q1")) and a.startswith("P") and len(a) == 1 + I.ID_LEN
    assert a != I.pattern_id(E("a q1 & b q3"))
    assert a != I.pattern_id(E("a q1 & b q2"), transform="ts_quintile")
    assert a != I.pattern_id(E("a q1 & b q2"), target="excess_1d")
    assert I.pattern_hash(E("a q1 & b q2")).startswith(a[1:])
    with pytest.raises(I.IdentityError):
        I.pattern_id(E("a q1"), transform="decile")
    with pytest.raises(I.IdentityError):
        I.pattern_id(E("a q1"), target="")


def test_ids_are_unique_across_every_small_expression():
    seen = {}
    feats = ["a", "b", "c", "d"]
    for f in feats:
        for q in range(5):
            seen[I.pattern_id(I.Expression.single(f, q))] = f"{f}{q}"
    for (f1, f2) in itertools.combinations(feats, 2):
        for q1, q2 in itertools.product(range(5), repeat=2):
            seen[I.pattern_id(I.Expression.make([I.Term(f1, q1), I.Term(f2, q2)]))] = (f1, q1, f2, q2)
    assert len(seen) == 20 + 6 * 25                      # no collisions in the space the miner searches


def test_canonical_expression_accepts_text_and_expression():
    assert I.canonical_expression("b q2 & a q1") == "a q1 & b q2"
    assert I.canonical_expression(E("b q2 & a q1")) == "a q1 & b q2"


# ------------------------------------------------------------------ miner keys and masks
def test_key_round_trip_for_all_three_miner_shapes():
    feats = FEATS
    keys = [("s", 2, 3), ("p", 0, 1, 4, 2), ("u", 5, 0, 1, 4, 3, 2)]
    for k in keys:
        e = I.Expression.from_key(k, feats)
        back = e.to_key(feats)
        assert I.Expression.from_key(back, feats) == e           # same idea (term order may differ)
    assert I.Expression.from_key(("p", 0, 1, 4, 2), feats) == I.Expression.from_key(("p", 4, 2, 0, 1), feats)
    with pytest.raises(I.IdentityError):
        I.Expression.from_key(("s", 99, 1), feats)
    with pytest.raises(I.IdentityError):
        I.Expression.from_key(("z", 1, 1), feats)
    with pytest.raises(I.IdentityError):
        E("a q1 & b q2 & c q3").to_key(["a", "b", "c"])          # no miner tuple form
    with pytest.raises(I.IdentityError):
        E("a q1").to_key(["b"])


def test_mask_equals_the_miners_mask_on_random_quantiles():
    rng = np.random.default_rng(0)
    Q = rng.integers(0, 5, size=(4000, len(FEATS))).astype(np.int8)
    m = PatternMiner()
    for key in [("s", 1, 2), ("p", 0, 1, 3, 4), ("u", 2, 0, 5, 3, 1, 4), ("u", 0, 4, 1, 4, 2, 0)]:
        expr = I.Expression.from_key(key, FEATS)
        assert (expr.mask(Q, FEATS) == m._mask_from_key(key, Q, FEATS)).all()
    with pytest.raises(I.IdentityError):
        E("nope q1").mask(Q, FEATS)


def test_unknown_level_minus_one_is_neither_a_match_nor_an_exception():
    Q = np.array([[2, 0], [2, 3], [2, -1], [1, 3]], dtype=np.int8)
    assert E("a q2 unless b q0").mask(Q, ["a", "b"]).tolist() == [False, True, False, False]
    assert E("a q2").mask(Q, ["a", "b"]).tolist() == [True, True, True, False]


# ------------------------------------------------------------------ scope and windows
def test_scope_contains_matches_the_miners_rule_and_round_trips():
    for lab, lo, hi, inside, outside in (("low", 1.0, 2.0, 0.5, 1.5), ("mid", 1.0, 2.0, 1.5, 3.0), ("high", 1.0, 2.0, 2.5, 1.0)):
        s = I.Scope("m_vix", lab, lo, hi)
        assert s.contains(inside) and not s.contains(outside)
        assert s.contains(inside) == PatternMiner._in_scope((0, lab, lo, hi), np.array([inside]))
        assert I.Scope.from_miner(s.to_miner(CTX), CTX) == s
        assert I.Scope.from_dict(s.to_dict()) == s
    assert not I.Scope("m_vix", "low", 1, 2).contains(float("nan")) and not I.Scope("m_vix", "low", 1, 2).contains(None)
    assert I.Scope.from_miner(None, CTX) is None and I.Scope.from_dict(None) is None
    for bad in (("m_vix", "top", 1, 2), ("m_vix", "low", 3, 2), ("m_vix", "low", float("nan"), 2)):
        with pytest.raises(I.IdentityError):
            I.Scope(*bad)
    with pytest.raises(I.IdentityError):
        I.Scope.from_miner((9, "low", 1, 2), CTX)


def test_window_orders_normalises_and_overlaps():
    w = I.Window("2020-01-06", "2020-03-31 09:30")
    assert (w.start, w.end) == ("2020-01-06", "2020-03-31") and "2020-02-01" in w and "2020-04-01" not in w
    assert w.overlaps(I.Window("2020-03-31", "2020-05-01")) and not w.overlaps(I.Window("2020-04-01", "2020-05-01"))
    with pytest.raises(I.IdentityError):
        I.Window("2020-05-01", "2020-01-01")
    with pytest.raises(I.IdentityError):
        I.Window("not a date", "2020-01-01")
    assert I.Window.from_any(None) is None and I.Window.from_any(["2020-01-01", "2020-02-01"]).end == "2020-02-01"


# ------------------------------------------------------------------ records
def rec(text="a q1 & b q2", **kw):
    kw.setdefault("stats", {"t_disc": 4.2, "p_real": 0.91, "effect": 0.004})
    return I.new_record(text, discovery=("2015-01-01", "2019-12-31"), validation=("2020-01-01", "2021-12-31"), **kw)


def test_record_id_hash_and_features_follow_the_expression():
    r = rec("b q2 & a q1")
    assert r.id == I.pattern_id(E("a q1 & b q2")) and r.hash.startswith(r.id[1:]) and r.features == ("a", "b")
    assert r.family == "pair" and r.state == "candidate" and not r.usable and r.text == "a q1 & b q2"
    assert rec(stats={"t_disc": 9}).id == r.id                         # statistics never change identity


def test_record_json_round_trip_is_lossless_and_stable():
    r = rec("a q1 & b q2 unless c q0", transform="xs_quintile", stats={"t_disc": 3.1, "fdr_pass": True, "n_rows": 1200})
    s = r.to_json()
    r2 = I.PatternRecord.from_json(s)
    assert r2 == r and r2.to_json() == s and json.loads(s)["schema"] == I.SCHEMA
    assert json.loads(s)["stats"]["fdr_pass"] is True and json.loads(s)["stats"]["n_rows"] == 1200


def test_nan_statistics_serialise_as_not_measured_and_bad_types_are_refused():
    r = rec(stats={"m_recent": float("nan"), "t_recent": np.float32(1.5), "n_rows": np.int64(9)})
    assert r.stats["m_recent"] is None and r.stats["t_recent"] == 1.5 and r.stats["n_rows"] == 9
    json.loads(r.to_json())                                            # strict JSON: no NaN literal
    with pytest.raises(I.IdentityError):
        rec(stats={"t_disc": "high"})


def test_tampered_payloads_are_refused():
    """PLANTED DEFECT: every field that feeds the hash, edited by hand after the id was issued, must fail loading."""
    d = rec().to_dict()
    for field, value in (("expression", "a q1 & b q3"), ("transform", "ts_quintile"), ("target", "excess_1d")):
        bad = dict(d, **{field: value})
        with pytest.raises(I.IdentityError, match="identity check failed"):
            I.PatternRecord.from_dict(bad)
    with pytest.raises(I.IdentityError, match="feature list"):
        I.PatternRecord.from_dict(dict(d, features=["a", "zzz"]))
    for missing in ("id", "expression", "state"):
        with pytest.raises(I.IdentityError, match="missing"):
            I.PatternRecord.from_dict({k: v for k, v in d.items() if k != missing})
    with pytest.raises(I.IdentityError, match="schema"):
        I.PatternRecord.from_dict(dict(d, schema=99))
    with pytest.raises(I.IdentityError):
        I.PatternRecord.from_json("{not json")
    d2 = dict(d, state="teleported")
    with pytest.raises(I.IdentityError):
        I.PatternRecord.from_dict(d2)


def test_windows_must_not_overlap_and_rescoped_needs_a_scope():
    with pytest.raises(I.IdentityError, match="strictly before"):
        I.new_record("a q1", discovery=("2015-01-01", "2020-06-30"), validation=("2020-01-01", "2021-12-31"))
    with pytest.raises(I.IdentityError, match="scope"):
        I.new_record("a q1", state="rescoped")
    ok = I.new_record("a q1", state="rescoped", scope=I.Scope("m_vix", "low", 1, 2))
    assert ok.scope.label == "low"
    with pytest.raises(I.IdentityError):
        I.new_record("a q1", target="")


def test_lifecycle_transitions_follow_the_state_machine_and_leave_a_trail():
    r = rec()
    r = r.transition("active", "2021-12-31", "passed P(real)")
    assert r.usable and r.history[-1] == ("2021-12-31", "candidate", "active", "passed P(real)")
    r = r.transition("failed", "2022-06-30", "recent stretch contradicts").transition("cause_search", "2022-06-30")
    sc = I.Scope("m_breadth", "high", 0.4, 0.7)
    r = r.transition("rescoped", "2022-07-01", "holds in high breadth", scope=sc)
    assert r.usable and r.scope == sc and len(r.history) == 4
    r = r.transition("failed", "2023-01-01").transition("cause_search", "2023-01-02").transition("discarded", "2023-01-03")
    assert not r.usable and r.scope is None
    for bad in ("rescoped", "watch", "failed", "nonsense"):
        with pytest.raises(I.IdentityError):
            rec().transition(bad, "2022-01-01")                        # candidate cannot jump to these
    with pytest.raises(I.IdentityError, match="earlier"):
        rec().transition("active", "2022-01-05").transition("watch", "2021-01-01")
    assert rec().transition("confirmed", "2022-01-05").state == "active"     # miner alias
    assert I.PatternRecord.from_json(r.to_json()) == r                # history survives serialisation


def test_records_are_immutable():
    r = rec()
    with pytest.raises(Exception):
        r.state = "active"
    r2 = r.with_stats(t_conf=2.0)
    assert r.stats.get("t_conf") is None and r2.stats["t_conf"] == 2.0 and r2.id == r.id


# ------------------------------------------------------------------ the book
def test_book_merges_same_identity_and_is_order_independent():
    a, b, c = rec("a q1"), rec("b q2 & c q0"), rec("d q4")
    b2 = b.with_stats(t_disc=7.0).transition("active", "2022-01-05")
    b1 = b.transition("rejected", "2022-01-02")
    book1, book2 = I.PatternBook([a, b1, c, b2]), I.PatternBook([c, b2, b1, a])
    assert len(book1) == 3 and book1.digest() == book2.digest()
    assert book1.get(b.id).state == "active"
    assert {h[2] for h in book1.get(b.id).history} == {"active", "rejected"}      # both audit trails survive the merge
    assert b in book1 and b.id in book1 and "Pdeadbeef" not in book1
    assert book1.counts() == {"active": 1, "candidate": 2}
    assert book1.find_expression("c q0 & b q2").id == b.id


def test_book_json_round_trip_and_diff():
    book = I.PatternBook([rec("a q1"), rec("b q2").transition("active", "2022-01-05")])
    again = I.PatternBook.from_json(book.to_json())
    assert again.digest() == book.digest() and book.diff(again) == {"only_here": [], "only_there": [], "changed": []}
    other = I.PatternBook([rec("a q1").with_stats(t_disc=1.0), rec("z q3")])
    d = book.diff(other)
    assert d["only_here"] == [rec("b q2").id] and d["only_there"] == [rec("z q3").id] and d["changed"] == [rec("a q1").id]
    with pytest.raises(I.IdentityError):
        I.PatternBook.from_json("[]")
    with pytest.raises(I.IdentityError):
        I.PatternBook.from_json("nope")
    dup = json.loads(book.to_json())
    dup["patterns"].append(dup["patterns"][0])
    with pytest.raises(I.IdentityError, match="duplicate id"):
        I.PatternBook.from_json(json.dumps(dup))
    assert len(I.PatternBook()) == 0 and I.PatternBook().counts() == {}


# ------------------------------------------------------------------ the miner bridge, on a real fit
def make_panel(n_dates=240, n_tick=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(n_tick)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.normal(size=(len(idx), len(FEATS))), index=idx, columns=FEATS)
    for c in CTX:
        X[c] = pd.Series(rng.normal(size=n_dates), index=dates).reindex(dates).repeat(n_tick).values
    top = X["f0"].groupby(level=0).rank(pct=True).values >= 0.8
    shock = pd.Series(rng.normal(0, 0.02, n_dates), index=dates).reindex(idx.get_level_values(0)).values
    return X, pd.Series(shock + 0.012 * top + rng.normal(0, 0.03, len(idx)), index=idx)


@pytest.fixture(scope="module")
def fitted():
    X, y = make_panel()
    m = PatternMiner({"max_pairs": 260, "max_unless": 40, "null_reps": 1, "min_n": 120}).fit(X, y, X.index.get_level_values(0).max())
    return X, y, m


def test_records_from_miner_keep_every_pattern_with_the_same_name_and_state(fitted):
    X, y, m = fitted
    dates = X.index.get_level_values(0)
    recs = I.records_from_miner(m.patterns, m.feats, CTX, dates)
    assert len(recs) == len(m.patterns)                                # nothing dropped, nothing merged
    names = dict(zip(m.patterns["key_named"], m.patterns["status"]))
    by_text = {r.text: r for r in recs}
    for named, status in names.items():
        canon = I.canonical_expression(named)
        assert by_text[canon].state == status
    top = by_text[I.canonical_expression("f0 q4")]
    assert top.state == "active" and top.stats["t_disc"] > 3 and top.stats["p_real"] >= 0.8
    disc, val = recs[0].discovery, recs[0].validation
    ud = np.sort(dates.unique())
    assert pd.Timestamp(disc.end) == ud[int(len(ud) * 0.7) - 1] and pd.Timestamp(val.start) == ud[int(len(ud) * 0.7)]
    assert I.PatternBook(recs).counts() == m.patterns["status"].value_counts().sort_index().to_dict()


def test_records_round_trip_through_json_at_miner_scale(fitted):
    X, y, m = fitted
    recs = I.records_from_miner(m.patterns, m.feats, CTX, X.index.get_level_values(0))
    book = I.PatternBook(recs)
    again = I.PatternBook.from_json(book.to_json())
    assert again.digest() == book.digest() and len(again) == len(recs) >= 100


def test_prior_built_from_records_is_accepted_by_the_miner_and_retested(fitted):
    X, y, m = fitted
    recs = I.records_from_miner(m.patterns, m.feats, CTX, X.index.get_level_values(0))
    prior = I.records_to_prior(recs)
    assert len(prior) == int(m.patterns["status"].isin(["active", "rescoped"]).sum()) and len(prior) >= 1
    assert set(prior.columns) == {"names", "effect", "p_real"}
    m2 = PatternMiner({"max_pairs": 20, "max_unless": 0, "null_reps": 1, "min_n": 120}).fit(
        X, y, X.index.get_level_values(0).max(), prior=prior)
    have = {I.canonical_expression(n) for n in m2.patterns["key_named"]}
    for names in prior["names"]:
        e = I.Expression.make([I.Term(names[i], names[i + 1]) for i in range(1, len(names), 2)][: 1 if names[0] == "s" else 2],
                              [I.Term(names[5], names[6])] if names[0] == "u" else [])
        assert e.text in have                                          # every banked pattern was re-tested


def test_records_from_miner_edge_cases():
    assert I.records_from_miner(pd.DataFrame(), FEATS, CTX, pd.bdate_range("2020-01-01", periods=10)) == []
    one = pd.DataFrame([{"key": ("s", 0, 1), "status": "rejected", "t_disc": 1.0, "t_conf": 0.5, "scope": None}])
    with pytest.raises(I.IdentityError):
        I.records_from_miner(one, FEATS, CTX, pd.bdate_range("2020-01-01", periods=1))
    two = pd.DataFrame([{"key": ("s", 0, 1), "status": "rejected", "t_disc": 1.0, "t_conf": 0.5, "scope": None},
                        {"key": ("s", 0, 1), "status": "active", "t_disc": 5.0, "t_conf": 4.0, "scope": None}])
    got = I.records_from_miner(two, FEATS, CTX, pd.bdate_range("2020-01-01", periods=50))
    assert len(got) == 1 and got[0].state == "active"                  # a repeated identity keeps its strongest row
    rescoped = pd.DataFrame([{"key": ("s", 2, 1), "status": "rescoped", "t_disc": 3.0, "t_conf": 3.0,
                              "scope": (1, "low", 0.1, 0.9)}])
    r = I.records_from_miner(rescoped, FEATS, CTX, pd.bdate_range("2020-01-01", periods=50))[0]
    assert r.scope == I.Scope("m_vix_term", "low", 0.1, 0.9)
    assert I.records_to_prior([r]).shape[0] == 1 and I.records_to_prior([]).empty
