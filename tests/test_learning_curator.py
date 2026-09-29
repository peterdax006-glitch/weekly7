"""Tests for engine/learning/curator.py and trader_view.py (canon C64; contract sections 28-30, 49, 55). Synthetic data only.
Each mechanism has a planted case it must catch, plus the empty case."""
import dataclasses
import datetime as dt
import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from engine.learning import curator as C
from engine.learning import trader_view as T
from engine.learning.archive import ChainCorrupt
from engine.learning.core import FirewallBreach, TemporalClass

FIXED = "2020-01-01T00:00:00"
CALM = {"m_vix": 12.0, "m_spy_r5": 0.01}
CRISIS = {"m_vix": 40.0, "m_spy_r5": -0.08}


def cur(**kw):
    return C.Curator(None, code_hash="testhash", created_real=FIXED, **kw)


def item(name, lean=0.02, horizon=5, extra=None):
    feats = {"rel_vix": 0.5, "rel_r5": -0.2, **(extra or {})}
    return T.TraderMemoryItem.make(T.opaque_token(name), "pattern", 1.0, feats, lean, horizon)


def fill(c, name, date, ctx, lean=0.02, matured=None, **kw):
    d = pd.Timestamp(date)
    return c.file(item(name, lean), d, d.year, ctx, matured_at=matured or (d + pd.Timedelta(days=10)), **kw)


def released_ids(rel):
    return [i.item_id for i in rel.items]


# ------------------------------------------------------------------------------------------ trader_view: refusal
BAD_PAYLOADS = [
    ("iso_in_id", {"item_id": "AAPL_2008-09-15"}),
    ("year_in_value", {"features": {"a": "since 2008"}}),
    ("year_float", {"features": {"a": 2008.0}}),
    ("year_int", {"features": {"a": 2008}}),
    ("fractional_year", {"features": {"a": 2008.5}}),
    ("datetime_obj", {"features": {"a": dt.datetime(2008, 9, 15)}}),
    ("date_obj", {"features": {"a": dt.date(2008, 9, 15)}}),
    ("timestamp_obj", {"features": {"a": pd.Timestamp("2008-09-15")}}),
    ("np_datetime", {"features": {"a": np.datetime64("2008-09-15")}}),
    ("compact_number", {"features": {"a": 20080915}}),
    ("epoch_seconds", {"features": {"a": 1221436800}}),
    ("epoch_millis", {"features": {"a": 1221436800000}}),
    ("date_key", {"features": {"2008-09-15": 0.1}}),
    ("year_key", {"features": {"year": 0.1}}),
    ("era_word", {"features": {"a": "lehman"}}),
    ("quarter", {"features": {"a": "Q3 08"}}),
    ("month_name", {"features": {"a": "Sep 15, 2008"}}),
    ("underscore_year", {"features": {"a": "AAPL_2008"}}),
    ("glued_year", {"features": {"a": "x2008y"}}),
    ("reason_field", {"features": {"reason_for_rank": 0.3}}),
]


@pytest.mark.parametrize("name,payload", BAD_PAYLOADS, ids=[b[0] for b in BAD_PAYLOADS])
def test_find_violations_catches_planted_dates(name, payload):
    assert T.find_violations(payload), f"{name} slipped through"


@pytest.mark.parametrize("name,payload", BAD_PAYLOADS, ids=[b[0] for b in BAD_PAYLOADS])
def test_scrub_output_is_always_clean(name, payload):
    clean, rep = T.scrub(payload)
    assert T.find_violations(clean) == []
    assert rep.removed >= 1 and not rep.clean


def test_clean_payload_is_untouched_and_counts_zero():
    payload = {"features": {"rel_vix": 0.4, "rel_r5": -1.2}, "lean": 0.02}
    clean, rep = T.scrub(payload)
    assert clean == payload and rep.clean and rep.removed == 0 and rep.nodes >= 4


def test_scrub_refuse_mode_raises():
    with pytest.raises(FirewallBreach):
        T.scrub({"features": {"a": 2008.0}}, refuse=True)
    T.scrub({"features": {"a": 0.5}}, refuse=True)


def test_scrub_empty_and_degenerate():
    assert T.scrub({})[0] == {} and T.scrub([])[0] == [] and T.scrub(None)[0] is None
    clean, rep = T.scrub("2008-09-15")
    assert clean is None and rep.total("iso_date") == 1


def test_scrub_ledger_accumulates_counts():
    led = T.ScrubLedger()
    led.scrub({"a": 2008.0, "b": 1.0})
    led.scrub({"c": "x_2009-01-02"})
    led.scrub({"d": 0.3})
    s = led.summary()
    assert s["calls"] == 3 and s["dirty_calls"] == 2 and s["by_category"]["year_number"] == 1


def test_number_reason_boundaries():
    assert T.number_reason(1899) is None and T.number_reason(1900) == "year_number" and T.number_reason(2100) == "year_number"
    assert T.number_reason(2101) is None and T.number_reason(0.5) is None and T.number_reason(True) is None
    assert T.number_reason(20081340) == "oversized_number"   # month 13 is not a date, but it is still an id-sized number
    assert T.number_reason(float("nan")) == "non_finite"


def test_opaque_token_has_no_digits_and_is_stable():
    toks = {T.opaque_token(("k", i)) for i in range(400)}
    assert len(toks) == 400 and all(t.isalpha() and t.islower() for t in toks)
    assert T.opaque_token("x") == T.opaque_token("x") and T.opaque_token("x") != T.opaque_token("x", salt="s")
    assert all(not T.find_violations(t) for t in toks)


# ------------------------------------------------------------------------------------------ trader_view: records
def test_trader_item_constructor_rejects_dates_everywhere():
    good = dict(item_id="abcdefgh", kind="pattern", weight=0.5, features={"rel_a": 0.1}, lean=0.01, horizon=5)
    T.TraderMemoryItem.make(**good)
    for field, val in (("item_id", "AAPL_2008-09-15"), ("item_id", "abc2008def"), ("item_id", "short"),
                       ("features", {"a": 2008.0}), ("features", {"a": "Q3 2009"}), ("features", {"date": 0.2}),
                       ("kind", "2008"), ("weight", 1.5), ("weight", float("nan")), ("lean", 3.0), ("horizon", 2008),
                       ("horizon", 0), ("horizon", True)):
        with pytest.raises(FirewallBreach):
            T.TraderMemoryItem.make(**{**good, field: val})


def test_trader_item_is_frozen():
    it = item("a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        it.weight = 0.9


def test_trader_situation_rejects_dates_and_negative_step():
    T.TraderSituation.make(3, {"rel_vix": 0.2})
    with pytest.raises(FirewallBreach):
        T.TraderSituation.make(3, {"rel_vix": 2010.0})
    with pytest.raises(FirewallBreach):
        T.TraderSituation.make(-1, {})
    with pytest.raises(FirewallBreach):
        T.TraderSituation.make(3, {"as_of": 0.1})


def test_trader_release_invariants():
    a, b = item("alpha").with_weight(0.25), item("beta").with_weight(0.75)
    lo, hi = sorted([a, b], key=lambda i: i.item_id)
    T.TraderRelease(1, (lo, hi))
    with pytest.raises(FirewallBreach):
        T.TraderRelease(1, (hi, lo))                      # order must not encode priority
    with pytest.raises(FirewallBreach):
        T.TraderRelease(1, (lo,))                         # weights must sum to 1
    with pytest.raises(FirewallBreach):
        T.TraderRelease(1, (lo, lo))
    assert len(T.TraderRelease(0, ())) == 0               # empty batch is legal


def test_blind_json_rechecks_and_release_hits_empty():
    rel = T.TraderRelease(2, (item("solo"),))
    assert T.release_year_hits(rel) == {} and "2008" not in T.blind_json(rel)
    with pytest.raises(FirewallBreach):
        T.blind_json({"note": "in 2008"})


def test_indistinguishable_ignores_nothing_but_equal_content():
    a, b = T.TraderRelease(1, (item("same"),)), T.TraderRelease(1, (item("same"),))
    assert T.indistinguishable(a, b)
    assert not T.indistinguishable(a, T.TraderRelease(1, (item("other"),)))


# ------------------------------------------------------------------------------------------ filing
def test_file_refuses_planted_dates_in_id_payload_and_key():
    c = cur()
    d = pd.Timestamp("2008-09-15")
    for bad in ({"item_id": "AAPL_2008-09-15", "features": {"rel_a": 0.1}, "lean": 0.01, "horizon": 5},
                {"features": {"rel_a": 2008.0}, "lean": 0.01, "horizon": 5},
                {"features": {"rel_a": "2008-09-15"}, "lean": 0.01, "horizon": 5},
                {"features": {"rel_a": 0.1}, "lean": 0.01, "horizon": 5, "note": "hi"},
                {"features": {"year": 0.1}, "lean": 0.01, "horizon": 5}):
        with pytest.raises(FirewallBreach):
            c.file(bad, d, 2008, CALM, matured_at="2008-09-25")
    assert len(c.store) == 0


def test_file_scrub_mode_strips_and_counts_and_reids():
    c = cur()
    d = pd.Timestamp("2008-09-15")
    dirty = {"item_id": "AAPL_2008-09-15", "features": {"rel_a": 0.1, "rel_b": 2008.0, "year": 3, "rel_c": "2008-09-15"},
             "lean": 0.01, "horizon": 5, "note": "x"}
    m = c.file(dirty, d, 2008, CALM, matured_at="2008-09-25", on_dirty="scrub")
    assert m.key.isalpha() and m.real_year == 2008
    assert m.payload["features"] == {"rel_a": 0.1}
    s = c.ledger.summary()["by_category"]
    assert s["year_number"] == 1 and s["forbidden_key"] == 1 and s["unknown_field"] == 1
    assert T.find_violations(m.payload) == []


def test_file_year_must_match_the_day_it_operated():
    c = cur()
    with pytest.raises(FirewallBreach):
        c.file(item("a"), "2010-03-01", 2011, CALM, matured_at="2010-03-10")


def test_file_context_must_be_m_columns_and_finite():
    c = cur()
    for ctx in ({"vix": 1.0}, {"m_vix": float("nan")}, {"m_vix": "high"}, {}):
        with pytest.raises(FirewallBreach):
            c.file(item("a"), "2010-03-01", 2010, ctx, matured_at="2010-03-10")


def test_file_is_idempotent_and_filed_by_year():
    c = cur()
    a = fill(c, "k", "2005-03-01", CALM)
    b = fill(c, "k", "2005-03-01", CALM)
    fill(c, "k2", "2009-03-01", CRISIS)
    assert a.mem_id == b.mem_id and len(c.store) == 2
    assert c.store.counts_by_year() == {2005: 1, 2009: 1} and c.store.years() == [2005, 2009]


def test_cannot_file_an_outcome_that_has_not_matured_yet():
    c = cur()
    c.day("2010-06-01", CALM)
    with pytest.raises(FirewallBreach):
        fill(c, "k", "2010-05-20", CALM, matured="2010-06-01")     # matures ON the clock day: not yet known
    with pytest.raises(FirewallBreach):
        fill(c, "k", "2010-05-20", CALM, matured="2010-06-09")
    fill(c, "k", "2010-05-20", CALM, matured="2010-05-31")


def test_store_persists_per_year_dirs_reopens_and_detects_tampering(tmp_path):
    root = tmp_path / "store"
    c = C.Curator(root, code_hash="testhash", created_real=FIXED)
    fill(c, "a", "2004-02-02", CALM)
    fill(c, "b", "2004-08-02", CRISIS)
    fill(c, "c", "2011-02-02", CALM)
    assert sorted(p.name for p in root.iterdir() if p.name.startswith("y")) == ["y2004", "y2011"]
    again = C.CuratorStore(root)
    assert len(again) == 3 and again.counts_by_year() == {2004: 2, 2011: 1} and again.verify()["ok"]
    chain = root / "y2004" / "chain.jsonl"
    text = chain.read_text(encoding="utf-8").replace('"reliability":0.5', '"reliability":0.9', 1)
    chain.write_text(text, encoding="utf-8")
    with pytest.raises(ChainCorrupt):
        C.CuratorStore(root)


def test_empty_store_release_is_empty_and_valid():
    c = cur()
    sit, rel = c.run_day("2012-01-03", CALM)
    assert len(rel) == 0 and sit.step == 0 and c.visible_count() == 0
    assert c.audit_frame().empty and c.year_share().empty


# ------------------------------------------------------------------------------------------ clock and daily release
def test_clock_only_moves_forward_and_release_needs_the_day():
    c = cur()
    c.day("2012-01-03", CALM)
    for again in ("2012-01-03", "2012-01-02", "2011-12-30"):
        with pytest.raises(FirewallBreach):
            c.day(again, CALM)
    with pytest.raises(FirewallBreach):
        c.release("2012-01-04", CALM)                      # that day has not been served
    c.day("2012-01-04", CALM)
    c.release("2012-01-04", CALM)
    with pytest.raises(FirewallBreach):
        cur().release("2012-01-04", CALM)                  # clock never started


def test_future_information_is_refused_before_use():
    c = cur()
    with pytest.raises(FirewallBreach):
        c.day("2012-01-03", CALM, info=[{"name": "px", "timestamp": "2012-01-04"}])
    with pytest.raises(FirewallBreach):                    # dated today but not public until later
        c.day("2012-01-03", CALM, info=[{"name": "f", "kind": "FILING", "timestamp": "2012-01-03"}])
    with pytest.raises(FirewallBreach):
        c.day("2012-01-03", CALM, info=[{"name": "x", "timestamp": "2012-01-03", "available_at": "2012-01-09"}])
    with pytest.raises(FirewallBreach):
        c.day("2012-01-03", CALM, info=[{"name": "no timestamp"}])
    assert c.now is None                                    # a refused day never advanced the clock
    c.day("2012-01-03", CALM, info=[{"name": "px", "timestamp": "2012-01-03"}, {"name": "old", "timestamp": "2011-12-01"}])


def test_situation_is_relative_and_year_free():
    c = cur()
    rng = np.random.default_rng(1)
    sits = []
    for i, d in enumerate(pd.bdate_range("2012-01-02", periods=90)):
        st = {"m_vix": 15 + rng.normal(), "m_spy_r5": 0.01 * rng.normal()}
        sits.append(c.day(d, st))
    assert [s.step for s in sits] == list(range(90))
    assert not sits[10].warm and sits[-1].warm
    spike = c.day(pd.bdate_range("2012-01-02", periods=91)[-1], {"m_vix": 60.0, "m_spy_r5": 0.0})
    assert spike.features["rel_vix"] > 4                     # far above its own history, in units of that history
    for s in sits + [spike]:
        assert T.find_violations(s.to_dict()) == [] and set(s.features) == {"rel_vix", "rel_spy_r5"}


def test_information_grows_day_by_day():
    c = cur()
    fill(c, "a", "2010-01-04", CALM, matured="2010-01-12")
    fill(c, "b", "2010-01-04", CALM, matured="2010-02-01")
    counts = []
    for d in ("2010-01-11", "2010-01-12", "2010-01-13", "2010-02-01", "2010-02-02"):
        c.day(d, CALM)
        counts.append(c.visible_count())
    assert counts == [0, 0, 1, 1, 2]                         # visible strictly AFTER the maturity day, never on it


# ------------------------------------------------------------------------------------------ C58 and relevance
def test_c58_unmatured_memory_is_never_released():
    c = cur()
    fill(c, "later", "2010-06-01", CALM, matured="2010-08-01")
    fill(c, "ok", "2010-01-04", CALM, matured="2010-01-15")
    for now, expect in (("2010-06-02", 1), ("2010-07-31", 1), ("2010-08-01", 1), ("2010-08-02", 2)):
        c.day(now, CALM)
        rel = c.release(now, CALM)
        assert len(rel) == expect
    early = cur()
    fill(early, "later", "2010-06-01", CALM, matured="2010-08-01")
    early.day("2010-07-15", CALM)
    assert len(early.release("2010-07-15", CALM)) == 0
    assert early.audit[-1].excluded_unmatured == 1


def test_ranking_changes_with_era_and_nothing_year_like_reaches_the_trader():
    c = cur(config=C.RelevanceConfig(min_priority=0.0))
    fill(c, "crisis_pattern", "2008-10-01", CRISIS, lean=-0.03, temporal=TemporalClass.PERSISTENT)
    fill(c, "calm_pattern", "2006-05-01", CALM, lean=0.02, temporal=TemporalClass.PERSISTENT)
    for extra in range(6):                                   # background so the robust scale is not degenerate
        fill(c, f"bg{extra}", "2007-03-0%d" % (extra + 1), {"m_vix": 14 + extra, "m_spy_r5": 0.0}, lean=0.01,
             temporal=TemporalClass.PERSISTENT)
    c.day("2012-01-03", CRISIS)
    p_crisis, _ = c.relevance("2012-01-03", CRISIS)
    p_calm, _ = c.relevance("2012-01-03", CALM)
    by = lambda ps, name: max(p.score for p in ps if p.key == T.opaque_token(name))
    assert by(p_crisis, "crisis_pattern") > by(p_crisis, "calm_pattern")
    assert by(p_calm, "calm_pattern") > by(p_calm, "crisis_pattern")
    r1 = c.release("2012-01-03", CRISIS, k=2)
    assert released_ids(r1)[0] != "" and len(r1) == 2
    top = max(r1.items, key=lambda i: i.weight)
    assert top.item_id == T.opaque_token("crisis_pattern")
    assert T.release_year_hits(r1) == {} and not any(str(y) in T.blind_json(r1) for y in range(1990, 2030))


def test_release_order_and_weights_do_not_reveal_priority():
    c = cur()
    names = [f"n{i}" for i in range(8)]
    for i, n in enumerate(names):
        fill(c, n, "2005-03-01", {"m_vix": 12 + 3 * i, "m_spy_r5": 0.0}, temporal=TemporalClass.PERSISTENT)
    c.day("2010-01-04", {"m_vix": 12.0, "m_spy_r5": 0.0})
    rel = c.release("2010-01-04", {"m_vix": 12.0, "m_spy_r5": 0.0}, k=8)
    ids = released_ids(rel)
    assert ids == sorted(ids) and abs(sum(i.weight for i in rel.items) - 1) < 1e-9
    weights = [i.weight for i in rel.items]
    assert weights != sorted(weights, reverse=True) or len(set(round(w, 9) for w in weights)) == 1   # id order, not priority order


def test_same_situation_in_different_years_gets_different_priority_but_identical_trader_view():
    def build(year):
        c = cur()
        fill(c, "same_lesson", f"{year}-03-01", CALM, temporal=TemporalClass.SLOW_DECAY)
        return c
    old, new = build(2001), build(2011)
    for c in (old, new):
        c.day("2012-06-01", CALM)
    p_old, _ = old.relevance("2012-06-01", CALM)
    p_new, _ = new.relevance("2012-06-01", CALM)
    assert C.priorities_differ(p_old, p_new)                 # the curator knows a decade-old fast-decaying lesson is stale
    assert p_new[0].score > p_old[0].score
    r_old, r_new = old.release("2012-06-01", CALM), new.release("2012-06-01", CALM)
    assert T.indistinguishable(r_old, r_new)                 # the trader sees the same item at the same weight
    assert T.blind_json(r_old) == T.blind_json(r_new)
    assert old.audit[-1].released[0]["years"] != new.audit[-1].released[0]["years"]     # only the trusted audit differs


def test_c59_consistent_across_years_beats_one_year_and_flip_flops():
    c = cur()
    for y in (2003, 2005, 2007, 2009):
        fill(c, "steady", f"{y}-04-01", CALM, lean=0.02, temporal=TemporalClass.PERSISTENT)
        fill(c, "flip", f"{y}-04-02", CALM, lean=0.02 if y % 4 == 3 else -0.02, temporal=TemporalClass.PERSISTENT)
    fill(c, "once", "2009-04-03", CALM, lean=0.02, temporal=TemporalClass.PERSISTENT)
    c.day("2012-01-03", CALM)
    prios, _ = c.relevance("2012-01-03", CALM)
    sc = {p.key: p for p in prios}
    steady, flip, once = (sc[T.opaque_token(n)] for n in ("steady", "flip", "once"))
    assert steady.universal and steady.n_years == 4 and not flip.universal and not once.universal
    assert steady.consistency > flip.consistency and steady.consistency > once.consistency
    assert steady.score > flip.score and steady.score > once.score


def test_universal_memory_is_only_mildly_gated_by_era():
    c = cur()
    for y in (2003, 2004, 2005):
        fill(c, "steady", f"{y}-04-01", CALM, temporal=TemporalClass.PERSISTENT)
        fill(c, "local", f"{y}-04-02", CALM, temporal=TemporalClass.PERSISTENT) if y == 2003 else None
    fill(c, "bg", "2004-06-01", CRISIS, temporal=TemporalClass.PERSISTENT)
    c.day("2010-01-04", CRISIS)
    prios, _ = c.relevance("2010-01-04", CRISIS)
    steady = max((p for p in prios if p.key == T.opaque_token("steady")), key=lambda p: p.score)
    local = max(p for p in prios if p.key == T.opaque_token("local"))
    assert steady.era < 0.3 and local.era < 0.3 and steady.universal and not local.universal
    assert steady.score > 2.5 * local.score


def test_reliability_and_temporal_class_move_the_priority():
    c = cur()
    fill(c, "solid", "2008-01-02", CALM, reliability=0.9, temporal=TemporalClass.PERSISTENT)
    fill(c, "shaky", "2008-01-03", CALM, reliability=0.9, calibration_error=0.8, temporal=TemporalClass.PERSISTENT)
    fill(c, "fast", "2008-01-04", CALM, reliability=0.9, temporal=TemporalClass.FAST_DECAY)
    c.day("2012-01-03", CALM)
    sc = {p.key: p.score for p in c.relevance("2012-01-03", CALM)[0]}
    solid, shaky, fast = (sc[T.opaque_token(n)] for n in ("solid", "shaky", "fast"))
    assert solid > shaky * 3 and solid > fast * 3


def test_seasonal_memory_returns_in_its_season():
    c = cur()
    fill(c, "santa", "2005-12-20", CALM, temporal=TemporalClass.SEASONAL)
    c.day("2010-12-22", CALM)
    winter = c.relevance("2010-12-22", CALM)[0][0].score
    c2 = cur()
    fill(c2, "santa", "2005-12-20", CALM, temporal=TemporalClass.SEASONAL)
    c2.day("2010-06-22", CALM)
    summer = c2.relevance("2010-06-22", CALM)[0][0].score
    assert winter > 5 * summer


def test_prefix_invariance_holds_and_a_leak_would_break_it():
    c = cur()
    rng = np.random.default_rng(3)
    for i in range(30):
        d = pd.Timestamp("2005-01-03") + pd.Timedelta(days=45 * i)
        fill(c, f"m{i % 7}", d, {"m_vix": 10 + 25 * rng.random(), "m_spy_r5": 0.0}, lean=float(rng.normal(0, .02)) or 0.01,
             matured=d + pd.Timedelta(days=int(rng.integers(5, 200))))
    dates = [pd.Timestamp("2006-06-01"), pd.Timestamp("2008-01-15"), pd.Timestamp("2010-03-30")]
    c.day("2011-01-03", CALM)
    assert C.prefix_invariance(c, dates, lambda d: CALM) == []
    leaky = c.truncated("2008-01-15")
    late = fill(c, "leak", "2007-12-01", CALM, matured="2009-01-01")
    leaky.store.add(late)                                    # a memory from the future slipped into the "past" store
    full, _ = leaky.relevance("2008-01-15", CALM)
    assert all(p.mem_id != late.mem_id for p in full)         # the matured-before filter still blocks it: the guard is per-record


def test_release_is_deterministic():
    def run():
        c = cur()
        for i in range(10):
            fill(c, f"d{i}", f"200{i % 5}-05-0{1 + i % 5}", {"m_vix": 10 + 3 * i, "m_spy_r5": 0.0})
        c.day("2012-01-03", CALM)
        return c.release("2012-01-03", CALM).digest()
    assert run() == run()


def test_audit_trail_is_trusted_side_only_and_chained(tmp_path):
    c = C.Curator(tmp_path / "s", code_hash="testhash", created_real=FIXED)
    fill(c, "a", "2004-01-05", CALM, temporal=TemporalClass.PERSISTENT)
    c.day("2010-01-04", CALM)
    rel = c.release("2010-01-04", CALM)
    df = c.audit_frame()
    assert len(df) == 1 and "2004" in df.iloc[0]["years"] and c.audit_verify()["ok"]
    assert c.year_share().to_dict() == {2004: 1.0}
    blob = T.blind_json(rel)
    assert "2004" not in blob and "audit" not in blob and "era" not in blob
    assert "curator report" in c.report_markdown().lower()


# ------------------------------------------------------------------------------------------ static path check
def test_real_repo_trader_path_does_not_reach_the_curator():
    n = T.assert_trader_path_clean()
    assert n >= 3
    assert "engine.livesim" in T.trader_closure()


def _tree(tmp_path, files):
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding="utf-8")
    return tmp_path


BASE = {"engine/__init__.py": "", "engine/learning/__init__.py": "", "engine/learning/curator.py": "class Curator: pass\n",
        "engine/livesim.py": "x = 1\n", "engine/adaptive.py": "y = 1\n", "engine/learning/learner.py": "z = 1\n"}


def test_static_check_passes_a_clean_tree(tmp_path):
    root = _tree(tmp_path, BASE)
    assert T.trader_path_violations(root=root) == []
    assert T.assert_trader_path_clean(root=root) == 3


@pytest.mark.parametrize("name,src,kind", [
    ("direct", "from engine.learning.curator import Curator\n", "import"),
    ("module", "import engine.learning.curator\n", "import"),
    ("from_pkg", "from engine.learning import curator\n", "import"),
    ("relative", "from . import curator\n", "import"),
    ("relative_from", "from .curator import Curator\n", "import"),
    ("lazy", "def f():\n    from engine.learning import curator\n    return curator\n", "import"),
    ("dynamic", "import importlib\nm = importlib.import_module('engine.learning.curator')\n", "dynamic_import"),
    ("dunder", "m = __import__('engine.learning.curator')\n", "dynamic_import"),
    ("store", "PATH = 'state/curator_store/y2008'\n", "store_read"),
    ("symbol", "def f(c):\n    return c.CuratorStore\n", "symbol"),
])
def test_static_check_catches_planted_trader_to_curator_link(tmp_path, name, src, kind):
    files = dict(BASE)
    files["engine/learning/learner.py"] = src
    v = T.trader_path_violations(root=_tree(tmp_path, files))
    assert v and any(x.kind == kind for x in v), f"{name}: {v}"
    with pytest.raises(FirewallBreach):
        T.assert_trader_path_clean(root=tmp_path)


def test_static_check_follows_transitive_imports(tmp_path):
    files = dict(BASE)
    files["engine/adaptive.py"] = "from engine import helper\n"
    files["engine/helper.py"] = "from engine.learning import curator\n"
    v = T.trader_path_violations(root=_tree(tmp_path, files))
    assert len(v) >= 1 and v[0].file == "engine/helper.py" and "engine.adaptive" in v[0].via


def test_static_check_does_not_flag_the_curator_importing_trader_view(tmp_path):
    files = dict(BASE)
    files["engine/learning/curator.py"] = "from .trader_view import TraderRelease\n"
    files["engine/learning/trader_view.py"] = "class TraderRelease: pass\n"
    files["engine/learning/learner.py"] = "from .trader_view import TraderRelease\n"
    assert T.trader_path_violations(root=_tree(tmp_path, files)) == []


def test_static_check_fails_closed_on_missing_entries(tmp_path):
    with pytest.raises(FirewallBreach):
        T.assert_trader_path_clean(root=_tree(tmp_path, {"engine/__init__.py": ""}))


# ------------------------------------------------------------------------------------------ channel 6
def frame(n=1500, seed=0, drift=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1998-01-01", periods=n)
    t = np.arange(n)
    level = np.log(100) + (0.0004 * t if drift else 0) + rng.normal(0, 0.01, n).cumsum() * 0.2
    return pd.DataFrame({"m_spy_level": level, "m_vix": 18 + 6 * np.sin(t / 60) + rng.normal(0, 1, n),
                         "other": rng.normal(size=n)}, index=idx)


def test_relative_context_drops_absolute_levels_and_other_columns():
    rel = C.relative_m_context(frame(), lookback=250, min_periods=30)
    assert set(rel.columns) == {"rel_spy_level", "pct_spy_level", "rel_vix", "pct_vix"}
    assert rel.iloc[:29].isna().all().all() and rel.iloc[300:].notna().all().all()
    assert rel.filter(like="pct_").max().max() <= 1 and rel.filter(like="pct_").min().min() >= 0
    assert rel.filter(like="rel_").abs().max().max() <= 6.0
    z = C.relative_m_context(frame(), lookback=250, min_periods=30, mode="z")
    assert set(z.columns) == {"rel_spy_level", "rel_vix"}


def test_relative_context_is_past_only_truncation_invariant():
    f = frame(700)
    full = C.relative_m_context(f, lookback=200, min_periods=30)
    for cut in (120, 333, 600):
        part = C.relative_m_context(f.iloc[:cut], lookback=200, min_periods=30)
        pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_relative_context_ignores_future_planted_change():
    f = frame(700)
    g = f.copy()
    g.iloc[500:, :] = g.iloc[500:, :] * 50 + 999        # rewrite the future
    a = C.relative_m_context(f, lookback=200, min_periods=30)
    b = C.relative_m_context(g, lookback=200, min_periods=30)
    pd.testing.assert_frame_equal(a.iloc[:500], b.iloc[:500])


def test_relative_context_empty_and_bad_mode():
    empty = C.relative_m_context(pd.DataFrame({"a": [1.0, 2.0]}, index=pd.bdate_range("2000-01-03", periods=2)))
    assert empty.shape == (2, 0)
    with pytest.raises(ValueError):
        C.relative_m_context(frame(50), mode="nope")
    assert len(C.relative_m_context(frame(5).iloc[:0])) == 0


def test_transform_report_shows_level_leak_removed_and_signal_kept():
    f = frame(n=7500, seed=4)                             # ~30 years, a drifting level identifies the year
    rng = np.random.default_rng(5)
    rel_vix = C.relative_m_context(f, lookback=250, min_periods=60, mode="z")["rel_vix"]
    target = (rel_vix.shift(1) * 0.01 + rng.normal(0, 0.01, len(f))).rename("t")   # vix stretch carries the signal
    rep = C.transform_report(f, target, lookback=250, months=12, seed=0, n_trees=60)
    assert rep["identifiability"]["raw"]["verdict"] == "identifiable"
    assert rep["identifiability"]["relative"]["skill"] < rep["identifiability"]["raw"]["skill"]
    assert rep["id_skill_removed"] > 0.2
    assert rep["signal"]["relative_abs_ic"] > 0.02 and rep["n_windows"] > 100
    assert rep["columns"] == ["m_spy_level", "m_vix"]


def test_transform_report_on_frame_without_drift_finds_little_to_remove():
    f = frame(n=7500, seed=6, drift=False)
    f["m_spy_level"] = 0.0 + np.random.default_rng(7).normal(size=len(f))
    target = pd.Series(np.random.default_rng(8).normal(size=len(f)), index=f.index)
    rep = C.transform_report(f, target, lookback=250, seed=0, n_trees=60)
    assert rep["identifiability"]["raw"]["skill"] < 0.25


def test_relevance_config_validation():
    assert C.RelevanceConfig().check() == []
    for bad in (dict(bandwidth=0), dict(universal_min_years=1), dict(min_history=1), dict(recency_floor=0)):
        assert C.RelevanceConfig(**bad).check()
    with pytest.raises(ValueError):
        C.Curator(None, C.RelevanceConfig(bandwidth=-1))


def test_era_similarity_and_scales_units():
    scales = {"m_vix": 5.0}
    same, known = C.era_similarity({"m_vix": 20.0}, {"m_vix": 20.0}, scales)
    far, _ = C.era_similarity({"m_vix": 20.0}, {"m_vix": 40.0}, scales)
    unknown = C.era_similarity({"m_spy_r5": 0.0}, {"m_vix": 40.0}, scales)
    assert same == 1.0 and known and far < 1e-3 and unknown == (0.5, False)
    assert C.robust_scale([1.0]) == 1.0 and C.robust_scale([5, 5, 5, 5]) == 1.0 and C.robust_scale([0, 1, 2, 3, 4]) > 1


# ------------------------------------------------------------------------------------------ paths, live objects, logging
def test_path_violations_catch_year_in_any_component():
    assert T.path_violations("data/cache/AAPL_2008-09-15.parquet")
    assert T.path_violations("C:\\state\\y2008\\chain.jsonl")
    assert T.path_violations("data/cache/prices.parquet") == []
    T.assert_blind_paths(["data/cache/prices.parquet", "state/run/feed.bin"])
    with pytest.raises(FirewallBreach):
        T.assert_blind_paths(["state/run/feed.bin", "state/2012/feed.bin"])


def test_object_leak_scan_finds_planted_attributes_and_is_cycle_safe():
    class Holder:
        def __init__(self):
            self.ok = [1.5, "fine"]
            self.meta = {"loaded_from": "cache_2009-03-02.pkl"}
            self.me = self

    v = T.object_leak_scan(Holder())
    assert any(x.category == "iso_date" for x in v)

    class Slotted:
        __slots__ = ("year_offset", "val")

        def __init__(self):
            self.year_offset, self.val = 3, 0.5

    assert any(x.category.startswith("attribute_") for x in T.object_leak_scan(Slotted()))
    assert T.object_leak_scan(Holder.__new__(Holder)) == []
    assert T.object_leak_scan(None) == [] and T.object_leak_scan({"a": 0.5}) == []


def test_object_leak_scan_of_a_real_release_is_clean():
    c = cur(config=C.RelevanceConfig(min_priority=0.0))
    fill(c, "a", "2004-01-05", CALM, temporal=TemporalClass.PERSISTENT)
    sit, rel = c.run_day("2010-01-04", CALM)
    assert T.object_leak_scan(T.TraderDay(sit, rel)) == []


def test_blind_log_refuses_or_scrubs_dates():
    strict = T.BlindLog()
    strict.log("step done", n=3)
    with pytest.raises(FirewallBreach):
        strict.log("finished 2008-09-15")
    loose = T.BlindLog(refuse=False)
    loose.log("finished 2008-09-15 fine", when=2008.0)
    assert "2008" not in loose.lines[0] and loose.ledger.summary()["dirty_calls"] == 1


def test_trader_day_requires_matching_steps():
    sit, rel = T.TraderSituation.make(4, {"rel_a": 0.1}), T.TraderRelease(4, ())
    assert json.loads(T.TraderDay(sit, rel).json())["situation"]["step"] == 4
    with pytest.raises(FirewallBreach):
        T.TraderDay(sit, T.TraderRelease(5, ()))


def test_release_entropy_and_comparison():
    assert T.weight_entropy(T.TraderRelease(0, ())) == 0.0
    a, b = sorted([item("x").with_weight(0.5), item("y").with_weight(0.5)], key=lambda i: i.item_id)
    assert abs(T.weight_entropy(T.TraderRelease(0, (a, b))) - np.log(2)) < 1e-9
    r1, r2 = T.TraderRelease(0, (a.with_weight(1.0),)), T.TraderRelease(0, (b.with_weight(1.0),))
    cmp = T.compare_releases(r1, r2)
    assert cmp["shared"] == 0 and cmp["jaccard"] == 0.0 and abs(cmp["weight_l1"] - 2.0) < 1e-9
    assert T.compare_releases(T.TraderRelease(0, ()), T.TraderRelease(0, ()))["jaccard"] == 1.0


# ------------------------------------------------------------------------------------------ trusted diagnostics and driver
def rich(seed=0):
    c = cur(config=C.RelevanceConfig(min_priority=0.0))
    rng = np.random.default_rng(seed)
    for i in range(24):
        y = 2003 + i % 6
        fill(c, f"k{i % 4}", f"{y}-0{1 + i % 9}-1{i % 5}", {"m_vix": 10 + 30 * rng.random(), "m_spy_r5": 0.0},
             lean=0.02 if (i + y) % 3 else -0.02, temporal=TemporalClass.PERSISTENT)
    return c


def test_key_history_and_year_summary_and_explain():
    c = rich()
    kh = C.key_history(c.store, T.opaque_token("k0"), "2012-01-03")
    assert list(kh.columns)[:2] == ["real_year", "n"] and kh["n"].sum() >= 4 and set(kh["sign"]) <= {-1, 0, 1}
    assert C.key_history(c.store, "nothing", "2012-01-03").empty
    ys = C.year_summary(c.store, "2012-01-03")
    assert ys["memories"].sum() == len(c.store) and (ys["knowable"] <= ys["memories"]).all() and "mean_m_vix" in ys
    c.day("2012-01-03", CALM)
    ex = C.explain(c, T.opaque_token("k1"), "2012-01-03", CALM)
    assert ex["eligible"] and 1 <= ex["rank"] <= ex["of"] and "consistency" in ex["best"]
    assert C.explain(c, "absentkey", "2012-01-03", CALM)["eligible"] is False


def test_era_sensitivity_ranking_moves_with_state():
    c = cur(config=C.RelevanceConfig(min_priority=0.0))
    for i, v in enumerate((11, 13, 15, 35, 40, 45)):
        fill(c, f"k{i}", f"200{3 + i}-05-01", {"m_vix": float(v), "m_spy_r5": 0.0}, temporal=TemporalClass.PERSISTENT)
    c.day("2012-01-03", CALM)
    tab = C.era_sensitivity(c, "2012-01-03", {"calm": {"m_vix": 12.0, "m_spy_r5": 0.0}, "crisis": {"m_vix": 42.0, "m_spy_r5": 0.0}}, k=3)
    assert tab.loc[0, "overlap_with_first"] == 1.0 and tab.loc[1, "overlap_with_first"] < 0.5
    assert tab.loc[0, "top"][0] != tab.loc[1, "top"][0]


def test_turnover_series_and_year_share_after_a_run():
    c = rich()
    states = pd.DataFrame({"m_vix": np.linspace(12, 40, 12), "m_spy_r5": 0.0}, index=pd.bdate_range("2012-01-02", periods=12))
    days = C.run_window(c, states, k=3)
    assert len(days) == 12 and [d.situation.step for d in days] == list(range(12))
    to = C.turnover_series(c)
    assert len(to) == 11 and to.between(0, 1).all() and to.max() > 0        # the state moved, so the released set moved
    ys = c.year_share()
    assert abs(ys.sum() - 1) < 1e-9 and set(ys.index) <= set(range(2003, 2009))


def test_run_window_output_is_clean_and_replays_identically():
    states = pd.DataFrame({"m_vix": np.linspace(12, 30, 8), "m_spy_r5": 0.01}, index=pd.bdate_range("2012-01-02", periods=8))
    assert C.replay_matches(rich, states, k=4)
    for d in C.run_window(rich(), states, k=4):
        assert T.find_violations(d.to_dict()) == []
    assert C.run_window(rich(), states.iloc[:0]) == []
    with pytest.raises(FirewallBreach):                                     # a plant: info stamped after the day it arrives
        C.run_window(rich(), states, info_of=lambda d: [{"name": "px", "timestamp": d + pd.Timedelta(days=3)}])


def test_file_batch_reports_refusals_without_losing_the_batch():
    c = cur()
    good = dict(item=item("g1"), real_date="2005-03-01", real_year=2005, context=CALM, matured_at="2005-03-09")
    bad_year = dict(good, real_year=2006)
    bad_ctx = dict(good, context={"vix": 1.0})
    bad_item = dict(good, item={"features": {"a": 2008.0}, "lean": 0.1, "horizon": 5})
    bad_arg = dict(good, nonsense=1)
    out = C.file_batch(c, [good, bad_year, bad_ctx, bad_item, dict(good, item=item("g2")), bad_arg])
    assert out["filed"] == 2 and [r["index"] for r in out["refused"]] == [1, 2, 3, 5] and out["store_size"] == 2
    assert C.file_batch(c, [])["filed"] == 0


def test_blind_equivalence_same_situations_different_year_label():
    states = pd.DataFrame({"m_vix": np.linspace(12, 30, 10), "m_spy_r5": 0.01}, index=pd.bdate_range("2012-01-02", periods=10))
    same = C.blind_equivalence(lambda: cur(), states, shift_days=3652)                 # empty store: nothing to differ
    assert same["same_steps"] and same["same_situations"] and same["mean_release_l1"] == 0.0
    moved = C.blind_equivalence(rich, states, shift_days=365 * 12)
    assert moved["same_steps"] and moved["same_situations"]


def test_trader_view_identifiability_probe_flags_a_planted_year_leak():
    rng = np.random.default_rng(0)
    starts = pd.date_range("1990-01-01", "2019-12-01", freq="MS")
    clean = pd.DataFrame({"a": rng.normal(size=len(starts)), "b": rng.normal(size=len(starts))})
    leaky = clean.assign(c=(starts.year.to_numpy() - 1990) / 30 + rng.normal(0, 0.02, len(starts)))
    ok = C.trader_view_identifiability(clean, starts, n_trees=60)
    bad = C.trader_view_identifiability(leaky, starts, n_trees=60)
    assert ok["verdict"] == "not identifiable" and bad["verdict"] == "identifiable" and bad["skill"] > ok["skill"] + 0.3
    with pytest.raises(ValueError):
        C.trader_view_identifiability(clean.iloc[:5], starts)


def test_release_window_features_and_empty_window():
    c = rich()
    states = pd.DataFrame({"m_vix": np.linspace(12, 30, 6), "m_spy_r5": 0.01}, index=pd.bdate_range("2012-01-02", periods=6))
    days = C.run_window(c, states, k=3)
    f = C.release_window_features([d.release for d in days], [d.situation for d in days])
    assert {"n_items", "entropy", "mean_lean", "rel_vix_mean", "rel_spy_r5_sd"} <= set(f) and f["n_items"] == 3
    e = C.release_window_features([], [])
    assert e == {"n_items": 0.0, "entropy": 0.0, "mean_lean": 0.0, "abs_lean": 0.0}


def test_store_manifest_hashes_config_and_chains():
    c = rich()
    m1 = C.store_manifest(c)
    assert m1["chains_ok"] and sum(m1["years"].values()) == len(c.store) and len(m1["config_hash"]) == 16
    assert C.store_manifest(cur(config=C.RelevanceConfig(bandwidth=2.0)))["config_hash"] != C.store_manifest(cur())["config_hash"]
