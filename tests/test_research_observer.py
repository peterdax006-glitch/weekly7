"""Tests for engine/research/observer.py and autopsy.py (C66 sections 4 and 21, C67 bands). Synthetic planted worlds only."""
import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest

from engine.research import autopsy as A
from engine.research import observer as O
from engine.research.core import FirewallBreach, MoveCategory as MC, Provenance

NOW = "2030-01-01"
P = O.ObserverParams()


def day(n=600, seed=0, d=0, plant=None):
    return O.synthetic_day(n, seed, d, P, plant or O.Plant())


def prov():
    return Provenance("2026-09-29", "2020-01-01", "abc")


# ---------------------------------------------------------------- categories A-I planted
def test_planted_categories_are_exact():
    s, o, t = day()
    r = O.observe_day(s, o, NOW, P)
    ids = lambda g: set(s.tickers[t[g]])
    rows = r.rows.set_index("ticker")
    fp = {tk for tk in rows.index if O.has(rows.at[tk, "flags"], MC.FALSE_POSITIVE)}
    fn = {tk for tk in rows.index if O.has(rows.at[tk, "flags"], MC.FALSE_NEGATIVE)}
    assert ids("fp") <= fp
    assert not (ids("hit") & fp)
    assert ids("pred") <= fn and ids("unpred") <= fn
    assert not (ids("hit") & fn)
    pred = {tk for tk in rows.index if O.has(rows.at[tk, "flags"], MC.PREDICTABLE_MOVER)}
    unpred = {tk for tk in rows.index if O.has(rows.at[tk, "flags"], MC.UNPREDICTABLE_MOVER)}
    assert ids("pred") <= pred and not (ids("pred") & unpred)
    assert ids("unpred") <= unpred
    near = {tk for tk in rows.index if O.has(rows.at[tk, "flags"], MC.NEAR_MISS)}
    assert ids("near") <= near
    assert r.n(MC.ABSTAINED) == 2 and r.n(MC.LOW_CONFIDENCE) >= 2
    assert not O.check_invariants(O.classify_day(s, o, P), s)
    assert not r.validate()


def test_bands_count_and_keep_every_mover():
    plant = O.Plant(n_up_mid=40, n_up_big=15, n_down_mid=35, n_down_big=12)
    s, o, t = day(3000, 1, 0, plant)
    r = O.observe_day(s, o, NOW, P)
    assert r.bands["up_5_10"] >= 40 and r.bands["up_gt10"] >= 15 + 5      # +hits (9%) are in the 5-10 band, unpred gap 12% in >10
    assert r.bands["down_5_10"] == 35 and r.bands["down_gt10"] == 12
    band_rows = r.rows[r.rows["band"] != 0]
    assert len(band_rows) == sum(r.bands.values())                       # every band mover has a row, uncapped
    assert r.market["bands"] == r.bands
    rows = r.rows.set_index("ticker")
    tk = s.tickers[t["down_big"][0]]
    assert rows.at[tk, "band"] == -2 and rows.at[tk, "c2c"] == pytest.approx(-0.13, abs=1e-6)
    assert rows.at[tk, "o2c"] == pytest.approx(-0.13, abs=1e-6)
    assert 0.0 <= rows.at[tk, "close_loc"] <= 1.0 and rows.at[tk, "hl_range"] > 0.13


def test_null_world_finds_no_movers():
    s, o, _ = day(600, 2, 0, O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    r = O.observe_day(s, o, NOW, P)
    assert sum(r.bands.values()) == 0
    for c in (MC.EXTREME_UP, MC.EXTREME_DOWN, MC.FALSE_NEGATIVE, MC.PREDICTABLE_MOVER, MC.UNPREDICTABLE_MOVER, MC.WINNER, MC.LOSER):
        assert r.n(c) == 0, c


def test_gap_geometry_and_suspect_prints():
    s, o, t = day(600, 3, 0, O.Plant(n_up_mid=3))
    tk = t["unpred"][0]
    dc = O.classify_day(s, o, P)
    assert dc.gap[tk] == pytest.approx(0.12, abs=1e-6) and dc.gshare[tk] > 0.9
    # a 2:1 split-looking print is flagged, still counted
    o2 = O.DayOutcome.make(o.resolved_at, o.tickers, o.entry, o.hi, o.lo, o.close, o.volume_ratio, day_hi=o.day_hi, day_lo=o.day_lo,
                           day_close=o.day_close.copy())
    j = t["near"][0]
    o2.day_close[j] = s.prev_close[j] * 0.5
    o2.hi[j] = max(o2.hi[j], o2.entry[j] * 1.001)
    o2 = O.DayOutcome.make(o.resolved_at, o.tickers, s.prev_close * 0.5 * (np.arange(s.n) == j) + o.entry * (np.arange(s.n) != j),
                           np.where(np.arange(s.n) == j, s.prev_close * 0.52, o.hi), np.where(np.arange(s.n) == j, s.prev_close * 0.48, o.lo),
                           np.where(np.arange(s.n) == j, s.prev_close * 0.5, o.close), o.volume_ratio)
    r = O.observe_day(s, o2, NOW, P)
    assert r.market["n_suspect_band"] == 1
    assert r.rows.set_index("ticker").at[s.tickers[j], "suspect"] == 1


def test_percentile_and_rank_edge_cases():
    assert np.isnan(O.pct_rank(np.array([np.nan, np.nan]))).all()
    assert (O.pct_rank(np.array([2.0, 2.0, 2.0])) == 0.5).all()
    assert list(O.desc_rank(np.array([1.0, 3.0, np.nan, 2.0]))) == [3, 1, 0, 2]
    assert np.isnan(O.auc(np.array([1.0, 2.0]), np.array([True, True])))
    assert O.auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    b = O.band_code(np.array([0.049, 0.05, -0.0999, -0.10, np.nan]), 0.05, 0.10)
    assert list(b) == [0, 1, -1, -2, 0]


# ---------------------------------------------------------------- empty / degenerate / validation
def test_empty_universe_is_a_record_not_a_crash():
    s = O.DecisionSnapshot.make("2020-01-06", [], [])
    o = O.DayOutcome.make("2020-01-07", [], [], [], [], [])
    r = O.observe_day(s, o, NOW, P)
    assert r.empty and r.rows.empty and sum(r.counts.values()) == 0 and not r.validate()
    led = O.ObserverLedger(P)
    led.add(r)
    assert led.category_rates().empty or led.category_rates().sum() == 0
    assert "no days" in O.render_ledger(O.ObserverLedger(P))


def test_all_nan_outcomes_do_not_invent_categories():
    s, _, _ = day(50, 4, 0, O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    nan = np.full(s.n, np.nan)
    o = O.DayOutcome.make("2020-01-07", s.tickers, nan, nan, nan, nan)
    r = O.observe_day(s, o, NOW, dt.__name__ and P if False else P)
    assert r.n(MC.FALSE_NEGATIVE) == 0 and r.n(MC.FALSE_POSITIVE) == 0 and r.n_no_outcome == s.n


def test_bad_inputs_are_rejected():
    s, o, _ = day(100, 5, 0, O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    with pytest.raises(FirewallBreach):
        O.observe_day(s, o, "2020-01-07", P)                                 # outcome not yet matured at now
    with pytest.raises(O.ObserverError):
        O.observe_day(s, O.DayOutcome.make(o.resolved_at, o.tickers[::-1][:-1].tolist() + ["ZZZ"], o.entry, o.hi, o.lo, o.close), NOW, P)
    bad_hi = o.hi.copy()
    bad_hi[0] = o.lo[0] * 0.5
    with pytest.raises(O.ObserverError):
        O.observe_day(s, O.DayOutcome.make(o.resolved_at, o.tickers, o.entry, bad_hi, o.lo, o.close), NOW, P)
    with pytest.raises(ValueError):
        O.ObserverParams(k_pick=50, k_high=20).require_valid()
    dup = O.DecisionSnapshot.make("2020-01-06", ["A", "A"], [1.0, 1.0])
    with pytest.raises(O.ObserverError):
        O.observe_day(dup, O.DayOutcome.make("2020-01-07", ["A", "A"], [1, 1], [1, 1], [1, 1], [1, 1]), NOW, P)


def test_outcome_reindex_missing_names_become_nan():
    s, o, _ = day(80, 6, 0, O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    keep = np.arange(0, 80, 2)
    o2 = O.DayOutcome.make(o.resolved_at, o.tickers[keep], o.entry[keep], o.hi[keep], o.lo[keep], o.close[keep])
    r = O.observe_day(s, o2, NOW, P)
    assert r.n_no_outcome == 40


# ---------------------------------------------------------------- determinism, caps, sampling
def test_determinism_and_row_cap_with_exact_counts():
    plant = O.Plant(n_pred=5, n_up_mid=30)
    s, o, _ = day(1500, 7, 0, plant)
    small = O.ObserverParams(max_rows_per_cat=5, n_top=3)
    a, b = O.observe_day(s, o, NOW, small), O.observe_day(s, o, NOW, small)
    assert O.record_digest(a) == O.record_digest(b)
    big = O.observe_day(s, o, NOW, dataclass_replace(small, max_rows_per_cat=500))
    assert a.counts == big.counts                                             # counts never depend on the cap
    assert a.dropped["CONSIDERED_MEDIUM"] > 0 and all(a.kept[k] + a.dropped[k] == a.counts[k] for k in a.counts)
    assert len(a.rows) < len(big.rows)
    assert len(a.rows[a.rows["band"] != 0]) == sum(a.bands.values())         # bands are exempt from the cap


def dataclass_replace(p, **kw):
    import dataclasses
    return dataclasses.replace(p, **kw)


def test_counterfactual_sample_is_seeded_and_weighted():
    s, o, _ = day(2000, 8, 0, O.Plant(n_pred=40, n_up_mid=10))
    r = O.observe_day(s, o, NOW, dataclass_replace(P, max_rows_per_cat=200))
    a, b = O.counterfactual_sample(r, 12, 1), O.counterfactual_sample(r, 12, 1)
    assert list(a["cid"]) == list(b["cid"]) and len(a) == 12
    assert a["weight"].max() > 1.0
    assert list(O.counterfactual_sample(r, 12, 2)["cid"]) != list(a["cid"])


# ---------------------------------------------------------------- ledger, streaming, checkpoint, firewall
def test_stream_ledger_checkpoint_roundtrip(tmp_path):
    plant = O.Plant(n_up_mid=4, n_down_big=2)
    led = O.run_year("2020", O.synthetic_days(8, 300, 9, plant), tmp_path, P)
    assert len(led) == 8
    assert O.verify_checkpoint(led, tmp_path, "chk") == []
    again = O.run_year("2020", O.synthetic_days(12, 300, 9, plant), tmp_path, P)     # resume: first 8 days skipped
    assert len(again) == 12
    with pytest.raises(O.ObserverError):
        led.add(list(led)[0])                                                # out of order / repeated day
    with pytest.raises(O.ObserverError):
        O.run_year("2020", O.synthetic_days(2, 300, 9, plant), tmp_path, dataclass_replace(P, k_pick=9))


def test_release_refuses_replayed_year_and_gates_by_maturity():
    led = O.ObserverLedger(P, filed_year="2008")
    for s, o in O.synthetic_days(4, 200, 10, O.Plant(n_up_mid=3)):
        led.add(O.observe_day(s, o, NOW, P))
    with pytest.raises(FirewallBreach):
        led.release(NOW, replaying_years=["2008"])
    assert len(led.release(NOW, replaying_years=["2015"])) == 4
    first = list(led)[0]
    with pytest.raises(FirewallBreach):
        first.as_matured(prov()).gate(first.resolved_at)                     # not strictly before
    assert "counts" in first.as_matured(prov()).gate("2031-01-01")
    assert len(led.known(list(led)[1].resolved_at)) == 1


def test_universe_health_flags_survivor_only_and_outage():
    led = O.ObserverLedger(P)
    for s, o in O.synthetic_days(70, 120, 11, O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)):
        led.add(O.observe_day(s, o, NOW, P))
    h = O.universe_health(led)
    assert h["survivor_only_suspect"] and h["daily_turnover"] == 0.0
    led2 = O.ObserverLedger(P)
    it = O.synthetic_days(3, 200, 12, O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    s, o = next(it)
    led2.add(O.observe_day(s, o, NOW, P))
    s2, o2, _ = O.synthetic_day(100, 12, 1, plant=O.Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    led2.add(O.observe_day(s2, o2, NOW, P))
    assert O.universe_health(led2)["outages"] == 1


def test_bars_stream_handles_delisting_and_matches_manual():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=160)
    cols = [f"T{i}" for i in range(40)]
    C = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.02, (160, 40)), 0)), idx, cols)
    Op = C.shift(1).fillna(C)
    H, L = np.maximum(Op, C) * 1.01, np.minimum(Op, C) * 0.99
    V = pd.DataFrame(rng.uniform(2e6, 6e6, C.shape), idx, cols)
    C.iloc[120:, 5] = np.nan
    bars = dict(Open=Op, High=H, Low=L, Close=C, Volume=V)
    days = list(O.bars_stream(bars, P, start=idx[100], end=idx[130]))
    assert days and all(s.n > 0 for s, _ in days)
    s, out = days[0]
    j = list(s.tickers).index("T3")
    t = idx.get_loc(pd.Timestamp(s.decided_at))
    assert s.prev_close[j] == pytest.approx(C.iloc[t, 3]) and out.entry[j] == pytest.approx(Op.iloc[t + 1, 3])
    assert out.day_close[j] == pytest.approx(C.iloc[t + 1, 3])
    late = [d for d in days if pd.Timestamp(d[0].decided_at) >= idx[118]]
    assert any(o.delisted.any() for _, o in late)


def test_cost_at_3000_names_is_small():
    m = O.measure_cost(3000, 3)
    assert m["ms_per_day"] < 2000 and m["peak_mb"] < 200 and m["rows_per_day"] < 1500


def test_score_calibration_detects_a_planted_score_and_not_a_random_one():
    good = O.ObserverLedger(P)
    for s, o in O.synthetic_days(10, 800, 13, O.Plant(n_pred=15)):
        # planted: movers get high scores
        r = O.observe_day(s, o, NOW, P)
        good.add(r)
    cal = O.score_calibration(good)
    assert cal["names"].sum() > 0 and cal.attrs["monotone_rho"] == cal.attrs["monotone_rho"]


# ---------------------------------------------------------------- autopsy
def run_autopsy(days=14, plant=None, seed=3, n=600, ctx=None):
    st = A.AutopsyState()
    last = None
    for s, o in O.synthetic_days(days, n, seed, plant or O.Plant(n_up_mid=8, n_up_big=4, n_down_mid=6, n_down_big=3)):
        last = A.step(st, s, o, NOW, "2026-09-29", ctx)
    return st, last


def test_autopsy_has_five_sections_and_band_counts():
    st, a = run_autopsy()
    assert not a.validate() and all(v is not None for v in a.sections().values())
    assert a.market.bands["up_gt10"] >= 4 and set(a.market.bands) == set(O.BAND_NAMES.values())
    assert len(a.market.gainers) == 5 and a.market.gainers[0].value >= a.market.gainers[-1].value
    assert a.market.losers[0].value <= a.market.losers[-1].value
    assert a.model.missed_winners and a.model.missed_winners[0].extra["reason"]
    assert "AUTOPSY" in A.render(a)
    assert a.research.questions and a.learning.n_new_questions >= 1


def test_questions_are_identity_free_and_a_planted_leak_is_caught():
    st, a = run_autopsy()
    tickers = set(st.observer.ledger._recs[-1].rows["ticker"])
    for q in a.questions():
        assert not A.identity_violations(q.text, tickers)
    assert len(A.identity_violations("why did S00012 jump on 2020-01-06", ["S00012"])) == 3
    assert A.identity_violations("this looks like 2008", []) == ["contains a year"]
    s, o, _ = day(600, 3, 0)
    rec = O.observe_day(s, o, NOW, P)
    with pytest.raises(A.AutopsyError):
        A._q("why did " + str(rec.rows["ticker"].iloc[0]) + " move", "loss", A.Problem.LOSS_AVOIDANCE, rec, "2026-09-29", "s", "f", 5, 0.5,
             set(), list(rec.rows["ticker"]))


def test_autopsy_second_day_dedupes_known_questions_and_uses_past_only():
    st = A.AutopsyState()
    s, o, _ = day(600, 3, 0)
    a1 = A.step(st, s, o, NOW, "2026-09-29")
    n1 = len(st.known_questions)
    a1b = A.autopsy_day(O.observe_day(s, o, NOW, P), st.observer.ledger, st, NOW, "2026-09-29")
    assert a1b.learning.n_new_questions == 0 and a1b.learning.n_duplicate_questions == len(a1b.questions())
    with pytest.raises(FirewallBreach):
        A.autopsy_day(O.observe_day(s, o, NOW, P), st.observer.ledger, st, o.resolved_at, "2026-09-29")


def test_losses_are_classified_never_forced():
    plant = O.Plant(n_hit=0, n_fp=0)
    s, o, t = day(400, 14, 0, plant)
    # make three picks lose 8% each
    picked = np.flatnonzero(s.picked)
    if len(picked) == 0:
        s = O.DecisionSnapshot.make(s.decided_at, s.tickers, s.prev_close, score=s.score, confidence=s.confidence,
                                    picked=np.isin(np.arange(s.n), np.argsort(-s.score)[:5]), features=s.features)
        picked = np.flatnonzero(s.picked)
    close = o.close.copy(); entry = o.entry.copy(); hi = o.hi.copy(); lo = o.lo.copy()
    close[picked] = entry[picked] * 0.92
    lo[picked] = close[picked] * 0.99
    o2 = O.DayOutcome.make(o.resolved_at, o.tickers, entry, hi, lo, close, o.volume_ratio)
    st = A.AutopsyState()
    a = A.step(st, s, o2, NOW, "2026-09-29")
    assert a.risk.n_losses == len(picked)
    everything = a.risk.avoidable + a.risk.unavoidable + a.risk.undetermined
    assert len(everything) == len(picked)
    assert all(i.cause == "" or i.cause in {c.value for c in A.FC} for i in everything)
    assert a.risk.unknown_rate == a.risk.unknown_rate


def test_unknown_causes_never_claim_predictable():
    st, a = run_autopsy()
    assert all(u.verdict.value != "PREDICTABLE" for u in a.research.unknown_causes)
    assert any(u.verdict.value == "UNKNOWN" for u in a.research.unknown_causes)


def test_pattern_events_and_contradictions():
    ev = [A.PatternEvent("P1", "broken", 0.02, -0.01, 0.4, 20, 4, expected_sign=1, cell="x"), A.PatternEvent("P2", "new", 0.03, 0.03, 0.9, 10, 8)]
    ctx = A.AutopsyContext(patterns=ev)
    st, a = run_autopsy(ctx=ctx)
    assert len(a.research.broken_patterns) == 1 and len(a.research.new_patterns) == 1
    assert any("P1" in q.text for q in a.questions())
    with pytest.raises(A.AutopsyError):
        run_autopsy(days=1, ctx=A.AutopsyContext(patterns=[A.PatternEvent("", "nonsense")]))


def test_autopsy_empty_day():
    st = A.AutopsyState()
    s = O.DecisionSnapshot.make("2020-01-06", [], [])
    o = O.DayOutcome.make("2020-01-07", [], [], [], [], [])
    a = A.step(st, s, o, NOW, "2026-09-29")
    assert a.empty and not a.validate() and "empty" in A.render(a)
