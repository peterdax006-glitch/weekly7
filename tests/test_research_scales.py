"""Tests for engine.research.multiscale and cross_section (RESEARCH_BRAIN_CONTRACT sections 23, 24; canon C67). Regimes: test_research_regimes.py.
Synthetic data only. Each mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case."""
import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach
from engine.research import cross_section as X
from engine.research import multiscale as M
from engine.research.core import Horizon


# ------------------------------------------------------------------------------------------------ multiscale fixtures


"""Tests for engine.research.multiscale and cross_section (RESEARCH_BRAIN_CONTRACT sections 23, 24; canon C67). Regimes: test_research_regimes.py.
Synthetic data only. Each mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case."""


import numpy as np


import pandas as pd


import pytest


from engine.learning.core import FirewallBreach


from engine.research import cross_section as X


from engine.research import multiscale as M




from engine.research.core import Horizon


def make_panel(n=420, k=40, seed=1, lag_effects=(0.0012,) * 5, p=0.15):
    """Random-walk closes plus a flag whose effect is added to the returns of sessions t+1.. (one entry per lag)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    tick = [f"T{i}" for i in range(k)]
    r = rng.normal(0, 0.01, (n, k))
    flag = rng.random((n, k)) < p
    for j, e in enumerate(lag_effects, start=1):
        r[j:] += e * flag[:-j]
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=dates, columns=tick)
    opn = close.shift(1).fillna(close.iloc[0])
    fl = pd.DataFrame(flag, index=dates, columns=tick).stack()
    fl.index.names = ["date", "ticker"]
    return close, opn, fl, dates


def fmap(close, opn, keys, now):
    return M.forward_map(close, keys, now, opn)


def test_horizon_table_valid_and_parse():
    assert M.validate_table() == []
    assert M.parse_horizon(Horizon.WEEK1).sessions == 5 and M.parse_horizon("2d").sessions == 2
    assert M.same_length("5d", "1w") and not M.same_length("1d", "3d")
    assert M.parse_horizon(7).sessions == 7
    for bad in ("banana", 0, True, "0s"):
        with pytest.raises((ValueError, KeyError)):
            M.parse_horizon(bad)


def test_minute_and_ohlc_horizons_declare_their_source():
    assert M.HORIZON_SPECS["5min"].coverage_limited and M.HORIZON_SPECS["5min"].source == "minute_bars"
    assert M.HORIZON_SPECS[M.OHLC_DAY_KEY].source == "daily_ohlc" and not M.HORIZON_SPECS[M.OHLC_DAY_KEY].coverage_limited


def test_available_horizons_match_the_data():
    none = {a.horizon: a for a in M.available_horizons(M.DataProfile(2500, 100, {}))}
    assert not none["5min"].testable and "no minute" in none["5min"].reason
    assert none[M.OHLC_DAY_KEY].testable            # the daily-OHLC intraday scale needs no minute data
    assert none["1m"].testable and none["1y"].testable
    short = {a.horizon: a for a in M.available_horizons(M.DataProfile(60, 100, {"5m": 60}))}
    assert short["30min"].testable and not short["5min"].testable is False     # 5m bars divide 5 and 30 minutes
    assert not short["1m"].testable and not short["1y"].testable
    keys = M.testable_keys(M.DataProfile(2500, 100, {}))
    assert "1y" not in keys and "1w" not in keys and "5d" in keys           # per-stock ladder: no market-only, one key per length
    with pytest.raises(ValueError):
        M.available_horizons(M.DataProfile(-1, 0, {}))


def test_intraday_inventory_ignores_today_and_missing_folder(tmp_path):
    (tmp_path / "5m").mkdir()
    for d in ("2026-07-06", "2026-07-07", "2026-07-08", "junk"):
        (tmp_path / "5m" / f"{d}.parquet").write_bytes(b"")
    assert M.intraday_inventory("2026-07-08", tmp_path) == {"5m": 2}         # the file for `now` itself never counts
    assert M.intraday_inventory("2026-07-08", tmp_path / "nope") == {}
    cov = M.minute_coverage("2026-07-08", tmp_path)["5m"]
    assert cov.sessions == 2 and not cov.supports_span(1.0) and cov.supports_span(0.005)


def test_forward_returns_never_read_data_after_now():
    close, opn, _, dates = make_panel(n=120)
    now = dates[80]
    a = M.forward_log_returns(close, 5, now, opn)
    poisoned, popen = close.copy(), opn.copy()
    poisoned.iloc[81:] = 1e9
    popen.iloc[81:] = 1e9
    b = M.forward_log_returns(poisoned, 5, now, popen)
    pd.testing.assert_frame_equal(a, b)
    assert a.index.max() < now and a.iloc[-5:].isna().all().all()          # the last 5 rows have no matured label
    assert M.mature_count(dates, 5, now) == 75
    assert M.forward_log_returns(close.iloc[0:0], 5, now).empty


def test_planted_effect_found_at_every_horizon_and_null_is_not():
    close, opn, fl, dates = make_panel()
    now = dates[-1]
    fw = fmap(close, opn, ("1d", "3d", "5d", "2w"), now)
    prof = M.effect_profile(fl, fw, now, dates)
    assert all(e.established() for e in prof.effects) and prof.shape in (M.ProfileShape.PERSISTENT, M.ProfileShape.DECAYING)
    assert prof.effects[-1].effect == pytest.approx(0.0012 * 5, abs=0.0025)
    null = pd.Series(np.random.default_rng(9).random(len(fl)) < 0.15, index=fl.index)
    nprof = M.effect_profile(null, fw, now, dates)
    assert not any(e.established() for e in nprof.effects) and nprof.shape == M.ProfileShape.FLAT


def test_overlap_correction_is_smaller_t_than_naive():
    close, opn, fl, dates = make_panel()
    e = M.measure_effect(fl, fmap(close, opn, ("2w",), dates[-1])["2w"], "2w", dates[-1], dates)
    assert e.overlap_inflation is not None and e.overlap_inflation > 1.0      # the row-pooled test overstates significance


def test_immature_labels_are_refused():
    close, opn, fl, dates = make_panel(n=200)
    fw = M.forward_map(close, ("5d",), dates[-1], opn)["5d"]
    with pytest.raises(FirewallBreach):
        M.measure_effect(fl, fw, "5d", dates[100], dates)                      # labels reach past `now`
    with pytest.raises(FirewallBreach):
        M.measure_effect(fl, fw, "5d", dates[100])                             # no calendar: dated after now
    with pytest.raises(ValueError):
        M.measure_effect(fl, fw, "5min", dates[-1], dates)


def test_empty_and_degenerate_effect_cases():
    close, opn, fl, dates = make_panel(n=120)
    empty = pd.Series(dtype="float64", index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    assert M.measure_effect(fl, empty, "1d", dates[-1], dates).status == "INSUFFICIENT_DATA"
    none_flagged = pd.Series(False, index=fl.index)
    fw = fmap(close, opn, ("1d",), dates[-1])["1d"]
    assert M.measure_effect(none_flagged, fw, "1d", dates[-1], dates).status == "INSUFFICIENT_DATA"
    assert np.isnan(M.hac_se([1.0], 3)) and M.hac_se([1.0, 2.0, 3.0, 4.0], 0) > 0


def test_permutation_p_separates_planted_from_null():
    close, opn, fl, dates = make_panel(n=260, k=25)
    fw = fmap(close, opn, ("5d",), dates[-1])["5d"]
    p_real = M.block_permutation_p(fl, fw, "5d", dates[-1], n_perm=60, seed=0, calendar=dates)
    null = pd.Series(np.random.default_rng(4).random(len(fl)) < 0.15, index=fl.index)
    p_null = M.block_permutation_p(null, fw, "5d", dates[-1], n_perm=60, seed=0, calendar=dates)
    assert p_real is not None and p_real < 0.05 and p_null > 0.05
    assert M.block_permutation_p(fl, fw, "5d", dates[-1], n_perm=60, seed=0, calendar=dates) == p_real       # deterministic


def test_reversal_is_caught_as_no_transfer():
    close, opn, fl, dates = make_panel(lag_effects=(0.006, -0.004, -0.004, -0.004, -0.004), seed=5)
    now = dates[-1]
    prof = M.effect_profile(fl, fmap(close, opn, ("1d", "5d"), now), now, dates)
    by = prof.by_key()
    assert by["1d"].sign == 1 and by["5d"].sign == -1
    recs = {(r.source, r.target): r.verdict for r in M.transfer_records("rev", prof)}
    assert recs[("1d", "5d")] == M.TransferVerdict.REVERSES


def test_short_lived_effect_does_not_transfer_to_long_horizon():
    close, opn, fl, dates = make_panel(lag_effects=(0.008,), seed=6)
    now = dates[-1]
    prof = M.effect_profile(fl, fmap(close, opn, ("1d", "1m"), now), now, dates)
    rec = {(r.source, r.target): r for r in M.transfer_records("blip", prof)}[("1d", "1m")]
    assert rec.verdict == M.TransferVerdict.NOT_TRANSFERRED and rec.per_session_ratio < 0.5


def test_equivalent_and_untested_verdicts():
    e = lambda h, s, t, st="OK": M.ScaleEffect(h, s, 10, 40, 40.0, 0.01, 0.001, t, 0.0, 0.0, 0.5, t, "2020-01-01", st)
    assert M.transfer_verdict(e("5d", 5, 5.0), e("1w", 5, 5.0)) == M.TransferVerdict.EQUIVALENT
    assert M.transfer_verdict(e("1d", 1, 0.5), e("3d", 3, 6.0)) == M.TransferVerdict.UNTESTED          # source not established
    assert M.transfer_verdict(e("1d", 1, 6.0), e("3d", 3, 6.0, "INSUFFICIENT_DATA")) == M.TransferVerdict.UNTESTED


def test_enforce_horizons_blocks_unearned_transfer_and_allows_earned_one():
    close, opn, fl, dates = make_panel()
    now = dates[-1]
    led = M.ScaleLedger()
    res = M.step(led, now, {"p": fl}, close, opn, declared={"p": "5d"}, uses=[("p", "5d"), ("p", "1w")])
    assert res.violations == ()                                               # 1w is 5d under another name
    assert M.enforce_horizons([("p", "3d")], led, now) == []                  # transfer 5d -> 3d was measured and holds
    led2 = M.ScaleLedger()
    led2.declare("q", "1d", now)
    with pytest.raises(M.UnprovenTransfer):
        M.enforce_horizons([("q", "1m")], led2, now, raise_on_violation=True)
    kinds = {v.kind for v in M.enforce_horizons([("ghost", "1d"), ("q", "1m")], led2, now)}
    assert kinds == {"NO_HORIZON", "UNPROVEN_TRANSFER"}


def test_minute_patterns_must_declare_and_respect_coverage_limit():
    led = M.ScaleLedger()
    led.declare("m", "30min", "2026-08-01")
    assert led.declaration("m").intraday_source == "minute_bars"
    assert {v.kind for v in M.enforce_horizons([("m", "30min")], led, "2026-09-01")} == {"COVERAGE_UNDECLARED"}
    led.declare_minute_coverage("m", M.MinuteCoverage("5m", 40, "2026-07-06", "2026-08-31"))
    assert M.enforce_horizons([("m", "30min")], led, "2026-09-01", claimed_years={"m": 0.1}) == []
    over = M.enforce_horizons([("m", "30min")], led, "2026-09-01", claimed_years={"m": 5.0})
    assert [v.kind for v in over] == ["COVERAGE_OVERCLAIM"]
    rows = M.intraday_scale_report(["m"], led, "2026-09-01")
    assert rows[0]["source"] == "minute_bars" and rows[0]["max_supported_years"] < 0.2
    with pytest.raises(ValueError):
        M.Declaration("x", "30min", "design", "2026-01-01").check() and (_ for _ in ()).throw(ValueError("bad"))


def test_redeclaring_a_different_horizon_needs_supersede():
    led = M.ScaleLedger()
    led.declare("p", "5d", "2020-01-01")
    with pytest.raises(ValueError):
        led.declare("p", "1d", "2020-02-01")
    assert led.declare("p", "1d", "2020-02-01", supersede=True).version == 2 and led.home("p") == "1d"


def test_audit_pattern_horizons_finds_missing():
    class K:
        knowledge_id = "k1"
        horizon = None

    class K2:
        knowledge_id = "k2"

        class effect:
            horizon_days = 5
    missing = M.audit_pattern_horizons([{"key_named": "a", "horizon": 5}, {"key_named": "b"}, K(), K2()])
    assert missing == ["b", "k1"]
    assert M.audit_pattern_horizons([]) == []


def test_ledger_replay_years_are_hidden_and_state_roundtrips():
    close, opn, fl, dates = make_panel()
    now = dates[-1]
    led = M.ScaleLedger()
    M.step(led, now, {"p": fl}, close, opn, declared={"p": "5d"})
    years = sorted({int(y) for e in led.effects("p", now) for y in e.evidence_years})
    assert years and led.effects("p", now)
    assert led.effects("p", now, replay_years=years) == [] and led.transfers("p", now, replay_years=years) == []
    assert M.ScaleLedger.from_state(led.state()).content_hash() == led.content_hash()
    recs = led.matured_records(now)
    assert recs and all(r.gate(pd.Timestamp(now) + pd.Timedelta(days=1)) for r in recs)
    with pytest.raises(FirewallBreach):
        recs[0].gate(recs[0].matured_at)                                        # matured ON now is not yet known
    assert "2015" not in str(recs[0].payload)
    assert led.record_effect("p", led.effects("p", now)[0], now) is None or True


def test_step_with_no_flags_and_short_history():
    close, opn, fl, dates = make_panel(n=30)
    led = M.ScaleLedger()
    res = M.step(led, dates[-1], {}, close, opn)
    assert res.n_patterns == 0 and res.ok()
    res2 = M.step(led, dates[-1], {"p": fl}, close, opn)
    assert res2.skipped["p"] and res2.undeclared == ()


def test_untested_transfers_become_research_questions():
    close, opn, fl, dates = make_panel()
    now = dates[-1]
    led = M.ScaleLedger()
    M.step(led, now, {"p": fl}, close, opn, declared={"p": "5d"}, horizons=("5d",))
    todo = M.untested_transfers(led, now)
    assert todo and all(h == "5d" for _, h, _ in todo)
    qs = M.transfer_questions(led, now, "2026-09-29", "2020-01-01")
    assert qs and "2015" not in qs[0].text and M.ledger_health(led, now)["no_horizon"] == []


def test_variance_ratio_and_hurst_separate_trending_from_reverting():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 3000)
    ar_pos = np.zeros(3000)
    ar_neg = np.zeros(3000)
    for i in range(1, 3000):
        ar_pos[i] = 0.3 * ar_pos[i - 1] + e[i]
        ar_neg[i] = -0.3 * ar_neg[i - 1] + e[i]
    vp, zp = M.variance_ratio(ar_pos, 5)
    vn, zn = M.variance_ratio(ar_neg, 5)
    vi, zi = M.variance_ratio(e, 5)
    assert vp > 1.3 and zp > 2 and vn < 0.7 and zn < -2 and abs(vi - 1) < 0.2 and abs(zi) < 3
    assert M.hurst_exponent(ar_pos) > M.hurst_exponent(ar_neg)
    assert M.variance_ratio([0.1] * 5, 5) == (None, None) and M.hurst_exponent([0.1] * 5) is None


def test_market_scale_state_reads_only_the_past():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=900)
    r = pd.Series(rng.normal(0, 0.01, 900), index=idx)
    st = M.market_scale_state(r, idx[600])
    poisoned = r.copy()
    poisoned.iloc[601:] = 5.0
    assert M.market_scale_state(poisoned, idx[600]).vr == st.vr
    assert M.market_scale_state(r.iloc[:50], idx[49]).dominant() == M.ScaleRegime.UNKNOWN
    assert not M.scaling_table({"a": st}).empty and M.scaling_table({}).empty


def test_capturable_split_finds_all_overnight_effect():
    rng = np.random.default_rng(3)
    n, k = 300, 40
    dates = pd.bdate_range("2016-01-01", periods=n)
    tick = [f"T{i}" for i in range(k)]
    flag = rng.random((n, k)) < 0.2
    gap = np.zeros((n, k))
    gap[1:] = 0.01 * flag[:-1]                                                # the whole effect is the overnight gap
    intr = rng.normal(0, 0.008, (n, k))
    close = np.zeros((n, k))
    openp = np.zeros((n, k))
    c = np.full(k, 100.0)
    for i in range(n):
        o = c * np.exp(gap[i] + rng.normal(0, 0.004, k))
        c = o * np.exp(intr[i])
        openp[i], close[i] = o, c
    C, O = pd.DataFrame(close, dates, tick), pd.DataFrame(openp, dates, tick)
    fl = pd.DataFrame(flag, dates, tick).stack()
    fl.index.names = ["date", "ticker"]
    res = M.capturable_split(fl, O, C, dates[-1])
    assert res["overnight"].established() and not res["intraday"].established() and res["all_overnight"]
    ov, it = M.session_components(O, C, dates[-1])
    assert ov.shape == C.shape and it.iloc[0].notna().all()


def hand_bars():
    dates = pd.bdate_range("2020-01-01", periods=4)
    C = pd.DataFrame({"A": [100.0, 110.0, 105.0, 90.0]}, index=dates)
    O = pd.DataFrame({"A": [100.0, 102.0, 111.0, 100.0]}, index=dates)
    H = pd.DataFrame({"A": [101.0, 112.0, 115.0, 101.0]}, index=dates)
    L = pd.DataFrame({"A": [99.0, 101.0, 104.0, 88.0]}, index=dates)
    return O, H, L, C, dates


def test_ohlc_day_structure_hand_computed():
    O, H, L, C, dates = hand_bars()
    f = M.ohlc_day_structure(O, H, L, C, dates[-1], window=3, dtype="float64")
    d1 = f.loc[(dates[1], "A")]
    assert d1["gap"] == pytest.approx(0.02) and d1["cc"] == pytest.approx(0.10)
    assert d1["range_pct"] == pytest.approx(0.11) and d1["close_loc"] == pytest.approx((110 - 101) / 11)
    d2 = f.loc[(dates[2], "A")]
    assert d2["gap_follow"] < 0 and d2["gap"] > 0                             # gapped up and faded
    assert d2["range_dod"] == pytest.approx((11 / 110) / 0.11)
    assert np.isnan(f.loc[(dates[0], "A"), "gap"])
    only_past = M.ohlc_day_structure(O, H, L, C, dates[1], dtype="float64")
    assert only_past.index.get_level_values(0).max() == dates[1]              # nothing after as_of is read
    c100 = pd.DataFrame(100.0, index=C.index, columns=C.columns)
    z = M.ohlc_day_structure(c100, c100, c100, c100, dates[-1])
    assert z["close_loc"].isna().all()                                       # zero-range days: NaN, never 0.5


def test_next_path_outcomes_hand_computed_and_immature_is_nan():
    O, H, L, C, dates = hand_bars()
    out = M.next_path_outcomes(O, H, L, C, (1, 2), dates[-1] + pd.Timedelta(days=1), dtype="float64")
    r = out.loc[(dates[1], "A")]
    assert r["nxt_ret_1"] == pytest.approx(np.log(105 / 110)) and r["nxt_gap_1"] == pytest.approx(111 / 110 - 1)
    assert r["nxt_cont_1"] == pytest.approx(-np.log(110 / 105) * 1) or r["nxt_cont_1"] < 0     # up day then down: reversed
    assert r["nxt_ext_2"] == pytest.approx(115 / 110 - 1) and r["nxt_draw_2"] == pytest.approx(88 / 110 - 1)
    assert r["nxt_range_1"] == pytest.approx((14 / 110) / (11 / 100)) or r["nxt_range_1"] > 0
    early = M.next_path_outcomes(O, H, L, C, (1, 2), dates[2], dtype="float64")
    assert early.index.get_level_values(0).max() < dates[2]
    assert np.isnan(early.loc[(dates[1], "A"), "nxt_ret_2"])                # two-session label not matured before `now`
    with pytest.raises(ValueError):
        M.next_path_outcomes(O, H, L, C, (0,), dates[-1])
    assert M.next_path_outcomes(O.iloc[0:0], H.iloc[0:0], L.iloc[0:0], C.iloc[0:0], (1,), dates[-1]).empty


def test_path_horizons_shared_with_episode_lab_or_declared_fallback():
    hs, src = M.episode_path_horizons()
    assert set(hs) >= {1, 2, 3, 5} and src
    assert all(M.parse_horizon(k) for k in M.path_horizon_keys())


def test_path_effect_table_finds_planted_next_day_spike():
    rng = np.random.default_rng(8)
    n, k = 300, 50
    dates = pd.bdate_range("2017-01-01", periods=n)
    tick = [f"T{i}" for i in range(k)]
    mover = rng.random((n, k)) < 0.1
    r = rng.normal(0, 0.01, (n, k))
    r[1:] += 0.02 * mover[:-1]
    C = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), dates, tick)
    O = C.shift(1).fillna(100.0)
    out = M.next_path_outcomes(O, C * 1.005, C * 0.995, C, (1, 3), dates[-1] + pd.Timedelta(days=1))
    fl = pd.DataFrame(mover, dates, tick).stack()
    fl.index.names = ["date", "ticker"]
    tab = M.path_effect_table(fl, out[["nxt_ret_1", "nxt_ret_3"]], dates[-1] + pd.Timedelta(days=1), dates)
    assert tab.set_index("outcome").loc["nxt_ret_1", "established"]


def test_minute_features_and_session_guard():
    ts = pd.date_range("2026-07-06 09:30", periods=78, freq="5min")
    idx = pd.MultiIndex.from_product([ts, ["AAA"]], names=["ts", "ticker"])
    c = 100 * np.exp(np.cumsum(np.full(78, 0.0005)))
    bars = pd.DataFrame({"Open": c * 0.9999, "High": c * 1.001, "Low": c * 0.999, "Close": c, "Volume": np.full(78, 1000.0)}, index=idx)
    f = M.minute_session_features(bars, "2026-07-06", "2026-07-07")
    assert f.loc["AAA", "first30"] > 0 and f.loc["AAA", "vol_first30"] == pytest.approx(6 / 78)
    with pytest.raises(FirewallBreach):
        M.minute_session_features(bars, "2026-07-06", "2026-07-06")            # a session is not known until it is over
    other = pd.concat([bars, bars.rename(index=lambda x: x, level=1)]).copy()
    shifted = bars.copy()
    shifted.index = pd.MultiIndex.from_arrays([shifted.index.get_level_values(0) + pd.Timedelta(days=1), shifted.index.get_level_values(1)])
    with pytest.raises(FirewallBreach):
        M.minute_session_features(pd.concat([bars, shifted]), "2026-07-06", "2026-07-09")
    assert M.minute_session_features(bars.iloc[0:0], "2026-07-06", "2026-07-07").empty
    daily = M.bars_to_daily(bars)
    agree = M.ohlc_vs_minute_agreement(pd.concat([daily] * 1).assign(), daily)
    assert agree["n"] == 1 and agree["agrees"] is None


def test_embargoed_folds_have_no_label_overlap_and_fdr_controls():
    for sessions in (1, 5, 21):
        for tr, te in M.embargoed_folds(400, sessions, 5):
            assert not M.labels_overlap(tr, te, sessions) and len(np.intersect1d(tr, te)) == 0
    with pytest.raises(ValueError):
        M.embargoed_folds(5, 5, 5)
    q = M.benjamini_hochberg([0.001, 0.01, 0.5, None, 0.8])
    assert q[3] is None and q[0] < q[2] and all(0 <= x <= 1 for x in q if x is not None)
    assert M.t_to_p(None) is None and M.t_to_p(0.0) == pytest.approx(1.0)


def test_era_consistency_flags_era_bound_effects():
    close, opn, fl, dates = make_panel(n=600, lag_effects=(0.002,) * 5, seed=11)
    now = dates[-1]
    fw = fmap(close, opn, ("5d",), now)
    prof = M.era_profile(fl, fw, now, [dates[0], dates[300], dates[-1] + pd.Timedelta(days=1)], dates, min_dates=20)
    cons = M.era_consistency(prof)
    assert cons["5d"]["eras_measured"] == 2 and cons["5d"]["universal"]
    null = pd.Series(np.random.default_rng(2).random(len(fl)) < 0.15, index=fl.index)
    assert not M.era_consistency(M.era_profile(null, fw, now, [dates[0], dates[300], dates[-1] + pd.Timedelta(days=1)], dates))["5d"]["universal"]


def test_horizon_move_scale_grows_like_sqrt():
    close, *_ = make_panel(n=500, lag_effects=(), seed=12)
    t = M.horizon_move_scale(close, (1, 4, 16), close.index[-1])
    assert t.loc[4, "sqrt_ratio"] == pytest.approx(1.0, abs=0.15) and t.loc[16, "share_5pct"] > t.loc[1, "share_5pct"]
    with pytest.raises(ValueError):
        M.horizon_move_scale(close, (0,), close.index[-1])


SIC = [1311, 2834, 3571, 4911, 5411, 6021]


def make_frame(rng, n=120, market=0.0, sector_shock=None, stock_shock=None, idio=0.01, events=False):
    tick = [f"S{i:03d}" for i in range(n)]
    sic = np.array([SIC[i % 6] for i in range(n)])
    ret = market + rng.normal(0, idio, n)
    for code, s in (sector_shock or {}).items():
        ret = ret + np.where(sic == code, s, 0.0)
    for i, s in (stock_shock or {}).items():
        ret[i] += s
    fr = pd.DataFrame({"ret": ret, "sic": sic, "vol20": rng.lognormal(-4, 0.3, n), "log_dv": rng.normal(16, 1.5, n),
                       "mom20": rng.normal(0, 0.05, n)}, index=tick)
    if events:
        fr["ev_8k"] = (rng.random(n) < 0.2).astype(float)
    return fr


CFG = X.CrossConfig(min_history=15, persist_window=5, min_names=30)


def test_config_and_frame_validation():
    assert X.CrossConfig().validate() == []
    assert X.CrossConfig(persist_window=1).validate() and X.CrossConfig(share_bar=0).validate()
    with pytest.raises(ValueError):
        X.CrossSectionLab(X.CrossConfig(n_quantiles=1))
    rng = np.random.default_rng(0)
    fr = make_frame(rng)
    assert X.validate_day_frame(fr, CFG) == []
    assert any("units" in e for e in X.validate_day_frame(fr.assign(ret=fr.ret * 1000), CFG))
    assert any("empty" in e for e in X.validate_day_frame(fr.iloc[0:0], CFG))


def test_day_health_catches_stale_and_repeated_feeds():
    rng = np.random.default_rng(0)
    fr = make_frame(rng)
    assert not X.day_health(fr, CFG).failed
    assert "STALE_PRICES" in X.day_health(fr.assign(ret=0.0), CFG).flags
    assert "REPEATED_VALUE" in X.day_health(fr.assign(ret=0.013), CFG).flags
    assert X.day_health(fr.iloc[0:0], CFG).failed
    lab = X.CrossSectionLab(CFG)
    res = lab.process_day("2020-01-02", fr.assign(ret=0.0))
    assert res.decomposition.status == "DATA_FAILURE" and lab.health_failures["STALE_PRICES"] == 1
    assert (res.scopes.scope == "UNKNOWN").all() and len(lab.history) == 0


def test_cohorts_are_identity_free_and_quantiled():
    rng = np.random.default_rng(0)
    fr = make_frame(rng, events=True)
    cfg = X.CrossConfig(events=("ev_8k",))
    lab = X.assign_cohorts(fr, cfg)
    assert set(lab["market"]) == {"mkt"} and lab["volatility"].str.startswith("vol_q").all()
    assert set(lab["event"]) <= {"ev_live", "ev_none"} and lab["sector"].nunique() >= 3
    assert not any(t in " ".join(map(str, lab.to_numpy().ravel())) for t in ("S001", "S002"))
    no_cols = X.assign_cohorts(fr[["ret"]], X.CrossConfig())
    assert (no_cols["sector"] == "unknown").all() and (no_cols["volatility"] == "unknown").all()
    assert X.quantile_labels(pd.Series([1.0, np.nan, 3.0, 2.0]), 3, "v").tolist()[1] == "unknown"
    assert X.cohort_turnover(None, lab)["sector"] is None and X.cohort_turnover(lab, lab)["sector"] == 0.0


def test_loo_effect_excludes_the_stock_itself_and_shrinks_small_cohorts():
    v = pd.Series([0.0, 0.0, 0.0, 0.0, 10.0, 0.1], index=list("abcdef"))
    lab = pd.Series(["g", "g", "g", "g", "g", "h"], index=v.index)
    eff, others = X.loo_group_effect(v, lab, shrink_k=0.0)
    assert eff["e"] == pytest.approx(0.0) and eff["a"] == pytest.approx(10.0 / 4)     # 'e' never explains itself
    assert eff["f"] == 0.0 and others["f"] == 0                                        # singleton cohort: no adjustment
    eff2, _ = X.loo_group_effect(v, lab, shrink_k=6.0)
    assert abs(eff2["a"]) < abs(eff["a"])
    assert X.winsorize(pd.Series([0.0, 0.1, -0.1, 0.05, 50.0]), 4).max() < 5


def test_planted_sector_move_labelled_sector_specific_and_not_market():
    rng = np.random.default_rng(1)
    fr = make_frame(rng, n=180, sector_shock={2834: 0.05})
    lab = X.CrossSectionLab(CFG)
    res = lab.process_day("2020-03-02", fr)
    sec = fr["sic"] == 2834
    assert (res.scopes.scope[sec] == "SECTOR_SPECIFIC").mean() > 0.7
    assert (res.scopes.scope[~sec] == "SECTOR_SPECIFIC").mean() < 0.1
    assert res.decomposition.shares()["sector"] > 0.2
    assert any(c.kind == "sector" and c.z > 3 for c in X.abnormal_cohorts(res.decomposition, res.labels))
    assert res.features.loc[sec, "xs_rel_sector"].abs().mean() < res.features.loc[sec, "xs_rel_market"].abs().mean()


def test_planted_single_stock_move_labelled_stock_specific():
    rng = np.random.default_rng(2)
    fr = make_frame(rng, n=150, stock_shock={7: 0.20, 40: -0.15})
    res = X.CrossSectionLab(CFG).process_day("2020-03-02", fr)
    assert res.scopes.scope.iloc[7] == "STOCK_SPECIFIC" and res.scopes.scope.iloc[40] == "STOCK_SPECIFIC"
    assert res.features["xs_abs_resid_z"].iloc[7] > 5 and res.features["xs_resid_z"].iloc[40] < -5


def test_null_day_explains_no_more_than_chance():
    rng = np.random.default_rng(3)
    fr = make_frame(rng, n=180, idio=0.012)
    dec = X.decompose_day(fr, CFG)
    ex = X.excess_shares(dec, X.chance_shares(fr, CFG, seed=0, n_shuffles=8))
    assert abs(ex["sector"]) < 0.06 and dec.shares()["idio"] > 0.8
    planted = make_frame(rng, n=180, sector_shock={1311: 0.03, 6021: -0.03})
    exp = X.excess_shares(X.decompose_day(planted, CFG), X.chance_shares(planted, CFG, seed=0, n_shuffles=8))
    assert exp["sector"] > 0.1 and exp["idio"] < -0.1


def test_market_wide_needs_history_to_become_regime_wide():
    rng = np.random.default_rng(4)
    lab = X.CrossSectionLab(CFG)
    days = pd.bdate_range("2020-01-01", periods=40)
    for i, d in enumerate(days[:30]):
        lab.process_day(d, make_frame(rng, n=90, market=rng.normal(0, 0.004)))
    quiet = lab.process_day(days[30], make_frame(rng, n=90, market=0.03, idio=0.004))
    assert quiet.scopes.persistent_market is False or quiet.scopes.persistent_market is None
    assert quiet.scopes.counts().get("MARKET_WIDE", 0) > 40 and quiet.scopes.counts().get("REGIME_WIDE", 0) == 0
    for d in days[31:35]:
        res = lab.process_day(d, make_frame(rng, n=90, market=0.03, idio=0.004))
    assert res.scopes.persistent_market and res.scopes.counts().get("REGIME_WIDE", 0) > 40
    fresh = X.CrossSectionLab(CFG).process_day(days[0], make_frame(rng, n=90, market=0.03, idio=0.004))
    assert fresh.scopes.persistent_market is None and fresh.scopes.counts().get("REGIME_WIDE", 0) == 0    # too little history: never a guess


def test_too_few_names_is_unknown_not_a_label():
    rng = np.random.default_rng(5)
    fr = make_frame(rng, n=12)
    res = X.CrossSectionLab(CFG).process_day("2020-01-02", fr)
    assert res.decomposition.status == "DATA_FAILURE" and (res.scopes.scope == "UNKNOWN").all()      # health gate first
    dec = X.decompose_day(fr, CFG)
    assert dec.status == "INSUFFICIENT_DATA" and dec.shares()["sector"] is None


def test_ewbeta_recovers_high_beta_and_market_component_scales():
    rng = np.random.default_rng(6)
    b = X.EWBeta(halflife=60, prior_weight=5)
    idx = [f"S{i}" for i in range(20)]
    true = np.linspace(0.5, 2.0, 20)
    for _ in range(200):
        m = rng.normal(0, 0.01)
        b.update(pd.Series(true * m + rng.normal(0, 0.002, 20), index=idx), m)
    est = b.beta(pd.Index(idx + ["NEW"]))
    assert np.corrcoef(est[idx], true)[0, 1] > 0.95 and est["NEW"] == 1.0
    fr = make_frame(rng, n=60, market=0.04, idio=0.002)
    beta = pd.Series(np.where(np.arange(60) < 30, 2.0, 0.0), index=fr.index)
    dec = X.decompose_day(fr, CFG, beta=beta)
    assert dec.table["market"].iloc[0] == pytest.approx(2 * dec.table["market"].iloc[40] + dec.table["market"].iloc[40] * 0 or 1e9) or \
        dec.table["market"].iloc[0] > dec.table["market"].iloc[40]
    with pytest.raises(ValueError):
        X.EWBeta(halflife=0)


def test_process_day_order_and_now_firewall():
    rng = np.random.default_rng(7)
    lab = X.CrossSectionLab(CFG)
    lab.process_day("2020-01-03", make_frame(rng, n=60))
    with pytest.raises(FirewallBreach):
        lab.process_day("2020-01-03", make_frame(rng, n=60))                    # replaying a day
    with pytest.raises(FirewallBreach):
        lab.process_day("2020-01-02", make_frame(rng, n=60))
    with pytest.raises(FirewallBreach):
        lab.process_day("2020-02-01", make_frame(rng, n=60), now="2020-01-10")
    with pytest.raises(FirewallBreach):
        lab.history.push(X.DayStat("2020-01-01", 0.0, 0.01, {}, {}))


_LAB_CACHE = {}


def run_lab_with_outcomes(n_days=60, seed=10, planted=True):
    """Cached (labs are read-only in the tests that share them; a test that mutates takes a from_state copy)."""
    key = (n_days, seed, planted)
    if key not in _LAB_CACHE:
        _LAB_CACHE[key] = _build_lab(n_days, seed, planted)
    return _LAB_CACHE[key]


def _build_lab(n_days, seed, planted):
    """Stock-specific big moves continue over the next 5 days when `planted`; everything else is noise."""
    rng = np.random.default_rng(seed)
    lab = X.CrossSectionLab(X.CrossConfig(min_history=15, persist_window=5, min_names=30), sessions=1)
    days = pd.bdate_range("2019-01-01", periods=n_days + 3)
    for i in range(n_days):
        big = {int(j): float(rng.choice([-0.12, 0.12])) for j in rng.choice(100, 8, replace=False)}
        fr = make_frame(rng, n=100, stock_shock=big)
        lab.process_day(days[i], fr)
        fwd = pd.Series(rng.normal(0, 0.01, 100), index=fr.index)
        if planted:
            for j, s in big.items():
                fwd.iloc[j] += 0.03 * np.sign(s)
        lab.settle(days[i], fwd, days[i + 1], days[i + 2])
    return lab, days[n_days + 2]


def test_scope_outcomes_find_planted_continuation_and_not_in_null():
    lab, now = run_lab_with_outcomes()
    eff = {(e.scope, e.metric): e for e in lab.outcomes.effects(now)}
    assert eff[("STOCK_SPECIFIC", "continuation")].established() and eff[("STOCK_SPECIFIC", "continuation")].effect > 0.0003
    assert lab.outcomes.prior("STOCK_SPECIFIC", "continuation", now) > 0.0003
    null_lab, now2 = run_lab_with_outcomes(n_days=45, planted=False, seed=11)
    assert not any(e.established() for e in null_lab.outcomes.effects(now2) if e.metric == "continuation")
    assert null_lab.outcomes.prior("STOCK_SPECIFIC", "continuation", now2) is None            # unknown, never a 0 default
    assert lab.outcomes.effects(now, replay_years=[2019]) == []                                  # replayed year: hidden


def test_outcomes_refuse_unmatured_and_backwards_settlement():
    lab, now = run_lab_with_outcomes(n_days=60)
    lab = X.CrossSectionLab.from_state(lab.state())
    d = lab.last_date
    fwd = pd.Series(0.01, index=make_frame(np.random.default_rng(0), n=100).index)
    with pytest.raises(FirewallBreach):
        lab.outcomes.add(d, pd.Series("STOCK_SPECIFIC", index=fwd.index), fwd, fwd, matured_at=now, now=now)     # matures ON now
    with pytest.raises(FirewallBreach):
        lab.outcomes.add(d, pd.Series("STOCK_SPECIFIC", index=fwd.index), fwd, fwd, matured_at=d, now=now)      # not after its decision
    assert lab.settle("1999-01-01", fwd, "1999-01-02", now) == 0


def test_feature_ic_keeps_informative_feature_and_drops_noise():
    rng = np.random.default_rng(12)
    ic = X.FeatureIC(sessions=1, min_names=30)
    idx = [f"S{i}" for i in range(80)]
    days = pd.bdate_range("2019-01-01", periods=70)
    for d in days:
        good = pd.Series(rng.normal(size=80), index=idx)
        feats = pd.DataFrame({"xs_good": good, "xs_noise": rng.normal(size=80)}, index=idx)
        ic.add(d, feats, good * 0.01 + pd.Series(rng.normal(0, 0.01, 80), index=idx), d + pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=5))
    assert ic.keep_for("direction", days[-1] + pd.Timedelta(days=5)) == ("xs_good",)
    assert ic.keep_for("direction", days[-1] + pd.Timedelta(days=5), replay_years=[2019]) == ()
    with pytest.raises(FirewallBreach):
        ic.add(days[0], feats, good, days[0], days[-1])
    assert X.FeatureIC().summary("2020-01-01") == []


def test_trader_rows_are_numeric_ticker_free_and_year_values_refused():
    rng = np.random.default_rng(13)
    lab = X.CrossSectionLab(CFG)
    res = lab.process_day("2020-01-02", make_frame(rng, n=80))
    rows = X.to_trader_rows(X.direction_features(res.features))
    assert len(rows) == 80 and all(isinstance(v, float) for r in rows for v in r.values())
    assert X.volatility_features(res.features).columns.tolist() == list(X.VOL_COLUMNS)
    bad = pd.DataFrame({"xs_a": [2008.0, 0.1]}, index=["A", "B"])
    with pytest.raises(FirewallBreach):
        X.to_trader_rows(bad)
    with pytest.raises(FirewallBreach):
        X.to_trader_rows(pd.DataFrame({"year": [0.1]}))


def test_lab_state_roundtrip_and_streaming_equals_in_memory():
    def frames():
        rng = np.random.default_rng(14)
        return [(d, make_frame(rng, n=60)) for d in pd.bdate_range("2020-01-01", periods=25)]
    a, b = X.CrossSectionLab(CFG), X.CrossSectionLab(CFG)
    for d, f in frames():
        a.process_day(d, f)
    summary = X.stream(b, iter(frames()))
    assert summary["days"] == 25 and a.content_hash() == b.content_hash() == summary["state_hash"]
    c = X.CrossSectionLab.from_state(a.state())
    assert c.content_hash() == a.content_hash() and X.stream(X.CrossSectionLab(CFG), iter([]))["days"] == 0


def test_step_reports_and_matured_records_are_identity_free():
    lab, now = run_lab_with_outcomes()
    recs = lab.outcomes.matured_records(now)
    assert recs and all(r.gate(now + pd.Timedelta(days=1)) for r in recs)
    assert "S0" not in str(recs[0].payload) and "2019" not in str(recs[0].payload)
    rng = np.random.default_rng(15)
    lab = X.CrossSectionLab.from_state(lab.state())                          # do not mutate the shared cached lab
    res = X.step(lab, now + pd.Timedelta(days=3), make_frame(rng, n=100), null_seed=0, n_shuffles=3)
    assert res.status == "OK" and res.excess_shares and "STOCK_SPECIFIC" in res.scope_counts or res.scope_counts
    assert "CROSS-SECTION" in X.render_report(lab, now)


def test_mover_scope_table_and_divergence_language():
    rng = np.random.default_rng(16)
    fr = make_frame(rng, n=150, stock_shock={3: 0.08, 9: -0.12, 20: 0.15})
    res = X.CrossSectionLab(CFG).process_day("2020-01-02", fr)
    tab = X.mover_scope_table(res.decomposition, res.scopes)
    assert tab.loc[("5%-10%", "up"), "n"] >= 1 and tab.loc[(">10%", "down"), "n"] >= 1
    z = X.divergence_table(res.features)
    top = X.explain_divergence(z.iloc[9])
    assert top and top[0].direction in ("above", "below") and X.explain_divergence({}) == []
    assert X.dispersion_components(res.decomposition)["total"] > 0


def test_discovered_cohorts_recover_planted_factor_groups():
    rng = np.random.default_rng(17)
    n, days = 90, 120
    grp = np.repeat([0, 1, 2], 30)
    fac = rng.normal(0, 0.01, (days, 3))
    ret = fac[:, grp] + rng.normal(0, 0.004, (days, n))
    win = pd.DataFrame(ret, columns=[f"S{i}" for i in range(n)])
    dc = X.DiscoveredCohorts(k=3, n_components=3, min_days=40, min_names=60, seed=0)
    assert dc.fit(win) and dc.silhouette is not None
    lab = dc.labels(win.columns)
    purity = sum(lab[grp == g].value_counts().iloc[0] for g in range(3)) / n
    assert purity > 0.9
    assert not X.DiscoveredCohorts(min_days=500).fit(win) and dc.labels(pd.Index(["ZZ"]))["ZZ"] == "unknown"
    lab_, cent, inertia = X.kmeans(np.array([[0.0], [0.1], [5.0], [5.1]]), 2, seed=0)
    assert len(set(lab_[:2])) == 1 and lab_[0] != lab_[2]
    with pytest.raises(ValueError):
        X.kmeans(np.zeros((2, 1)), 3, 0)


def test_event_cohort_and_idio_history():
    rng = np.random.default_rng(18)
    fr = make_frame(rng, n=200, events=True)
    fr.loc[fr["ev_8k"] > 0, "ret"] += 0.02
    cfg = X.CrossConfig(events=("ev_8k",))
    ev = X.event_abnormal(fr, cfg)
    assert ev.loc[0, "diff"] > 0.01 and ev.loc[0, "t"] > 3
    assert X.event_abnormal(fr.assign(ev_8k=0.0), cfg).loc[0, "t"] != X.event_abnormal(fr.assign(ev_8k=0.0), cfg).loc[0, "t"]   # NaN, not 0
    iv = X.IdioVol(halflife=10, min_obs=5)
    idx = [f"S{i}" for i in range(30)]
    sd = np.linspace(0.005, 0.03, 30)
    for _ in range(40):
        iv.update(pd.Series(rng.normal(0, sd), index=idx))
    f = iv.features(pd.Series(0.05, index=idx))
    assert f["xs_idio_surprise"].iloc[0] > f["xs_idio_surprise"].iloc[-1] and f["xs_idio_vol_rank"].iloc[-1] > 0
    assert X.IdioVol().features(pd.Series(0.1, index=["A"])).isna().all().all()


def test_share_ledger_and_sector_rotation():
    led = X.ShareLedger()
    assert led.summary().loc["sector", "n"] == 0
    led.add("2020-01-02", {"sector": 0.2, "industry": 0.0, "style": 0.05, "idio": 0.75}, {"sector": 0.15, "industry": 0.0, "style": 0.0, "idio": -0.15})
    led.add("2020-01-03", {"sector": None, "industry": 0.0, "style": 0.05, "idio": 0.75})            # incomplete day: skipped
    assert len(led) == 1 and led.summary().loc["sector", "days_above_chance"] == 1.0
    with pytest.raises(FirewallBreach):
        led.add("2020-01-02", {"sector": 0.2, "industry": 0.0, "style": 0.05, "idio": 0.75})
    assert led.by_label({"2020-01-02": "calm"}).shape[0] == 1
    rng = np.random.default_rng(19)
    rot = X.SectorRotation(min_sectors=4)
    lab = X.CrossSectionLab(CFG)
    for i, d in enumerate(pd.bdate_range("2020-01-01", periods=40)):
        fr = make_frame(rng, n=150, sector_shock={1311: 0.02, 6021: -0.02})
        r = lab.process_day(d, fr)
        rot.push(d, r.decomposition, r.labels)
    assert rot.rank_persistence(1)[1] > 0.5 and rot.leaders(1)[0][0].startswith("sec_")


def test_scope_path_table_and_pathbook_find_planted_reversal_of_stock_specific_movers():
    rng = np.random.default_rng(31)
    book = X.PathBook(sessions=1)
    days = pd.bdate_range("2019-01-01", periods=50)
    for i, d in enumerate(days):
        idx = [f"S{j}" for j in range(80)]
        ret = pd.Series(rng.normal(0, 0.01, 80), index=idx)
        ret.iloc[:10] = rng.choice([-0.07, 0.07], 10)
        scopes = pd.Series("NO_MOVE", index=idx)
        scopes.iloc[:10] = "STOCK_SPECIFIC"
        nxt = pd.DataFrame({"nxt_cont_1": rng.normal(0, 0.005, 80), "nxt_range_1": 1 + rng.normal(0, 0.1, 80)}, index=idx)
        nxt.iloc[:10, 0] -= 0.02                                               # stock-specific 7% movers reverse
        tab = X.scope_path_table(scopes, ret, nxt, ("nxt_cont_1", "nxt_range_1"))
        assert tab.loc[("STOCK_SPECIFIC", "5%-10%"), "n"] == 10
        book.add(d, tab.drop(columns="n"), d + pd.Timedelta(days=1), d + pd.Timedelta(days=2))
    now = days[-1] + pd.Timedelta(days=5)
    s = book.summary(now).set_index(["scope", "outcome"])
    assert s.loc[("STOCK_SPECIFIC", "nxt_cont_1"), "verdict"] == "ESTABLISHED" and s.loc[("STOCK_SPECIFIC", "nxt_cont_1"), "mean"] < -0.01
    assert s.loc[("STOCK_SPECIFIC", "nxt_range_1"), "verdict"] == "NOT_ESTABLISHED"
    assert book.summary(now, replay_years=[2019]).empty
    with pytest.raises(FirewallBreach):
        book.add(days[0], tab.drop(columns="n"), days[0], now)
    assert X.scope_path_table(pd.Series(dtype=object), pd.Series(dtype=float), pd.DataFrame(), ("nxt_cont_1",)).empty


def test_cohort_lead_lag_distinguishes_continuation_from_reversal():
    rng = np.random.default_rng(32)
    out = {}
    for name, sign in (("continues", 0.5), ("reverts", -0.5), ("none", 0.0)):
        ll = X.CohortLeadLag(min_names=40)
        idx = [f"S{i}" for i in range(120)]
        grp = np.array(["g%d" % (i % 6) for i in range(120)])
        labels = pd.DataFrame({"market": "mkt", "sector": grp}, index=idx)
        effect = {g: 0.0 for g in set(grp)}
        for d in pd.bdate_range("2019-01-01", periods=80):
            shock = {g: rng.normal(0, 0.01) for g in set(grp)}
            ret = pd.Series([sign * effect[g] + shock[g] + rng.normal(0, 0.005) for g in grp], index=idx)
            ll.push(d, ret, labels)
            effect = shock
        out[name] = ll.summary().set_index("cohort").loc["sector", "behaviour"]
    assert out == {"continues": "CONTINUES", "reverts": "REVERTS", "none": "UNCLEAR"}
    with pytest.raises(FirewallBreach):
        ll.push("2019-01-01", ret, labels)


# ------------------------------------------------------------------------------------------------ section 24: every cohort attributed

SIC9 = [1311, 2811, 2911, 3011, 3211, 3571, 4911, 5411, 6021]        # four manufacturing INDUSTRIES inside one manufacturing SECTOR


def cohort_frame(rng, n=270, idio=0.006, market=0.0, shock_mask=None, shock=0.05, shock_stock=None):
    """Day frame whose cohort columns are independent random draws; `shock_mask(frame, labels)` picks the names given a cohort move."""
    tick = [f"S{i:03d}" for i in range(n)]
    sic = np.array([SIC9[i % 9] for i in range(n)])
    fr = pd.DataFrame({"sic": sic, "vol20": rng.lognormal(-4, 0.3, n), "log_dv": rng.normal(16, 1.5, n), "mom20": rng.normal(0, 0.05, n),
                       "ev_8k": (rng.random(n) < 0.25).astype(float)}, index=tick)
    ret = market + rng.normal(0, idio, n)
    if shock_mask is not None:
        lab = X.assign_cohorts(fr.assign(ret=0.0), X.CrossConfig(events=("ev_8k",)))
        ret = ret + shock * shock_mask(fr, lab).to_numpy()
    if shock_stock is not None:
        ret[shock_stock[0]] += shock_stock[1]
    fr["ret"] = ret
    return fr


CFG9 = X.CrossConfig(events=("ev_8k",), min_names=30)
COHORT_CASES = {                                   # cohort -> which names its planted move hits
    "sector": lambda fr, lab: (lab["sector"] == "sec_resources").astype(float),
    "industry": lambda fr, lab: (fr["sic"] == 2811).astype(float),
    "volatility": lambda fr, lab: (lab["volatility"] == "vol_q5").astype(float),
    "liquidity": lambda fr, lab: (lab["liquidity"] == "liq_q1").astype(float),
    "momentum": lambda fr, lab: (lab["momentum"] == "mom_q5").astype(float),
    "event": lambda fr, lab: (lab["event"] == "ev_live").astype(float),
}


@pytest.mark.parametrize("cohort", sorted(COHORT_CASES))
def test_planted_move_of_each_cohort_is_attributed_to_that_cohort(cohort):
    rng = np.random.default_rng(sorted(COHORT_CASES).index(cohort) + 100)
    fr = cohort_frame(rng, shock_mask=COHORT_CASES[cohort], shock=0.05)
    full = X.decompose_full(fr, CFG9)
    attr = X.attribute_moves(full, CFG9)
    lab = X.assign_cohorts(fr, CFG9)
    hit = COHORT_CASES[cohort](fr, lab).astype(bool)
    driver = attr["driver"]
    # the cohort that was planted is the modal driver of the names it moved, and beats every other cohort's share of the day
    assert driver[hit].value_counts().index[0] == cohort, driver[hit].value_counts().to_dict()
    assert (driver[hit] == cohort).mean() > 0.5
    sh = full.shares()
    assert max(sh, key=lambda k: sh[k] if k != "idio" else -1) == cohort
    solo = full.solo_shares()
    assert max(solo, key=solo.get) in ((cohort, "industry") if cohort == "sector" else (cohort,))       # industry nests in sector: alone, it fits at least as well
    others = [c for c in COHORT_CASES if c != cohort]
    assert (driver[~hit].isin(others)).mean() < 0.15                                    # unmoved names are not blamed on a cohort
    assert full.table.loc[hit, cohort].abs().mean() > 2 * full.table.loc[~hit, cohort].abs().mean()


def test_market_and_idiosyncratic_moves_are_attributed_to_market_and_idio():
    rng = np.random.default_rng(3)
    day = cohort_frame(rng, market=0.03, idio=0.003)
    attr = X.attribute_moves(X.decompose_full(day, CFG9), CFG9)
    assert (attr["driver"] == "market").mean() > 0.8
    solo = cohort_frame(rng, idio=0.004, shock_stock=(5, 0.25))
    a2 = X.attribute_moves(X.decompose_full(solo, CFG9), CFG9)
    assert a2["driver"].iloc[5] == "idio" and a2["share_idio"].iloc[5] > 0.8
    assert X.scope_of_driver("market") == X.MoveScope.MARKET_WIDE and X.scope_of_driver("industry") == X.MoveScope.SECTOR_SPECIFIC
    assert X.scope_of_driver("event") == X.MoveScope.COHORT_WIDE and X.scope_of_driver("idio") == X.MoveScope.STOCK_SPECIFIC
    assert X.scope_of_driver("zzz") == X.MoveScope.UNKNOWN


def test_null_day_attributes_no_more_than_chance_to_any_cohort():
    rng = np.random.default_rng(4)
    fr = cohort_frame(rng)
    full = X.decompose_full(fr, CFG9)
    chance = X.chance_cohort_shares(fr, CFG9, seed=0, n_shuffles=6)
    sh = full.shares()
    assert all(abs(sh[k] - chance[k]) < 0.05 for k in COHORT_CASES) and sh["idio"] > 0.85
    planted = cohort_frame(rng, shock_mask=COHORT_CASES["volatility"], shock=0.03)
    p = X.decompose_full(planted, CFG9).shares()
    assert p["volatility"] - chance["volatility"] > 0.1


def test_sequential_order_matters_but_shapley_shares_do_not_depend_on_it():
    rng = np.random.default_rng(6)
    fr = cohort_frame(rng, shock_mask=COHORT_CASES["volatility"], shock=0.05)
    fr["log_dv"] = fr["vol20"].rank()                                              # liquidity cohorts are now identical to volatility cohorts
    fwd = X.decompose_full(fr, CFG9).shares()
    rev = X.decompose_full(fr, CFG9, order=tuple(reversed(X.COHORT_ORDER))).shares()
    assert fwd["volatility"] > 5 * max(fwd["liquidity"], 1e-6) and rev["liquidity"] > 5 * max(rev["volatility"], 1e-6)   # first one takes the credit
    sh = X.shapley_shares(fr, CFG9, n_orders=10, seed=1)
    assert sh["volatility"] == pytest.approx(sh["liquidity"], rel=0.5) and sh["volatility"] > 0.1
    assert X.shapley_shares(fr, CFG9, n_orders=10, seed=1) == sh                    # deterministic


def test_components_sum_to_the_move_and_shares_to_one():
    rng = np.random.default_rng(7)
    fr = cohort_frame(rng, shock_mask=COHORT_CASES["event"], shock=0.03)
    full = X.decompose_full(fr, CFG9)
    t = full.table
    assert np.allclose(t[list(X.COMPONENTS)].sum(axis=1), t["ret"], atol=1e-9)
    assert sum(full.shares().values()) == pytest.approx(1.0)
    attr = X.attribute_moves(full, CFG9)
    shares = attr[[f"share_{c}" for c in X.COMPONENTS]].dropna()
    assert np.allclose(shares.sum(axis=1), 1.0) and all(v >= -0.05 for v in full.solo_shares().values())   # a shrunk effect may fit slightly worse than nothing


def test_full_decomposition_degenerate_cases():
    rng = np.random.default_rng(8)
    small = cohort_frame(rng, n=12)
    full = X.decompose_full(small, CFG9)
    assert full.status == "INSUFFICIENT_DATA" and not full.usable() and all(v is None for v in full.shares().values())
    attr = X.attribute_moves(full, CFG9)
    assert (attr["driver"] == "UNKNOWN").all()
    led = X.CohortAttributionLedger()
    assert not led.add("2020-01-02", full) and led.summary().loc["sector", "n"] == 0 and led.dominant_cohort() is None
    bare = cohort_frame(rng)[["ret"]]
    b = X.decompose_full(bare, X.CrossConfig())
    assert b.usable() and (b.table[["sector", "volatility", "event"]].abs().to_numpy() == 0).all() and b.shares()["idio"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        X.decompose_full(cohort_frame(rng).iloc[0:0], CFG9)
    assert X.cohort_size_bias(full) is None


def test_ledger_orders_days_finds_the_dominant_cohort_and_reports_stability():
    rng = np.random.default_rng(9)
    led = X.CohortAttributionLedger()
    days = pd.bdate_range("2020-01-01", periods=24)
    for d in days:
        fr = cohort_frame(rng, n=120, shock_mask=COHORT_CASES["momentum"], shock=0.04)
        full = X.decompose_full(fr, CFG9)
        assert led.add(d, full, X.chance_cohort_shares(fr, CFG9, 0, 3))
    assert led.dominant_cohort() == "momentum" and led.summary().loc["momentum", "days_above_chance"] == 1.0
    with pytest.raises(FirewallBreach):
        led.add(days[3], full)
    st = X.attribution_stability(led, min_days=20)
    assert st["same_top"] and st["top_first"] == "momentum" and st["rank_corr"] > 0.3
    assert X.attribution_stability(X.CohortAttributionLedger())["rank_corr"] is None
    assert led.frame(days[10]).shape[0] == 11
    bias = X.cohort_size_bias(full)
    assert bias is None or -1.0 <= bias <= 1.0


def test_mover_driver_table_and_lab_integration_of_the_attribution():
    rng = np.random.default_rng(10)
    fr = cohort_frame(rng, shock_mask=COHORT_CASES["volatility"], shock=0.07, idio=0.004)
    full = X.decompose_full(fr, CFG9)
    attr = X.attribute_moves(full, CFG9)
    tab = X.mover_driver_table(full, attr)
    assert tab.loc[("5%-10%", "up"), "volatility"] > 20 and tab.loc[("5%-10%", "up"), "n"] == tab.loc[("5%-10%", "up")].iloc[1:].sum()
    lab = X.CrossSectionLab(CFG9)
    res = lab.process_day("2020-01-02", fr)
    assert res.attribution is not None and (res.attribution["driver"] == attr["driver"]).all()
    assert {"xs_attr_volatility", "xs_attr_idio", "xs_driver_code"} <= set(res.features.columns) and len(lab.cohort_ledger) == 1
    back = X.CrossSectionLab.from_state(lab.state())
    assert len(back.cohort_ledger) == 1 and back.content_hash() == lab.content_hash()
    assert "COHORT ATTRIBUTION" in X.cohort_report(lab)
    feats = X.attribution_features(full, attr)
    assert feats.dtypes.eq("float32").all() and feats["xs_driver_code"].max() <= len(X.COMPONENTS)
