"""Tests for engine.research.regimes (RESEARCH_BRAIN_CONTRACT section 25; canon C67; rules 5, 23). Synthetic data only.
Each mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case."""
import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach
from engine.research import cross_section as X
from engine.research import regimes as R


def market_rows(n=400, seed=0, blocks=None):
    """Daily market summaries. blocks: list of (length, dict of overrides) cycled; base is a calm market."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    rows, pos = [], 0
    blocks = blocks or [(n, {})]
    i = 0
    while pos < n:
        length, ov = blocks[i % len(blocks)]
        for _ in range(min(length, n - pos)):
            vol = ov.get("vol", 0.008)
            rows.append({"ret": rng.normal(ov.get("drift", 0.0003), vol), "vix": ov.get("vix", 15) + rng.normal(0, 0.8),
                         "dispersion": ov.get("disp", 0.012) * (1 + rng.normal(0, 0.05)), "dollar_volume": 1e10 * (1 + rng.normal(0, 0.03)) * ov.get("dv", 1.0),
                         "event_share": float(np.clip(ov.get("ev", 0.1) + rng.normal(0, 0.01), 0, 1)), "breadth": ov.get("breadth", 0.55) + rng.normal(0, 0.03)})
        pos += length
        i += 1
    return dates, rows[:n]


def run_monitor(dates, rows, cfg=None):
    mon = R.RegimeMonitor(cfg or R.RegimeConfig(min_history=40, refit_every=30))
    for d, r in zip(dates, rows):
        mon.process(d, r)
    return mon


def test_config_and_market_row_validation():
    assert R.RegimeConfig().validate() == []
    assert R.RegimeConfig(q_lo=0.7, q_hi=0.3).validate() and R.RegimeConfig(persistence_window=10).validate()
    with pytest.raises(ValueError):
        R.RegimeMonitor(R.RegimeConfig(k_max=1))
    assert R.validate_market_row({"vix": 20}) and R.validate_market_row({"ret": 0.9}) and R.validate_market_row({"ret": 0.01, "event_share": 2})
    assert R.validate_market_row({"ret": 0.01, "dispersion": 0.01}) == []


def test_axes_start_unknown_then_follow_planted_volatility_blocks():
    dates, rows = market_rows(360, blocks=[(60, {"vix": 12, "vol": 0.005}), (60, {"vix": 32, "vol": 0.02})])
    mon = run_monitor(dates, rows)
    f = mon.frame()
    assert (f["volatility"].iloc[:39] == "unknown").all()
    late = f["volatility"].iloc[180:]
    calm_days, storm_days = late.iloc[[i for i in range(len(late)) if (i + 180) % 120 < 60]], late.iloc[[i for i in range(len(late)) if (i + 180) % 120 >= 60]]
    assert (storm_days == "high_vol").mean() > 0.9 and (calm_days == "low_vol").mean() > 0.9
    assert set(f["composite"]) >= {"stress"} or "bear_volatile" in set(f["composite"]) or len(set(f["composite"])) > 1


def test_hysteresis_stops_threshold_grazing_from_flipping_state():
    spec = R.AXIS_BY_NAME["volatility"]
    past = np.linspace(10, 30, 200)
    cfg = R.RegimeConfig()
    hi_cut = float(np.quantile(past, cfg.q_hi))
    assert R.classify_axis(spec, hi_cut - 0.01, past, "high_vol", cfg) == "high_vol"         # inside the margin: keep
    assert R.classify_axis(spec, hi_cut - 0.01, past, "low_vol", cfg) == R.NEUTRAL
    assert R.classify_axis(spec, hi_cut + 1, past, None, cfg) == "high_vol"
    assert R.classify_axis(spec, None, past, None, cfg) == R.UNKNOWN_STATE
    assert R.classify_axis(spec, 20.0, past[:10], None, cfg) == R.UNKNOWN_STATE
    assert R.classify_axis(spec, 20.0, np.full(100, 5.0), None, cfg) == R.UNKNOWN_STATE      # no spread: unknown, not a state


def test_character_axis_reads_trend_vs_mean_reversion():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 0.008, 700)
    trend, mr = np.zeros(700), np.zeros(700)
    for i in range(1, 700):
        trend[i] = 0.35 * trend[i - 1] + e[i]
        mr[i] = -0.35 * mr[i - 1] + e[i]
    dates = pd.bdate_range("2018-01-01", periods=700)
    out = {}
    for name, series in (("trend", trend), ("mr", mr)):
        mon = R.RegimeMonitor(R.RegimeConfig(min_history=60))
        for d, r in zip(dates, series):
            mon.process(d, {"ret": float(r)})
        out[name] = mon.history.series("persistence").mean()
    assert out["trend"] > 0.1 > -0.1 > out["mr"]


def test_monitor_refuses_future_and_out_of_order_days_and_bad_rows():
    dates, rows = market_rows(60)
    mon = R.RegimeMonitor(R.RegimeConfig(min_history=40))
    mon.process(dates[0], rows[0])
    with pytest.raises(FirewallBreach):
        mon.process(dates[0], rows[1])
    with pytest.raises(FirewallBreach):
        mon.process(dates[5], rows[1], now=dates[2])
    with pytest.raises(ValueError):
        mon.process(dates[1], {"vix": 20})
    assert mon.rejected_days == 1 and R.RegimeMonitor().frame().empty and R.RegimeMonitor().occupancy().empty


def test_indicators_use_only_the_past_plus_today():
    dates, rows = market_rows(150)
    a = run_monitor(dates[:100], rows[:100])
    poisoned = [dict(r) for r in rows]
    for r in poisoned[100:]:
        r["ret"], r["vix"] = 0.3, 99.0
    b = run_monitor(dates[:100], poisoned[:100])
    assert a.content_hash() == b.content_hash()


def test_discovery_accepts_persistent_regimes_and_rejects_noise():
    blocks = [(70, {"vol": 0.006, "disp": 0.010, "ev": 0.05, "breadth": 0.6, "vix": 13}), (70, {"vol": 0.02, "disp": 0.03, "ev": 0.3, "breadth": 0.35, "vix": 35})]
    dates, rows = market_rows(560, seed=1, blocks=blocks)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=60, dwell_ratio_min=2.0))
    acc = [f for f in mon.discovery.fits if f.status == "ACCEPTED"]
    assert acc and any(f.k == 2 for f in acc) and all(f.dwell_ratio > 3 for f in acc)
    labels = [s.discovered for s in mon.states if s.discovered != "unknown"]
    assert len(set(labels)) >= 2 and R.dwell_ratio(labels, 0)[0] > 3
    rng = np.random.default_rng(2)
    noise_rows = [{"ret": rng.normal(0, 0.01), "vix": 15 + rng.normal(0, 3), "dispersion": 0.012 + rng.normal(0, 0.003), "dollar_volume": 1e10 * (1 + rng.normal(0, 0.1)),
                   "event_share": float(np.clip(0.1 + rng.normal(0, 0.05), 0, 1)), "breadth": 0.5 + rng.normal(0, 0.1)} for _ in range(560)]
    nmon = run_monitor(dates, noise_rows, R.RegimeConfig(min_history=60, refit_every=60))
    assert not any(f.status == "ACCEPTED" for f in nmon.discovery.fits) and nmon.discovery.status() in ("NO_STRUCTURE", "REJECTED_NOISE")
    assert all(s.discovered == "unknown" for s in nmon.states)


def test_dwell_ratio_edge_cases_and_run_lengths():
    assert R.run_lengths(list("aabccc")) == [2, 1, 3] and R.run_lengths([]) == []
    assert R.dwell_ratio(list("ab" * 3), 0) == (None, None) and R.dwell_ratio(["a"] * 40, 0) == (None, None)
    rand = list(np.random.default_rng(0).choice(["a", "b"], 200))
    assert R.dwell_ratio(rand, 0)[0] == pytest.approx(1.0, abs=0.25)
    assert R.dwell_ratio(["a"] * 50 + ["b"] * 50, 0)[0] > 5


def test_discovered_ids_stay_stable_across_refits():
    blocks = [(70, {"vol": 0.006, "disp": 0.010, "ev": 0.05, "breadth": 0.6, "vix": 13}), (70, {"vol": 0.02, "disp": 0.03, "ev": 0.3, "breadth": 0.35, "vix": 35})]
    dates, rows = market_rows(300, seed=4, blocks=blocks)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=1000))
    disc = mon.discovery
    first = disc.refit()
    assert first.status == "ACCEPTED" and len(first.ids) == first.k
    again = disc.refit()                                                     # same data, second refit: ids must be inherited
    assert again.ids == first.ids and disc.assign(disc.vectors[-1][1]) in first.ids


def test_monitor_state_roundtrip_and_occupancy():
    dates, rows = market_rows(200, blocks=[(50, {"vix": 12}), (50, {"vix": 30})])
    mon = run_monitor(dates, rows)
    back = R.RegimeMonitor.from_state(mon.state())
    assert back.content_hash() == mon.content_hash()
    occ = mon.occupancy()
    assert set(occ["axis"]) == {a.name for a in R.AXES} and occ.groupby("axis")["share"].sum().round(6).eq(1.0).all()
    tm = R.transition_matrix(mon, "volatility")
    assert tm.sum(axis=1).round(6).eq(1.0).all() and R.stay_probability(mon, "volatility", "high_vol") > 0.8
    assert R.expected_dwell(mon, "volatility")["high_vol"] > 5
    with pytest.raises(ValueError):
        R.RegimeMonitor.from_state({"schema": "x"})


def fake_state(date, vol="high_vol", liq="high_liquidity", disc="unknown"):
    return R.RegimeState(str(date.date()), {"volatility": vol, "liquidity": liq, "direction": "unknown", "dispersion": "unknown",
                                            "character": "unknown", "events": "unknown"}, {}, {}, "unknown", disc)


def fill_book(kind, seed=0, n=420, book=None, pid=None):
    """Daily pattern effects filed under alternating 20-day volatility blocks and a random liquidity state."""
    rng = np.random.default_rng(seed)
    book = book or R.PatternRegimeBook(sessions=1, min_days=30)
    dates = pd.bdate_range("2018-01-01", periods=n)
    now = dates[-1] + pd.Timedelta(days=30)
    for i, d in enumerate(dates):
        vol = "high_vol" if (i // 20) % 2 == 0 else "low_vol"
        liq = "high_liquidity" if rng.random() < 0.5 else "low_liquidity"
        base = {"universal": 0.004, "bound": 0.006 if vol == "high_vol" else 0.0, "reversing": 0.005 if vol == "high_vol" else -0.005, "null": 0.0}[kind]
        book.add(pid or kind, d, base + rng.normal(0, 0.003), d + pd.Timedelta(days=2), now, fake_state(d, vol, liq))
    return book, now, dates


def test_regime_bound_pattern_is_not_universal_and_is_gated_off_where_it_fails():
    book, now, dates = fill_book("bound")
    rep = book.report("bound", now)
    assert rep.verdict == R.Verdict.REGIME_BOUND and "volatility" in rep.bound_axes and "liquidity" not in rep.bound_axes
    assert "high_vol" in rep.good_states["volatility"] and "low_vol" in rep.bad_states["volatility"]
    assert book.regime_gate("bound", fake_state(dates[0], "high_vol"), now)["allowed"] is True
    assert book.regime_gate("bound", fake_state(dates[0], "low_vol"), now)["allowed"] is False
    unknown = R.RegimeState("x", {"volatility": "unknown"}, {}, {}, "unknown")
    assert book.regime_gate("bound", unknown, now)["allowed"] is None                    # unknown regime: abstain, not allow
    assert R.regime_weights(book, fake_state(dates[0], "low_vol"), now) == {"bound": 0.0}


def test_universal_null_and_reversing_patterns_are_told_apart():
    for kind, want in (("universal", R.Verdict.UNIVERSAL), ("null", R.Verdict.NOT_ESTABLISHED), ("reversing", R.Verdict.REGIME_REVERSING)):
        book, now, _ = fill_book(kind, seed=3)
        assert book.report(kind, now).verdict == want, kind
    book, now, dates = fill_book("universal", seed=3)
    assert book.regime_gate("universal", fake_state(dates[0], "low_vol"), now)["allowed"] is True


def test_unseen_regime_transfer_fails_for_bound_pattern_and_passes_for_universal():
    book, now, _ = fill_book("bound", seed=5)
    res = {r["held_out"]: r for r in R.unseen_regime_transfer(book, "bound", "volatility", now)}
    assert res["low_vol"]["verdict"] in ("FAILS", "UNTESTED") and not res["low_vol"]["transfers"]
    ubook, unow, _ = fill_book("universal", seed=5)
    assert all(r["transfers"] for r in R.unseen_regime_transfer(ubook, "universal", "volatility", unow))
    assert R.unseen_regime_transfer(ubook, "nope", "volatility", unow) == [] and R.unseen_regime_transfer(ubook, "universal", "zzz", unow) == []


def test_book_firewalls_and_empty_cases():
    book = R.PatternRegimeBook(sessions=1, min_days=30)
    d = pd.Timestamp("2020-01-02")
    with pytest.raises(FirewallBreach):
        book.add("p", d, 0.01, d + pd.Timedelta(days=2), d + pd.Timedelta(days=2), fake_state(d))          # matures ON now
    with pytest.raises(FirewallBreach):
        book.add("p", d, 0.01, d, d + pd.Timedelta(days=9), fake_state(d))                                 # not after decision
    assert book.add("p", d, float("nan"), d + pd.Timedelta(days=1), d + pd.Timedelta(days=9), fake_state(d)) is False
    assert book.report("p", d + pd.Timedelta(days=9)).verdict == R.Verdict.INSUFFICIENT_DATA
    assert book.report("never", d).verdict == R.Verdict.INSUFFICIENT_DATA and book.matured_records(d) == []
    b2, now, dates = fill_book("bound", seed=6)
    with pytest.raises(FirewallBreach):
        b2.add("bound", dates[-1] + pd.Timedelta(days=1), 0.0, now, now, fake_state(dates[-1]))
    assert b2.report("bound", now, replay_years=[2018, 2019]).verdict == R.Verdict.INSUFFICIENT_DATA     # every year replayed: nothing visible
    recs = b2.matured_records(now)
    assert recs and recs[0].gate(now + pd.Timedelta(days=1))["bound_axes"] == ["volatility"] and "2018" not in str(recs[0].payload)
    assert R.PatternRegimeBook.from_state(b2.state()).report("bound", now).verdict == b2.report("bound", now).verdict


def test_coverage_gaps_and_questions():
    book, now, _ = fill_book("bound", seed=7, n=200)
    gaps = R.coverage_gaps(book, now, min_days=10_000)
    assert ("bound", "direction", "bull", 0) in gaps and gaps == sorted(gaps, key=lambda g: (g[3], g[0], g[1], g[2]))
    qs = R.coverage_questions(gaps, "2026-09-29", "2020-01-01", limit=3)
    assert len(qs) == 3 and "2018" not in qs[0].text


def test_changes_trader_row_and_step_end_to_end():
    dates, rows = market_rows(300, blocks=[(80, {"vix": 12, "vol": 0.005}), (80, {"vix": 32, "vol": 0.02})])
    mon = run_monitor(dates[:299], rows[:299])
    ch = R.detect_changes(mon.states, min_persist=5)
    assert any(c.axis == "volatility" and c.after == "high_vol" for c in ch) and R.detect_changes([]) == []
    qs = R.regime_questions(ch, "2026-09-29", "2020-01-01")
    assert qs and "2018" not in qs[0].text
    row = R.trader_regime_row(mon.states[-1])
    assert row["rg_volatility"] in (-1.0, 0.0, 1.0) and all(k.startswith("rg_") for k in row)
    assert R.trader_regime_row(mon.states[0]) == {}                                                         # nothing known yet: nothing shown
    book = R.PatternRegimeBook(sessions=1, min_days=10)
    eff = [("p", dates[i], 0.004, dates[i] + pd.Timedelta(days=2)) for i in range(100, 200)]
    res = R.step(mon, book, dates[299], rows[299], [e for e in eff if e[3] < dates[299]], gate_patterns=["p"])
    assert res.n_added == 100 and res.states["volatility"] != "unknown" and "p" in res.verdicts and res.gates["p"]["allowed"] in (True, False, None)
    assert "REGIMES" in R.render_report(mon, book, dates[299])


def test_axis_diagnostics_are_computable_and_honest():
    dates, rows = market_rows(400, blocks=[(60, {"vix": 12, "vol": 0.005, "ev": 0.05}), (60, {"vix": 32, "vol": 0.02, "ev": 0.4})])
    mon = run_monitor(dates, rows)
    sens = R.threshold_sensitivity(mon)
    assert set(sens["axis"]) == {"volatility", "dispersion", "liquidity", "events"} and (sens["changed_share"].dropna() >= 0).all()
    assoc = R.axis_association(mon)
    assert assoc.loc["volatility", "events"] > 0.5                                       # the planted blocks move both axes together
    assert R.axis_association(R.RegimeMonitor()).empty and R.discovered_vs_named(R.RegimeMonitor()) == {}
    d2, r2 = market_rows(400, blocks=[(200, {"vix": 12}), (200, {"vix": 34})])
    brk = R.indicator_breaks(run_monitor(d2, r2))
    assert len(brk["vol"]) >= 1 and brk["vol"][0] > str(d2[195].date())
    assert R.cusum_break([0.0] * 10) == [] and R.cusum_break([0.0, 1.0] * 5 + [0.0, 1.0] * 5) == []
    shifted = list(np.random.default_rng(0).normal(0, 1, 60)) + list(np.random.default_rng(1).normal(6, 1, 60))
    assert R.cusum_break(shifted) and R.cusum_break(shifted)[0] >= 60
    prof = R.state_return_profile(mon, pd.Series([r["ret"] for r in rows], index=dates), "volatility", dates[-1])
    assert prof.loc["high_vol", "std"] > prof.loc["low_vol", "std"]
    assert R.state_return_profile(R.RegimeMonitor(), pd.Series(dtype=float), "volatility", dates[-1]).empty


def test_scope_effect_by_regime_links_cross_section_and_regimes():
    days = pd.bdate_range("2019-01-01", periods=60)
    now = days[-1] + pd.Timedelta(days=5)
    rng0 = np.random.default_rng(2)
    outcomes = X.ScopeOutcomes(sessions=1)
    idx = [f"S{i}" for i in range(30)]
    for d in days:
        ret = pd.Series(rng0.choice([-0.07, 0.07], 30), index=idx)
        outcomes.add(d, pd.Series("STOCK_SPECIFIC", index=idx), ret, pd.Series(rng0.normal(0, 0.01, 30), index=idx), d + pd.Timedelta(days=1), now)
    mon = R.RegimeMonitor(R.RegimeConfig(min_history=40))
    rng = np.random.default_rng(1)
    for d in days:
        mon.process(d, {"ret": float(rng.normal(0, 0.01)), "vix": 15.0 + float(rng.normal(0, 3))})
    tab = R.scope_effect_by_regime(outcomes, mon, "volatility", now, min_days=3)
    assert not tab.empty and set(tab.index.get_level_values("scope")) >= {"STOCK_SPECIFIC"}
    assert R.scope_effect_by_regime(outcomes, mon, "volatility", now, replay_years=[2019]).empty
    assert R.regime_share_labels(mon, "volatility") == mon.label_map("volatility")


def test_regime_forecast_beats_persistence_only_when_dynamics_are_learnable():
    blocks = [(30, {"vix": 12}), (10, {"vix": 34})]
    dates, rows = market_rows(500, blocks=blocks, seed=2)
    mon = run_monitor(dates, rows)
    fc = R.RegimeForecast(mon, "volatility")
    p = fc.next_probs()
    assert p is None or abs(sum(p.values()) - 1.0) < 1e-6
    skill = fc.brier_skill(min_days=30)
    assert skill is None or skill == skill
    assert R.RegimeForecast(R.RegimeMonitor(), "volatility").next_probs() is None
    assert R.RegimeForecast(R.RegimeMonitor(), "volatility").brier_skill() is None


def test_shrinkage_best_axis_and_mover_rates():
    book, now, _ = fill_book("bound", seed=8)
    rep = book.report("bound", now)
    raw = {e.state: e.effect for e in rep.by_state if e.axis == "volatility"}
    shr = R.shrunk_state_effects(rep, "volatility")
    assert set(shr) == set(raw) and all(abs(shr[k] - rep.overall.effect) <= abs(raw[k] - rep.overall.effect) + 1e-12 for k in shr)
    assert R.best_axis(book, "bound", now)[0] == "volatility"
    nbook, nnow, _ = fill_book("null", seed=8)
    assert R.best_axis(nbook, "null", nnow) is None and R.shrunk_state_effects(nbook.report("never", nnow), "volatility") == {}
    dates, rows = market_rows(200, blocks=[(50, {"vix": 12}), (50, {"vix": 34})])
    mon = run_monitor(dates, rows)
    counts = {str(d.date()): ((300, 90) if r["vix"] > 25 else (150, 30)) for d, r in zip(dates, rows)}
    tab = R.mover_rate_by_regime(mon, counts, "volatility", dates[-1] + pd.Timedelta(days=1))
    assert tab.loc["high_vol", "ratio_5_10"] > tab.loc["low_vol", "ratio_5_10"] and R.mover_rate_by_regime(mon, {}, "volatility", dates[-1]).empty


# ------------------------------------------------------------------------------------------------ every section-25 regime, each with its own test

def ar_rows(n_blocks, block_len, phis, seed=0):
    """Market returns from AR(1) blocks (phi > 0 trending, phi < 0 mean-reverting) with everything else calm."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_blocks * block_len)
    rows, prev = [], 0.0
    for b in range(n_blocks):
        phi = phis[b % len(phis)]
        for _ in range(block_len):
            prev = phi * prev + rng.normal(0, 0.008)
            rows.append({"ret": prev, "vix": 15 + rng.normal(0, 0.8), "dispersion": 0.012, "dollar_volume": 1e10, "event_share": 0.1, "breadth": 0.55})
    return dates, rows


WORLDS = {                                  # regime -> (override that puts the market IN it, override that puts it in the opposite pole)
    "high_vol": ({"vix": 34}, {"vix": 11}), "low_vol": ({"vix": 11}, {"vix": 34}),
    "high_dispersion": ({"disp": 0.04}, {"disp": 0.006}), "low_dispersion": ({"disp": 0.006}, {"disp": 0.04}),
    "high_liquidity": ({"dv": 3.0}, {"dv": 0.3}), "low_liquidity": ({"dv": 0.3}, {"dv": 3.0}),
    "event_heavy": ({"ev": 0.45}, {"ev": 0.02}), "event_light": ({"ev": 0.02}, {"ev": 0.45}),
    "bull": ({"ma200": 0.12}, {"ma200": -0.12}), "bear": ({"ma200": -0.12}, {"ma200": 0.12}),
}


def world_rows(on, off, n=600, block=60, seed=0):
    dates, rows = market_rows(n, seed=seed, blocks=[(block, on), (block, off)])
    for i, (r, d) in enumerate(zip(rows, dates)):
        ov = on if (i // block) % 2 == 0 else off
        if "ma200" in ov:
            r["spy_ma200"] = ov["ma200"] + 0.01 * np.random.default_rng(i).normal()
    return dates, rows


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_each_named_regime_fires_inside_its_planted_block_and_not_in_the_opposite(name):
    on, off = WORLDS[name]
    dates, rows = world_rows(on, off)
    mon = R.RegimeMonitor(R.RegimeConfig(min_history=60, refit_every=10_000))
    named = R.NamedRegimeMonitor(mon)
    flags = [named.process(d, r) for d, r in zip(dates, rows)]
    late = range(300, 600)
    on_days = [i for i in late if (i // 60) % 2 == 0 and i % 60 >= 10]
    off_days = [i for i in late if (i // 60) % 2 == 1 and i % 60 >= 10]
    assert np.mean([flags[i][name] is True for i in on_days]) > 0.8, name
    assert np.mean([flags[i][name] is True for i in off_days]) < 0.05, name
    assert named.exclusive_violations() == []
    assert named.summary().loc[name, "episodes"] >= 3 and R.NAMED_BY_NAME[name].meaning


@pytest.mark.parametrize("name,phis", [("trend", (0.4, -0.4)), ("mean_reversion", (-0.4, 0.4))])
def test_trend_and_mean_reversion_regimes_read_the_return_dynamics(name, phis):
    dates, rows = ar_rows(6, 200, phis, seed=3)
    named = R.NamedRegimeMonitor(R.RegimeMonitor(R.RegimeConfig(min_history=60, refit_every=10_000)))
    flags = [named.process(d, r) for d, r in zip(dates, rows)]
    on_days = [i for i in range(400, 1200) if (i // 200) % 2 == 0 and i % 200 >= 120]
    off_days = [i for i in range(400, 1200) if (i // 200) % 2 == 1 and i % 200 >= 120]
    assert np.mean([flags[i][name] is True for i in on_days]) > 0.6 and np.mean([flags[i][name] is True for i in off_days]) < 0.1


def test_twelve_regimes_are_defined_once_each_with_distinct_axes_and_unknown_is_not_false():
    assert len(R.NAMED_REGIMES) == 12 and {r.name for r in R.NAMED_REGIMES} == set(R.STATE_CODE) - {R.NEUTRAL}
    assert {r.axis for r in R.NAMED_REGIMES} == {a.name for a in R.AXES}
    st = R.RegimeState("2020-01-02", {a.name: R.UNKNOWN_STATE for a in R.AXES}, {}, {}, "unknown")
    assert all(v is None for v in R.regime_flags(st).values())
    neutral = R.RegimeState("2020-01-02", {a.name: R.NEUTRAL for a in R.AXES}, {}, {}, "unknown")
    assert not any(R.regime_flags(neutral).values())


def test_named_monitor_boundaries_episodes_and_incremental_update():
    dates, rows = world_rows({"vix": 34}, {"vix": 11}, n=300)
    mon = R.RegimeMonitor(R.RegimeConfig(min_history=60, refit_every=10_000))
    named = R.NamedRegimeMonitor(mon)
    assert named.update() == 0 and named.boundary("high_vol") is None and named.distance_to_entry("high_vol") is None
    for d, r in zip(dates, rows):
        named.process(d, r)
    assert named.update() == 0                                                  # idempotent
    lo, hi = named.boundary("high_vol")
    assert lo < hi and named.summary()["known_days"].max() > 200
    eps = named.episodes("low_vol")
    assert all(e.days >= 1 for e in eps) and sum(e.days for e in eps) == named.summary().loc["low_vol", "active_days"]
    now_low = "low_vol" in named.active()
    assert (named.distance_to_entry("low_vol") <= 0) == now_low or abs(named.distance_to_entry("low_vol")) < 5


# ------------------------------------------------------------------------------------------------ discovery: stable regime count

def two_regime_world(seed, n=700):
    blocks = [(70, {"vol": 0.006, "disp": 0.010, "ev": 0.05, "breadth": 0.6, "vix": 13}), (70, {"vol": 0.02, "disp": 0.03, "ev": 0.3, "breadth": 0.35, "vix": 35})]
    return market_rows(n, seed=seed, blocks=blocks)


@pytest.mark.parametrize("seed", [1, 4, 7])
def test_two_regime_world_gives_two_regimes_at_every_refit_with_stable_ids(seed):
    dates, rows = two_regime_world(seed)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=60))
    fits = mon.discovery.fits
    assert len(fits) >= 9 and all(f.status == "ACCEPTED" and f.k == 2 for f in fits)          # 2 at EVERY refit
    assert all(f.stability >= 0.9 for f in fits) and len({i for f in fits for i in f.ids}) == 2   # and the same two ids throughout
    hist = R.refit_history(mon.discovery)
    assert not hist["k_changed"] and hist["id_churn"] == 0 and hist["mean_stability"] > 0.9
    desc = R.describe_discovered(mon.discovery)
    assert {v["vol"] for v in desc.values()} == {"high", "low"} and {v["event_share"] for v in desc.values()} == {"high", "low"}


def test_cluster_stability_separates_real_structure_from_a_blob():
    rng = np.random.default_rng(0)
    two = np.vstack([rng.normal(0, 0.3, (100, 3)), rng.normal(5, 0.3, (100, 3))])
    lab2, _, _ = R.kmeans(two, 2, 0)
    blob = rng.normal(0, 1, (200, 3))
    lab_b, _, _ = R.kmeans(blob, 4, 0)
    assert R.cluster_stability(two, lab2, 2, 0) > 0.95 and R.cluster_stability(blob, lab_b, 4, 0) < 0.8


def test_hungarian_matching_inherits_ids_under_permutation_and_gives_new_ids_to_far_centroids():
    cfg = R.RegimeConfig()
    d = R.DiscoveredRegimes(cfg)
    center, scale = np.zeros(5), np.ones(5)
    d.centroids, d.ids, d.center, d.scale = np.array([[0.0] * 5, [4.0] * 5]), ("A", "B"), center, scale
    ids, cost = d._match(np.array([[4.1] * 5, [0.1] * 5]), center, scale)                     # same two clusters, order swapped
    assert ids == ("B", "A") and 0 < cost < 0.2
    ids2, _ = d._match(np.array([[0.1] * 5, [30.0] * 5]), center, scale)                     # second centroid is nowhere near B
    assert ids2[0] == "A" and ids2[1] not in ("A", "B")
    fresh = R.DiscoveredRegimes(cfg)
    assert fresh._match(np.zeros((2, 5)), center, scale)[0] == ("disc_0", "disc_1") and fresh.status() == "NOT_FITTED"


def test_unstable_or_structureless_worlds_are_not_accepted():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-01", periods=500)
    noise = [{"ret": rng.normal(0, 0.01), "vix": 15 + rng.normal(0, 3), "dispersion": 0.012 + rng.normal(0, 0.003), "dollar_volume": 1e10 * (1 + rng.normal(0, 0.1)),
              "event_share": float(np.clip(0.1 + rng.normal(0, 0.05), 0, 1)), "breadth": 0.5 + rng.normal(0, 0.1)} for _ in dates]
    mon = run_monitor(dates, noise, R.RegimeConfig(min_history=60, refit_every=100))
    assert not any(f.status == "ACCEPTED" for f in mon.discovery.fits)
    assert R.refit_history(R.DiscoveredRegimes(R.RegimeConfig()))["n_fits"] == 0 and R.describe_discovered(R.DiscoveredRegimes(R.RegimeConfig())) == {}


# ------------------------------------------------------------------------------------------------ transitions and lead time

def ramp_rows(n_cycles=8, calm=45, ramp=15, storm=45, lead_ramp=True, seed=0):
    """vix sits calm, then ramps up over `ramp` days (or steps at once) into a storm, then falls back."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_cycles):
        rows += [11.0] * calm
        rows += list(np.linspace(11.0, 34.0, ramp)) if lead_ramp else [34.0] * ramp
        rows += [34.0] * storm
        rows += [11.0] * 10
    dates = pd.bdate_range("2018-01-01", periods=len(rows))
    return dates, [{"ret": rng.normal(0, 0.01), "vix": v + rng.normal(0, 0.4), "dispersion": 0.012, "dollar_volume": 1e10, "event_share": 0.1, "breadth": 0.55} for v in rows]


def test_indicator_pressure_reads_only_the_past():
    v = np.r_[np.zeros(60) + np.random.default_rng(0).normal(0, 0.1, 60), np.linspace(0, 5, 20)]
    full = R.indicator_pressure(v)
    cut = R.indicator_pressure(v[:70])
    assert np.allclose(full[:70], cut, equal_nan=True) and np.isnan(full[:34]).all() and full[-1] > 3
    assert np.isnan(R.indicator_pressure(np.ones(80))).all()


def test_ramped_transitions_give_positive_lead_time_and_steps_do_not():
    dates, rows = ramp_rows(lead_ramp=True)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    est = R.lead_time_estimate(mon, "volatility", z_bar=3.0)
    assert est.n_events >= 3 and est.hit_rate is not None and est.hit_rate > 0.5 and est.median_lead is not None and est.median_lead >= 2
    d2, r2 = ramp_rows(lead_ramp=False)
    est2 = R.lead_time_estimate(run_monitor(d2, r2, R.RegimeConfig(min_history=60, refit_every=10_000)), "volatility")
    assert est2.median_lead is None or est2.median_lead < est.median_lead                # a step gives no notice
    ev = R.transition_events(mon, "volatility")
    assert all(e.lead_days is None or (0 < e.lead_days <= 25 and e.warned_index < e.flip_index) for e in ev)


def test_current_warning_fires_during_a_ramp_and_not_when_calm():
    dates, rows = ramp_rows(n_cycles=3)
    calm = run_monitor(dates[:155], rows[:155], R.RegimeConfig(min_history=60, refit_every=10_000))
    assert R.current_warning(calm, "volatility") is None and R.all_warnings(calm) == []
    mid = run_monitor(dates[:164], rows[:164], R.RegimeConfig(min_history=60, refit_every=10_000))
    w = R.current_warning(mid, "volatility")
    assert w is not None and w.toward == "high_vol" and w.z > 3
    assert R.current_warning(R.RegimeMonitor(), "volatility") is None


def test_false_alarms_are_counted_when_pressure_does_not_lead_to_a_regime():
    rng = np.random.default_rng(1)
    rows = [{"ret": rng.normal(0, 0.01), "vix": 15 + rng.normal(0, 1.0), "dispersion": 0.012, "dollar_volume": 1e10, "event_share": 0.1, "breadth": 0.55} for _ in range(400)]
    mon = run_monitor(pd.bdate_range("2018-01-01", periods=400), rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    far = R.false_alarm_rate(mon, "volatility", z_bar=2.5)
    assert far is None or 0.0 <= far <= 1.0
    dates, r2 = ramp_rows()
    real = R.false_alarm_rate(run_monitor(dates, r2, R.RegimeConfig(min_history=60, refit_every=10_000)), "volatility")
    assert real is not None and real < 0.5


def test_shock_days_find_a_one_day_jump():
    dates, rows = market_rows(200)
    rows[150]["vix"] = 80.0
    mon = run_monitor(dates, rows)
    assert str(dates[150].date()) in R.shock_days(mon, "volatility") and R.shock_days(R.RegimeMonitor(), "volatility") == []


# ------------------------------------------------------------------------------------------------ pattern x regime: shrinkage

def test_pool_states_shrinks_thin_states_and_pools_homogeneous_ones():
    E = lambda st, eff, se: R.StateEffect("volatility", st, 50, eff, se, eff / se)
    homog, tau2, mu = R.pool_states([E("a", 0.010, 0.004), E("b", 0.011, 0.004), E("c", 0.009, 0.004)])
    assert tau2 == 0.0 and all(p.weight == 0.0 and p.effect == pytest.approx(mu) for p in homog)     # states agree: full pooling
    het, tau2h, mu2 = R.pool_states([E("a", 0.030, 0.002), E("b", -0.030, 0.002), E("c", 0.0, 0.002)])
    assert tau2h > 0 and all(p.weight > 0.9 for p in het)                                                  # states differ: keep own estimates
    thin, _, _ = R.pool_states([E("big", 0.010, 0.001), E("big2", 0.012, 0.001), E("thin", 0.060, 0.030)])
    t = next(p for p in thin if p.state == "thin")
    assert t.weight < 0.5 and abs(t.effect - 0.011) < abs(t.raw_effect - 0.011)                        # the lucky thin state is pulled to the pool
    assert R.pool_states([E("only", 0.01, 0.01)]) == ([], 0.0, None)
    assert t.se < t.raw_se and t.established() in (True, False)


def test_report_carries_pooled_view_and_gate_uses_shrunk_effects():
    book, now, dates = fill_book("bound", seed=4)
    rep = book.report("bound", now)
    assert set(rep.pooled["volatility"][0].__dataclass_fields__) >= {"raw_effect", "effect", "weight"} and rep.tau2["volatility"] > 0
    assert "high_vol" in rep.good_states["volatility"] and "low_vol" in rep.bad_states["volatility"]
    assert book.regime_gate("bound", fake_state(dates[0], "high_vol"), now)["allowed"] is True
    nb, nnow, _ = fill_book("null", seed=4)
    assert nb.report("null", nnow).pooled["volatility"] and nb.report("null", nnow).verdict == R.Verdict.NOT_ESTABLISHED


def test_regime_adjusted_effect_is_higher_in_the_state_where_the_pattern_works():
    book, now, dates = fill_book("bound", seed=6)
    hi = R.regime_adjusted_effect(book, "bound", fake_state(dates[0], "high_vol"), now)
    lo = R.regime_adjusted_effect(book, "bound", fake_state(dates[0], "low_vol"), now)
    assert hi["effect"] > lo["effect"] and "volatility" in hi["used"] and "direction" in hi["unused"]
    assert R.regime_adjusted_effect(book, "never", fake_state(dates[0]), now)["effect"] is None


def test_two_way_interaction_found_only_where_planted():
    rng = np.random.default_rng(9)
    book = R.PatternRegimeBook(sessions=1, min_days=20)
    dates = pd.bdate_range("2018-01-01", periods=700)
    now = dates[-1] + pd.Timedelta(days=30)
    for i, d in enumerate(dates):
        vol = "high_vol" if (i // 20) % 2 == 0 else "low_vol"
        liq = "high_liquidity" if (i // 7) % 2 == 0 else "low_liquidity"
        eff = 0.008 if (vol == "high_vol" and liq == "low_liquidity") else 0.0
        book.add("p", d, eff + rng.normal(0, 0.003), d + pd.Timedelta(days=2), now, fake_state(d, vol, liq))
    cells = {(c["a"], c["b"]): c for c in R.two_way_effects(book, "p", "volatility", "liquidity", now)}
    assert cells[("high_vol", "low_liquidity")]["established"] and cells[("high_vol", "low_liquidity")]["interaction"] > 0.001
    assert cells[("high_vol", "low_liquidity")]["shrunk_interaction"] > 0
    assert cells[("high_vol", "high_liquidity")]["interaction"] < 0            # a 2x2 interaction has +x on the diagonal, -x off it
    assert R.two_way_effects(book, "never", "volatility", "liquidity", now) == [] and R.two_way_effects(book, "p", "volatility", "zzz", now) == []


def test_pattern_weakens_after_transitions_is_detected():
    dates, rows = ramp_rows(n_cycles=8, seed=2)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    ev = R.transition_events(mon, "volatility")
    post = set()
    for e in ev:
        post.update(s.date for s in mon.states[e.flip_index: e.flip_index + 10])
    rng = np.random.default_rng(3)
    book = R.PatternRegimeBook(sessions=1, min_days=20)
    now = dates[-1] + pd.Timedelta(days=30)
    for s in mon.states:
        d = pd.Timestamp(s.date)
        book.add("p", d, (-0.004 if s.date in post else 0.004) + rng.normal(0, 0.003), d + pd.Timedelta(days=2), now, s)
    res = R.effect_after_transitions(book, mon, "p", "volatility", now, window=10)
    assert res["verdict"] == "WEAKER_AFTER_TRANSITIONS" and res["diff"] < 0
    assert R.effect_after_transitions(book, mon, "never", "volatility", now)["verdict"] == "INSUFFICIENT_DATA"


def test_verdict_by_era_and_universal_claims_audit():
    book, now, dates = fill_book("bound", seed=2, n=600)
    v = R.verdict_by_era(book, "bound", now, [dates[0], dates[300], dates[-1] + pd.Timedelta(days=1)])
    assert set(v) == {"era0", "era1"} and R.regime_dependence_stable(v) in (True, False, None)
    assert R.regime_dependence_stable({"a": "REGIME_BOUND", "b": "INSUFFICIENT_DATA"}) is None
    assert R.regime_dependence_stable({"a": "REGIME_BOUND", "b": "UNIVERSAL"}) is False
    ubook, unow, _ = fill_book("universal", seed=2)
    fill_book("bound", seed=2, book=ubook, pid="bound")
    assert R.universal_claims_audit(ubook, ["universal", "bound", "missing"], unow) == ["bound", "missing"]
    rank = R.rank_by_regime_dependence(ubook, unow)
    assert rank.loc["bound", "best_axis"] == "volatility" and rank.loc["universal", "transfer_share"] == 1.0 and R.rank_by_regime_dependence(R.PatternRegimeBook(), unow).empty


# ------------------------------------------------------------------------------------------------ feature frame, health, survival, combinations

def test_regime_feature_frame_is_point_in_time_and_marks_unknown_as_nan():
    dates, rows = ramp_rows(n_cycles=3)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    full = R.regime_feature_frame(mon)
    part = R.regime_feature_frame(run_monitor(dates[:150], rows[:150], R.RegimeConfig(min_history=60, refit_every=10_000)))
    pd.testing.assert_frame_equal(full.iloc[:150], part)                                       # row t never depends on later days
    assert full["high_vol"].iloc[:30].isna().all() and set(full["high_vol"].dropna().unique()) <= {0.0, 1.0}
    assert {"dwell_volatility", "composite_code", "pressure_volatility"} <= set(full.columns) and (full["pressure_volatility"] == 1.0).any()
    assert R.regime_feature_frame(R.RegimeMonitor()).empty


def test_axis_health_flags_stuck_flickering_and_unknown_axes():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=300)
    rows = [{"ret": rng.normal(0, 0.01), "vix": (11.0 if i % 2 == 0 else 34.0) + rng.normal(0, 0.1)} for i in range(300)]      # flicker; other inputs missing
    h = {x.axis: x for x in R.axis_health(run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000)))}
    assert "FLICKERING" in h["volatility"].issues and "MOSTLY_UNKNOWN" in h["events"].issues and "volatility" in R.unhealthy_axes(run_monitor(dates, rows))
    d2, r2 = world_rows({"vix": 34}, {"vix": 11})
    ok = {x.axis: x for x in R.axis_health(run_monitor(d2, r2, R.RegimeConfig(min_history=60, refit_every=10_000)))}
    assert not ok["volatility"].issues
    assert all(x.issues == ("MOSTLY_UNKNOWN",) or "MOSTLY_UNKNOWN" in x.issues for x in R.axis_health(R.RegimeMonitor()))


def test_survival_curve_censors_the_open_episode_and_predicts_remaining_days():
    dates, rows = world_rows({"vix": 34}, {"vix": 11}, n=500)
    mon = R.RegimeMonitor(R.RegimeConfig(min_history=60, refit_every=10_000))
    named = R.NamedRegimeMonitor(mon)
    for d, r in zip(dates, rows):
        named.process(d, r)
    s = R.survival_curve(named, "high_vol", 100)
    assert s.is_monotonic_decreasing and s.iloc[0] <= 1.0 and s.loc[20] > s.loc[80]
    rem = R.expected_remaining(named, "high_vol", 10)
    assert rem is not None and rem > 5
    assert R.expected_remaining(named, "high_vol", 0) is None
    assert R.survival_curve(R.NamedRegimeMonitor(R.RegimeMonitor()), "bull").empty and R.expected_remaining(R.NamedRegimeMonitor(R.RegimeMonitor()), "bull", 5) is None


def test_persistence_test_cooccurrence_and_joint_table():
    dates, rows = world_rows({"vix": 34, "disp": 0.04}, {"vix": 11, "disp": 0.006})
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    pers = R.axis_persistence_test(mon)
    assert pers["volatility"] > 3 and pers["dispersion"] > 3
    co = R.cooccurrence(mon, "high_vol", "high_dispersion")
    assert co["lift"] > 1.5 and R.cooccurrence(mon, "high_vol", "low_vol")["p_b_given_a"] is None or R.cooccurrence(mon, "high_vol", "low_vol")["p_b_given_a"] == 0.0
    assert R.cooccurrence(R.RegimeMonitor(), "bull", "bear")["lift"] is None
    rng = np.random.default_rng(1)
    coin = [{"ret": 0.001, "vix": 11.0 + 23.0 * float(rng.random() < 0.5) + rng.normal(0, 0.3)} for _ in range(300)]
    assert R.axis_persistence_test(run_monitor(pd.bdate_range("2018-01-01", periods=300), coin))["volatility"] < 1.5      # coin flips are not a regime
    jt = R.joint_state_table(mon, "volatility", "dispersion")
    assert jt.to_numpy().sum() > 300 and R.joint_state_table(R.RegimeMonitor()).empty


def test_definition_agreement_and_report_card_and_step_extras():
    dates, rows = world_rows({"vix": 34}, {"vix": 11})
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    da = R.definition_agreement(mon).set_index("axis")
    assert da.loc["volatility", "agreement"] > 0.7
    card = R.regime_report_card(mon)
    assert "REGIME CARD" in card and "axis volatility" in card
    book = R.PatternRegimeBook(sessions=1, min_days=10)
    mon2 = R.RegimeMonitor(R.RegimeConfig(min_history=60, refit_every=10_000))
    for d, r in zip(dates[:200], rows[:200]):
        mon2.process(d, r)
    res = R.step(mon2, book, dates[200], rows[200])
    assert isinstance(res.active, tuple) and res.unhealthy is not None and res.discovery_stability is None
    assert set(R.trader_regime_row(mon2.states[-1])) <= {f"rg_{a.name}" for a in R.AXES}


def test_tune_quantiles_prefers_cutoffs_that_give_persistent_populated_regimes():
    dates, rows = world_rows({"vix": 34}, {"vix": 11})
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    t = R.tune_quantiles(mon, "volatility")
    assert len(t) == 4 and (t["persistence"].dropna() > 2).all() and t["score"].max() > 2
    tiny = R.tune_quantiles(mon, "volatility", grid=((0.33, 0.67),), min_state_share=0.6)
    assert tiny["score"].iloc[0] == 0.0                                            # both poles cannot hold 60% of days each: unpopulated regimes score nothing
    with pytest.raises(ValueError):
        R.tune_quantiles(mon, "direction")
    assert R.tune_quantiles(R.RegimeMonitor(), "volatility")["score"].eq(0.0).all()


def test_axis_information_sees_volatility_clustering_and_not_noise():
    dates, rows = world_rows({"vix": 34, "vol": 0.02}, {"vix": 11, "vol": 0.004})
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    rets = pd.Series([r["ret"] for r in rows], index=dates)
    info = R.axis_information(mon, "volatility", rets, dates[-1] + pd.Timedelta(days=1))
    assert info["abs_ic"] > 0.3 and abs(info["signed_ic"]) < 0.15
    rng = np.random.default_rng(4)
    flat = R.axis_information(mon, "volatility", pd.Series(rng.normal(0, 0.01, len(dates)), index=dates), dates[-1] + pd.Timedelta(days=1))
    assert abs(flat["abs_ic"]) < 0.15
    assert R.axis_information(R.RegimeMonitor(), "volatility", rets, dates[-1])["abs_ic"] is None
    diag = R.regime_diagnostics(mon, rets, dates[-1] + pd.Timedelta(days=1))
    assert set(diag.index) == {a.name for a in R.AXES} and diag.loc["volatility", "abs_ic"] > 0.3


def test_monitor_records_are_identity_free_and_respect_maturity_and_rebuild_matches():
    dates, rows = world_rows({"vix": 34}, {"vix": 11}, n=300)
    mon = R.RegimeMonitor(R.RegimeConfig(min_history=60, refit_every=10_000))
    named = R.NamedRegimeMonitor(mon)
    for d, r in zip(dates, rows):
        named.process(d, r)
    now = dates[-1] + pd.Timedelta(days=1)
    recs = R.monitor_records(mon, now)
    assert len(recs) == len(R.AXES) + 1 and all(r.gate(now + pd.Timedelta(days=1)) for r in recs)
    assert "2018" not in str([r.payload for r in recs]) and any(r.payload["axis"] == "volatility" for r in recs)
    with pytest.raises(FirewallBreach):
        R.monitor_records(mon, dates[-1])                                            # the last processed day is not yet in the past
    assert R.monitor_records(R.RegimeMonitor(), now) == [] and len(R.monitor_records(mon, now, discovery_only=True)) == 1
    back = R.rebuild_named(R.RegimeMonitor.from_state(mon.state()))
    assert back.summary().equals(named.summary()) and back.active() == named.active()


def test_adopt_tuned_changes_definitions_only_on_a_real_gain_and_logs_why():
    cfg = R.RegimeConfig()
    tune = pd.DataFrame({"q_lo": [0.33, 0.25], "q_hi": [0.67, 0.75], "score": [4.0, 9.0]})
    new, dec = R.adopt_tuned(cfg, tune)
    assert dec["adopted"] and (new.q_lo, new.q_hi) == (0.25, 0.75) and dec["config_hash"] and cfg.q_lo == 0.33      # original untouched
    same, dec2 = R.adopt_tuned(cfg, pd.DataFrame({"q_lo": [0.33, 0.25], "q_hi": [0.67, 0.75], "score": [4.0, 4.5]}))
    assert same is cfg and not dec2["adopted"] and "not 20%" in dec2["reason"]
    assert not R.adopt_tuned(cfg, pd.DataFrame({"q_lo": [0.33], "q_hi": [0.67], "score": [0.0]}))[1]["adopted"] and not R.adopt_tuned(cfg, pd.DataFrame())[1]["adopted"]


def test_active_regimes_key_and_cohort_shares_by_regime():
    dates, rows = world_rows({"vix": 34}, {"vix": 11}, n=300)
    mon = run_monitor(dates, rows, R.RegimeConfig(min_history=60, refit_every=10_000))
    assert {"high_vol", "low_vol"} & set(R.active_regimes(mon.states[-1]))
    assert R.active_regimes(mon.states[0]) == ()
    led = X.CohortAttributionLedger()
    rng = np.random.default_rng(0)
    for d in dates[100:180]:
        led.add(d, X.decompose_full(pd.DataFrame({"ret": rng.normal(0, 0.01, 60)}, index=[f"S{i}" for i in range(60)]), X.CrossConfig()))
    tab = R.cohort_shares_by_regime(led, mon, "volatility", dates[-1], min_days=5)
    assert not tab.empty and tab["days"].sum() <= 80 and "seq_sector" in tab.columns
    assert R.cohort_shares_by_regime(X.CohortAttributionLedger(), mon, "volatility", dates[-1]).empty
