"""Tests for engine.research.market_expectations, change_points and regime_memory (PREDICTION_ERROR_ADDITION checklists F, G, I, R, W, X;
canon C68, C63). Synthetic data only. Every mechanism has a planted case it must catch, a null case where it must find nothing, and the
empty case. Checklist X gets its own proof: scrambling everything after t leaves the detector state at t bit-identical, and a
deliberately leaky runner is caught by the same check."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach
from engine.research import change_points as CP
from engine.research import market_expectations as ME
from engine.research import regime_memory as RM
from engine.research import regimes as RG
from engine.research.change_points import Scope, Target


# ---------------------------------------------------------------------------------------------- market expectations helpers

def dates_of(n, start="2020-01-06"):
    return [d.date().isoformat() for d in pd.bdate_range(start, periods=n)]


def ar1(rng, n, phi=0.5, sd=1.0):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sd)
    return x


LEVELS = {"volatility": (1.0, 0.1), "momentum_persistence": (1.0, 0.1), "dispersion": (1.0, 0.1), "gap_behavior": (1.0, 0.1), "holding_period": (3.0, 0.3),
          "optimal_exit_days": (2.0, 0.2), "microstructure": (1.0, 0.1), "reversal_probability": (0.5, 0.03), "sector_leadership": (0.4, 0.03),
          "correlation": (0.3, 0.03), "breadth": (0.5, 0.03), "event_share": (0.2, 0.02), "pattern_overlap": (0.3, 0.03)}


def make_history(n=200, step_at=None, driver="volatility", seed=0, n_after=5, hit_drop=False, small=False):
    """Market observations. opportunity = 0.10 + link * standardised driver + noise; the driver(s) step at step_at (5 sd, or 1.6 sd each
    across six drivers when small=True, so no single one is significant but together they move opportunity by the same 0.04)."""
    rng = np.random.default_rng(seed)
    ds = dates_of(n)
    q = {k: lv + sd * ar1(rng, n, 0.4) for k, (lv, sd) in LEVELS.items()}
    drivers = ("volatility",) if not small else ("volatility", "dispersion", "correlation", "breadth", "gap_behavior", "microstructure")
    jump, link = (5.0, 0.008) if not small else (1.6, 0.0042)
    opp = 0.10 + rng.normal(0, 0.004, n)
    if driver:
        for d in drivers:
            lv, sd = LEVELS[d]
            if step_at is not None:
                q[d][step_at:] += jump * sd
            opp += link * (q[d] - lv) / sd
    hit = 0.30 + rng.normal(0, 0.03, n)
    if hit_drop and step_at is not None:
        hit[step_at:] -= 0.15
    obs = []
    for i in range(n):
        v = {"opportunity": float(opp[i]), "n_qualifying": float(3000 * opp[i] * 0.7), "n_movers": float(3000 * opp[i]),
             "pick_hit_rate": float(np.clip(hit[i], 0.01, 0.99)), "n_picked": 100.0}
        v.update({k: float(q[k][i]) for k in q})
        obs.append(ME.MarketObs(ds[i], v))
    return obs


def surprise_at(history, exp_value, q="opportunity"):
    a = history[-1].get(q)
    sc = 0.004 if q == "opportunity" else 0.03
    return ME.MarketSurprise(q, exp_value, a, sc, a - exp_value, (a - exp_value) / sc, 9.0, (a - exp_value) / exp_value)


# ---------------------------------------------------------------------------------------------- checklist F / G

def test_config_and_observation_validation():
    assert ME.ExpectationConfig().validate() == []
    assert ME.ExpectationConfig(ewma_lambda=1.5).validate() and ME.ExpectationConfig(recent=1).validate()
    with pytest.raises(ValueError):
        ME.MarketExpectationEngine(ME.ExpectationConfig(recent=1))
    ok = ME.MarketObs("2020-01-06", {"opportunity": 0.1, "n_movers": 300, "n_qualifying": 200, "breadth": 0.5})
    assert ok.validate() == []
    assert ME.MarketObs("2020-01-06", {"opportunity": 1.4}).validate()
    assert ME.MarketObs("2020-01-06", {"n_movers": 100, "n_qualifying": 200}).validate()      # qualifying cannot exceed movers
    assert ME.MarketObs("2020-01-06", {"opportunity": float("nan")}).validate()
    assert ME.MarketObs("2020-01-06", {}, {"a": 0.8, "b": 0.6}).validate()
    assert ME.MarketObs("not a date", {}).validate()


def test_expectation_ledger_is_immutable_and_hash_chained():
    eng, _ = ME.run_history(make_history(60))
    assert len(eng.ledger) >= 30 and eng.ledger.verify() == []
    first = eng.ledger.entries()[3]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.for_period = "2030-01-01"
    # a rewrite of an old expectation (planted) is found
    forged = dataclasses.replace(first, quantities=tuple((n, e * 2, s) for n, e, s in first.quantities))
    eng.ledger._items[3] = forged
    assert 3 in eng.ledger.verify()
    # appending something that does not extend the head, or a stale period, is refused
    eng2, _ = ME.run_history(make_history(40))
    last = eng2.ledger.entries()[-1]
    with pytest.raises(FirewallBreach):
        eng2.ledger.append(dataclasses.replace(last, prev_hash="genesis", digest=""))
    bad = dataclasses.replace(last, prev_hash=eng2.ledger.head())
    bad = dataclasses.replace(bad, digest=bad.compute_digest())
    with pytest.raises(FirewallBreach):
        eng2.ledger.append(bad)                                 # same period again
    with pytest.raises(ValueError):
        eng2.ledger.append(dataclasses.replace(last, made_at=last.for_period))


def test_engine_is_forward_only_and_deterministic():
    hist = make_history(50)
    eng = ME.MarketExpectationEngine()
    for o in hist[:30]:
        eng.step(o.date, o)
    with pytest.raises(FirewallBreach):
        eng.step(hist[29].date, hist[31])                       # observation from the future of `now`
    with pytest.raises(FirewallBreach):
        eng.step(hist[29].date, hist[10])                       # not after the last processed observation
    with pytest.raises(ValueError):
        eng.step(hist[40].date, ME.MarketObs(hist[40].date, {"opportunity": 5.0}))
    a, _ = ME.run_history(hist)
    b, _ = ME.run_history(hist)
    assert a.content_hash() == b.content_hash()
    with pytest.raises(ValueError):
        a._forecast(pd.Timestamp("2020-06-01").date(), "2020-05-01")


def test_no_expectation_before_enough_history_and_empty_engine():
    eng = ME.MarketExpectationEngine()
    res = eng.step("2020-01-06", ME.MarketObs("2020-01-06", {"opportunity": 0.1}))
    assert res.next_expectation is None and "fewer than" in res.reason_no_expectation
    assert eng.surprise_history("opportunity").tolist() == [float("nan")] or np.isnan(eng.surprise_history("opportunity")).all()
    assert ME.forecast_quantity(np.array([]), ME.ExpectationConfig()) is None
    assert ME.expectation_skill(eng, "opportunity")["skill"] is None


def test_resolve_matches_the_period_and_standardises_errors():
    eng, res = ME.run_history(make_history(60))
    exp = eng.ledger.entries()[-1]
    obs = ME.MarketObs(exp.for_period, {"opportunity": exp.expected("opportunity") * 1.5})
    out = ME.resolve(exp, obs, exp.for_period)
    s = out["opportunity"]
    assert s.move == ME.Move.EXPANSION and s.relative_gap == pytest.approx(0.5) and s.z == pytest.approx(s.error / s.scale) and s.bits > 0
    assert "volatility" not in out                                              # missing quantity is absent, not zero
    with pytest.raises(ValueError):
        ME.resolve(exp, ME.MarketObs("2031-01-01", {"opportunity": 0.1}), "2031-01-01")
    with pytest.raises(FirewallBreach):
        ME.resolve(exp, obs, "2019-01-01")


def test_null_market_never_triggers_an_investigation():
    eng, res = ME.run_history(make_history(260, step_at=None, driver=None, seed=3))
    assert [r for r in eng.reports if r.target == "opportunity"] == []                # nothing changed in the market itself
    assert len(eng.reports) <= 6                                                      # chance z >= 2.5 on the noisy hit rate is rare
    assert not any(ME.may_claim_strategy_stopped(r) for r in eng.reports)             # and never turns into 'the strategy stopped working'
    sk = ME.expectation_skill(eng, "opportunity")
    assert sk["n"] >= 100 and sk["skill"] is not None


def test_planted_volatility_expansion_is_explained_by_volatility_only():
    hist = make_history(160, step_at=150, seed=1)
    h = hist[:155]
    exp = float(np.mean([o.get("opportunity") for o in h[:150]]))
    rep = ME.investigate(h, surprise_at(h, exp), h[-1].date, seed=1)
    assert rep.move == ME.Move.EXPANSION
    sup = [p.name for p in rep.supported()]
    assert "volatility_change" in sup and set(sup) <= {"volatility_change"}, sup
    assert rep.joint_share is not None and 0.6 < rep.joint_share < 1.6
    assert rep.conclusion == ME.Conclusion.EXPLAINED
    assert not ME.may_claim_strategy_stopped(rep)
    txt = ME.render_report(rep)
    assert "volatility_change" in txt and "EXPLAINED" in txt
    other = {p.name: p for p in rep.probes}["breadth_change"]
    assert other.finding == ME.Finding.NOT_SUPPORTED


def test_contraction_driven_by_a_quantity_uses_the_same_tests():
    hist = make_history(160, step_at=150, seed=2)
    for o in hist[150:]:                                                         # flip: volatility falls, opportunity falls with it
        o.values["volatility"] = 2.0 - o.values["volatility"]
        o.values["opportunity"] = 0.10 - (o.values["opportunity"] - 0.10)
    h = hist[:155]
    exp = float(np.mean([o.get("opportunity") for o in h[:150]]))
    rep = ME.investigate(h, surprise_at(h, exp), h[-1].date, seed=2)
    assert rep.move == ME.Move.CONTRACTION
    assert "volatility_change" in [p.name for p in rep.supported()]


def test_untestable_is_not_read_as_not_supported():
    hist = make_history(160, step_at=150, seed=1)
    for o in hist:                                                               # the caller never measured these
        for k in ("holding_period", "optimal_exit_days", "microstructure"):
            o.values.pop(k)
    h = hist[:155]
    rep = ME.investigate(h, surprise_at(h, 0.10), h[-1].date)
    by = {p.name: p for p in rep.probes}
    assert by["holding_period_change"].finding == ME.Finding.UNTESTABLE and by["microstructure_change"].finding == ME.Finding.UNTESTABLE
    assert "holding_period_change" in rep.untestable() and any("untestable" in n for n in rep.notes)
    qs = ME.investigation_questions(rep, "2026-09-29", "2026-09-28")
    assert any("holding_period_change" in q.text for q in qs)


def test_many_small_changes_combine_but_no_single_one_is_significant():
    hist = make_history(170, step_at=150, seed=4, small=True)
    h = hist[:156]
    exp = float(np.mean([o.get("opportunity") for o in h[:150]]))
    rep = ME.investigate(h, surprise_at(h, exp), h[-1].date, seed=4)
    assert rep.combined.finding == ME.Finding.SUPPORTED or len(rep.supported()) >= 2       # the many-small-changes route or joint route
    assert rep.conclusion in (ME.Conclusion.EXPLAINED, ME.Conclusion.PARTLY_EXPLAINED)
    few = ME.combined_small_changes(rep.probes[:2], ME.ExpectationConfig())
    assert few.finding == ME.Finding.UNTESTABLE                                   # fewer than three usable probes


def test_combined_small_changes_pool_detects_what_no_single_probe_does():
    cfg = ME.ExpectationConfig()
    mk = lambda i, z: ME.ProbeResult(f"p{i}", "q", ME.BreakCause.RANDOM_VARIATION, ME.Finding.NOT_SUPPORTED, z, 0.2, 0.4, 1.0, 3.0, 0.1, 0.1, "")
    pooled = ME.combined_small_changes([mk(i, 1.6) for i in range(6)], cfg)         # six 1.6-sigma pushes the same way
    assert pooled.finding == ME.Finding.SUPPORTED and pooled.shift_z > 3.5
    mixed = ME.combined_small_changes([mk(i, 1.6 * (-1) ** i) for i in range(6)], cfg)   # pushes cancel
    assert mixed.finding == ME.Finding.NOT_SUPPORTED


def test_candidate_selection_failure_is_an_explicit_alternative():
    hist = make_history(170, step_at=150, driver=None, seed=5, hit_drop=True)
    h = hist[:158]
    exp = float(np.mean([o.get("pick_hit_rate") for o in h[:150]]))
    rep = ME.investigate(h, surprise_at(h, exp, "pick_hit_rate"), h[-1].date)
    assert rep.selection.attribution == ME.Attribution.SELECTION_FAILURE and rep.selection.hit_z < -4
    assert rep.conclusion == ME.Conclusion.SELECTION_FAILURE
    assert not ME.may_claim_strategy_stopped(rep)
    # the market did contract and the hit rate held: MARKET_CHANGE
    hist2 = make_history(170, step_at=150, seed=6)
    for o in hist2[150:]:
        o.values["n_qualifying"] = o.values["n_qualifying"] * 0.5
    sel = ME.selection_check(hist2[:158], ME.Move.CONTRACTION, 5, ME.ExpectationConfig())
    assert sel.attribution == ME.Attribution.MARKET_CHANGE
    # no pick data at all: UNKNOWN, never assumed fine
    for o in hist2:
        o.values.pop("pick_hit_rate"), o.values.pop("n_picked")
    assert ME.selection_check(hist2[:158], ME.Move.CONTRACTION, 5, ME.ExpectationConfig()).attribution == ME.Attribution.UNKNOWN


def test_units_patterns_sectors_stocks_are_classified_and_the_strategy_claim_needs_specifics():
    rng = np.random.default_rng(0)
    n, k = 100, 5
    base = lambda m: m + rng.normal(0, 0.01, n)
    series = {"weakens": base(0.03), "reverses": base(0.03), "stable": base(0.03), "wakes": base(0.0), "stops": base(0.03), "grows": base(0.02)}
    series["weakens"][-k:] -= 0.02
    series["reverses"][-k:] -= 0.06
    series["wakes"][-k:] += 0.04
    series["stops"][-k:] -= 0.024
    series["grows"][-k:] += 0.05
    out = {u.unit: u for u in ME.unit_shifts(series, k, ME.ExpectationConfig())}
    assert out["stable"].status == ME.UnitStatus.STABLE
    assert out["reverses"].status == ME.UnitStatus.REVERSED
    assert out["stops"].status == ME.UnitStatus.STOPPED_RESPONDING
    assert out["weakens"].status in (ME.UnitStatus.WEAKENED, ME.UnitStatus.STOPPED_RESPONDING)
    assert out["grows"].status == ME.UnitStatus.STRENGTHENED and out["wakes"].newly_active
    assert ME.unit_shifts({"short": [0.1] * 10}, k, ME.ExpectationConfig())[0].status == ME.UnitStatus.INSUFFICIENT
    assert ME.unit_shifts({}, k, ME.ExpectationConfig()) == []
    # the claim: only with a named deteriorated pattern
    hist = make_history(160, step_at=150, seed=1)
    h = hist[:155]
    surprise = ME.MarketSurprise("opportunity", 0.10, 0.06, 0.004, -0.04, -10.0, 9.0, -0.4)
    with_pat = ME.investigate(h, surprise, h[-1].date, pattern_effects=series)
    without = ME.investigate(h, surprise, h[-1].date)
    assert with_pat.move == ME.Move.CONTRACTION and any(s.startswith("pattern:") for s in with_pat.specifics())
    assert ME.may_claim_strategy_stopped(with_pat) or with_pat.selection.attribution == ME.Attribution.SELECTION_FAILURE
    assert not ME.may_claim_strategy_stopped(without) and without.specifics() == []


def test_precursor_scan_finds_a_replicated_lagged_cause_and_nothing_in_noise():
    rng = np.random.default_rng(7)
    n = 320
    hist = make_history(n, step_at=None, driver=None, seed=8)
    ev = np.array([o.get("event_share") for o in hist])
    z = np.full(n, np.nan)
    z[2:] = 2.0 * (ev[:-2] - ev.mean()) / ev.std() + rng.normal(0, 1, n - 2)         # error at t follows event_share two periods earlier
    cfg = ME.ExpectationConfig(n_perm=300)
    res = ME.precursor_scan(hist, z, cfg, seed=1)
    good = [p for p in res if p.usable]
    assert any(p.probe == "external_information" and p.lag == 2 for p in good)
    null = ME.precursor_scan(hist, np.r_[np.nan, rng.normal(0, 1, n - 1)], cfg, seed=1)
    assert [p for p in null if p.usable] == []
    assert ME.precursor_scan(hist[:10], np.zeros(10), cfg) == []


def test_engine_end_to_end_investigates_the_first_big_error_and_feeds_the_tracker():
    from engine.learning.surprise import SurpriseTracker
    hist = make_history(170, step_at=150, seed=9)
    eng = ME.MarketExpectationEngine(seed=9)
    trk = SurpriseTracker()
    fed = 0
    for i, o in enumerate(hist):
        res = eng.step(o.date, o, regime_changed=True)
        if i + 1 < len(hist):
            fed += eng.feed_tracker(trk, res, hist[i + 1].date)
    opp = [r for r in eng.reports if r.target == "opportunity"]
    assert opp and hist[150].date <= opp[0].period <= hist[153].date, [r.period for r in opp][:3]        # caught within days of the step
    assert fed > 0 and len(trk) == fed
    rec = eng.matured_report(opp[0], "2026-09-29")
    assert "period" not in rec.payload and opp[0].period not in str(rec.payload)
    with pytest.raises(FirewallBreach):
        rec.gate(opp[0].period)
    assert rec.gate(hist[160].date)["target"] == "opportunity"
    assert opp[0].regime_changed is True
    assert ME.sector_concentration({"1": [10, 8], "2": [10, 1], "3": [10, 1]}) > 0.3
    assert ME.sector_concentration({"1": [10, 1]}) is None and ME.sector_concentration({}) is None


def test_observation_from_day_record_reads_the_observer_summary():
    from engine.research import observer as OB
    days = list(OB.synthetic_days(2, n=400, seed=1))
    rec = OB.observe_day(days[0][0], days[0][1], days[0][1].date if hasattr(days[0][1], "date") else "2020-01-10")
    obs = ME.observation_from_day_record(rec, {"correlation": 0.3})
    assert obs.get("correlation") == 0.3 and obs.get("opportunity") is not None and obs.validate() == []
    assert obs.get("holding_period") is None                                        # not derivable from the record: stays None, not 0


# ---------------------------------------------------------------------------------------------- checklist I

def null_streams(n=250, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return {f"s{i}": rng.normal(0, 1, n) for i in range(k)}


def test_threshold_is_calibrated_and_false_alarm_rate_is_bounded():
    cfg = CP.ChangeConfig()
    h = CP.calibrate_h(cfg)
    assert h == CP.calibrate_h(cfg) and 5 < h < 40
    assert CP.calibrate_h(dataclasses.replace(cfg, alpha=0.01)) > h                  # tighter false-alarm level, higher bar
    rng = np.random.default_rng(1)
    alarms = 0
    n_series = 150
    days = dates_of(cfg.warmup + cfg.horizon)
    for i in range(n_series):
        d = CP.StreamDetector("x", Target.VOLATILITY, cfg)
        for j, v in enumerate(rng.normal(0, 1, cfg.warmup + cfg.horizon)):
            if d.update(days[j], v) is not None:
                alarms += 1
                break
    assert alarms / n_series < 0.12                                                  # stated 5% + Monte Carlo slack
    with pytest.raises(ValueError):
        CP.StreamDetector("x", Target.VOLATILITY, CP.ChangeConfig(warmup=5))


def test_planted_shift_is_detected_early_located_and_carries_its_information_cutoff():
    rng = np.random.default_rng(2)
    n, cp = 250, 120
    ds = dates_of(n)
    x = rng.normal(0, 1, n)
    x[cp:] += 2.0
    run = CP.run_forward(ds, {"a": x}, {"a": Target.CORRELATION})
    assert len(run.detections) >= 1
    d = run.detections[0]
    assert d.direction == 1 and cp <= d.alarm_index <= cp + 25 and abs(d.change_index - cp) <= 10
    assert d.validate() == [] and d.information_through == d.alarm_date and d.latency >= 0
    sc = CP.evaluate_detections(run.detections, {"a": cp})[0]
    assert sc.detected and 0 <= sc.latency <= 25 and sc.false_alarms == 0
    down = rng.normal(0, 1, n)
    down[cp:] -= 2.0
    assert CP.run_forward(ds, {"a": down}, {"a": Target.CORRELATION}).detections[0].direction == -1


def test_null_streams_raise_no_alarm_and_a_missing_or_constant_stream_is_safe():
    ds = dates_of(250)
    s = null_streams(seed=11)
    run = CP.run_forward(ds, s, {k: Target.BREADTH for k in s})
    assert len(run.detections) <= 1
    s2 = {"const": np.ones(250), "gappy": np.where(np.arange(250) % 3 == 0, np.nan, 1.0), "empty": np.full(250, np.nan)}
    r2 = CP.run_forward(ds, s2, {k: Target.BREADTH for k in s2})
    assert r2.detections == () and r2.monitor.detectors["empty"].phase == "warmup"
    with pytest.raises(ValueError):
        CP.run_forward(ds, {"short": np.zeros(10)}, {"short": Target.BREADTH})
    assert CP.evaluate_detections([], {}) == [] and CP.summarize([]) == {}


def test_a_stream_that_changes_a_constant_reference_is_still_caught():
    x = np.r_[np.ones(60), np.ones(30) * 1.5]
    run = CP.run_forward(dates_of(90), {"c": x}, {"c": Target.HOLDING_PERIOD})
    assert run.detections and run.detections[0].alarm_index >= 60


def test_detector_refuses_out_of_order_values_and_the_monitor_refuses_the_future():
    d = CP.StreamDetector("x", Target.VOLATILITY)
    d.update("2020-01-06", 0.1)
    with pytest.raises(FirewallBreach):
        d.update("2020-01-06", 0.1)
    m = CP.ChangeMonitor()
    m.register("a", Target.BREADTH)
    m.update("2020-01-06", {"a": 1.0}, now="2020-01-06")
    with pytest.raises(FirewallBreach):
        m.update("2020-01-07", {"a": 1.0}, now="2020-01-06")
    with pytest.raises(FirewallBreach):
        m.update("2020-01-06", {"a": 1.0})
    with pytest.raises(ValueError):
        m.update("2020-01-07", {"unregistered": 1.0})


def test_checkpoint_round_trip_continues_bit_identically():
    ds = dates_of(200)
    x = null_streams(200, 3, seed=5)
    x["s0"][120:] += 3
    tg = {k: Target.BREADTH for k in x}
    full = CP.run_forward(ds, x, tg)
    half = CP.run_forward(ds, x, tg, upto=99)
    restored = CP.ChangeMonitor.from_state(half.monitor.state())
    assert restored.digest() == half.monitor.digest()
    for i in range(100, 200):
        restored.update(ds[i], {k: x[k][i] for k in x}, now=ds[i])
    assert restored.digest() == full.monitor.digest()
    with pytest.raises(ValueError):
        CP.ChangeMonitor.from_state({"schema": "other"})


def test_no_retrospective_cheating_scrambling_the_future_leaves_the_past_bit_identical():
    ds = dates_of(240)
    s = null_streams(240, 5, seed=3)
    s["s1"][130:] += 2.5
    s["s2"][130:] += 2.5
    tg = {k: Target.VOLATILITY for k in s}
    t = 140
    assert CP.check_no_lookahead(ds, s, tg, t=t, seed=1) == []
    base = CP.run_forward(ds, s, tg)
    for mode in CP.SCRAMBLES:                                                        # every scramble really changed the future
        alt = CP.scramble_after(s, t, mode, 1)
        assert not np.array_equal(np.nan_to_num(alt["s0"][t + 1:]), s["s0"][t + 1:])
        assert np.array_equal(alt["s0"][: t + 1], s["s0"][: t + 1])
    # and the digest at t is exact: differs after t (state depends on the data that was there)
    alt = CP.run_forward(ds, CP.scramble_after(s, t, "level_shift", 1), tg)
    assert base.digests[t] == alt.digests[t] and base.digests[-1] != alt.digests[-1]


def test_the_lookahead_check_can_fail_a_leaky_detector_is_caught():
    """A runner that standardises with the FULL-sample mean and scale (a classic hindsight leak) must be flagged."""
    def leaky(dates, streams, targets, cfg=None):
        z = {k: (np.asarray(v, dtype="float64") - np.nanmean(v)) / (np.nanstd(v) or 1.0) for k, v in streams.items()}
        return CP.run_forward(dates, z, targets, cfg)
    ds = dates_of(240)
    s = null_streams(240, 3, seed=4)
    bad = CP.check_no_lookahead(ds, s, {k: Target.VOLATILITY for k in s}, t=120, seed=1, runner=leaky)
    assert bad and any("state digest differs" in b for b in bad)


def test_detectability_claims_cannot_use_hindsight():
    x = np.r_[np.zeros(80), np.ones(80) * 3.0] + np.random.default_rng(0).normal(0, 0.5, 160)
    run = CP.run_forward(dates_of(160), {"a": x}, {"a": Target.MOMENTUM})
    d = run.detections[0]
    assert CP.detectable_at(run.detections, "a", d.alarm_index) and not CP.detectable_at(run.detections, "a", 80)
    assert CP.earlier_claim_valid(d, d.alarm_index) and not CP.earlier_claim_valid(d, 80)
    forged = dataclasses.replace(d, information_through="2099-01-01")
    assert forged.validate() and not CP.earlier_claim_valid(forged, 200)


def test_latency_curve_is_monotone_and_stated():
    cur = CP.latency_curve(CP.ChangeConfig(), shifts=(1.0, 3.0), n_sim=60, seed=1)
    assert cur[1].median_delay < cur[0].median_delay and cur[1].detect_prob >= cur[0].detect_prob and cur[1].detect_prob > 0.95


def test_relationship_statistics_find_planted_relationships_and_not_null_ones():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 400)
    assert CP.daily_ic(y + rng.normal(0, 1, 400), y) > 0.5 and abs(CP.daily_ic(rng.normal(0, 1, 400), y)) < 0.2
    assert np.isnan(CP.daily_ic([1, 2], [1, 2])) and np.isnan(CP.daily_ic(np.ones(50), y[:50]))
    fired = rng.random(400) < 0.3
    assert CP.pattern_t(y + 1.5 * fired, fired) > 5 and abs(CP.pattern_t(y, fired)) < 3.5 and np.isnan(CP.pattern_t(y, np.zeros(400, bool)))
    common = rng.normal(0, 1, (1, 40))
    assert CP.mean_pair_correlation(common + 0.3 * rng.normal(0, 1, (20, 40))) > 0.7
    assert abs(CP.mean_pair_correlation(rng.normal(0, 1, (20, 40)))) < 0.1 and np.isnan(CP.mean_pair_correlation(np.zeros((2, 40))))
    p = np.full(300, 0.8)
    assert CP.calibration_residual(p, (rng.random(300) < 0.5).astype(float)) > 0.5      # overconfident
    assert abs(CP.calibration_residual(p, (rng.random(300) < 0.8).astype(float))) < 0.2


def test_build_streams_covers_all_thirteen_relationships_and_omits_what_it_lacks():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(0, 0.02, n)
    day = CP.DayInputs(outcome=y, features={"f1": y + rng.normal(0, 0.02, n)}, patterns={"pA": rng.random(n) < 0.3},
                       sector=rng.integers(0, 4, n), returns_window=rng.normal(0, 0.01, (15, 30)), day_abs_move=0.012, dispersion=0.02,
                       breadth_up=0.55, direction_hits=60, direction_n=100, past_long=rng.normal(0, 1, n), past_short=rng.normal(0, 1, n),
                       holding_days=3.2, optimal_exit_days=2.5, confidence_p=np.full(n, 0.7), confidence_hit=(rng.random(n) < 0.6).astype(float),
                       error_z=rng.normal(0, 1, 30), confident_failures=2, confident_n=40)
    v, tg, un = CP.build_streams(day)
    cov = CP.coverage_of_checklist_i(tg)
    assert all(cov.values()), [k for k, ok in cov.items() if not ok]
    assert Target.DISPERSION in tg.values() and Target.CONFIDENT_FAILURES in tg.values()
    assert any(s.startswith("sector:") for s in v) and all(np.isfinite(x) for x in v.values() if x is not None)
    v2, tg2, _ = CP.build_streams(CP.DayInputs())
    assert v2 == {} and not any(CP.coverage_of_checklist_i(tg2).values())


def test_regime_layer_streams_feed_the_detector():
    rng = np.random.default_rng(1)
    n = 380
    ds = pd.bdate_range("2018-01-01", periods=n)
    mon = RG.RegimeMonitor(RG.RegimeConfig(min_history=40, refit_every=30))
    for i, d in enumerate(ds):
        hi = i >= 250
        mon.process(d, {"ret": rng.normal(0.0003, 0.02 if hi else 0.006), "vix": (34 if hi else 14) + rng.normal(0, 0.8),
                        "dispersion": 0.012 * (1 + rng.normal(0, 0.05)), "dollar_volume": 1e10 * (1 + rng.normal(0, 0.03)),
                        "event_share": float(np.clip(0.1 + rng.normal(0, 0.01), 0, 1))})
    dates, series, tg = CP.streams_from_regime_monitor(mon)
    run = CP.run_forward(dates, series, tg)
    vol = [d for d in run.detections if d.stream == "vol"]
    assert vol and vol[0].direction == 1 and 250 <= vol[0].alarm_index <= 285
    assert isinstance(CP.regime_precursors(mon), tuple)


# ---------------------------------------------------------------------------------------------- scope and early warning (checklist R)

def universe(n_sectors=8, per=25):
    return {f"u{s}_{i}": f"S{s}" for s in range(n_sectors) for i in range(per)}


def test_scope_single_stock_sector_market_and_none():
    cfg = CP.ChangeConfig()
    uni = universe()
    p0 = CP.null_alarm_rate(cfg, cfg.scope_window)
    assert 0 < p0 < 0.01
    assert CP.classify_scope({}, uni, p0, cfg).scope == Scope.NONE
    lone = CP.classify_scope({"u3_1": "S3"}, uni, p0, cfg)
    assert lone.scope == Scope.SINGLE_STOCK and lone.units == ("u3_1",)
    sect = {u: s for u, s in uni.items() if s == "S2"}
    assert CP.classify_scope({u: s for u, s in list(sect.items())[:14]}, uni, p0, cfg).scope == Scope.SECTOR
    spread = {u: s for i, (u, s) in enumerate(uni.items()) if i % 4 == 0}
    v = CP.classify_scope(spread, uni, p0, cfg)
    assert v.scope == Scope.MARKET_WIDE and len(v.sectors) >= 2
    mk = CP.classify_scope({}, uni, p0, cfg, market_alarms=5, market_streams=6)
    assert mk.scope == Scope.MARKET_WIDE and mk.market_alarms == 5
    assert CP.classify_scope({}, {}, p0, cfg).scope == Scope.NONE
    # chance-level alarms across many sectors are not a regime change
    few = {"u0_0": "S0", "u4_3": "S4"}
    assert CP.classify_scope(few, uni, p0, cfg).scope in (Scope.SINGLE_STOCK, Scope.UNCLEAR)


def make_world(n=200, n_units=30, cp=120, kind="market", seed=0):
    rng = np.random.default_rng(seed)
    ds = dates_of(n)
    sectors = {f"u{i}": f"S{i % 3}" for i in range(n_units)}
    market = {k: rng.normal(0, 1, n) for k in ("volatility", "breadth", "correlation", "error_level", "calibration", "holding_period")}
    units = {u: rng.normal(0, 1, n) for u in sectors}
    if kind == "market":
        for k in market:
            market[k][cp:] += 2.5
        for u in units:
            units[u][cp:] += 2.0
    elif kind == "single":
        units["u4"][cp:] += 4.0
    elif kind == "sector":
        for u, s in sectors.items():
            if s == "S1":
                units[u][cp:] += 3.0
    return ds, market, units, sectors


def run_world(kind, seed=0):
    ds, market, units, sectors = make_world(kind=kind, seed=seed)
    sys_ = CP.EarlyWarningSystem()
    tg = {k: {"volatility": Target.VOLATILITY, "breadth": Target.BREADTH, "correlation": Target.CORRELATION, "error_level": Target.ERROR_DISTRIBUTION,
              "calibration": Target.CALIBRATION, "holding_period": Target.HOLDING_PERIOD}[k] for k in market}
    out = []
    for i, d in enumerate(ds):
        out.append(CP.step(sys_, d, {k: market[k][i] for k in market}, tg, unit_values={u: units[u][i] for u in units}, unit_sector=sectors))
    return sys_, out


def test_early_warning_market_wide_regime_change_reaches_alert_and_names_precursors():
    sys_, out = run_world("market")
    before, after = out[:120], out[120:]
    assert sum(1 for e in before if e.level in (CP.Level.WARNING, CP.Level.ALERT)) <= 3
    peak = [e for e in after if e.level == CP.Level.ALERT]
    assert peak and peak[0].scope.scope == Scope.MARKET_WIDE
    fired = set().union(*[e.fired() for e in after])
    assert {"volatility_change", "breadth_change", "correlation_change", "prediction_error_shift"} <= fired
    assert sys_.digest() == run_world("market")[0].digest()                             # deterministic


def test_early_warning_single_stock_and_sector_are_told_apart_from_the_market():
    _, single = run_world("single", seed=1)
    hits = [e for e in single if e.scope.scope != Scope.NONE]
    assert all(e.level != CP.Level.ALERT for e in single) and all(e.scope.scope in (Scope.SINGLE_STOCK, Scope.NONE) for e in single)
    assert any(e.scope.scope == Scope.SINGLE_STOCK and e.scope.units == ("u4",) for e in hits)
    _, sector = run_world("sector", seed=2)
    assert any(e.scope.scope == Scope.SECTOR and e.scope.sectors == ("S1",) for e in sector)
    assert not any(e.scope.scope == Scope.MARKET_WIDE for e in sector)
    _, null = run_world("null", seed=3)
    assert all(e.level in (CP.Level.NONE, CP.Level.WATCH) for e in null) and not any(e.scope.scope == Scope.MARKET_WIDE for e in null)


# ---------------------------------------------------------------------------------------------- checklist W / X (regime memory)

def regime_world(shift_len=None, n=330, cp=160, seed=0):
    """Four market quantities that all move at cp (for shift_len steps, or for good), a decoy that spikes then reverts before the
    declaration, three patterns (one weakens, one strengthens, one keeps working) and error z scores that jump after cp."""
    rng = np.random.default_rng(seed)
    ds = dates_of(n)
    q = {k: rng.normal(0, 1, n) for k in ("volatility", "breadth", "correlation", "error_level", "decoy")}
    end = n if shift_len is None else cp + shift_len
    for k in ("volatility", "breadth", "correlation", "error_level"):
        q[k][cp:end] += 3.0
    q["decoy"][cp - 22: cp - 8] += 3.0
    pats = {"P_weak": rng.normal(0.03, 0.01, n), "P_strong": rng.normal(0.0, 0.01, n), "P_keep": rng.normal(0.02, 0.01, n)}
    pats["P_weak"][cp:end] -= 0.027
    pats["P_strong"][cp:end] += 0.03
    z = rng.normal(0, 1, n)
    z[cp:end] += 3.0
    tg = {"volatility": Target.VOLATILITY, "breadth": Target.BREADTH, "correlation": Target.CORRELATION, "error_level": Target.ERROR_DISTRIBUTION,
          "decoy": Target.HOLDING_PERIOD}
    return ds, q, pats, z, tg


def evidence_at(ds, q, pats, z, tg, t, run=None):
    run = run or CP.run_forward(ds, q, tg)
    dets = [d for d in run.detections if d.alarm_index <= t]
    return RM.RegimeEvidence(ds[: t + 1], dets, {k: v[: t + 1] for k, v in q.items()}, {k: v[: t + 1] for k, v in pats.items()}, z[: t + 1]), run


def test_regime_record_has_every_checklist_w_field():
    ds, q, pats, z, tg = regime_world()
    ev, run = evidence_at(ds, q, pats, z, tg, 230)
    recs = RM.build_record(ev, ds[230], RM.MemoryConfig(), scope=CP.ScopeVerdict(Scope.MARKET_WIDE, 30, 20, ("S0", "S1"), (), 1e-9, 4, "test"))
    assert len(recs) == 1
    r = recs[0]
    assert r.validate() == [] and r.scope == Scope.MARKET_WIDE.value
    assert abs(ds.index(r.change_date) - 160) <= 10                                     # what the new regime looks like starts where it did
    assert ds.index(r.detected_at) >= ds.index(r.change_date) and r.detection_latency >= 0 and r.information_through == r.detected_at
    assert r.earliest_evidence.alarm_date <= r.detected_at
    assert r.old_regime.get("volatility").mean < 1 < r.new_regime.get("volatility").mean and not r.old_regime.partial
    assert r.patterns_weakened == ("P_weak",) and r.patterns_strengthened == ("P_strong",) and "P_keep" not in r.patterns_affected
    assert set(r.patterns_affected) == {"P_weak", "P_strong"}
    eb = r.errors_before_detection
    assert eb.n > 0 and eb.mean_z > 1.5 and eb.first_big_step is not None
    assert {"volatility", "breadth", "correlation", "error_level"} <= {s.stream for s in r.signals}
    assert 0.3 < r.confidence < 0.95 and dict(r.confidence_parts)["new_regime_seen"] > 0.5
    assert any(c.name == "volatility" for c in r.return_conditions) and any(c.kind == "pattern" and c.name == "P_weak" for c in r.return_conditions)
    assert "REGIME CHANGE" in RM.render_record(r) and "P_weak" in RM.render_record(r)


def test_misleading_signal_is_the_decoy_that_reverted():
    ds, q, pats, z, tg = regime_world()
    ev, run = evidence_at(ds, q, pats, z, tg, 230)
    assert any(d.stream == "decoy" for d in run.detections)                              # the decoy really alarmed
    r = RM.build_record(ev, ds[230])[0]
    assert "decoy" in [s.stream for s in r.misleading_signals] and "decoy" not in [s.stream for s in r.signals]
    lead = [s.stream for s in r.earlier_signals]
    assert lead and all(s.lead > 0 for s in r.earlier_signals)


def test_a_single_relationship_moving_is_not_a_regime_change_and_empty_evidence_is_safe():
    ds, q, pats, z, tg = regime_world()
    only = {k: v for k, v in q.items() if k == "volatility"}
    ev, _ = evidence_at(ds, only, pats, z, {"volatility": Target.VOLATILITY}, 230)
    assert RM.build_record(ev, ds[230]) == []
    empty = RM.RegimeEvidence([], [], {})
    assert RM.build_record(empty, "2020-01-06") == []
    mem = RM.RegimeMemory()
    st = RM.step(mem, "2020-01-06", empty)
    assert st.new_records == () and st.advice is None and len(mem) == 0
    assert mem.advice(["volatility_change"], "2030-01-01").known is False and mem.advice([], "2030-01-01").known is False


def test_evidence_from_the_future_is_refused_fail_closed():
    ds, q, pats, z, tg = regime_world()
    ev, run = evidence_at(ds, q, pats, z, tg, 230)
    with pytest.raises(FirewallBreach):
        RM.build_record(ev, ds[200])                                                     # evidence runs past `now`
    last = max(d.alarm_index for d in ev.detections)
    m = last - 2                                                                          # `now` before the newest alarm, which stays in the list
    later = dataclasses.replace(ev, dates=ds[: m + 1], quantities={k: v[: m + 1] for k, v in q.items()},
                                pattern_effects={k: v[: m + 1] for k, v in pats.items()}, error_z=z[: m + 1])
    with pytest.raises(FirewallBreach):
        RM.build_record(later, ds[m])                                                    # detections dated after now
    bad = dataclasses.replace(ev, error_z=z[:10])
    with pytest.raises(FirewallBreach):
        RM.build_record(bad, ds[230])
    with pytest.raises(ValueError):
        RM.build_record(ev, ds[230], RM.MemoryConfig(min_streams=1))
    r = RM.build_record(ev, ds[230])[0]
    with pytest.raises(FirewallBreach):
        RM.RegimeMemory().add(r, ds[150])                                                # filed as of a date before it was declared


def test_memory_is_append_only_hash_chained_and_tamper_evident():
    ds, q, pats, z, tg = regime_world()
    ev, _ = evidence_at(ds, q, pats, z, tg, 230)
    r = RM.build_record(ev, ds[230])[0]
    mem = RM.RegimeMemory()
    assert mem.add(r, ds[230]) is True and mem.add(r, ds[230]) is False and len(mem) == 1
    assert mem.status(r.record_id) == RM.RegimeStatus.CANDIDATE and mem.verify() == []
    with pytest.raises(FirewallBreach):
        mem.set_status(r.record_id, RM.RegimeStatus.CONFIRMED, r.detected_at)             # not after the declaration
    assert mem.set_status(r.record_id, RM.RegimeStatus.CONFIRMED, ds[260]) and mem.set_status(r.record_id, RM.RegimeStatus.CONFIRMED, ds[261]) is False
    with pytest.raises(KeyError):
        mem.set_status("nope", RM.RegimeStatus.CONFIRMED, ds[270])
    h = mem.content_hash()
    mem._events[0] = dataclasses.replace(mem._events[0], status="CONFIRMED")               # planted rewrite of history
    assert mem.verify() and mem.content_hash() != h
    mem2 = RM.RegimeMemory()
    mem2.add(r, ds[230])
    object.__setattr__(mem2._records[r.record_id], "confidence", 0.99)                      # planted edit of a stored record
    assert -1 in mem2.verify()


def test_status_resolution_confirms_a_real_change_and_refutes_a_false_one():
    ds, q, pats, z, tg = regime_world()
    ev, run = evidence_at(ds, q, pats, z, tg, 200)
    r = RM.build_record(ev, ds[200])[0]
    assert RM.resolve_status(r, {k: v[:186] for k, v in q.items()}, ds[:186], ds[185]) == RM.RegimeStatus.CANDIDATE    # too soon
    assert RM.resolve_status(r, q, ds[:300], ds[300]) == RM.RegimeStatus.CONFIRMED
    with pytest.raises(FirewallBreach):
        RM.resolve_status(r, q, ds[:300], ds[250])
    fds, fq, fp, fz, ftg = regime_world(shift_len=14, seed=1)                            # planted FALSE regime change: 14 steps, then normal
    fev, frun = evidence_at(fds, fq, fp, fz, ftg, 175)
    frecs = RM.build_record(fev, fds[175])
    assert frecs, "the short shift must still be declared, or the test proves nothing"
    assert RM.resolve_status(frecs[0], fq, fds[:300], fds[300]) == RM.RegimeStatus.FALSE_ALARM


def test_a_false_regime_change_does_not_disable_a_successful_pattern():
    fds, fq, fp, fz, ftg = regime_world(shift_len=14, seed=1)
    fev, _ = evidence_at(fds, fq, fp, fz, ftg, 175)
    rec = RM.build_record(fev, fds[175])[0]
    status = RM.resolve_status(rec, fq, fds[:300], fds[300])
    keep = fp["P_keep"]
    for st in (RM.RegimeStatus.CANDIDATE, RM.RegimeStatus.FALSE_ALARM, status, RM.RegimeStatus.CONFIRMED, None):
        g = RM.pattern_guard("P_keep", st, keep[100:160], keep[160:300], advice_weakens=True)
        assert g.action == RM.PatternAction.KEEP, (st, g)
    # ...whereas a pattern that really collapsed is not protected by a false alarm, but is at most REDUCED without confirmation
    rng = np.random.default_rng(0)
    before, after = rng.normal(0.03, 0.01, 60), rng.normal(0.005, 0.01, 60)
    assert RM.pattern_guard("p", RM.RegimeStatus.CANDIDATE, before, after).action == RM.PatternAction.REDUCE
    assert RM.pattern_guard("p", RM.RegimeStatus.FALSE_ALARM, before, after).action == RM.PatternAction.REDUCE
    assert RM.pattern_guard("p", RM.RegimeStatus.CONFIRMED, before, after).action == RM.PatternAction.SUSPEND
    assert RM.pattern_guard("p", RM.RegimeStatus.CONFIRMED, before, -after).action == RM.PatternAction.SUSPEND
    assert RM.pattern_guard("p", RM.RegimeStatus.CONFIRMED, before, before + rng.normal(0, 0.001, 60)).action == RM.PatternAction.KEEP
    thin = RM.pattern_guard("p", RM.RegimeStatus.CONFIRMED, before, after[:5], advice_weakens=True)
    assert thin.action == RM.PatternAction.REDUCE and RM.pattern_guard("p", RM.RegimeStatus.CANDIDATE, before, after[:5]).action == RM.PatternAction.KEEP
    assert RM.pattern_guard("p", RM.RegimeStatus.CONFIRMED, [], []).action == RM.PatternAction.KEEP
    assert RM.pattern_guard("p", RM.RegimeStatus.CONFIRMED, np.ones(30), np.ones(30)).action == RM.PatternAction.KEEP


def test_return_conditions_detect_a_return_to_the_old_regime_and_only_then():
    ds, q, pats, z, tg = regime_world()
    ev, _ = evidence_at(ds, q, pats, z, tg, 230)
    r = RM.build_record(ev, ds[230])[0]
    still_new = {c.name: [4.0] * 10 for c in r.return_conditions}
    assert RM.returned(r, still_new) is False
    old = r.old_regime
    back = {c.name: [old.get(c.name).mean if c.kind == "quantity" else c.lo + 0.01] * 10 for c in r.return_conditions}
    assert RM.returned(r, back) is True
    partial = dict(back)
    partial[r.return_conditions[0].name] = [99.0] * 10
    assert RM.returned(r, partial) is False and RM.returned(r, {}) is False
    assert RM.returned(dataclasses.replace(r, return_conditions=()), back) is False


def clone(rec, rid, day, precursors_from=None, weak=("P_weak",), strong=("P_strong",)):
    sigs = tuple(dataclasses.replace(s, precursor=p, alarm_date=day) for s, p in zip(rec.signals, precursors_from or [s.precursor for s in rec.signals]))
    early = tuple(s for s in sigs if s.role == RM.SignalRole.LEADING)
    return dataclasses.replace(rec, record_id=rid, detected_at=day, change_date=day, information_through=day, signals=sigs, earlier_signals=early,
                               earliest_evidence=dataclasses.replace(rec.earliest_evidence, alarm_date=day), patterns_weakened=weak, patterns_strengthened=strong,
                               patterns_affected=tuple(sorted(weak + strong)))


def test_memory_learns_which_signals_lead_and_states_the_advice_with_its_false_alarm_history():
    ds, q, pats, z, tg = regime_world()
    ev, _ = evidence_at(ds, q, pats, z, tg, 230)
    base = RM.build_record(ev, ds[230])[0]
    pre = ["volatility_change", "breadth_change", "correlation_change", "prediction_error_shift"][: len(base.signals)]
    mem = RM.RegimeMemory()
    for i, day in enumerate(("2019-03-01", "2019-09-02", "2020-03-02")):
        r = clone(base, f"R{i}", day, pre)
        mem.add(r, day)
        mem.set_status(f"R{i}", RM.RegimeStatus.CONFIRMED, "2020-12-01")
    far = clone(base, "RF", "2020-06-01", pre, weak=("P_other",), strong=())
    mem.add(far, "2020-06-01")
    mem.set_status("RF", RM.RegimeStatus.FALSE_ALARM, "2020-12-01")
    adv = mem.advice(pre[:2], "2021-01-04")
    assert adv.known and adv.support == 4 and adv.false_alarm_rate == pytest.approx(0.25)
    assert adv.weaken == ("P_weak",) and adv.strengthen == ("P_strong",) and "P_weak" in adv.text and 0 < adv.confidence < 1
    assert mem.advice(pre[:2], "2019-09-02").known is False                                # only earlier records are usable at that date
    unrelated = mem.advice(["abnormal_dispersion"], "2021-01-04")
    assert unrelated.known is False and "UNKNOWN" in unrelated.text
    rel = {r.precursor: r for r in mem.signal_reliability("2021-01-04")}
    assert rel["volatility_change"].n_leading >= 3 and rel["volatility_change"].false_alarm_records == 1
    assert mem.signal_reliability("2018-01-01") == []
    one = RM.RegimeMemory()
    one.add(clone(base, "R0", "2019-03-01", pre), "2019-03-01")
    assert one.advice(pre[:2], "2021-01-04").known is False                                # a single memory is not enough


def test_matured_records_are_identity_free_and_gated():
    ds, q, pats, z, tg = regime_world()
    ev, _ = evidence_at(ds, q, pats, z, tg, 230)
    r = RM.build_record(ev, ds[230])[0]
    mem = RM.RegimeMemory()
    mem.add(r, ds[230])
    recs = mem.matured_records(ds[300], "2026-09-29")
    assert len(recs) == 1
    text = str(dict(recs[0].payload))
    assert r.detected_at not in text and "decoy" not in text and r.record_id not in text
    with pytest.raises(FirewallBreach):
        recs[0].gate(r.detected_at)
    assert recs[0].gate(ds[300])["scope"] == r.scope
    assert mem.matured_records(r.detected_at, "x") == []


def test_step_runs_forward_files_confirms_and_never_uses_the_future():
    ds, q, pats, z, tg = regime_world()
    run = CP.run_forward(ds, q, tg)
    mem = RM.RegimeMemory()
    seen = {}
    for t in (150, 175, 200, 260, 320):
        ev, _ = evidence_at(ds, q, pats, z, tg, t, run)
        out = RM.step(mem, ds[t], ev, active_precursors=["volatility_change"] if t > 200 else ())
        seen[t] = (out, len(mem))
    assert seen[150][1] == 0 and seen[320][1] == 1
    rid = mem.records()[0].record_id
    assert mem.status(rid) == RM.RegimeStatus.CONFIRMED and any(rid == c[0] for o, _ in seen.values() for c in o.status_changes)
    assert all(as_d <= ds[320] for as_d in [mem.get(rid).information_through])
    assert mem.verify() == []
    # records filed at t=175 are identical whether or not the data after 175 is scrambled (the state at t depends on nothing later)
    s2 = CP.scramble_after(q, 175, "level_shift", 3)
    run2 = CP.run_forward(ds, s2, tg)
    ev_a, _ = evidence_at(ds, q, pats, z, tg, 175, run)
    ev_b, _ = evidence_at(ds, {k: s2[k] for k in q}, pats, z, tg, 175, run2)
    ra, rb = RM.build_record(ev_a, ds[175]), RM.build_record(ev_b, ds[175])
    assert [CP.exact_digest(x) for x in ra] == [CP.exact_digest(x) for x in rb]
