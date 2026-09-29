"""Bible Phase 16 gap risk: event calendar, EVT tail, conditional model vs the old per-ticker model, loss-cap analysis."""
import numpy as np
import pandas as pd
import pytest
from test_exits import D, make_paths

from engine import gaprisk as G
from engine import stops as S


def gap_world(seed=0, weeks=170, per_week=40, n_tick=200, xi=0.35, event_mult=3.0):
    """Quarterly earnings filings (07:00-08:00 ET) whose gap night is fat-tailed and ~3x bigger than a normal night's.
    Returns Paths, events frame, and the TRUE event-week flags."""
    rng = np.random.default_rng(seed)
    p = make_paths(seed, weeks, per_week, kinds={"k": ([0.0] * 5, 0.012, 1)}, gap_sd=0.0005)
    tick = np.array([f"T{i:03d}" for i in range(n_tick)])
    p.ticker = np.concatenate([rng.choice(tick, per_week, replace=False) for _ in range(weeks)]).astype(object)
    phase = {t: int(rng.integers(0, 91)) for t in tick}
    start = np.datetime64("2009-01-05", "D")
    rows = [(t, start + phase[t] + 91 * k) for t in tick for k in range(0, 150)]
    ev = pd.DataFrame({"ticker": [r[0] for r in rows], "kind": "EARN",
                       "accepted": pd.to_datetime([f"{r[1]} 12:00" for r in rows], utc=True)})   # 12:00 UTC = before the open
    cal = G.EventCalendar(ev)
    real = cal.realized(p)
    u = rng.random(p.o.shape)
    z = np.where(real, np.where(u < 0.7, (rng.random(p.o.shape) ** -xi - 1) / xi * event_mult, 0.0),
                 np.where(u < 0.10, (rng.random(p.o.shape) ** -xi - 1) / xi, 0.0))
    g = np.clip(z * p.atr[:, None], 0, 0.9)
    fac = np.cumprod(1 - g, axis=1)                 # a gap at night d shifts every later bar
    for a in (p.o, p.h, p.l, p.c):
        a *= fac
    return p, ev, real.any(1)


@pytest.fixture(scope="module")
def world():
    return gap_world()


def test_calendar_effective_session_rules():
    ev = pd.DataFrame({"ticker": ["A", "A", "A", "B"], "kind": ["EARN", "EARN", "EARN", "OFFERING"],
                       "accepted": pd.to_datetime(["2020-03-03 13:00", "2020-03-03 22:00", "2020-03-07 15:00", "2020-03-03 13:00"], utc=True)})
    c = G.EventCalendar(ev)                  # 13:00Z=08:00 ET Tue (same day); 22:00Z=17:00 ET Tue (next day); Sat -> later Monday
    assert list(c.by_ticker["A"]) == [np.datetime64("2020-03-03"), np.datetime64("2020-03-04"), np.datetime64("2020-03-10")]
    assert "B" not in c.by_ticker and G.EventCalendar(ev.iloc[:0]).by_ticker == {}


def test_expected_event_is_point_in_time_and_finds_the_planted_rhythm(world):
    p, ev, truth = world
    cal = G.EventCalendar(ev)
    exp = cal.expected_in_week(p)
    late = p.week > np.datetime64("2015-06-01")
    assert exp[truth & late].mean() > 0.9 and exp[~truth & late].mean() < 0.35
    # PIT: deleting every filing dated after the cut cannot change predictions for entries before it
    cut = pd.Timestamp("2016-12-01", tz="UTC")
    trimmed = G.EventCalendar(ev[ev.accepted < cut])
    early = p.week < np.datetime64("2016-12-01")
    assert (trimmed.expected_in_week(p)[early] == exp[early]).all()
    assert not G.EventCalendar(ev.iloc[:0]).expected_in_week(p).any()


def test_gpd_recovers_the_planted_shape_and_quantile_inverts_the_tail():
    rng = np.random.default_rng(0)
    xi, beta = 0.4, 1.0
    exc = beta / xi * (rng.random(20000) ** -xi - 1)
    xh, bh = G.fit_gpd(exc)
    assert xh == pytest.approx(xi, abs=0.08) and bh == pytest.approx(beta, rel=0.25)
    xs, _ = G.fit_gpd(exc[:12])                                  # thin sample: shape shrinks toward the prior
    assert abs(xs - G.XI_PRIOR) <= abs(xh - G.XI_PRIOR) + 0.15
    for t in (0.05, 0.01, 0.001):
        z = G.gpd_quantile(t, 2.0, xh, bh, 0.1)
        assert G.gpd_sf(np.array([z]), 2.0, xh, bh, 0.1)[0] == pytest.approx(t, rel=1e-6)
    assert G.gpd_sf(np.array([1.0]), 2.0, 0.3, 1.0, 0.1)[0] == 0.1        # below the threshold: floor


def test_conditional_model_covers_fat_event_tails_the_old_model_misses(world):
    p, ev, _ = world
    cal = G.EventCalendar(ev)
    fac = {"old": lambda tr, pr: S.fit_gap_model(tr, q=pr).tail,
           "conditional": lambda tr, pr: (lambda m: (lambda x: m.night_quantile(x, pr)))(G.ConditionalGapModel(cal).fit(tr))}
    cov = G.walk_forward_coverage(p, fac, probs=(0.95, 0.99), min_train_weeks=60, block_weeks=25)
    s = G.coverage_summary(cov).set_index(["model", "prob"])
    for pr in (0.95, 0.99):
        assert 0.6 <= s.loc[("conditional", pr), "exceed_rate"] / (1 - pr) <= 1.6, s
    assert {"era", "ratio", "ci_lo", "ci_hi", "nights"} <= set(cov.columns)
    # unconditionally both look fine; the old model breaks exactly in the expected-event weeks
    ev_cov = G.walk_forward_coverage(p, fac, probs=(0.99,), min_train_weeks=60, block_weeks=25,
                                     label_fn=lambda t: np.where(cal.expected_in_week(t), "event_week", "normal_week"))
    r = ev_cov.set_index(["model", "era"]).ratio
    assert r[("old", "event_week")] > 2.0 and 0.5 <= r[("conditional", "event_week")] <= 1.7
    assert r[("old", "normal_week")] < r[("conditional", "normal_week")] + 0.3


def test_event_flag_matters_and_kappa_calibrates(world):
    p, ev, truth = world
    cal = G.EventCalendar(ev)
    tr = p.until(np.datetime64("2016-06-01"))
    te = p.take(np.flatnonzero(p.week > np.datetime64("2016-06-01")))
    m = G.ConditionalGapModel(cal).fit(tr)
    blind = G.ConditionalGapModel(None).fit(tr)                  # same model without the calendar
    exp = cal.expected_in_week(te)
    q_ev, q_no = m.night_quantile(te, 0.99), blind.night_quantile(te, 0.99)
    assert q_ev[exp].mean() > 1.5 * q_ev[~exp].mean()
    assert abs(q_no[exp].mean() / q_no[~exp].mean() - 1) < 0.25     # the blind model cannot tell the weeks apart
    assert 0.5 <= m.kappa <= 4.0 and (m.p_night(te, 0.10) >= 0).all()
    assert m.p_position(te, 0.10).max() <= 1 and (m.p_position(te, 0.05) >= m.p_position(te, 0.15) - 1e-12).all()
    assert np.allclose(m.tail(te), m.night_quantile(te, 0.95))


def test_loss_cap_analysis_shows_avoidance_and_that_sizing_cannot_fix_it(world):
    p, ev, _ = world
    lc = G.loss_cap_analysis(p, G.EventCalendar(ev), cap=0.10, stop_k=2.0, min_train_weeks=60, block_weeks=25)
    assert len(lc) > 3000 and lc.realized.any()
    tab = G.policy_table(lc, taus=(0.30, 0.10), target=0.01)
    allp = tab[tab.era == "ALL"].set_index("policy")
    assert allp.loc["avoid_event_weeks", "p_breach"] < allp.loc["no_filter", "p_breach"]
    assert allp.loc["avoid_events+model_p<=0.1", "p_breach"] <= allp.loc["model_p<=0.1", "p_breach"] + 1e-9
    assert allp.loc["model_p<=0.1", "kept_share"] < 1.0
    cal = G.calibration_of_p(lc, 4)
    assert cal.realized.iloc[-1] > cal.realized.iloc[0]            # the model's ranking carries information
    ra = G.required_avoidance(lc, target=0.05)
    assert "feasible" in ra and (not ra["feasible"] or ra["p_breach"] <= 0.05)
    assert G.weight_cap_for_portfolio_hit(0.03, 0.20) == pytest.approx(0.15)


def test_required_avoidance_reports_infeasible_when_no_cut_reaches_the_target():
    n = 1000
    rng = np.random.default_rng(0)
    lc = G.LossCap(rng.random(n), np.zeros(n, bool), rng.random(n) < 0.2, np.full(n, 2015), np.array(["k"] * n, object), np.zeros(n), 0.2)
    assert G.required_avoidance(lc, 0.01)["feasible"] is False      # 20% breach regardless of the ranking
    lc.realized[:] = False
    assert G.required_avoidance(lc, 0.05)["feasible"] is True


def test_empty_and_unfitted_and_stop_rule_hook(world):
    p, ev, _ = world
    e = p.take(np.array([], int))
    m = G.ConditionalGapModel(None).fit(e)
    assert m.night_quantile(e, 0.99).shape == (0,) and G.walk_forward_coverage(e, {}).empty
    assert len(G.loss_cap_analysis(e, None)) == 0 and G.policy_table(G.loss_cap_analysis(e, None)).empty
    with pytest.raises(RuntimeError):
        G.ConditionalGapModel(None).night_quantile(p.take(np.arange(5)), 0.99)
    rule = S.GapAwareStop(name="g", family="gap", gap_model_fn=G.conditional_gap_factory(G.EventCalendar(ev), calibrate=False))
    fitted = rule.fit(p.take(np.arange(4000)))
    assert isinstance(fitted.gap, G.ConditionalGapModel) and (fitted.dist(p.take(np.arange(50))) >= S.MIN_DIST).all()
