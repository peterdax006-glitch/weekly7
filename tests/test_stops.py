"""Bible Phase 16 (stop / loss engine): gap-through fills, no-guarantee reporting, gap-aware sizing and filtering,
walk-forward recovery of a planted crash type."""
import numpy as np
import pytest
from test_exits import D, Z, make_paths, one

from engine import exits as E
from engine import stops as S
from engine.exits import ExitSpec, Paths, run_exit


def plant_slide(p, rows, rate=0.93):
    """Rows lose `rate` per day in a straight line (no gaps): a slow crash a stop can actually catch."""
    p0 = p.prev_close[rows]
    lvl = p0[:, None] * rate ** np.arange(1, D + 1)[None, :]
    opn = np.concatenate([p0[:, None], lvl[:, :-1]], axis=1)
    p.o[rows], p.c[rows], p.h[rows], p.l[rows] = opn, lvl, opn, lvl * 0.995


def plant_gap(p, rows, day=2, size=0.30):
    """Rows gap down `size` at the open of `day` (previous close untouched) and stay there."""
    for a in (p.o, p.h, p.l, p.c):
        a[np.ix_(rows, range(day, D))] *= 1 - size


def test_wilson_and_empty():
    lo, hi = S.wilson(0, 100)
    assert lo == 0 and 0.03 < hi < 0.04
    assert S.wilson(0, 0) == (0.0, 1.0)
    lo, hi = S.wilson(50, 100)
    assert lo < 0.5 < hi


def test_loss_risk_never_claims_a_guarantee_and_shows_gap_damage():
    p = make_paths(1, 60, kinds={"k": ([0.0] * 5, 0.015, 1)})
    rows = np.flatnonzero(np.random.default_rng(0).random(len(p)) < 0.10)
    plant_gap(p, rows, 2, 0.30)
    res = S.AtrStop(name="a", family="atr", k=2.0).fit(p).run(p)          # a 6%-ish stop that the gap jumps through
    r = S.loss_risk(res)
    assert r["guarantee"] is False and "No stop guarantees" in r["note"]
    assert r["p_loss_gt20"] > 0.03 and r["worst_observed"] < -0.25         # the stop did NOT cap the loss at -20%
    assert r["p_loss_gt20_lo"] <= r["p_loss_gt20"] <= r["p_loss_gt20_hi"]
    assert r["gap_through_share"] > 0 and r["overshoot_max"] > 0.15
    e = S.loss_risk(res.take(np.array([], int)))
    assert e == {"n": 0, "guarantee": False, "note": S.NO_GUARANTEE}


def test_gap_stats_measure_frequency_and_severity():
    p = make_paths(2, 30, kinds={"k": ([0.0] * 5, 0.01, 1)}, gap_sd=0.002)
    calm = S.gap_stats(p)
    plant_gap(p, np.arange(0, len(p), 10), 1, 0.25)
    hot = S.gap_stats(p)
    assert hot["freq_gt20"] > calm["freq_gt20"] == 0 and hot["severity_max"] > 0.24
    assert S.gap_stats(p.take(np.array([], int))) == {"nights": 0}


@pytest.fixture(scope="module")
def gappy():
    """Half the tickers are gap-prone (and their gaps are the catastrophic losses); half are calm."""
    p = make_paths(3, 120, kinds={"k": ([0.0] * 5, 0.012, 1)}, gap_sd=0.003)
    bad = np.isin(p.ticker, [f"T{i:03d}" for i in range(0, 200, 4)])            # 25% of names
    rng = np.random.default_rng(1)
    hit = np.flatnonzero(bad & (rng.random(len(p)) < 0.5))
    plant_gap(p, hit, 2, 0.28)
    return p, bad


def test_gap_model_flags_the_gap_prone_names_and_filtering_cuts_catastrophes(gappy):
    p, bad = gappy
    half = len(p) // 2
    train, test = p.take(np.arange(half)), p.take(np.arange(half, len(p)))
    gm = S.fit_gap_model(train)
    tail = gm.tail(test)
    tbad = np.isin(test.ticker, np.unique(p.ticker[bad]))
    assert tail[tbad].mean() > tail[~tbad].mean() + 0.02
    keep = S.filter_candidates(test, gm, 0.05)
    assert keep[~tbad].mean() > 0.9 and keep[tbad].mean() < 0.3
    res = run_exit(test, ExitSpec(), Z, stop_dist=np.full(len(test), 0.10))
    eff = S.filter_effect(res, keep)
    assert eff["kept"]["cat_rate"] < 0.3 * eff["dropped"]["cat_rate"]
    assert eff["kept"]["worst"] > eff["dropped"]["worst"]
    assert eff["kept_share"] == pytest.approx(keep.mean())


def test_unknown_ticker_falls_back_to_type_then_overall():
    p = make_paths(4, 20, kinds={"k": ([0.0] * 5, 0.01, 1)})
    gm = S.fit_gap_model(p)
    q = one(**{"o": [100] * 5, "h": [101] * 5, "l": [99] * 5, "c": [100] * 5})
    assert gm.tail(q)[0] == pytest.approx(gm.kind["k"] if "k" in gm.kind else gm.overall)
    q.kind[:] = "unseen"
    assert gm.tail(q)[0] == pytest.approx(gm.overall)
    assert S.fit_gap_model(p.take(np.array([], int))).overall == 0.0


def test_position_sizing_shrinks_with_gap_tail_and_respects_the_budget():
    p = make_paths(0, 2)
    dist = np.full(len(p), 0.08)
    tail = np.linspace(0.0, 0.25, len(p))
    w = S.size_positions(p, dist, tail, loss_budget=0.02, max_weight=0.25)
    assert (np.diff(w) <= 1e-12).all() and w.max() <= 0.25
    assert ((dist + tail) * w <= 0.02 + 1e-12).all()


def test_stop_rule_distances_are_sane_and_learned_only_after_fit(gappy):
    p, bad = gappy
    vp = S.VolPercentileStop(name="v", family="vp", q=0.9)
    with pytest.raises(RuntimeError):
        vp.dist(p)
    d = vp.fit(p).dist(p)
    assert (d >= S.MIN_DIST).all() and (d <= S.MAX_DIST).all()
    atr = S.AtrStop(name="a", family="atr", k=3.0).dist(p)
    ga = S.GapAwareStop(name="g", family="gap", k=3.0).fit(p)
    gd = ga.dist(p)
    assert (gd <= atr + 1e-12).all()
    badmask = np.isin(p.ticker, np.unique(p.ticker[bad]))
    assert gd[badmask].mean() <= gd[~badmask].mean()                    # gap-prone names get tighter stops
    heavy = p.take(np.arange(200))
    heavy.weight[:] = 0.5
    sz = S.SizeAwareStop(name="s", family="size", k=3.0, risk_budget=0.012).dist(heavy)
    assert (sz <= max(S.MIN_DIST, 0.012 / 0.5) + 1e-12).all()


def test_every_default_stop_rule_runs_and_skip_filter_goes_to_cash(gappy):
    p, _ = gappy
    p = p.take(np.arange(3000))
    p.fail = np.zeros((len(p), D), bool)
    rules = S.default_stop_rules(True)
    assert rules[0].name == "no_stop" and len({r.name for r in rules}) == len(rules)
    for r in rules:
        res = r.fit(p).run(p)
        assert np.isfinite(res.net).all() and res.days.min() >= 0
    hf = next(r for r in rules if r.name == "hybrid_filter").fit(p).run(p)
    assert (hf.days == 0).any() and (hf.net[hf.days == 0] == 0).all()


def slide_world(seed=6, weeks=160):
    p = make_paths(seed, weeks, kinds={"slide": ([0.0] * 5, 0.015, 1), "calm": ([0.0] * 5, 0.012, 1)})
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero((p.kind == "slide") & (rng.random(len(p)) < 0.12))
    plant_slide(p, rows, 0.93)                                          # ~30% slow-crash losers with no gap
    return p, rows


def test_walk_forward_stops_recover_the_slow_crash_and_leave_the_calm_type_alone():
    p, rows = slide_world()
    wf = S.walk_forward_stops(p, min_train_weeks=60, block_weeks=20, seed=4)
    rep = S.risk_report(wf).set_index("kind")
    s = rep.loc["slide"]
    assert s["base_p_loss_gt20"] > 0.05 and s["learned_p_loss_gt20"] < 0.5 * s["base_p_loss_gt20"]
    assert s["learned_worst_observed"] > s["base_worst_observed"] + 0.05
    assert bool(rep["guarantee"].any()) is False
    picks = [c["slide"] for _, c in wf.fold_choices if "slide" in c]
    assert all(x != "no_stop" for x in picks)
    calm = wf.report().set_index("kind").loc["calm"]
    assert calm["verdict"] != "worse"


def test_feasibility_gate_rejects_a_rule_with_too_many_catastrophes():
    f = S.make_feasible(0.05)
    assert f({}, {"n": 500, "cat_rate": 0.0}) is True
    assert f({}, {"n": 500, "cat_rate": 0.10}) is False
    assert f({}, {"n": 0, "cat_rate": 0.0}) is False


def test_thin_and_empty_inputs_fall_back_to_baseline_and_empty_report():
    p = make_paths(0, 8, kinds={"k": ([0.0] * 5, 0.02, 1)})
    pol = E.learn(p, S.default_stop_rules(False), "2030-01-01")
    assert pol.selections["k"].chosen == "no_stop" and pol.selections["k"].reason == "neutral:insufficient_data"
    wf = S.walk_forward_stops(p, min_train_weeks=52)
    assert S.risk_report(wf).empty


def test_distance_curve_type_gap_table_era_and_reports():
    p, rows = slide_world(weeks=80)
    cur = S.stop_distance_curve(p.take(np.flatnonzero(p.kind == "slide")))
    assert cur.iloc[0].stop is None or np.isnan(cur.iloc[0].stop)
    assert cur.iloc[0].p_loss_gt20 > cur.iloc[3].p_loss_gt20             # a 10% stop catches the slow crash
    assert (cur.p_loss_gt20_hi >= cur.p_loss_gt20).all()
    gt = S.type_gap_table(p)
    assert set(gt.kind) == {"slide", "calm"}
    wf = S.walk_forward_stops(p, min_train_weeks=40, block_weeks=20, seed=1)
    txt = S.format_risk_report(wf)
    assert "No stop guarantees" in txt and "slide" in txt
    era = S.stop_era_breakdown(wf)
    assert not era.empty and (era.learned_p_gt20 <= era.base_p_gt20 + 1e-9).all()
    assert "not enough history" in S.format_risk_report(S.walk_forward_stops(p, min_train_weeks=10 ** 4))
    assert S.stop_era_breakdown(S.walk_forward_stops(p, min_train_weeks=10 ** 4)).empty


def test_stop_does_not_shrink_gap_severity_only_probability_of_slow_losses():
    """Planted: every catastrophe is a gap. A stop must then leave P(loss>20%) essentially unchanged - the engine may
    not pretend otherwise (this is the Phase-16 no-guarantee rule as a measurable statement)."""
    p = make_paths(9, 100, kinds={"k": ([0.0] * 5, 0.012, 1)})
    rows = np.flatnonzero(np.random.default_rng(2).random(len(p)) < 0.08)
    plant_gap(p, rows, 3, 0.30)
    base = S.loss_risk(run_exit(p, ExitSpec(), Z))
    stopped = S.loss_risk(run_exit(p, ExitSpec(), Z, stop_dist=np.full(len(p), 0.06)))
    assert abs(stopped["p_loss_gt20"] - base["p_loss_gt20"]) < 0.01
    assert stopped["gap_through_share"] > 0.05 and stopped["worst_observed"] < -0.25


def test_gap_aware_sizing_cuts_worst_week_when_gap_prone_names_are_the_losers(gappy):
    p, bad = gappy
    half = len(p) // 2
    train, test = p.take(np.arange(half)), p.take(np.arange(half, len(p)))
    gm = S.fit_gap_model(train)
    dist = np.full(len(test), 0.08)
    res = run_exit(test, ExitSpec(), Z, stop_dist=dist)
    tab = S.sizing_effect(test, res, dist, gm.tail(test), loss_budget=0.01).set_index("weights")
    assert tab.loc["gap_aware_sized", "cvar5"] > tab.loc["equal_weight", "cvar5"]
    assert tab.loc["gap_aware_sized", "mean_weight"] < 0.25 and S.sizing_effect(test.take(np.array([], int)), res.take(np.array([], int)), dist[:0], dist[:0]).empty


def test_gap_calibration_is_honest_on_stationary_gaps_and_exposes_a_regime_break():
    p = make_paths(12, 160, kinds={"k": ([0.0] * 5, 0.012, 1)}, gap_sd=0.01)
    half = len(p) // 2
    train, test = p.take(np.arange(half)), p.take(np.arange(half, len(p)))
    gm = S.fit_gap_model(train)
    cal = S.gap_calibration(gm, test)
    assert cal.exceed_rate.mean() == pytest.approx(0.05, abs=0.02)
    plant_gap(test, np.flatnonzero(np.random.default_rng(3).random(len(test)) < 0.3), 1, 0.12)   # gaps the model never saw
    broken = S.gap_calibration(gm, test)
    assert broken.exceed_rate.mean() > cal.exceed_rate.mean() + 0.02 and broken.excess.max() > 0.04
    assert S.gap_calibration(gm, test.take(np.array([], int))).empty
