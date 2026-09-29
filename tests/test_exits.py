"""Bible Phase 15 (exit learner): fill mechanics, no look-ahead, tiered selection, planted-effect recovery."""
import numpy as np
import pandas as pd
import pytest

from engine import exits as E
from engine.exits import CostModel, ExitSpec, Paths, run_exit

Z = CostModel(0.0, 0.0)
D = 5


def make_paths(seed=0, weeks=120, per_week=30, kinds=None, gap_sd=0.005, start="2015-01-05"):
    """Synthetic positions. kinds: {name: (daily_mu list, daily_sd, share)} - the planted per-type intra-week shape."""
    kinds = kinds or {"flat": ([0.0] * D, 0.02, 1.0)}
    rng = np.random.default_rng(seed)
    names = list(kinds)
    share = np.array([kinds[k][2] for k in names], float)
    wk0 = np.datetime64(start, "D") + np.arange(weeks) * 7
    n = weeks * per_week
    kind = rng.choice(names, n, p=share / share.sum())
    week = np.repeat(wk0, per_week)
    mu = np.array([kinds[k][0] for k in kind])
    sd = np.array([kinds[k][1] for k in kind])[:, None]
    p0 = rng.uniform(20, 80, n)
    gap = rng.normal(0, gap_sd, (n, D))
    intra = mu + rng.normal(0, 1, (n, D)) * sd
    o, c = np.empty((n, D)), np.empty((n, D))
    prev = p0.copy()
    for d in range(D):
        o[:, d] = prev * (1 + gap[:, d])
        c[:, d] = o[:, d] * (1 + intra[:, d])
        prev = c[:, d]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.4, (n, D))) * sd)
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.4, (n, D))) * sd)
    vol = np.broadcast_to(sd[:, 0], (n,)).copy()
    return Paths(o, h, l, c, p0, vol, vol * 1.1, week, week + 4, kind, np.array([f"T{i % 200:03d}" for i in range(n)]),
                 fail=None)


def one(o, h, l, c, prev=100.0, vol=0.02, fail=None):
    a = lambda x: np.array([x], float)
    return Paths(a(o), a(h), a(l), a(c), [prev], [vol], [vol], ["2020-01-06"], ["2020-01-10"], ["k"], ["X"],
                 None if fail is None else np.array([fail]))


BASE = dict(o=[100, 100, 100, 100, 100], h=[101] * 5, l=[99] * 5, c=[100] * 5)


# ---------------------------------------------------------------- data validation
def test_validate_catches_planted_defects():
    p = make_paths(weeks=5)
    assert p.validate() == []
    p.h[3, 2] = p.c[3, 2] * 0.9                   # high below the close
    p.o[7, 0] = np.nan
    bad = " ".join(p.validate())
    assert "high below" in bad and "non-finite o" in bad
    with pytest.raises(ValueError):
        p.check()


# ---------------------------------------------------------------- fill mechanics (hand-built paths, zero cost)
def test_week_end_exit_fills_last_close():
    p = one(**{**BASE, "c": [100, 101, 102, 103, 108]})
    r = run_exit(p, ExitSpec(), Z)
    assert r.days[0] == D and r.gross[0] == pytest.approx(0.08)


def test_target_intraday_fills_at_level_and_gap_fills_at_open():
    intraday = one([100, 100, 100, 100, 100], [100, 111, 100, 100, 100], [99] * 5, [100] * 5)
    r = run_exit(intraday, ExitSpec(target=0.10), Z)
    assert r.gross[0] == pytest.approx(0.10) and r.days[0] == 2 and E.REASONS[r.reason[0]] == "target"
    gapped = one([100, 115, 100, 100, 100], [100, 116, 100, 100, 100], [99, 114, 99, 99, 99], [100, 115, 100, 100, 100])
    r = run_exit(gapped, ExitSpec(target=0.10), Z)
    assert r.gross[0] == pytest.approx(0.15) and E.REASONS[r.reason[0]] == "gap_target"


def test_stop_gap_through_loses_more_than_the_stop():
    """The planted defect a naive simulator hides: the open gaps below the stop and the fill is the OPEN."""
    p = one([100, 70, 100, 100, 100], [100, 71, 100, 100, 100], [99, 69, 99, 99, 99], [100, 70, 100, 100, 100])
    r = run_exit(p, ExitSpec(), Z, stop_dist=np.array([0.10]))
    assert r.gross[0] == pytest.approx(-0.30) and E.REASONS[r.reason[0]] == "gap_stop"
    assert r.stop_overshoot[0] == pytest.approx(0.20)


def test_stop_and_target_in_same_bar_takes_the_stop():
    p = one([100] * 5, [112, 100, 100, 100, 100], [88, 99, 99, 99, 99], [100] * 5)
    r = run_exit(p, ExitSpec(target=0.10, stop=0.10), Z)
    assert r.gross[0] == pytest.approx(-0.10)


def test_trailing_uses_only_prior_highs():
    # day1 high 112 arms nothing yet on day 1 itself; from day 2 the level is 112*(1-.05)=106.4
    p = one([100, 100, 108, 100, 100], [100, 112, 110, 100, 100], [99, 99, 105, 99, 99], [100, 110, 108, 100, 100])
    r = run_exit(p, ExitSpec(trail=0.05), Z)
    assert r.days[0] == 3 and r.gross[0] == pytest.approx(106.4 / 100 - 1) and E.REASONS[r.reason[0]] == "trail"


def test_pattern_failure_fills_next_open_and_never_same_close():
    fail = [False, True, False, False, False]
    p = one([100, 100, 93, 100, 100], [101] * 5, [90] * 5, [100, 96, 100, 100, 100], fail=fail)
    r = run_exit(p, ExitSpec(fail_exit=True), Z)
    assert r.days[0] == 3 and r.gross[0] == pytest.approx(-0.07)     # decided at day-2 close, filled day-3 open
    with pytest.raises(ValueError):
        run_exit(one(**BASE), ExitSpec(fail_exit=True), Z)


def test_hold_days_and_costs():
    p = one([100] * 5, [101] * 5, [99] * 5, [100, 100, 104, 100, 100])
    r = run_exit(p, ExitSpec(hold_days=3), CostModel(fee_bps=10, slip_bps=0))
    assert r.days[0] == 3 and r.net[0] == pytest.approx(0.04 - 0.002)
    r2 = run_exit(p, ExitSpec(hold_days=3), CostModel(fee_bps=0, slip_bps=100))
    assert r2.gross[0] == pytest.approx(104 * 0.99 / (100 * 1.01) - 1)


# ---------------------------------------------------------------- no look-ahead
def test_bars_after_the_exit_cannot_change_the_outcome():
    p = make_paths(seed=3, weeks=20)
    spec = ExitSpec(target=0.04, trail=0.05, arm=0.02)
    r0 = run_exit(p, spec, Z)
    q = make_paths(seed=3, weeks=20)
    for i in range(len(q)):                      # scramble everything after each exit day
        d = r0.days[i]
        if d < D:
            q.o[i, d:], q.h[i, d:], q.l[i, d:], q.c[i, d:] = 1.0, 2.0, 0.5, 1.5
    r1 = run_exit(q, spec, Z)
    assert np.array_equal(r0.net, r1.net) and np.array_equal(r0.days, r1.days)


def test_learn_ignores_positions_unfinished_at_as_of():
    kinds = {"fader": ([0.03, 0.03, -0.03, -0.03, -0.03], 0.015, 1)}
    p = make_paths(1, 100, kinds=kinds)
    as_of = np.datetime64("2016-12-30")
    rules = E.default_rules(False)
    a = E.learn(p, rules, as_of, seed=1)
    late = p.take(np.flatnonzero(p.end > as_of))
    late.c[:] *= 0.5                                # poison the future
    late.o[:] *= 1.7
    poisoned = Paths(np.vstack([p.o[p.end <= as_of], late.o]), np.vstack([p.h[p.end <= as_of], late.h]),
                     np.vstack([p.l[p.end <= as_of], late.l]), np.vstack([p.c[p.end <= as_of], late.c]),
                     np.r_[p.prev_close[p.end <= as_of], late.prev_close], np.r_[p.vol[p.end <= as_of], late.vol],
                     np.r_[p.atr[p.end <= as_of], late.atr], np.r_[p.week[p.end <= as_of], late.week],
                     np.r_[p.end[p.end <= as_of], late.end], np.r_[p.kind[p.end <= as_of], late.kind],
                     np.r_[p.ticker[p.end <= as_of], late.ticker])
    b = E.learn(poisoned, rules, as_of, seed=1)
    assert {k: s.chosen for k, s in a.selections.items()} == {k: s.chosen for k, s in b.selections.items()}


# ---------------------------------------------------------------- tiered objective
def m(**kw):
    base = dict(weeks=50, mean_dev=0.01, in_band=0.5, cvar5=-0.05, maxdd=0.1, cat_rate=0.0, below_band=0.3, pos_weeks=0.6,
                hit10=0.1, precision=0.55)
    return {**base, **kw}


def test_lower_tier_gain_cannot_buy_a_higher_tier_loss():
    a = m(cvar5=-0.20, hit10=0.60, precision=0.9, pos_weeks=0.95)       # huge tier-3 gains, worse tier-2
    b = m()
    assert E.compare(a, b) == -1 and E.compare(b, a) == 1
    assert E.compare(m(mean_dev=0.001), m(mean_dev=0.03, cvar5=0.05)) == 1   # tier 1 outranks a tier-2 gain
    assert E.compare(m(hit10=0.105), m()) == 0                           # within tolerance is a tie


def test_raw_return_alone_never_wins():
    """Two rules, one with a far higher raw mean but a catastrophic tail: the tiered objective prefers the safer one."""
    n = 600
    codes = np.repeat(np.arange(60), 10)
    rng = np.random.default_rng(0)
    safe = rng.normal(0.07, 0.01, n)
    wild = safe + 0.03
    wild[::25] = -0.5
    ms, mw = (E.band_metrics(E.week_table(x, codes, 60)) for x in (safe, wild))
    assert wild.mean() > safe.mean() - 0.03 and mw["cat_rate"] > ms["cat_rate"]
    assert E.compare(ms, mw) == 1


def test_bootstrap_win_is_even_for_identical_rules_and_decisive_for_a_real_gap():
    rng = np.random.default_rng(1)
    codes = np.repeat(np.arange(80), 12)
    a = rng.normal(0.02, 0.03, len(codes))
    assert E.bootstrap_win(a, a.copy(), codes, 80, np.random.default_rng(0)) == 0.5
    better = a + 0.05
    assert E.bootstrap_win(better, a, codes, 80, np.random.default_rng(0)) > 0.9


# ---------------------------------------------------------------- planted effects through walk-forward
FADER = ([0.03, 0.03, -0.03, -0.03, -0.03], 0.015, 1)


def test_learner_finds_the_planted_fade_and_beats_week_end_out_of_sample():
    kinds = {"fader": FADER, "flat": ([0.0] * 5, 0.02, 1)}
    p = make_paths(5, 160, kinds=kinds)
    wf = E.walk_forward(p, E.default_rules(False), min_train_weeks=60, block_weeks=20, seed=2)
    rep = wf.report().set_index("kind")
    assert rep.loc["fader", "verdict"] == "better"
    assert rep.loc["fader", "learned_mean_week"] > rep.loc["fader", "base_mean_week"] + 0.02
    picked = [c["fader"] for _, c in wf.fold_choices if "fader" in c]
    assert sum(x != "week_end" for x in picked) >= len(picked) - 1
    assert rep.loc["fader", "learned_days"] < D                       # it really exits early


def test_martingale_world_no_rule_manufactures_return():
    """On a driftless walk an exit can reshape risk but cannot create return: the learned policy must not be worse
    out of sample and must not claim a mean-week gain beyond noise."""
    p = make_paths(11, 160, kinds={"flat": ([0.0] * 5, 0.02, 1)})
    wf = E.walk_forward(p, E.default_rules(False), min_train_weeks=60, block_weeks=20, seed=3)
    row = wf.report().set_index("kind").loc["flat"]
    assert row["verdict"] != "worse"
    assert row["learned_mean_week"] < row["base_mean_week"] + 0.005


def test_pattern_failure_rules_only_offered_with_flags_and_used_when_informative():
    assert not any(r.family == "pattern_failure" for r in E.default_rules(False))
    assert any(r.family == "pattern_failure" for r in E.default_rules(True))
    p = make_paths(2, 150, kinds={"k": ([0.0] * 5, 0.02, 1)})
    # planted: a score path that collapses exactly on the days before a large drop
    rng = np.random.default_rng(0)
    dropper = rng.random(len(p)) < 0.3
    for a in (p.o, p.h, p.l, p.c):
        a[dropper, 4:] *= 0.85
    scores = np.ones((len(p), D))
    scores[dropper, 2:] = 0.1
    p.fail = E.pattern_fail_flags(scores, np.ones(len(p)), 0.5)
    r = run_exit(p, ExitSpec(fail_exit=True), Z)
    base = run_exit(p, ExitSpec(), Z)
    assert r.net[dropper].mean() > base.net[dropper].mean() + 0.10   # exits before the drop, at the day-3 open


# ---------------------------------------------------------------- empty and degenerate
def test_empty_and_thin_inputs():
    empty = make_paths(weeks=1, per_week=1).take(np.array([], int))
    assert len(run_exit(empty, ExitSpec(target=0.1), Z).net) == 0
    assert E.trade_metrics(run_exit(empty, ExitSpec(), Z)) == {"n": 0}
    r, sel = E.select_rule(empty, E.default_rules(False))
    assert r.name == "week_end" and sel.reason == "neutral:insufficient_data"
    small = make_paths(0, 10, kinds={"fader": FADER})
    pol = E.learn(small, E.default_rules(False), "2030-01-01")
    assert pol.selections["fader"].reason == "neutral:insufficient_data"
    wf = E.walk_forward(small, E.default_rules(False), min_train_weeks=52)
    assert not wf.tested.any() and wf.report().empty


def test_deterministic_given_seed_and_metrics_shape():
    p = make_paths(4, 120, kinds={"fader": FADER})
    a = E.learn(p, E.default_rules(False), "2017-06-30", seed=9)
    b = E.learn(p, E.default_rules(False), "2017-06-30", seed=9)
    assert a.selections["fader"].chosen == b.selections["fader"].chosen
    tm = E.trade_metrics(run_exit(p, ExitSpec(target=0.05, trail=0.04)))
    for k in ("hit_rate", "avg_gain", "avg_loss", "tail5_mean", "days_held", "early_exit_rate", "cost_drag_bps"):
        assert k in tm
    assert tm["avg_loss"] <= 0 <= tm["avg_gain"] and abs(sum(tm["reasons"].values()) - 1) < 1e-9
    bm = E.band_metrics(E.week_table(run_exit(p, ExitSpec()).net, *[np.unique(p.week, return_inverse=True)[1], 120]))
    assert 0 <= bm["in_band"] <= 1 and bm["cvar5"] <= bm["mean_week"]


# ---------------------------------------------------------------- diagnostics
def test_lookahead_audit_passes_for_the_simulator_and_catches_a_cheating_one(monkeypatch):
    p = make_paths(8, 30)
    assert E.audit_no_lookahead(p, ExitSpec(target=0.04, trail=0.05, arm=0.02, stop=0.06)) == 0
    real = E.run_exit

    def cheat(pp, spec, cost=CostModel(), stop_dist=None):        # exits at the FINAL close if the last close is higher
        r = real(pp, spec, cost, stop_dist)
        better = pp.c[:, -1] > pp.c[:, 0] * 1.05
        r.net = np.where(better, pp.c[:, -1] / pp.o[:, 0] - 1, r.net)
        r.days = np.where(better, 1, r.days)
        return r
    monkeypatch.setattr(E, "run_exit", cheat)
    assert E.audit_no_lookahead(p, ExitSpec()) > 0


def test_evaluate_rules_and_cost_sensitivity():
    p = make_paths(6, 80, kinds={"fader": FADER})
    tab = E.evaluate_rules(p, E.default_rules(False))
    assert tab.iloc[0].rule == "week_end" and {"hit_rate", "in_band", "cvar5", "days_held"} <= set(tab.columns)
    assert tab.set_index("rule").loc["target_5", "days_held"] < D
    cs = E.cost_sensitivity(p, ExitSpec(target=0.05))
    assert cs["mean"].is_monotonic_decreasing and len(cs) == 5


def test_gain_interval_excludes_zero_for_planted_effect_and_reports_render():
    p = make_paths(5, 160, kinds={"fader": FADER, "flat": ([0.0] * 5, 0.02, 1)})
    wf = E.walk_forward(p, E.default_rules(False), min_train_weeks=60, block_weeks=20, seed=2)
    ci = E.oos_gain_interval(wf, "fader")
    assert ci["significant"] and ci["lo"] > 0
    txt = E.format_report(wf)
    assert "fader" in txt and "latest rules" in txt
    assert E.oos_gain_interval(wf, "missing") == {"n_weeks": 0}
    assert E.format_report(E.walk_forward(p, E.default_rules(False), min_train_weeks=10 ** 4)).startswith("Exit learner: no")


def test_era_breakdown_localises_an_effect_that_only_exists_in_one_era():
    """Plant the fade in the first half of history only; the era table must show the gain there and none after."""
    early = make_paths(5, 80, kinds={"fader": FADER}, start="2010-01-04")
    late = make_paths(6, 80, kinds={"fader": ([0.0] * 5, 0.015, 1)}, start="2016-01-04")
    p = Paths(*(np.concatenate([getattr(early, k), getattr(late, k)]) for k in
                ("o", "h", "l", "c", "prev_close", "vol", "atr", "week", "end", "kind", "ticker")))
    wf = E.walk_forward(p, E.default_rules(False), min_train_weeks=30, block_weeks=10, seed=1)
    era = E.era_breakdown(wf, years_per_era=1)
    assert not era.empty
    gain = (era.learned_mean_week - era.base_mean_week)[era.kind == "ALL"]
    yrs = era.era[era.kind == "ALL"]
    assert gain[yrs <= 2012].max() > 0.02          # planted era: exit helps
    assert E.era_breakdown(E.walk_forward(p, E.default_rules(False), min_train_weeks=10 ** 4)).empty


def test_rule_usage_reports_fold_choices():
    p = make_paths(5, 160, kinds={"fader": FADER})
    wf = E.walk_forward(p, E.default_rules(False), min_train_weeks=60, block_weeks=20, seed=2)
    u = E.rule_usage(wf)
    assert u.groupby("kind")["share"].sum().round(6).eq(1.0).all() and u.folds.sum() == len(wf.fold_choices)


# ---------------------------------------------------------------- selection guards
def test_selection_stability_is_high_for_a_real_winner_and_zero_for_a_clone():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(60), 12)
    base = rng.normal(0.0, 0.05, len(codes))
    good = np.where(base < -0.05, -0.05, base)                     # a genuinely thinner left tail
    tabs = [E.week_table(x, codes, 60) for x in (base, good, base.copy())]
    assert E.selection_stability(tabs, [True] * 3, 1, 60, np.random.default_rng(1), 60) > 0.6
    assert E.selection_stability(tabs, [True] * 3, 2, 60, np.random.default_rng(1), 60) == 0.0


def test_infeasible_candidates_can_never_be_chosen():
    p = make_paths(5, 120, kinds={"fader": FADER})
    rules = E.default_rules(False)
    r, sel = E.select_rule(p, rules, feasible=lambda band, tm: False)
    assert r.name == "week_end" and not sel.table.set_index("rule").drop("week_end").feasible.any()
    r2, sel2 = E.select_rule(p, rules, seed=1)
    assert sel2.reason == "selected" and sel2.stability >= 0.25 and r2.name != "week_end"


def test_thin_type_borrows_the_pooled_rule_only_if_the_pooled_rule_passed_every_gate():
    kinds = {"big": FADER, "rare": FADER}
    p = make_paths(3, 110, kinds={"big": (*FADER[:2], 0.96), "rare": (*FADER[:2], 0.04)})
    pol = E.learn(p, E.default_rules(False), "2017-06-30", seed=2, min_trades=150)
    assert pol.selections["rare"].reason == "pooled_fallback" and pol.selections["big"].reason == "selected"
    assert pol.by_kind["rare"].name == pol.by_kind["big"].name or pol.selections["rare"].chosen != "week_end"
    off = E.learn(p, E.default_rules(False), "2017-06-30", seed=2, min_trades=150, pool_fallback=False)
    assert off.selections["rare"].reason == "neutral:insufficient_data" and off.by_kind["rare"].name == "week_end"
    flat = make_paths(3, 110, kinds={"big": ([0.0] * 5, 0.02, 0.96), "rare": ([0.0] * 5, 0.02, 0.04)})
    pf = E.learn(flat, E.default_rules(False), "2017-06-30", seed=2, min_trades=150)
    if pf.selections["big"].reason != "selected":                  # nothing real to pool -> the thin type stays neutral
        assert pf.selections["rare"].reason == "neutral:insufficient_data"


def test_policy_applies_per_type_and_unknown_types_get_the_baseline():
    p = make_paths(7, 110, kinds={"fader": FADER, "flat": ([0.0] * 5, 0.02, 1)})
    pol = E.learn(p, E.default_rules(False), "2017-06-30", seed=2)
    mixed = pol.apply(p)
    for k in ("fader", "flat"):
        m = p.kind == k
        assert np.allclose(mixed.net[m], pol.by_kind[k].run(p.take(np.flatnonzero(m))).net)
    q = p.take(np.flatnonzero(p.kind == "fader"))
    q.kind[:] = "never_seen"
    assert np.allclose(pol.apply(q).net, run_exit(q, ExitSpec(), CostModel()).net)


def test_training_window_limits_to_recent_weeks_and_log_writes_provenance(tmp_path, monkeypatch):
    early = make_paths(5, 60, kinds={"k": FADER}, start="2010-01-04")
    late = make_paths(6, 60, kinds={"k": ([0.0] * 5, 0.015, 1)}, start="2013-01-07")
    p = Paths(*(np.concatenate([getattr(early, k), getattr(late, k)]) for k in
                ("o", "h", "l", "c", "prev_close", "vol", "atr", "week", "end", "kind", "ticker")))
    rules = E.default_rules(False)
    full = E.learn(p, rules, "2014-01-01", seed=1)
    recent = E.learn(p, rules, "2014-01-01", seed=1, train_window_weeks=50)
    assert recent.selections["k"].n_weeks <= 50 < full.selections["k"].n_weeks
    from engine import improve
    reg = tmp_path / "reg.jsonl"
    monkeypatch.setattr(improve, "REG", reg)
    wf = E.walk_forward(early, rules, min_train_weeks=30, block_weeks=10, seed=1)
    E.log_walk_forward(wf, {"x": 1}, 5)
    rec = __import__("json").loads(reg.read_text().splitlines()[-1])
    assert rec["event"] == "exit_walk_forward" and rec["seed"] == 5 and "code_hash" in rec and rec["report"]
