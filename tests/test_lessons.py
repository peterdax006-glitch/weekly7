"""Phase 10 lesson memory: planted trap is learned, one-offs and identity proxies are not, trust/expiry work."""
import numpy as np
import pandas as pd
import pytest

from engine.lessons import (Episode, IdentityLeak, LessonBook, audit_identity, identity_proxy_columns, post_mortem)


def make_panel(seed=0, n_days=160, n_tk=40, trap=True, trap_days=None):
    """f0 is the model's signal. In the trap (m_vix high and f2 > 0.3) the signal reverses. px is a static per-ticker
    price level; day_idx trends with the calendar (both identity proxies)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tks = [f"T{i:02d}" for i in range(n_tk)]
    idx = pd.MultiIndex.from_product([dates, tks], names=["date", "ticker"])
    n = len(idx)
    vix = np.repeat(rng.normal(size=n_days), n_tk)
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n), "f2": rng.normal(size=n),
                      "f3": rng.normal(size=n), "m_vix": vix,
                      "m_breadth": np.repeat(rng.normal(size=n_days), n_tk),
                      "px": np.tile(rng.uniform(10, 500, n_tk), n_days),
                      "day_idx": np.repeat(np.arange(n_days, dtype=float), n_tk)}, index=idx)
    y = 0.02 * X["f0"] + rng.normal(scale=0.03, size=n)
    if trap:
        region = (X["m_vix"] > 0.3) & (X["f2"] > 0.3)
        if trap_days is not None:
            region &= X["day_idx"].isin(trap_days)
        y = y.where(~region, -0.04 * X["f0"] + rng.normal(scale=0.03, size=n))
    return X, y


def score_fn(Xd):
    return Xd["f0"].copy()


def base_frame(X, y, k=5, horizon=7):
    from engine.antimemo import play_window
    _, fr = play_window(X, y, score_fn, None, k=k, collect_frame=True, horizon_days=horizon)
    return fr


def learn_book(seed=0, **kw):
    X, y = make_panel(seed, **kw)
    fr = base_frame(X, y)
    eps = post_mortem(fr, X, fr["resolved"].max(), seed=seed)
    b = LessonBook(seed=seed)
    b.record(eps)
    b.learn()
    return b, X, y, eps


def test_episode_refuses_identity_fields():
    ok = dict(eid="e", situation="s", features={"f0": 1.0}, context={}, decision={"side": 1.0},
              outcome={"pnl": 0.0}, category="ok", confidence=0.5, counterfactual={})
    Episode(**ok)
    with pytest.raises(IdentityLeak):
        Episode(**{**ok, "features": {"ticker": 1.0}})
    with pytest.raises(IdentityLeak):
        Episode(**{**ok, "context": {"m_x": "2021-01-04"}})
    with pytest.raises(ValueError):
        Episode(**{**ok, "category": "vibes"})


def test_identity_proxy_columns_flag_price_level_and_calendar_trend_only():
    X, _ = make_panel(1)
    bad = identity_proxy_columns(X)
    assert bad.get("px") == "ticker-constant"
    assert bad.get("day_idx") == "calendar-trend"
    assert not {"f0", "f1", "f2", "f3", "m_vix", "m_breadth"} & set(bad)


def test_post_mortem_fields_categories_and_no_lookahead():
    X, y = make_panel(2)
    fr = base_frame(X, y)
    cut = fr["resolved"].sort_values().iloc[len(fr) // 2]
    eps = post_mortem(fr, X, cut, seed=2)
    assert eps and all(e.origin["week"] != "" for e in eps)
    n_expected = int(((fr["resolved"] <= cut) & fr["taken"]).sum())
    assert sum(e.decision["taken"] == 1.0 for e in eps) == n_expected      # nothing resolved after `now` is used
    cats = {e.category for e in eps}
    assert {"ok", "false_positive"} <= cats and cats & {"false_negative", "missed_winner"}
    e = next(e for e in eps if e.category == "false_positive")
    assert e.counterfactual["best"] in ("no_trade", "flip") and e.counterfactual["regret"] > 0
    for e in eps:
        assert not any(isinstance(v, str) for d in (e.features, e.context, e.decision, e.outcome) for v in d.values())
        assert "px" not in e.features and "day_idx" not in e.features


def test_planted_trap_is_learned_as_abstract_rule():
    b, X, y, _ = learn_book(3)
    assert b.lessons, b.rejected_mining
    used = {f for l in b.lessons.values() for f, _, _ in l.conds}
    assert used & {"m_vix", "f2"} and not used & {"px", "day_idx"}
    L = min((l for l in b.lessons.values() if l.direction == -1), key=lambda l: l.p)   # the trap itself, as a downweight
    assert L.direction == -1 and L.n_weeks >= 6 and L.n_tickers >= 8
    trap = ((X["m_vix"] > 0.3) & (X["f2"] > 0.3)).to_numpy()
    f = b.factor(X).to_numpy()
    assert f[~trap].mean() > f[trap].mean() + 0.08
    assert audit_identity(b, tickers=X.index.get_level_values(1).unique(), dates=X.index.get_level_values(0)[:50]) == []


def test_one_week_event_is_not_generalised():
    b, X, _, _ = learn_book(4, trap_days=list(range(50, 53)))          # the trap exists on 3 days only
    used = [l for l in b.lessons.values() if any(f in ("m_vix", "f2") for f, _, _ in l.conds)]
    assert not [l for l in used if l.n_weeks < 6]
    assert b.rejected_mining.get("not_distinct", 0) + b.rejected_mining.get("fdr", 0) > 0


def test_identity_only_effect_cannot_become_a_lesson():
    """Losses concentrated in a few tickers that only px identifies: px is excluded so no lesson can name them."""
    X, y = make_panel(5, trap=False)
    names = X.index.get_level_values(1)
    victims = X.loc[X.index.get_level_values(0)[0]].sort_values("px").index[:4]
    y = y.where(~names.isin(victims), -0.04 * X["f0"])
    fr = base_frame(X, y)
    b = LessonBook(seed=5)
    b.record(post_mortem(fr, X, fr["resolved"].max(), seed=5))
    b.learn()
    assert not any(f in ("px", "day_idx") for l in b.lessons.values() for f, _, _ in l.conds)


def test_trust_falls_with_bad_feedback_and_lesson_retires():
    b, X, y, _ = learn_book(6)
    L = min(b.lessons.values(), key=lambda l: l.p)
    t0 = L.trust
    days = X.index.get_level_values(0).unique()[:60]
    for d in days:                                    # feedback where the lesson's rows do the OPPOSITE of the lesson
        Xd = X[X.index.get_level_values(0) == d]
        m = L.mask(Xd)
        pnl = pd.Series(np.where(m, -L.direction * 0.05, 0.0), index=Xd.index)
        b.feedback(Xd, pnl)
        if L.status != "active":
            break
    assert L.trust < t0 and L.status == "retired"
    assert b.factor(X).eq(1.0).all() or L.lid not in [l.lid for l in b.active()]


def test_expiry_by_own_clock_and_roundtrip():
    b, X, _, _ = learn_book(7)
    n = len(b.active())
    assert n > 0
    b2 = LessonBook.from_json(b.to_json())
    assert [l.lid for l in b2.active()] == [l.lid for l in b.active()]
    assert np.allclose(b2.factor(X), b.factor(X))
    b.tick(b.p["ttl"] + 1)
    assert not b.active() and b.factor(X).eq(1.0).all()


def test_audit_catches_a_planted_ticker():
    b, X, _, _ = learn_book(8)
    L = next(iter(b.lessons.values()))
    L.category = "T07"
    assert audit_identity(b, tickers=["T07"]) == ["T07"]


def test_empty_and_degenerate_inputs():
    X, y = make_panel(9, n_days=5, n_tk=3)
    assert post_mortem(pd.DataFrame(), X, pd.Timestamp("2030-01-01")) == []
    fr = base_frame(X, y)
    assert post_mortem(fr, X, pd.Timestamp("2000-01-01")) == []        # nothing resolved yet
    b = LessonBook()
    assert b.learn() == [] and b.rejected_mining["too_few_episodes"] == 0
    assert b.factor(X).eq(1.0).all() and b.items() == []
    b.record(post_mortem(fr, X, fr["resolved"].max()))
    assert b.learn() == []                                              # 15 rows cannot support a lesson


def _frame_for(X, y, **cols):
    fr = base_frame(X, y)
    for k, v in cols.items():
        fr[k] = v(fr) if callable(v) else v
    return fr


def test_regime_failure_is_named_when_the_loss_happens_in_an_unprecedented_market():
    X, y = make_panel(21, n_days=260)
    d = X.index.get_level_values(0)
    late = d >= d.unique()[210]
    X.loc[late, "m_vix"] += 6.0                                     # a market the earlier reference never saw
    y = y.where(~late, -0.05 * X["f0"])
    fr = base_frame(X, y)
    fr = fr[fr.index.get_level_values(0) >= d.unique()[150]]         # decisions from day 150; days 0-149 are the reference
    eps = post_mortem(fr, X, fr["resolved"].max(), seed=1)
    cats = pd.Series([e.category for e in eps if e.decision["taken"] == 1.0 and e.pnl < -0.01]).value_counts()
    assert cats.get("regime_failure", 0) > cats.get("false_positive", 0)


def test_pattern_failure_names_the_pattern_that_loses_on_average():
    X, y = make_panel(22, trap=False)
    fr = base_frame(X, y)
    bad = np.arange(len(fr)) % 2 == 1
    fr["pattern"] = np.where(bad, "P_bad", "P_good")
    fr.loc[bad, "y"] = -fr.loc[bad, "side"] * fr.loc[bad, "y"].abs()          # P_bad always loses, P_good is untouched
    fr.loc[~bad, "y"] = fr.loc[~bad, "side"] * fr.loc[~bad, "y"].abs() * 0.5   # and P_good always wins
    eps = post_mortem(fr, X, fr["resolved"].max(), seed=2)
    lost = [e for e in eps if e.decision["taken"] == 1.0 and e.pnl < -0.01]
    assert lost and all(e.category in ("pattern_failure", "regime_failure") for e in lost)   # only P_bad can lose
    assert sum(e.category == "pattern_failure" for e in lost) > 0.8 * len(lost)
    assert not any(e.category == "pattern_failure" for e in eps if e.pnl >= -0.01)
    fr["pattern"] = None                                                       # no pattern info: falls back to generic
    eps2 = post_mortem(fr, X, fr["resolved"].max(), seed=2)
    assert not any(e.category == "pattern_failure" for e in eps2)


def test_counterfactual_tight_stop_and_flip_are_computed_from_the_outcome():
    from engine.lessons import _counterfactual
    cf = _counterfactual(True, 1.0, -0.06, -0.0605, 0.0005, mae=-0.08, stop=0.02)
    assert cf["tight_stop"] == -0.0205 and cf["flip"] == 0.06 - 0.0005 and cf["best"] == "flip"
    assert abs(cf["regret"] - (0.0595 + 0.0605)) < 1e-12
    cf2 = _counterfactual(False, 1.0, 0.05, 0.0495, 0.0005)
    assert cf2["best"] == "take" and abs(cf2["regret"] - 0.0495) < 1e-12
    cf3 = _counterfactual(True, 1.0, 0.04, 0.0395, 0.0005, mae=-0.01, stop=0.02)       # stop never hit: unchanged
    assert cf3["tight_stop"] == 0.0395 and cf3["regret"] == 0.0


def test_multiple_testing_control_keeps_noise_from_becoming_lessons():
    books_with_lessons = 0
    for seed in range(8):
        X, _ = make_panel(30 + seed, n_days=110, n_tk=30, trap=False)
        y = pd.Series(np.random.default_rng(seed).normal(scale=0.03, size=len(X)), index=X.index)
        fr = base_frame(X, y)
        b = LessonBook(seed=seed)
        b.record(post_mortem(fr, X, fr["resolved"].max(), seed=seed))
        books_with_lessons += bool(b.learn())
    assert books_with_lessons <= 1                                   # FDR 5% over ~100 conditions: rarely any


def test_conjunction_effect_is_mined_and_pair_must_beat_its_parents():
    X, _ = make_panel(41, trap=False)
    rng = np.random.default_rng(41)
    both = (X["f1"] < -0.3) & (X["f3"] > 0.3)
    y = (0.02 * X["f0"] + rng.normal(scale=0.03, size=len(X))).where(~both, -0.05 * X["f0"] + rng.normal(scale=0.03, size=len(X)))
    fr = base_frame(X, y)
    b = LessonBook(seed=41)
    b.record(post_mortem(fr, X, fr["resolved"].max(), seed=41))
    b.learn()
    assert b.lessons
    top = min(b.lessons.values(), key=lambda l: l.p)
    assert top.direction == -1 and {f for f, _, _ in top.conds} <= {"f1", "f3"}
    assert all(l.n <= 0.5 * len(b.episodes) for l in b.lessons.values())      # no complement-of-the-region "lessons"


def test_untaken_sample_is_uniform_so_the_baseline_is_not_selected_on_outcome():
    X, y = make_panel(42, trap=False)
    fr = base_frame(X, y)
    eps = post_mortem(fr, X, fr["resolved"].max(), params={"untaken_frac": 0.3}, seed=3)
    untaken = [e for e in eps if e.decision["taken"] == 0.0]
    n_un = int((~fr["taken"]).sum())
    assert abs(len(untaken) / n_un - 0.3) < 0.02
    mean_all = float((fr["side"] * fr["y"])[~fr["taken"]].mean())
    assert abs(np.mean([e.outcome["y"] * e.decision["side"] for e in untaken]) - mean_all) < 0.003


def test_deterministic_given_seed():
    b1, *_ = learn_book(50)
    b2, *_ = learn_book(50)
    assert b1.to_json() == b2.to_json()


def _episodes(seed, **kw):
    X, y = make_panel(seed, **kw)
    fr = base_frame(X, y)
    return post_mortem(fr, X, fr["resolved"].max(), seed=seed), X


def test_stability_gate_rejects_an_effect_confined_to_one_stretch_of_time():
    X, y = make_panel(70, trap=False)
    d = X.index.get_level_values(0)
    early = d < d.unique()[55]
    region = early & (X["m_vix"] > 0).to_numpy() & (X["f2"] > 0).to_numpy()
    late_region = (~early) & (X["m_vix"] > 0).to_numpy() & (X["f2"] > 0).to_numpy()
    y = y.where(~region, -0.10 * X["f0"]).where(~late_region, 0.05 * X["f0"])   # strong early, then it reverses
    fr = base_frame(X, y)
    eps = post_mortem(fr, X, fr["resolved"].max(), seed=70)
    strict, lax = LessonBook(seed=1), LessonBook({"stab_min": 0.0}, seed=1)
    for b in (strict, lax):
        b.record(eps)
        b.learn()
    assert strict.rejected_mining.get("unstable", 0) > 0
    assert min((l.stability for l in lax.lessons.values()), default=1.0) < 0.75     # the lax book kept a one-stretch rule
    assert all(l.stability >= 0.75 for l in strict.lessons.values())


def test_revalidation_confirms_a_lesson_that_still_holds_and_retires_one_that_reversed():
    eps_a, X = _episodes(71)
    b = LessonBook(seed=1)
    b.record(eps_a)
    b.learn()
    assert b.lessons
    down = min((l for l in b.lessons.values() if l.direction == -1), key=lambda l: l.p)
    a0 = down.a
    eps_same, _ = _episodes(171)                                    # same process, new window
    rep = {r["lid"]: r for r in b.revalidate(eps_same)}
    assert rep[down.lid]["action"] == "confirmed" and down.a > a0 and down.status == "active"
    X2, y2 = make_panel(172, trap=False)                            # the trap region now REWARDS f0
    reg = (X2["m_vix"] > 0.3) & (X2["f2"] > 0.3)
    y2 = y2.where(~reg, 0.06 * X2["f0"])
    fr = base_frame(X2, y2)
    eps_rev = post_mortem(fr, X2, fr["resolved"].max(), seed=3)
    rep2 = {r["lid"]: r for r in b.revalidate(eps_rev)}
    assert rep2[down.lid]["action"] == "retired" and down.status == "retired"
    assert b.revalidate([]) == []


def test_refresh_retests_before_recording_and_ticks_the_clock():
    eps_a, X = _episodes(73)
    b = LessonBook(seed=1)
    b.record(eps_a)
    b.learn()
    n_before, tick_before = len(b.episodes), b.tick_n
    eps_b, _ = _episodes(173)
    rev, new = b.refresh(eps_b)
    assert rev and len(b.episodes) > n_before and b.tick_n == tick_before + 1
    assert {r["lid"] for r in rev}.isdisjoint({l.lid for l in new})      # a lesson is never tested on its own batch


def test_explain_lists_the_firing_lessons_on_exactly_the_rows_they_touch():
    b, X, _, _ = learn_book(74)
    s = X["f0"]
    ex = b.explain(X, s)
    assert np.allclose(ex["adjusted"], s * ex["factor"])
    fired = ex["lessons"] != ""
    assert fired.any() and (ex.loc[fired, "factor"] != 1.0).all() and (ex.loc[~fired, "factor"] == 1.0).all()


def test_diagnostics_flags_opposite_lessons_on_overlapping_rows():
    from engine.lessons import Lesson
    X, _ = make_panel(75, n_days=20, n_tk=10)
    mk = lambda lid, conds, direction: Lesson(lid, conds, direction, 0.5 if direction < 0 else 1.5, "ok", 50, 8, 9,
                                              0.01 * direction, 3.0, 0.01, 0.01, 9.0, 1.0, 0, 100)
    b = LessonBook()
    b.lessons = {"dn": mk("dn", [("f0", ">", 0.0)], -1), "up": mk("up", [("f0", ">", 1.0)], +1),
                 "far": mk("far", [("f1", "<=", -50.0)], +1)}
    dg = b.diagnostics(X)
    assert dg["conflicts"] and dg["conflicts"][0]["rows"] == int((X["f0"] > 1.0).sum())
    assert dg["coverage"]["far"] == 0.0 and 0 < dg["rows_touched"] < 1
    assert LessonBook().diagnostics(X)["conflicts"] == []
