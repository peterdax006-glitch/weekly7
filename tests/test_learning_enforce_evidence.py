"""F10 (C69 W-07 continued; C62 C03, E04, I16, I18): the learning-claim gate stays 'enforce' and keeps its bar; the learner now SUPPLIES
the evidence it was deadlocked on.  Tests: the portfolio-level card (risk, drawdown, band share, calibration of the learner's book
against a no-knowledge book) and its merge; the evidence card on the learner's shadow book (a planted book passes the unchanged gate,
a null book and a one-year book do not); no learner is judged on another learner's card; the multi-year planted world; the shared
weekly retirement check with its two re-opened doors; the false-degrade study that justified retire_window 8 -> 16; the degrade
audit; and the honest identity-blocker wording.  Each mechanism has a planted case it must catch, a null case and the empty case.
Synthetic data only.  Status: IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from engine.learning import identity_firewall as IDF
from engine.learning import learner as LN
from engine.learning import planted_world as PW
from engine.learning import retirement as RT
from engine.learning import scorecard as SC
from engine.learning import transfer as TR
from engine.learning import wiring as W
from engine.learning.core import FirewallBreach
from engine.learning.firewalls import LayerName, fail

import test_learning_learner as T


# ------------------------------------------------------------------------------------------------ synthetic books

def book_rows(years=5, n=40, beta=0.02, informative=True, seed=0, start="2009-01-02"):
    """One decision unit per (week, name): raw return = market + beta on the planted top cell + noise; the learner's ex-ante
    expectation marks the planted cell (informative) or a random 20% of names (null)."""
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.date_range(start, periods=52 * years, freq="W-FRI"):
        f = rng.standard_normal(n)
        mkt = rng.normal(0, 0.02)
        raw = mkt + beta * (f > 0.84) + rng.normal(0, 0.05, n)
        mark = (f > 0.84) if informative else (rng.random(n) < 0.2)
        for i in range(n):
            out.append(LN._BookRow(str(d.date()), str((d + pd.Timedelta(days=7)).date()), f"T{i:02d}", "hi" if mkt > 0 else "lo",
                                   0.01 if mark[i] else None, 0.02, float(raw[i] - raw.mean()), float(raw[i])))
    return out


def frame(rows):
    return pd.DataFrame({"date": [r.decided for r in rows], "mature": [r.matured for r in rows], "ticker": [r.ident for r in rows],
                         "ret": [r.raw_ret for r in rows], "expected": [np.nan if r.expected is None else r.expected for r in rows],
                         "scale": [r.scale for r in rows]})


def learner_with(rows):
    L = LN.LegitimateLearner(T.make_cfg(seed=1), code_hash_fn=lambda: "pinned-test-code")
    L._book = list(rows)
    last = max(pd.Timestamp(r.matured) for r in rows) if rows else pd.Timestamp("2009-01-02")
    return L, last + pd.Timedelta(days=7)


@pytest.fixture(scope="module")
def planted_book():
    return book_rows(informative=True, seed=0)


@pytest.fixture(scope="module")
def null_book():
    return book_rows(informative=False, seed=0)


# ------------------------------------------------------------------------------------------------ weekly books

def test_weekly_books_pick_by_expectation_and_the_baseline_never_reads_it(planted_book):
    f = frame(planted_book[: 40 * 20])
    now = pd.Timestamp(f["mature"].max()) + pd.Timedelta(days=1)
    b = SC.weekly_books(f, now, top_n=5, seed=3)
    assert len(b) == 20 and (b["n_units"] == 40).all()
    wk = f[f["date"] == f["date"].min()]
    top = wk[wk["expected"].notna()].sort_values("expected", ascending=False)
    assert b["n_learner"].iloc[0] == min(5, len(top))
    blind = SC.weekly_books(f.assign(expected=np.nan), now, top_n=5, seed=3)
    assert np.allclose(blind["baseline"], b["baseline"])                        # the baseline book is the same without knowledge
    assert (blind["learner"] == 0.0).all() and (blind["n_learner"] == 0).all()     # no expectation: cash, not a default pick
    assert len(SC.weekly_books(f.iloc[:0], now, top_n=5)) == 0                      # empty case


def test_weekly_books_refuse_an_outcome_that_has_not_matured(planted_book):
    f = frame(planted_book[:400])
    with pytest.raises(FirewallBreach):
        SC.weekly_books(f, pd.Timestamp(f["mature"].max()), top_n=5)            # matures ON now: not yet known
    with pytest.raises(ValueError):
        SC.weekly_books(f.drop(columns=["expected"]), pd.Timestamp("2030-01-01"), top_n=5)


# ------------------------------------------------------------------------------------------------ portfolio card

def test_portfolio_card_measures_a_planted_book_and_the_learner_book_beats_a_blind_one(planted_book):
    f = frame(planted_book)
    now = pd.Timestamp(f["mature"].max()) + pd.Timedelta(days=1)
    pc = SC.portfolio_card(f, now, top_n=5, scale="scale", seed=1, n_boot=200)
    assert pc.untested() == [] and pc.n_windows == 20
    assert pc.drawdown_change.significantly_positive and pc.risk_change.significantly_positive   # the planted edge is visible
    assert pc.learner_mean_week > pc.baseline_mean_week
    assert pc.calibration.measured and pc.calibration.n > 1000


def test_portfolio_card_leaves_short_empty_and_pickless_books_untested(planted_book):
    f = frame(planted_book)
    short = f[pd.to_datetime(f["date"]) < pd.Timestamp("2009-10-01")]              # 39 weeks = 3 windows of 13 < 4
    now = pd.Timestamp("2011-01-01")
    pc = SC.portfolio_card(short, now, top_n=5, scale="scale")
    assert {"risk_change", "drawdown_change", "band_share"} <= set(pc.untested()) and "windows" in pc.band_share.note
    blind = SC.portfolio_card(f.assign(expected=np.nan), pd.Timestamp("2015-01-01"), top_n=5, scale="scale")
    assert set(blind.untested()) == set(SC.PORTFOLIO_FIELDS) and "never held a pick" in blind.risk_change.note
    empty = SC.portfolio_card(f.iloc[:0], now, top_n=5)
    assert set(empty.untested()) == set(SC.PORTFOLIO_FIELDS) and empty.n_weeks == 0


def test_calibration_sees_an_overconfident_scale(planted_book):
    f = frame(planted_book)
    now = pd.Timestamp(f["mature"].max()) + pd.Timedelta(days=1)
    ok = SC.portfolio_card(f, now, top_n=5, scale=0.05, n_boot=100).calibration
    bad = SC.portfolio_card(f, now, top_n=5, scale=0.0005, n_boot=100).calibration   # planted defect: p ~ 0.99 on every mark
    assert ok.measured and bad.measured and bad.value > ok.value + 0.1


def test_merge_never_overwrites_a_measurement_and_untested_stays_untested(planted_book):
    import test_learning_scorecard as TS
    card = TS.good_card(band_share=SC.Measured.untested("x"), risk_change=SC.Measured.untested("x"))
    f = frame(planted_book)
    pc = SC.portfolio_card(f, card.now, top_n=5, scale="scale", n_boot=100) if pd.Timestamp(card.now) > pd.Timestamp(f["mature"].max()) else None
    pc = dataclasses.replace(pc, drawdown_change=SC.Measured(-9.0, -9.1, -8.9, 5, SC.MStatus.MEASURED), calibration=SC.Measured.untested("none"))
    m = SC.merge_portfolio(card, pc)
    assert m.band_share == pc.band_share and m.risk_change == pc.risk_change            # filled from the portfolio card
    assert m.drawdown_change == card.drawdown_change                                   # already measured: never overwritten
    assert m.calibration == card.calibration
    with pytest.raises(ValueError):
        SC.merge_portfolio(card, dataclasses.replace(pc, now=dt.date(2020, 1, 1)))
    unt = SC.merge_portfolio(dataclasses.replace(card, calibration=SC.Measured.untested("x")), pc)
    assert not unt.calibration.measured                                                # untested + untested = untested, never 0


# ------------------------------------------------------------------------------------------------ the evidence card

def test_a_planted_book_supplies_every_field_and_passes_the_unchanged_gate(planted_book):
    L, now = learner_with(planted_book)
    card = L.evidence_card(now)
    assert card.check() == []
    assert set(SC.PORTFOLIO_FIELDS).isdisjoint(card.untested_fields())
    d = SC.gate_improvement_claim(card)
    assert d.allowed, d.blockers
    assert L.cfg.learning_claim == "enforce" == LN.LearnerConfig().learning_claim


def test_a_null_book_supplies_the_evidence_and_is_refused(null_book):
    L, now = learner_with(null_book)
    card = L.evidence_card(now)
    assert card.check() == []                                                          # the evidence exists ...
    d = SC.gate_improvement_claim(card)
    assert not d.allowed and "learning_gain" in [c.name for c in d.checks if c.blocking and not c.ok]   # ... and says no


def test_a_one_year_book_cannot_supply_it_and_nothing_is_registered(planted_book):
    L, now = learner_with([r for r in planted_book if r.decided < "2010-01-01"])
    card = L.evidence_card(now)
    assert card.check()                                                                # the F07 deadlock: no forward-year fold
    W.HUB.scorecards["K-f10-probe"] = SC.LearningScorecard("someone-else", dt.date(2009, 1, 1), "c", 0)
    L._tick = 7
    L._register_evidence("K-f10-probe", now)
    assert "K-f10-probe" not in W.HUB.scorecards                                       # never judged on another learner's card
    assert L.evidence_log[-1]["valid"] is False and L.counters["evidence_cards_refused"] == 1
    E, _ = learner_with([])
    assert E.evidence_card(pd.Timestamp("2009-06-01")) is None                         # empty case


def test_a_valid_card_is_registered_once_per_tick(planted_book):
    L, now = learner_with(planted_book)
    L._tick = 3
    L._register_evidence("K-f10-a", now)
    L._register_evidence("K-f10-b", now)
    assert L.counters["evidence_cards"] == 1 and W.HUB.scorecards["K-f10-a"] is W.HUB.scorecards["K-f10-b"]
    assert L.evidence_log[-1]["allowed"] is True and L.evidence_log[-1]["blockers"] == []


def test_the_shadow_book_is_written_before_outcomes_and_never_by_a_frozen_learner():
    w = PW.make_world(T.mini_spec(14, 30), seed=2)
    L = T.run(T.new_learner(), LN.WorldFeed(w), range(len(w.dates)))
    assert L._book and all(b.decided < b.matured for b in L._book)
    n = len(L._book)
    L.freeze()
    LN.score_decisions(L, LN.WorldFeed(PW.reidentify(w, 5, tickers=True, shift_years=3).world), range(2), "probe")
    assert len(L._book) == n
    assert L.book_frame(pd.Timestamp(w.dates[0])).empty                             # nothing matured before the first week


# ------------------------------------------------------------------------------------------------ multi-year world

def test_multi_year_world_has_forward_year_folds_and_the_null_world_the_same_length():
    spec, null = PW.multi_year_spec(5), PW.multi_year_spec(5, noise=True)
    assert spec.weeks == null.weeks == 260 and [i.item_id for i in null.items] == ["noise_a", "noise_b"]
    dec = spec.item("decaying")
    assert dec.change_week(spec.weeks) == 130 and dec.multipliers(spec.weeks)[-1] == 0.0
    d = pd.date_range(spec.start, periods=spec.weeks, freq="W-FRI")
    u = pd.DataFrame({"date": d, "mature": d + pd.Timedelta(days=7), "ticker": "T", "base": 0.0, "learned": 0.0})
    folds = TR.make_folds(TR.prepare_units(u, d[-1] + pd.Timedelta(days=30)), TR.Axis.YEAR)
    assert len(folds) >= 3                                                             # stability needs three held-out years
    four = TR.make_folds(TR.prepare_units(u[u["date"] < "2013-01-01"], d[-1] + pd.Timedelta(days=30)), TR.Axis.YEAR)
    assert len(four) == 2                                                              # four years: stability stays UNTESTED
    with pytest.raises(ValueError):
        PW.multi_year_spec(3)
    with pytest.raises(ValueError):
        PW.multi_year_spec(5, decay_years=(3, 3))


def test_live_for_degrade_and_the_signed_cell_series():
    spec = PW.multi_year_spec(5)
    assert PW.live_for_degrade(spec, "strong", 250) and PW.live_for_degrade(spec, "regime", 5)
    assert PW.live_for_degrade(spec, "decaying", 129) and not PW.live_for_degrade(spec, "decaying", 130)
    assert not PW.live_for_degrade(spec, "noise_a", 10)
    w = PW.make_world(PW.multi_year_spec(4, 30), seed=1)
    neg = w.cell_weekly("negative", (0, 104))
    assert len(neg) == 104 and neg.mean() > 0                                          # signed by the planted (negative) direction
    assert len(w.cell_weekly("noise_a", (0, 0))) == 0


# ------------------------------------------------------------------------------------------------ the weekly retirement check

def _weeks(start, n):
    return [d.date().isoformat() for d in pd.date_range(start, periods=n, freq="7D")]


def _parked(led, kid="K1"):
    led.register(kid, "2009-01-02")
    led.transition(kid, RT.State.DEGRADED, "2009-03-06", "DEGRADE", "planted dip")
    led.transition(kid, RT.State.DORMANT, "2009-05-08", "DORMANT", "still weak")
    return kid


def test_a_parked_item_can_come_back_through_probation_and_the_old_call_could_not():
    led = RT.RetirementLedger(RT.RetirementPolicy(min_n=8, recover_min_n=16))
    kid = _parked(led)
    rng = np.random.default_rng(0)
    d = _weeks("2009-01-09", 60)
    vals = [(-0.01 if x <= "2009-05-08" else 0.02) + rng.normal(0, 0.005) for x in d]
    now = "2009-10-02"
    upto = [i for i, x in enumerate(d) if x < now]
    ds, vs = [d[i] for i in upto], [vals[i] for i in upto]
    window_ev = RT.series_evidence(ds[-8:], vs[-8:], "2009-05-08", now)
    assert "insufficient" in led.attempt_recovery(kid, window_ev, now).reason      # the old learner call: 8 < 16, for ever
    v = RT.window_check(led, kid, ds, vs, now, 8)
    assert v.kind == "RECOVER_PROBATION" and led.state(kid, "2009-10-03") is RT.State.DEGRADED
    later = "2010-02-05"
    ds2 = [x for x in d if x < later]
    v2 = RT.window_check(led, kid, ds2, vals[:len(ds2)], later, 8)
    assert v2.kind == "RECOVER_FULL" and led.state(kid, "2010-02-06") is RT.State.ACTIVE


def test_probation_that_fails_goes_back_to_dormant_and_weak_evidence_never_recovers():
    led = RT.RetirementLedger(RT.RetirementPolicy(min_n=8, recover_min_n=16))
    kid = _parked(led)
    led.transition(kid, RT.State.DEGRADED, "2009-10-02", "RECOVER_PROBATION", "planted")
    rng = np.random.default_rng(1)
    d = _weeks("2009-10-09", 12)
    v = RT.window_check(led, kid, d, list(-0.01 + rng.normal(0, 0.004, 12)), "2010-01-02", 8)
    assert v.kind == "REVERT" and led.state(kid, "2010-01-03") is RT.State.DORMANT
    led2 = RT.RetirementLedger(RT.RetirementPolicy(min_n=8, recover_min_n=16))
    k2 = _parked(led2, "K2")
    d2 = _weeks("2009-05-15", 30)
    v2 = RT.window_check(led2, k2, d2, list(0.001 + rng.normal(0, 0.02, 30)), "2009-12-15", 8)   # null: noise stays parked
    assert not v2.changes and led2.state(k2, "2009-12-16") is RT.State.DORMANT
    assert RT.window_check(led2, k2, d2[:5], [0.01] * 5, "2009-12-15", 8) is None               # shorter than the window
    assert RT.window_check(led2, "never-registered", d2, [0.01] * 30, "2009-12-15", 8) is None
    with pytest.raises(ValueError):
        RT.window_check(led2, k2, d2, [0.01] * 3, "2009-12-15", 8)


# ------------------------------------------------------------------------------------------------ false-degrade study (the evidence)

def test_an_8_week_window_falsely_degrades_a_true_item_and_16_weeks_does_far_less():
    pol8, pol16 = (RT.RetirementPolicy(min_n=w, recover_min_n=2 * w) for w in (8, 16))
    true8 = RT.degrade_study(0.013, 0.016, 260, pol8, 8, n_sims=25, seed=1)
    true16 = RT.degrade_study(0.013, 0.016, 260, pol16, 16, n_sims=25, seed=1)
    assert true8.any_degrade >= 0.9 and true8.share_weeks_not_active > 0.3                # planted true item: removed half the time
    assert true16.share_weeks_not_active < 0.5 * true8.share_weeks_not_active
    null8 = RT.degrade_study(0.0, 0.016, 104, pol8, 8, n_sims=15, seed=2)
    null16 = RT.degrade_study(0.0, 0.016, 104, pol16, 16, n_sims=15, seed=2)
    assert null8.share_weeks_not_active > 0.85 and null16.share_weeks_not_active > 0.85   # noise is still demoted
    with pytest.raises(ValueError):
        RT.degrade_study(0.01, 0.0, 50, pol8, 8)


def test_a_real_decay_is_detected_by_the_16_week_window_without_the_false_alarms():
    item = PW.multi_year_spec(5).item("decaying")
    kw = dict(multipliers=item.multipliers(260), change_week=item.change_week(260), n_sims=20, seed=3)
    s8 = RT.degrade_study(0.015, 0.016, 260, RT.RetirementPolicy(min_n=8, recover_min_n=16), 8, **kw)
    s16 = RT.degrade_study(0.015, 0.016, 260, RT.RetirementPolicy(min_n=16, recover_min_n=32), 16, **kw)
    assert s8.false_before_change >= 0.4 and s16.false_before_change <= 0.2
    assert s16.detected_after_change >= 0.9 and s16.detect_delay_median <= 40


def test_the_learner_uses_the_16_week_window_the_study_supports():
    assert LN.LearnerConfig().retire_window == 16 and T.make_cfg().retire_window == 16
    L = T.new_learner()
    assert L.retirement.policy.min_n == 8 and L.retirement.policy.recover_min_n == 32   # floor kept for the lifecycle writer


def test_audit_degrades_sorts_calls_by_the_truth_at_the_time():
    led = RT.RetirementLedger(RT.RetirementPolicy(min_n=8, recover_min_n=16))
    for kid, at in (("live", "2009-03-06"), ("dead", "2009-03-06"), ("other", "2009-03-06")):
        led.register(kid, "2009-01-02")
        led.transition(kid, RT.State.DEGRADED, at, "DEGRADE", "planted")
    led.transition("live", RT.State.ACTIVE, "2009-06-05", "RECOVER_FULL", "planted")
    led.transition("live", RT.State.DEGRADED, "2009-09-04", "DEGRADE", "planted")
    a = RT.audit_degrades(led, lambda k, at: {"live": True, "dead": False}.get(k))
    assert (a.on_live, a.on_dead, a.unknown, a.items_live) == (2, 1, 1, 1) and a.false_degrade_share == pytest.approx(2 / 3)
    e = RT.audit_degrades(RT.RetirementLedger(), lambda k, at: True)
    assert (e.on_live, e.on_dead, e.items_seen) == (0, 0, 0) and np.isnan(e.false_degrade_share)


def test_degrade_audit_maps_learner_items_to_planted_truth():
    w = PW.make_world(PW.multi_year_spec(4, 30), seed=1)
    L = T.new_learner()
    for iid in ("strong", "decaying"):
        pid = L.pattern_id(*w.spec.item(iid).conds[0])
        kid = f"K-{iid}"
        L._pid_of[kid], L._kid_of[pid] = pid, kid
        L.retirement.register(kid, str(w.dates[0].date()))
    L.retirement.transition("K-strong", RT.State.DEGRADED, str(w.dates[20].date()), "DEGRADE", "planted false degrade")
    L.retirement.transition("K-decaying", RT.State.DEGRADED, str(w.dates[180].date()), "DEGRADE", "planted true degrade")
    a = LN.degrade_audit(w, L)
    assert a["false_degrades"] == 1 and a["correct_degrades"] == 1 and a["items"]["strong"]["false_degrades"] == 1
    assert a["items"]["decaying"]["weeks_not_active"] == len(w.dates) - 180
    assert LN.degrade_audit(w, T.new_learner())["items"] == {}                        # empty case


# ------------------------------------------------------------------------------------------------ identity blocker wording

def _verdict(status, kind="ticker_shuffle", mode="eval"):
    return IDF.AttackVerdict(kind, mode, status, 0.05, 0.05, 1.0)


def test_identity_blocker_says_what_is_true_of_the_report():
    noop = IDF.IdentityReport((_verdict("OK"),), 0.05, 0.3, True, 0,
                              (fail(LayerName.IDENTITY, "no-op-transform", "episode_substitution", "only 0.0% of rows changed identity"),))
    msg = W.identity_failure_reason(noop)
    assert "collapsed" not in msg and "no-op-transform" in msg                          # F07: the old text said 'collapsed'
    col = IDF.IdentityReport((_verdict("COLLAPSE", "episode_substitution"), _verdict("OK")), 0.05, 0.3, True, 0)
    assert W.identity_failure_reason(col).startswith("collapsed under episode_substitution[eval]")
    assert "nondeterministic" in W.identity_failure_reason(IDF.IdentityReport((_verdict("OK"),), 0.05, 0.3, False, 0))
    assert "not judged" in W.identity_failure_reason(IDF.IdentityReport((_verdict("NO_SKILL"),), 0.0, 0.0, True, 0))
    assert W.identity_failure_reason(IDF.IdentityReport((), 0.0, 0.0, True, 0)) == "no attack was run"   # empty case
    chk = W._check_identity(W.LearningEvidence(identity=noop))
    assert not chk.ok and chk.detail == msg
