"""Tests for engine.learning.planted_world (contract 62 tests 1-12, 63). Synthetic only; small worlds; seconds."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning import planted_world as pw
from engine.learning.core import FirewallBreach

W, S = 104, 60
SEED = 7


@pytest.fixture(scope="module")
def world():
    return pw.make_world(pw.standard_spec(W, S, scale=2.0), SEED)


@pytest.fixture(scope="module")
def nullworld():
    return pw.make_world(pw.noise_only_spec(80, 60, 8), 3)


def item(world, iid):
    return world.spec.item(iid)


# ------------------------------------------------------------------ spec validation (each defect must be caught)
def test_standard_spec_valid_and_covers_every_kind():
    spec = pw.standard_spec(W, S)
    assert spec.validate() == []
    assert {i.kind for i in spec.items} == set(pw.KINDS)


@pytest.mark.parametrize("mut,frag", [
    (lambda s: dataclasses.replace(s, discovery_end=s.confirm_end), "splits"),
    (lambda s: dataclasses.replace(s, stocks=10), "stocks"),
    (lambda s: dataclasses.replace(s, start="2009-01-01"), "Friday"),
    (lambda s: dataclasses.replace(s, items=s.items + (s.items[0],)), "duplicate item ids"),
    (lambda s: dataclasses.replace(s, items=s.items[:1] + (dataclasses.replace(s.items[1], item_id="x", conds=s.items[0].conds),) + s.items[2:]), "share a condition key"),
    (lambda s: dataclasses.replace(s, items=(dataclasses.replace(s.items[0], conds=(("f99", 4),)),)), "unknown feature"),
    (lambda s: dataclasses.replace(s, items=(dataclasses.replace(s.items[0], conds=(("f0", 7),)),)), "quintile"),
    (lambda s: dataclasses.replace(s, items=(dataclasses.replace(s.items[0], profile="flip", params=(s.weeks + 3,)),)), "outside the sample"),
    (lambda s: dataclasses.replace(s, items=(pw.Item("d", pw.DUPLICATE, (("f14", 4),)),)), "redundant_with"),
    (lambda s: dataclasses.replace(s, items=(pw.Item("n", pw.NOISE, (("f0", 4),), 0.01),)), "no effect"),
    (lambda s: dataclasses.replace(s, items=(pw.Item("r", pw.REGIME, (("f0", 4),), 0.01),)), "needs a gate"),
    (lambda s: dataclasses.replace(s, items=(pw.Item("u", pw.UNLESS, (("f0", 4),), 0.01),)), "unless"),
    (lambda s: dataclasses.replace(s, items=(pw.Item("w", pw.WEAK, (("f0", 4),), 0.0001),)), "reporting floor"),
    (lambda s: dataclasses.replace(s, items=(pw.Item("g", pw.REGIME, (("f0", 4),), 0.01, gate=pw.Gate("m", "m_nope")),)), "market column"),
])
def test_spec_defects_are_caught(mut, frag):
    bad = mut(pw.standard_spec(W, S))
    errs = bad.validate()
    assert any(frag in e for e in errs), errs
    with pytest.raises(ValueError):
        bad.check()


# ------------------------------------------------------------------ determinism
def test_same_seed_same_world_different_seed_different(world):
    again = pw.make_world(pw.standard_spec(W, S, scale=2.0), SEED)
    assert again.content_hash() == world.content_hash()
    assert np.array_equal(again.y_raw.to_numpy(), world.y_raw.to_numpy())
    other = pw.make_world(pw.standard_spec(W, S, scale=2.0), SEED + 1)
    assert other.content_hash() != world.content_hash()
    assert other.ledger.content_hash() != world.ledger.content_hash()   # gate fractions/row counts are realised, so differ


def test_panel_shape_and_conventions(world):
    assert isinstance(world.X.index, pd.MultiIndex) and world.X.index.names == ["date", "ticker"]
    assert len(world.X) == W * S == len(world.y)
    assert all(c.startswith("m_") for c in world.market_columns()) and len(world.market_columns()) == 3
    assert world.canary_columns() == [pw.CANARY_COL]
    assert abs(world.y.groupby(level=0).mean()).max() < 1e-12            # excess return is demeaned per week
    assert all(d.weekday() == 4 for d in world.dates)


# ------------------------------------------------------------------ the ledger is exact
def test_ledger_internally_consistent(world):
    assert world.ledger.check() == []
    assert len(world.ledger.to_frame()) == len(world.spec.items) * W


def test_ledger_effect_equals_realised_contribution_exactly(world):
    """For every (item, week) the ledger's true_effect equals the mean planted contribution over the item's rows."""
    for j, it in enumerate(world.spec.items):
        m = world._row_mask(it)
        by_week = pd.Series(world.contrib[m, j]).groupby(world.week_idx[m]).mean()
        led = world.ledger.series(it.item_id).set_index("week")["true_effect"]
        for w, v in by_week.items():
            assert led[w] == pytest.approx(v, abs=1e-12), (it.item_id, w)


def test_y_is_noise_plus_exactly_the_planted_contributions(world):
    """An oracle that knows the truth removes every planted effect and is left with what the noise generator produced."""
    resid = world.y_raw.to_numpy() - world.contrib.sum(axis=1)
    g = pw._rng(SEED, 3)
    df = world.spec.tail_df
    idio = g.standard_t(df, W * S) * world.spec.noise_sd / np.sqrt(df / (df - 2))
    sector = g.integers(0, world.spec.n_sectors, S)
    sec = g.normal(0, world.spec.sector_sd, (W, world.spec.n_sectors))[:, sector].reshape(-1)
    week = np.repeat(g.normal(0, world.spec.week_sd, W), S)
    assert np.allclose(resid, idio + week + sec, atol=1e-12)


def test_strong_weak_negative_pair_are_constant_and_signed(world):
    for iid, sign in (("strong", 1), ("weak", 1), ("negative", -1), ("pair", 1)):
        s = world.ledger.series(iid)
        assert (s["active"] == (s["n_rows"] > 0)).all(), iid       # a week with no affected row has nothing to be active on
        assert s["active"].mean() > 0.8
        assert (np.sign(s["true_effect"][s["n_rows"] > 0]) == sign).all()
        assert s["condition_state"].isna().all()
    assert world.ledger.effect("strong", 0) == pytest.approx(0.024)
    assert world.ledger.effect("weak", 5) == pytest.approx(0.008)


def test_regime_item_follows_the_observable_state(world):
    s = world.ledger.series("regime").set_index("week")
    vix = world.X["m_vix"].groupby(level=0).first().to_numpy()
    for w in range(W):
        on = vix[w] > 0
        assert s.loc[w, "condition_state"] == on
        assert s.loc[w, "active"] == on
        assert s.loc[w, "true_effect"] == pytest.approx(0.02 if on else 0.0)
    assert 0 < s["active"].sum() < W                                       # both states occur


def test_conditional_unless_and_pair_row_effects(world):
    # conditional: bigger where f8 is in {3,4}, base elsewhere - and never zero on its key rows
    j = [i.item_id for i in world.spec.items].index("conditional")
    it = item(world, "conditional")
    m = world._row_mask(it)
    gate = it.gate.mask(world.X, world._Q)
    assert np.allclose(world.contrib[m & gate, j], 0.028) and np.allclose(world.contrib[m & ~gate, j], 0.008)
    # unless: rows with the exception carry nothing
    ju = [i.item_id for i in world.spec.items].index("unless")
    u = item(world, "unless")
    both = (world._Q["f9"].to_numpy() == 4) & (world._Q["f10"].to_numpy() == 4)
    exc = world._Q["f11"].to_numpy() == 4
    assert np.allclose(world.contrib[both & ~exc, ju], 0.04) and np.allclose(world.contrib[both & exc, ju], 0.0)
    assert both.sum() > 0 and (both & exc).sum() > 0
    # pair: neither condition alone carries any effect
    jp = [i.item_id for i in world.spec.items].index("pair")
    a = world._Q["f3"].to_numpy() == 4
    b = world._Q["f4"].to_numpy() == 0
    assert np.allclose(world.contrib[a ^ b, jp], 0.0) and np.allclose(world.contrib[a & b, jp], 0.04)


def test_hallucination_exists_only_in_discovery(world):
    s = world.ledger.series("hallucination")
    d, c = world.spec.discovery_end, world.spec.confirm_end
    assert s[s["week"] < d]["active"].all()
    assert not s[s["week"] >= d]["active"].any()
    assert world.exact_item_effect("hallucination", (0, d)) > 0.015
    assert world.exact_item_effect("hallucination", (d, W)) == 0.0
    assert world.ledger.change_week("hallucination") == d


def test_reversal_flips_sign_at_the_known_week(world):
    t0 = item(world, "reversal").params[0]
    assert world.ledger.change_week("reversal") == t0
    assert world.ledger.sign_at("reversal", t0 - 1) == 1 and world.ledger.sign_at("reversal", t0) == -1
    assert world.ledger.effect("reversal", W - 1) == pytest.approx(-world.ledger.effect("reversal", 0))
    assert world.spec.phase_of(t0) == "confirmation"


def test_decay_is_monotone_and_dead_by_evaluation(world):
    e = world.ledger.series("decaying")["true_effect"].to_numpy()
    assert (np.diff(e) <= 1e-15).all() and e[0] == pytest.approx(0.02) and e[-1] == 0.0
    cw = world.ledger.change_week("decaying")
    assert abs(e[cw]) <= 0.5 * abs(e[0]) + 1e-12 and abs(e[cw - 1]) > 0.5 * abs(e[0])
    assert not world.ledger.is_live("decaying", world.spec.confirm_end + 13)


def test_hidden_item_present_in_training_absent_in_evaluation(world):
    c = world.spec.confirm_end
    s = world.ledger.series("hidden")
    assert s[s["week"] < c]["active"].all() and not s[s["week"] >= c]["active"].any()
    assert world.ledger.change_week("hidden") == c
    assert "hidden" in world.ledger.should_hold(c - 1) and "hidden" not in world.ledger.should_hold(c + 13)


def test_using_the_hidden_item_hurts_in_evaluation(world):
    c = world.spec.confirm_end
    assert world.use_pnl("hidden", (0, c), cost=0.01, exact=True) > 0.004          # earned in training (net of demeaning)
    assert world.use_pnl("hidden", (c, W), cost=0.01, exact=True) == pytest.approx(-0.01, abs=1e-12)
    assert world.use_pnl("hidden", (c, W), cost=0.01, exact=True) < 0


def test_duplicate_and_noise_inject_nothing_but_duplicate_correlates(world):
    for iid in ("duplicate", "noise_a", "noise_b"):
        assert not world.ledger.series(iid)["active"].any()
    r = np.corrcoef(world.X["f0"], world.X["f14"])[0, 1]
    assert r > 0.9
    src, dup = item(world, "strong"), item(world, "duplicate")
    assert dup.redundant_with == "strong"
    # the copy "works" only because it overlaps the real pattern: no contribution of its own
    j = [i.item_id for i in world.spec.items].index("duplicate")
    assert np.all(world.contrib[:, j] == 0)
    apparent = world.exact_key_effect(dup.key, (0, W))
    assert apparent > 0.005                                                        # visible by overlap, redundant by truth
    overlap = ((world._Q["f0"].to_numpy() == 4) & (world._Q["f14"].to_numpy() == 4)).sum() / (world._Q["f14"].to_numpy() == 4).sum()
    assert overlap > 0.6


def test_should_hold_sets_at_each_phase(world):
    c = world.spec.confirm_end
    at_eval = set(world.ledger.should_hold(c + 13))
    assert at_eval == {"strong", "weak", "negative", "pair", "regime", "conditional", "unless", "reversal"}
    assert set(world.ledger.traps_at(c + 13)) == {"decaying", "hallucination", "duplicate", "hidden", "noise_a", "noise_b"}
    early = set(world.ledger.should_hold(5))
    assert {"hallucination", "hidden", "decaying"} <= early and "reversal" in early
    assert "duplicate" not in early and "noise_a" not in early


def test_ledger_queries_fail_loudly(world):
    with pytest.raises(KeyError):
        world.ledger.entry("strong", W + 5)
    with pytest.raises(KeyError):
        world.ledger.entry("nope", 0)
    with pytest.raises(IndexError):
        world.spec.phase_of(W)
    e = world.ledger.entry("regime", 3)
    assert e.phase == "discovery" and e.item_id == "regime" and isinstance(e.condition_state, bool)


# ------------------------------------------------------------------ oracle recovers every kind; null learner sees none
def test_oracle_scores_perfectly_at_every_phase(world):
    for wk in (10, world.spec.discovery_end + 3, world.spec.confirm_end - 1, world.spec.confirm_end + 13, W - 1):
        sc = pw.score_claims(world, pw.oracle_claims(world, wk), wk)
        assert sc.recall == 1.0 and sc.precision == 1.0 and sc.false_discovery_rate == 0.0, (wk, sc)
        assert sc.missed == [] and sc.fp == 0 and all(v == 1.0 for v in sc.recall_by_kind.values())
        assert all(v == 0.0 for v in sc.false_claim_by_kind.values())


def test_oracle_recall_by_kind_names_every_live_kind(world):
    sc = pw.score_claims(world, pw.oracle_claims(world, world.spec.confirm_end + 13), world.spec.confirm_end + 13)
    assert set(sc.recall_by_kind) == {pw.STRONG, pw.WEAK, pw.NEGATIVE, pw.PAIR, pw.REGIME, pw.CONDITIONAL, pw.UNLESS, pw.REVERSAL}
    assert set(sc.false_claim_by_kind) == {pw.DECAYING, pw.HALLUCINATION, pw.DUPLICATE, pw.HIDDEN_EVAL, pw.NOISE}


def test_oracle_claims_carry_signs_and_conditions(world):
    cl = {c.name: c for c in pw.oracle_claims(world, world.spec.confirm_end + 13)}
    assert cl["f2 q4"].effect < 0 and cl["f13 q4"].effect < 0                      # negative and the REVERSED pattern
    assert cl["f5 q4"].condition == ("m", "m_vix", 0.0, np.inf)
    assert cl[item(world, "unless").key_named].condition == ("q", "f11", (4,))


def test_null_learner_recalls_nothing_and_claims_no_falsehood(world):
    wk = world.spec.confirm_end + 13
    sc = pw.score_claims(world, pw.null_claims(world, wk), wk)
    assert sc.n_claims == 0 and sc.recall == 0.0 and sc.precision == 1.0 and sc.false_discovery_rate == 0.0
    assert sc.missed == sorted(world.ledger.should_hold(wk))
    assert all(v == 0.0 for v in sc.recall_by_kind.values())


def test_random_guessing_has_low_precision_and_recall(world):
    wk = world.spec.confirm_end + 13
    sc = pw.score_claims(world, pw.random_claims(world, 12, seed=1), wk)
    assert sc.precision < 0.6 and sc.recall < 0.6
    assert pw.random_claims(world, 12, 1) == pw.random_claims(world, 12, 1)


def test_every_trap_claimed_is_a_false_discovery_with_its_kind(world):
    wk = world.spec.confirm_end + 13
    traps = [item(world, i).key_named for i in ("hallucination", "hidden", "decaying", "duplicate", "noise_a")]
    sc = pw.score_claims(world, traps, wk)
    assert sc.tp == 0 and sc.fp == 5 and sc.false_discovery_rate == 1.0
    assert {r for _, r in sc.false_claims} == {"trap:hallucination", "trap:hidden_eval", "trap:decaying", "trap:duplicate", "trap:noise"}
    assert sc.false_claim_by_kind[pw.HALLUCINATION] == 1.0 and sc.false_claim_by_kind[pw.NOISE] == 0.5


def test_hallucination_is_legitimately_true_during_discovery_only(world):
    key = item(world, "hallucination").key_named
    early = pw.score_claims(world, [key], 20)
    late = pw.score_claims(world, [key], world.spec.confirm_end + 13)
    assert early.tp == 1 and late.fp == 1


def test_wrong_sign_claim_on_the_reversal_is_a_false_discovery(world):
    wk = world.spec.confirm_end + 13
    old = pw.Claim("f13 q4", effect=+0.01)                                         # the pre-reversal belief
    new = pw.Claim("f13 q4", effect=-0.01)
    assert pw.score_claims(world, [old], wk).sign_errors == ["reversal"]
    assert pw.score_claims(world, [old], wk).fp == 1
    assert pw.score_claims(world, [new], wk).tp == 1
    assert pw.score_claims(world, [old], 5).tp == 1                                # and it was right before the flip


def test_unplanted_claims_are_judged_by_exact_realised_effect(world):
    wk = world.spec.confirm_end + 13
    # f0 q4 & f16 q4: a sub-condition of the strong pattern - truly positive though never planted as such
    sub = pw.score_claims(world, ["f0 q4 & f16 q4"], wk)
    assert sub.unplanted["f0 q4 & f16 q4"]["real"] and sub.tp == 1
    junk = pw.score_claims(world, ["f16 q1", "f17 q3 & f16 q2"], wk)
    assert junk.fp == 2 and all(not v["real"] for v in junk.unplanted.values())


def test_unparsable_and_duplicate_claims_and_bad_week(world):
    wk = world.spec.confirm_end + 13
    sc = pw.score_claims(world, ["garbage", "f0 q4", "f0 q4", pw.Claim("f0 q4")], wk)
    assert sc.n_claims == 2 and sc.tp == 1 and ("garbage", "unparsable") in sc.false_claims
    with pytest.raises(IndexError):
        pw.score_claims(world, [], W)
    with pytest.raises(TypeError):
        pw.as_claim(3.14)


def test_as_claim_accepts_strings_mappings_and_duck_types():
    class D:
        key_named, effect, condition = "f1 q4", 0.5, None
    assert pw.as_claim("f1 q4").name == "f1 q4"
    assert pw.as_claim({"key_named": "f1 q4", "effect": 0.1}).effect == 0.1
    assert pw.as_claim(D()).effect == 0.5


def test_memoriser_is_distinguishable_from_the_oracle(world):
    wk = world.spec.confirm_end + 13
    mem = pw.score_claims(world, pw.memoriser_claims(world), wk)
    ora = pw.score_claims(world, pw.oracle_claims(world, wk), wk)
    assert mem.false_discovery_rate > 0.2 and ora.false_discovery_rate == 0.0
    assert mem.false_claim_by_kind[pw.HALLUCINATION] == 1.0 and mem.false_claim_by_kind[pw.HIDDEN_EVAL] == 1.0


# ------------------------------------------------------------------ degradation-detection delay
def test_oracle_detections_have_zero_delay(world):
    rep = pw.degradation_delays(world, pw.oracle_detections(world))
    assert set(rep.per_item) == {"decaying", "reversal", "hidden", "hallucination"}
    assert rep.median_delay == 0.0 and rep.detection_rate == 1.0
    assert rep.missed == [] and rep.premature == [] and rep.false_alarms == []


def test_late_detection_measures_the_delay(world):
    rep = pw.degradation_delays(world, pw.oracle_detections(world, delay=6))
    assert rep.median_delay == 6.0 and rep.mean_delay == 6.0


def test_missed_premature_false_alarm_and_first_flag_counts(world):
    cw = world.ledger.change_week("reversal")
    det = {"reversal": cw - 10, "strong": 30, "hidden": world.ledger.change_week("hidden") + 2}
    rep = pw.degradation_delays(world, det)
    assert rep.premature == ["reversal"] and rep.false_alarms == ["strong"]
    assert set(rep.missed) == {"decaying", "hallucination"}
    assert rep.per_item["hidden"]["delay"] == 2 and rep.detection_rate == pytest.approx(0.25)
    assert pw.degradation_delays(world, {"reversal": cw - 10}, tolerance=10).premature == []
    dated = pw.degradation_delays(world, {"f13 q4": world.dates[cw + 1]})            # key + date accepted
    assert dated.per_item["reversal"]["delay"] == 1


def test_detection_validation(world):
    with pytest.raises(KeyError):
        pw.degradation_delays(world, {"ghost": 3})
    with pytest.raises(IndexError):
        pw.degradation_delays(world, {"reversal": 10 ** 4})
    none = pw.degradation_delays(world, {})
    assert none.detection_rate == 0.0 and none.median_delay is None and len(none.missed) == 4


# ------------------------------------------------------------------ condition recovery
def test_oracle_recovers_every_condition(world):
    sc = pw.condition_recovery(world, pw.oracle_conditions(world))
    assert set(sc) == {"regime", "conditional", "unless"}
    for s in sc.values():
        assert s.accuracy == 1.0 and s.balanced_accuracy == 1.0 and not s.degenerate
    assert sc["regime"].boundary_error == 0.0


def test_always_on_and_wrong_variable_score_chance(world):
    always = {k: (lambda X, Q: np.ones(len(X), bool)) for k in ("regime", "conditional")}
    sc = pw.condition_recovery(world, always)
    assert sc["regime"].balanced_accuracy == pytest.approx(0.5) and sc["conditional"].balanced_accuracy == pytest.approx(0.5)
    assert sc["regime"].accuracy < 0.9                                             # plain accuracy would flatter it
    wrong = pw.condition_recovery(world, {"regime": ("m", "m_breadth", 0.0, np.inf), "conditional": ("q", "f16", (3, 4))})
    assert wrong["regime"].balanced_accuracy < 0.7 and wrong["conditional"].balanced_accuracy < 0.6


def test_shifted_threshold_is_penalised_and_measured(world):
    sc = pw.condition_recovery(world, {"regime": ("m", "m_vix", 0.8, np.inf)})
    assert 0.6 < sc["regime"].balanced_accuracy < 1.0
    assert sc["regime"].boundary_error == pytest.approx(0.8)


def test_condition_recovery_by_key_and_errors(world):
    sc = pw.condition_recovery(world, {"f5 q4": pw.item(world, "regime").gate if hasattr(pw, "item") else item(world, "regime").gate})
    assert sc["regime"].balanced_accuracy == 1.0
    with pytest.raises(ValueError):
        pw.condition_recovery(world, {"strong": ("m", "m_vix", 0.0, 1.0)})
    with pytest.raises(KeyError):
        pw.condition_recovery(world, {"ghost": ("m", "m_vix", 0.0, 1.0)})
    with pytest.raises(ValueError):
        pw.condition_recovery(world, {"regime": ("bogus", 1)})
    with pytest.raises(TypeError):
        pw.condition_recovery(world, {"regime": 5})


# ------------------------------------------------------------------ statistical recoverability without the ledger
def test_strong_effect_is_recoverable_from_data_and_null_world_is_silent(nullworld):
    spec = pw.single_item_spec(pw.Item("s", pw.STRONG, (("f0", 4),), 0.012), weeks=104, stocks=80)
    w = pw.make_world(spec, 5)
    est = w.estimate_key_effect(w.spec.item("s").key)
    assert est["t"] > 6 and est["effect"] == pytest.approx(0.012, abs=0.004)
    share = w._key_mask(w.spec.item("s").key).mean()                                # in-minus-out = demeaned effect / (1 - share)
    assert est["effect"] == pytest.approx(w.exact_key_effect(w.spec.item("s").key) / (1 - share), abs=0.004)
    # invisible under the null: 8 features x 5 quintiles, none planted
    ts = [abs(nullworld.estimate_key_effect((( f"f{f} q{q}",), None))["t"]) for f in range(8) for q in range(5)]
    assert max(ts) < 4.0 and np.mean(ts) < 1.6
    assert nullworld.ledger.should_hold(10) == []


def test_exact_effect_of_noise_world_is_exactly_zero(nullworld):
    assert np.all(nullworld.contrib == 0) and nullworld.exact_key_effect((("f0 q4",), None)) == 0.0
    assert nullworld.ledger.frame["active"].sum() == 0


def test_estimate_edge_cases(world):
    assert world.estimate_key_effect((("f0 q9",), None))["n_weeks"] == 0           # key selects nothing
    assert world.estimate_key_effect((("f0 q4",), None), (0, 2))["effect"] == 0.0  # < 3 weeks: no estimate
    assert world.exact_key_effect((("nope q1",), None)) == 0.0


def test_each_planted_kind_recoverable_from_data_in_its_live_window(world):
    """A key-aware statistician (no ledger) sees the strong/negative/pair/unless effects in the right direction."""
    for iid, sign in (("strong", 1), ("negative", -1), ("pair", 1), ("unless", 1)):
        est = world.estimate_key_effect(item(world, iid).key)
        assert np.sign(est["effect"]) == sign and abs(est["t"]) > 3, (iid, est)
    # hallucination is visible in discovery, gone in evaluation
    hk = item(world, "hallucination").key
    d = world.spec.discovery_end
    assert world.estimate_key_effect(hk, (0, d))["t"] > 4
    assert abs(world.estimate_key_effect(hk, (d, W))["t"]) < 3
    # reversal: opposite signs either side of the flip
    rk, t0 = item(world, "reversal").key, item(world, "reversal").params[0]
    assert world.estimate_key_effect(rk, (0, t0))["effect"] > 0 > world.estimate_key_effect(rk, (t0, W))["effect"]
    # regime: pooled effect is diluted, in-state weeks carry it
    vix = world.X["m_vix"].groupby(level=0).first().to_numpy()
    on_w = [w for w in range(W) if vix[w] > 0]
    assert len(on_w) >= 10


# ------------------------------------------------------------------ test 8: identities
def test_reidentify_changes_names_dates_and_order_but_not_truth(world):
    r = pw.reidentify(world, seed=4, shift_years=5)
    w2 = r.world
    assert set(r.ticker_map.values()).isdisjoint(set(r.ticker_map.keys())) and len(set(r.ticker_map.values())) == S
    assert r.date_shift_days == 5 * 364 and w2.dates[0] == world.dates[0] + pd.Timedelta(days=5 * 364)
    assert all(d.weekday() == 4 for d in w2.dates)
    # rows shuffled within each date: new order is not the old order under the renaming
    inv = {v: k for k, v in r.ticker_map.items()}
    back = [inv[t] for t in w2.X.index.get_level_values(1)[:S]]
    assert back != list(world.X.index.get_level_values(1)[:S]) and sorted(back) == sorted(world.X.index.get_level_values(1)[:S])
    # values travel with their stock: align on (week, original ticker)
    a = world.y_raw.copy(); a.index = pd.MultiIndex.from_arrays([world.week_idx, world.X.index.get_level_values(1)])
    b = w2.y_raw.copy(); b.index = pd.MultiIndex.from_arrays([w2.week_idx, [inv[t] for t in w2.X.index.get_level_values(1)]])
    assert np.allclose(a.sort_index().to_numpy(), b.sort_index().to_numpy())
    # truth is identical week by week, only the dates moved
    l1, l2 = world.ledger.to_frame(), w2.ledger.to_frame()
    assert np.array_equal(l1["true_effect"].to_numpy(), l2["true_effect"].to_numpy())
    assert l2["date"].iloc[0] == str(w2.dates[0].date()) and l1["date"].iloc[0] != l2["date"].iloc[0]
    # a pattern-knowing learner scores identically; a statistician recovers the same effect
    wk = W - 1
    assert pw.score_claims(w2, pw.oracle_claims(w2, wk), wk).recall == 1.0
    k = item(world, "strong").key
    assert w2.estimate_key_effect(k)["effect"] == pytest.approx(world.estimate_key_effect(k)["effect"], abs=1e-12)


def test_reidentify_is_deterministic_and_seed_sensitive(world):
    a, b, c = pw.reidentify(world, 1), pw.reidentify(world, 1), pw.reidentify(world, 2)
    assert dict(a.ticker_map) == dict(b.ticker_map) != dict(c.ticker_map)
    assert a.world.content_hash() == b.world.content_hash()
    keep = pw.reidentify(world, 1, tickers=False, shuffle_rows=False)
    assert dict(keep.ticker_map) == {t: t for t in keep.ticker_map} and keep.date_shift_days == 0
    assert keep.world.y_raw.equals(world.y_raw)


def test_name_memoriser_fails_after_reidentification(world):
    """A learner that stored 'ticker T007 rises' is right in the original world and no better than chance renamed."""
    hi = list(world.y.groupby(level=1).mean().sort_values().index[-6:])
    r = pw.reidentify(world, 9)
    renamed = [r.ticker_map[t] for t in hi]
    mean_orig = world.y[world.X.index.get_level_values(1).isin(hi)].mean()
    assert not r.world.X.index.get_level_values(1).isin(hi).any()                  # the remembered names select nothing now
    assert r.world.y[r.world.X.index.get_level_values(1).isin(renamed)].mean() == pytest.approx(mean_orig, abs=1e-12)
    # and a name-keyed lookup table built on the old world is empty when applied to the new one
    table = {t: 1.0 for t in hi}
    assert sum(table.get(t, 0.0) for t in r.world.X.index.get_level_values(1)) == 0.0


# ------------------------------------------------------------------ test 9: years
def test_year_swap_keeps_situation_classes_changes_everything_else(world):
    sw = pw.year_swap(world, seed=99, years=6, vol_scale=1.5)
    assert pw.same_situation_classes(world, sw)
    assert sw.dates[0] == world.dates[0] + pd.Timedelta(days=6 * 364)
    assert sw.content_hash() != world.content_hash()
    assert sw.y_raw.std() > world.y_raw.std()
    assert not np.allclose(sw.X["f0"].to_numpy(), world.X["f0"].to_numpy())
    # the conditions and effects carry over: the oracle still scores perfectly, the ledger keeps its shape
    wk = W - 1
    assert pw.score_claims(sw, pw.oracle_claims(sw, wk), wk).recall == 1.0
    assert sw.ledger.change_week("reversal") == world.ledger.change_week("reversal")
    assert sw.spec.name.endswith("@+6y")


def test_year_swap_accepts_a_spec_and_rejects_bad_scale():
    spec = pw.noise_only_spec(60, 40, 6)
    w = pw.year_swap(spec, 1, years=2)
    assert w.dates[0] == pd.Timestamp(spec.start) + pd.Timedelta(days=728)
    with pytest.raises(ValueError):
        pw.year_swap(spec, 1, vol_scale=0)


def test_effects_transfer_across_eras_for_a_situation_learner():
    spec = pw.single_item_spec(pw.Item("s", pw.STRONG, (("f0", 4),), 0.02), weeks=80, stocks=80)
    a, b = pw.make_world(spec, 1), pw.year_swap(spec, 2, years=8, vol_scale=1.2)
    k = spec.items[0].key
    assert a.estimate_key_effect(k)["t"] > 5 and b.estimate_key_effect(k)["t"] > 5


# ------------------------------------------------------------------ test 11: future information
def test_canary_is_a_near_copy_of_the_forward_return_and_hidden_from_the_view(world):
    c = np.corrcoef(world.X[pw.CANARY_COL], world.y_raw)[0, 1]
    assert c > 0.9
    now = world.dates[60]
    v = world.learner_view(now)
    assert pw.CANARY_COL not in v.X.columns and v.dropped == (pw.CANARY_COL,)
    assert pw.CANARY_COL in world.learner_view(now, include_canary=True).X.columns


def test_firewall_rejects_canary_columns(world):
    pw.assert_no_canary(world.feature_columns() + world.market_columns())
    with pytest.raises(FirewallBreach):
        pw.assert_no_canary(list(world.X.columns))
    with pytest.raises(FirewallBreach):
        pw.assert_no_canary(["f1", "canary_anything"])


def test_leak_suspicion_flags_the_canary_not_honest_signal(world):
    leaky = pw.leak_suspicion(world.X[pw.CANARY_COL], world)
    assert leaky["leak"] and leaky["ic"] > 0.5
    honest = pw.leak_suspicion(world.X["f0"], world)
    assert not honest["leak"] and abs(honest["ic"]) < 0.15
    assert pw.leak_suspicion(world.X["f0"].iloc[:10], world)["n"] == 10
    assert not pw.leak_suspicion(world.X["f0"].iloc[:10], world)["leak"]
    part = pw.leak_suspicion(world.X[pw.CANARY_COL], world, weeks=(80, W))
    assert part["leak"]


def test_learner_view_never_shows_the_future(world):
    for k in (1, 30, W - 1):
        now = world.dates[k]
        v = world.learner_view(now)
        assert v.X.index.get_level_values(0).max() == now
        assert v.y.index.get_level_values(0).max() == world.dates[k - 1] if k > 0 else len(v.y) == 0
        # every outcome shown matured at or before now: its NEXT week's date is <= now
        nxt = {d: world.dates[i + 1] for i, d in enumerate(world.dates[:-1])}
        assert all(nxt[d] <= now for d in v.y.index.get_level_values(0).unique())
    assert len(world.learner_view(world.dates[0]).y) == 0
    with pytest.raises(FirewallBreach):
        world.learner_view(world.dates[0] - pd.Timedelta(days=7))


def test_learner_view_between_weeks_and_after_end(world):
    mid = world.dates[10] + pd.Timedelta(days=3)
    assert world.learner_view(mid).X.index.get_level_values(0).max() == world.dates[10]
    late = world.learner_view(world.dates[-1] + pd.Timedelta(days=400))
    assert late.X.index.get_level_values(0).max() == world.dates[-1]
    assert late.y.index.get_level_values(0).max() == world.dates[-2]               # the last week's return has not matured


# ------------------------------------------------------------------ degenerate cases
def test_world_with_no_items():
    spec = pw.WorldSpec("empty", (), weeks=40, stocks=30, n_feat=4, discovery_end=15, confirm_end=30)
    w = pw.make_world(spec, 1)
    assert w.ledger.to_frame().empty and w.ledger.should_hold(5) == [] and w.ledger.check() == []
    sc = pw.score_claims(w, [], 5)
    assert sc.recall == 1.0 and sc.precision == 1.0 and sc.false_discovery_rate == 0.0
    assert w.exact_key_effect((("f0 q4",), None)) == 0.0
    assert pw.degradation_delays(w, {}).detection_rate == 1.0
    assert pw.condition_recovery(w, {}) == {}
    assert "empty" in w.summary()["name"]


def test_claims_against_a_world_with_no_signal_are_all_false(nullworld):
    sc = pw.score_claims(nullworld, ["f0 q4", "f3 q2", "f5 q0 & f6 q1"], 60)
    assert sc.tp == 0 and sc.fp == 3 and sc.recall == 1.0


# ------------------------------------------------------------------ manifest, hashes, stamps (test 12)
def test_manifest_lists_every_item_and_is_json_safe(world):
    import json
    m = world.manifest()
    assert len(m["items"]) == len(world.spec.items) and m["seed"] == SEED
    json.dumps(m)
    by = {i["id"]: i for i in m["items"]}
    assert by["reversal"]["change_week"] == world.ledger.change_week("reversal") and by["strong"]["change_week"] is None


def test_stale_stamp_is_refused(world):
    st = pw.stamp_world(world, code_hash="aaa")
    assert pw.check_stamp(st, world, code_hash="aaa") == []
    errs = pw.check_stamp(st, world, code_hash="bbb")
    assert errs and "code changed" in errs[0]
    other = pw.make_world(pw.standard_spec(W, S, scale=2.0), SEED + 1)
    assert any("differs" in e for e in pw.check_stamp(st, other, code_hash="aaa"))


def test_render_reports_mention_every_item_and_score(world):
    txt = pw.render_ledger(world)
    for it in world.spec.items:
        assert it.item_id in txt
    assert "YES" in txt and "no" in txt
    wk = world.spec.confirm_end + 13
    out = pw.render_score(pw.score_claims(world, pw.oracle_claims(world, wk), wk))
    assert "recall=1.00" in out and "FDR=0.00" in out


def test_gate_and_item_primitives():
    g = pw.Gate("q", "f1", qs=(0, 1))
    assert g.validate() == [] and g.spec() == ("q", "f1", (0, 1))
    assert pw.Gate("q", "f1", qs=(7,)).validate() and pw.Gate("z", "f1").validate() and pw.Gate("m", "m_vix", 1, 1).validate()
    it = pw.Item("d", pw.DECAYING, (("f0", 4),), 0.01, profile="decay", params=(10, 30))
    m = it.multipliers(40)
    assert m[9] == 1 and m[10] == 1 and m[20] == pytest.approx(0.5) and m[30] == 0 and m[39] == 0
    assert it.change_week(40) == 20
    assert pw.Item("c", pw.STRONG, (("f0", 4),), 0.01).change_week(40) is None
    late = pw.Item("l", pw.WEAK, (("f0", 4),), 0.01, profile="window", params=(5, 9))
    assert late.change_week(40) == 5 and late.multipliers(12).tolist()[4:10] == [0, 1, 1, 1, 1, 0]
    assert it.key_named == "f0 q4"
    assert pw.Item("p", pw.PAIR, (("f4", 0), ("f3", 4)), 0.01).key_named == "f3 q4 & f4 q0"
