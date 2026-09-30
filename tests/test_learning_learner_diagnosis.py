"""F07 (C69 W-07): tests for the three losses the learner diagnosis found, each with a case that FAILS on the old behaviour.

1. Retrieval skill starved (learner.shadow_monitoring): the walk-forward skill monitor only saw predictions made after an item was
   CHAMPION.  Now held, not-yet-production items are scored in shadow; shadow retrieval never reaches a decision.
2. One-way degradation (learner.RecoveringLedger + stage_store): a DEGRADED item that was never parked had no way back, and the
   knowledge object never followed a recovery.  Now the ledger's own recovery bar (n >= recover_min_n after the degrade,
   t >= recover_t > degrade_t) restores it through the ledger's only door.
3. Identity harness no-op (loop_hooks.episode_attack_kwargs): on the learner's 4 held-out dates the 20-date episode attack
   changed nothing, raised a FAIL finding and failed every true item without attacking it.
Plus the truth trace (learner.truth_trace) and the acceptance summary's null-world accounting.  Synthetic data only.
Status: IMPLEMENTED - NOT VALIDATED."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from engine.learning import identity_firewall as IDF
from engine.learning import learner as LN
from engine.learning import loop_hooks as LH
from engine.learning import planted_world as PW
from engine.learning import retirement as RT
from engine.learning.core import FirewallBreach, Promotion

import test_learning_learner as T


# ------------------------------------------------------------------------------------------------ 2. degraded items can recover

def _ledger():
    return LN.RecoveringLedger(RT.RetirementPolicy(min_n=8, recover_min_n=16))


def _weeks(start, n):
    return [d.date().isoformat() for d in pd.date_range(start, periods=n, freq="7D")]


def _degraded(led, kid="K1"):
    led.register(kid, "2009-01-02")
    led.transition(kid, RT.State.DEGRADED, "2009-03-06", "DEGRADE", "planted dip")
    return kid


def test_a_degraded_item_with_strong_evidence_after_the_degrade_recovers_and_the_old_ledger_could_not():
    led = _ledger()
    kid = _degraded(led)
    d = _weeks("2009-03-13", 20)
    rng = np.random.default_rng(0)
    ev = RT.series_evidence(d, 0.02 + rng.normal(0, 0.01, 20), "2009-03-06", "2009-08-01")
    old = RT.RetirementLedger(RT.RetirementPolicy(min_n=8, recover_min_n=16))       # the old behaviour: no door back
    _degraded(old)
    assert old.attempt_recovery(kid, ev, "2009-08-01").to_state is None
    assert old.evaluate(kid, ev, "2009-08-01").to_state is None
    v = led.recover_degraded(kid, ev, "2009-08-01", apply=True)
    assert v.to_state is RT.State.ACTIVE and v.kind == "RECOVER_FULL"
    assert led.state(kid, "2009-08-02") is RT.State.ACTIVE and led.influence(kid, "2009-08-02") == 1.0
    assert led.history(kid)[-1].evidence["t"] >= led.policy.recover_t


def test_recovery_refuses_weak_thin_old_or_future_evidence_and_items_that_are_not_degraded():
    led = _ledger()
    kid = _degraded(led)
    rng = np.random.default_rng(1)
    d = _weeks("2009-03-13", 20)
    weak = RT.series_evidence(d, 0.002 + rng.normal(0, 0.02, 20), "2009-03-06", "2009-08-01")
    assert weak.t < led.policy.recover_t and led.recover_degraded(kid, weak, "2009-08-01").to_state is None   # null: noise stays down
    thin = RT.series_evidence(d[:10], [0.03] * 5 + [0.031] * 5, "2009-03-06", "2009-08-01")
    assert "insufficient" in led.recover_degraded(kid, thin, "2009-08-01").reason
    old_d = _weeks("2009-01-09", 20)                                                  # starts before the degrade: cannot count
    stale = RT.series_evidence(old_d, 0.02 + rng.normal(0, 0.005, 20), "2009-01-02", "2009-08-01")
    assert "not after the degrade" in led.recover_degraded(kid, stale, "2009-08-01").reason
    empty = RT.series_evidence([], [], "2009-03-06", "2009-08-01")
    assert led.recover_degraded(kid, empty, "2009-08-01").to_state is None                          # empty case
    with pytest.raises(FirewallBreach):
        future = RT.Evidence(20, 0.02, 0.004, "2009-03-13", "2009-09-01")
        led.recover_degraded(kid, future, "2009-08-01")
    led2 = _ledger()
    led2.register("K2", "2009-01-02")
    assert "not degraded" in led2.recover_degraded("K2", RT.series_evidence(d, [0.03] * 20, "2009-03-06", "2009-08-01"), "2009-08-01").reason
    assert "not degraded" in led2.recover_degraded("K9", RT.series_evidence([], [], "2009-03-06", "2009-08-01"), "2009-08-01").reason


def test_the_learner_asks_for_recovery_of_a_degraded_item_and_the_object_follows_the_ledger():
    L = LN.LegitimateLearner(T.make_cfg(retire_window=8), code_hash_fn=lambda: "pinned-test-code")   # the door, not the window (F10: 16)
    pid, kid = "f0:q4", "K-planted"
    L._pid_of[kid], L._kid_of[pid] = pid, kid
    rng = np.random.default_rng(2)
    d = _weeks("2009-01-09", 30)
    vals = [(-0.01 if i < 8 else 0.02) + rng.normal(0, 0.005) for i in range(30)]  # a planted early dip, then a true effect
    L.retirement.register(kid, "2009-01-02")
    for i in range(8, 30):
        L._weekly[pid] = [(d[j], vals[j], 0.01, 8) for j in range(i + 1)]
        L._retire_check(kid, pid, 1, pd.Timestamp(d[i]) + pd.Timedelta(days=1))
    kinds = [t.kind for t in L.retirement.history(kid)]
    assert "DEGRADE" in kinds and kinds[-1] == "RECOVER_FULL", kinds                  # old code: stuck at DEGRADE for ever
    assert L.counters.get("recovered_from_degraded") == 1


# ------------------------------------------------------------------------------------------------ 3. identity harness sizing

def _panel(n_dates, n_tick=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2009-01-02", periods=n_dates, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, [f"T{i:02d}" for i in range(n_tick)]])
    X = pd.DataFrame({"f0": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(0.03 * (X["f0"] > 0.8) + rng.normal(0, 0.01, len(idx)), index=idx)
    return X, y


def _cell_rule(Xt, yt, Xe, seed):
    return (Xe["f0"].groupby(level=0).rank(pct=True) > 0.8).astype(float)


def test_episode_attack_is_sized_so_it_really_runs_on_the_learners_short_audit_window():
    X, y = _panel(8)
    ev = X.index.get_level_values(0) >= X.index.get_level_values(0).unique()[4]
    old = IDF.episode_substitution(X[ev], y[ev], 100)
    assert old.changed_frac == 0.0                                                   # old: one 20-date block, nothing moved
    kw = LH.episode_attack_kwargs(4)
    assert kw == {"episode_substitution": {"block": 2}}
    assert IDF.episode_substitution(X[ev], y[ev], 100, **kw["episode_substitution"]).changed_frac > 0.5
    base = IDF.IdentityHarness(_cell_rule, seed=0, boot=50, min_dates=2, modes=("eval",)).run(X[~ev], y[~ev], X[ev], y[ev])
    assert not base.passed and any("no-op" in f.check for f in base.findings if f.is_fail)
    fixed = IDF.IdentityHarness(_cell_rule, seed=0, boot=50, min_dates=2, modes=("eval",), attack_kwargs=kw).run(X[~ev], y[~ev], X[ev], y[ev])
    assert not any(f.is_fail for f in fixed.findings)
    assert fixed.passed                                                               # an identity-free true rule survives


def test_the_resized_attack_still_catches_a_memoriser_and_degenerate_windows_are_left_alone():
    X, y = _panel(8, seed=3)
    ev = X.index.get_level_values(0) >= X.index.get_level_values(0).unique()[4]
    kw = LH.episode_attack_kwargs(4)

    def stamp_lookup(Xt, yt, Xe, seed):                                                # planted defect: keys on (date, ticker)
        table = {k: v for k, v in zip(Xe.index, y.reindex(Xe.index).to_numpy())}
        return pd.Series([table[k] for k in Xe.index], index=Xe.index)
    rep = IDF.IdentityHarness(stamp_lookup, seed=0, boot=50, min_dates=2, modes=("eval",), attack_kwargs=kw).run(X[~ev], y[~ev], X[ev], y[ev])
    assert not rep.passed and "episode_substitution" in {v.kind for v in rep.verdicts if v.status == "COLLAPSE"}
    assert LH.episode_attack_kwargs(1) == {} and LH.episode_attack_kwargs(0) == {}
    assert LH.episode_attack_kwargs(400) == {"episode_substitution": {"block": 20}}   # long windows keep the default


# ------------------------------------------------------------------------------------------------ 1. shadow skill monitoring

@pytest.fixture(scope="module")
def small_world():
    return PW.make_world(T.mini_spec(26, 30), seed=4)


@pytest.fixture(scope="module")
def pair(small_world):
    """The same short episode learned with and without shadow monitoring (the old behaviour)."""
    out = {}
    for flag in (True, False):
        L = LN.LegitimateLearner(T.make_cfg(seed=4, shadow_monitoring=flag), code_hash_fn=lambda: "pinned-test-code")
        out[flag] = T.run(L, LN.WorldFeed(small_world), range(len(small_world.dates)))
    return out


def test_shadow_predictions_give_the_skill_monitor_evidence_before_anything_is_in_production(pair):
    new, old = pair[True], pair[False]
    assert new._pid_of, "the planted world must produce held knowledge for this test to mean anything"
    assert new.counters.get("shadow_only_predictions", 0) > 0
    n_new = new.monitor.status(pd.Timestamp("2011-01-01"))["n"]
    n_old = old.monitor.status(pd.Timestamp("2011-01-01"))["n"]
    assert n_new > n_old, (n_new, n_old)                                             # old: the gate never saw the shadow period
    if not old.production_ids():
        assert n_old == 0


def test_shadow_retrieval_never_carries_a_decision(pair):
    L = pair[True]
    champions = {kid for kid in L._pid_of for v in L.store.history(kid) if v.promotion == Promotion.CHAMPION}
    for d in L.decisions:
        assert set(d.knowledge_ids) <= champions                                     # only production knowledge carried weight
    if not champions:
        assert all(d.action == "ABSTAIN" for d in L.decisions)
    assert L.shadow_retriever.monitor is None


def test_a_frozen_learner_and_a_probe_register_no_shadow_predictions(pair, small_world):
    L = pair[True]
    before = (L.counters.get("predictions", 0), len(L.monitor._pred))
    L.freeze()
    feed = LN.WorldFeed(PW.reidentify(small_world, 9, tickers=True, shift_years=3).world)
    LN.score_decisions(L, feed, range(3), "probe")
    assert (L.counters.get("predictions", 0), len(L.monitor._pred)) == before


def test_truth_trace_names_a_stage_for_every_planted_item(pair, small_world):
    L = pair[True]
    feed = LN.WorldFeed(PW.reidentify(small_world, 10, tickers=True, shift_years=3).world)
    probe = LN.score_decisions(L, feed, range(4), "probe")
    rows = LN.truth_trace(small_world, L, probe, feed.world.dates[0])
    assert [r["item"] for r in rows] == [it.item_id for it in small_world.spec.items]
    assert all(r["stage"] in LN.TRACE_STAGES for r in rows)
    for r in rows:
        if r["born"]:
            assert r["stage"] != "never found" and r["role"] is not None
        else:
            assert r["stage"] == "never found" and r["why"]
    fresh = T.new_learner().freeze()                                                  # empty case: nothing learned, nothing found
    empty = LN.truth_trace(small_world, fresh, LN.score_decisions(fresh, feed, range(2), "none"), feed.world.dates[0])
    assert {r["stage"] for r in empty} == {"never found"}


def test_truth_trace_explains_a_multi_condition_item_it_cannot_search():
    spec = PW.WorldSpec("pair", (PW.Item("pair", PW.PAIR, (("f0", 4), ("f1", 4)), 0.02),), weeks=12, stocks=30, n_feat=6,
                        discovery_end=6, confirm_end=9).check()
    w = PW.make_world(spec, seed=1)
    L = T.new_learner().freeze()
    feed = LN.WorldFeed(w)
    rows = LN.truth_trace(w, L, LN.score_decisions(L, feed, range(1), "none"), w.dates[0])
    assert rows[0]["stage"] == "never found" and "single-cell" in rows[0]["why"]


# ------------------------------------------------------------------------------------------------ acceptance summary accounting

def _row(world, improved, imp, seed=1, changed=0):
    return {"seed": seed, "world": world, "improved": improved, "verdict": "IMPROVED" if improved else "NO CHANGE", "improvement_vs_none": imp,
            "improvement_vs_control": imp, "identity_invariant": True, "changed_without_knowledge": 0, "changed": changed, "production": 0,
            "trace": [{"planted_sign": 1, "stage": "acted on, right side"}, {"planted_sign": 0, "stage": "never found"}]}


def test_acceptance_summary_counts_a_false_improvement_on_the_null_world():
    import acceptance_mini as AM
    s = AM.summarise([_row("planted", True, 0.01), _row("planted", False, None, 2), _row("null", True, 0.004, 1, 12)])
    assert s["n_seeds"] == 2 and s["n_improved"] == 1 and s["null_false_improvements"] == 1 and s["null_changed_decisions"] == 12
    assert s["mean_improvement_vs_none"] == pytest.approx(0.01) and s["mean_improvement_vs_none_all_seeds"] == pytest.approx(0.005)
    assert s["planted_true_items_by_stage"] == {"acted on, right side": 2}
    e = AM.summarise([])
    assert e["n_seeds"] == 0 and e["mean_improvement_vs_none"] is None and e["null_false_improvements"] == 0
