"""Bible Phase 18: the weekly adapter. Guard rails (minimum evidence, cooling off, significance, one-step movement,
switch budget, rapid revert with lockout, hands-off after sealing), the per-knob state table, the hash-chained audit
trail (tamper, drop, hand-edited settings), replay determinism, and the Session's fill records (T13 / canon C33).
Planted cases: a world where a neighbour is really better (the adapter must move), one where nothing is (it must
not), a deviation that stops working (it must revert), and deliberate tampering (the audit must notice)."""
import copy

import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from engine.memory import CTX

CFG = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
       "pick": "top", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
       "trend_filter": None, "trend_gross": 0.0}
Z = np.zeros(len(CTX))


def world(n=260, seed=0, nt=12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-07", periods=n)
    tk = [f"T{i:02d}" for i in range(nt)]
    closes = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, nt)), axis=0)), index=idx, columns=tk)
    opens = closes.shift(1).fillna(closes.iloc[0]) * np.exp(rng.normal(0, 0.01, closes.shape))
    return closes, opens


def snaps(closes, fn):
    out = {}
    for d in closes.index:
        out[str(d.date())] = pd.DataFrame({"mu_raw": fn(d, closes.loc[:d]), "evidence": 0.5, "vol20": 0.02, "max20": 0.05,
                                           "log_dv": 18.0, "ev_red_flag": 0.0, "ev_offering": 0.0, "r5": 0.0, "m_vix": 0.5,
                                           "m_vix_term": 0.9, "m_spy_ma200": 1.05},
                                          index=pd.Index(closes.columns, name="ticker"))
    return out


def momentum(d, c):
    return c.iloc[-5:].pct_change().sum().values if len(c) > 5 else np.zeros(c.shape[1])


_CACHE = {}


def replay(seed=3, n=260, meta=None, cfg=CFG, fn=momentum, bps=5.0, fresh=False):
    """A full adaptive replay; memoised (a replay is ~5 s and deterministic). Tests that mutate deep-copy first.
    seed 3 is a world in which the adapter switches k 2->3, is reverted, and switches again."""
    key = (seed, n, repr(sorted((meta or {}).items())), repr(sorted(cfg.items())), fn.__name__, bps)
    if fresh or key not in _CACHE:
        closes, opens = world(n, seed)
        S = A.replay(cfg, snaps(closes, fn), closes, bps, {}, adaptive=True, meta=meta, opens=opens)
        if fresh:
            return S
        _CACHE[key] = S
    return _CACHE[key]


def switches(S):
    return [a for a in S.adapter.log if a["action"] == "switch"]


# ---------------------------------------------------------------- grid helpers
def test_one_step_accepts_only_grid_neighbours():
    assert A.one_step("k", 2, 3) and A.one_step("k", 3, 2) and A.one_step("w_model", 0.5, 0.7)
    assert not A.one_step("k", 2, 4) and not A.one_step("k", 2, 2) and not A.one_step("k", 5, 6)   # off-grid start
    assert not A.one_step("nope", 1, 2)


def test_validate_cfg_rejects_off_grid_and_ignores_unset_knobs():
    A.validate_cfg({"k": 3, "w_model": 0.5}, ["k", "w_model", "w_move"])            # w_move unset: fine
    with pytest.raises(A.GuardRailError):
        A.validate_cfg({"k": 5}, ["k"])


def test_off_grid_start_freezes_the_knob_and_says_so_in_the_audit():
    S = replay(seed=3, cfg={**CFG, "k": 5})
    ad = S.adapter
    assert ad.frozen == ["k"] and ad.audit[0]["action"] == "frozen_knobs"
    assert all(s["knob"] != "k" for s in switches(S))


# ---------------------------------------------------------------- hands off
def test_manual_change_allowed_before_the_first_decision_and_refused_after():
    ad = A.Adapter(CFG)
    ad.set_knob("k", 3, who="preseason study")
    assert ad.cfg["k"] == 3 and ad.default["k"] == 3 and ad.audit[-1]["action"] == "preseason_set"
    with pytest.raises(A.GuardRailError):
        ad.set_knob("k", 5)                                                    # off the grid, even before sealing
    closes, _ = world(30)
    sn = snaps(closes, momentum)
    d = closes.index[10]
    ad.step(d, sn[str(d.date())], closes.loc[:d], {}, [])
    assert ad.sealed
    with pytest.raises(A.ManualInterventionError):
        ad.set_knob("k", 4, who="the owner, mid-year")
    assert ad.cfg["k"] == 3


def test_time_going_backwards_is_refused():
    ad = A.Adapter(CFG)
    closes, _ = world(40)
    sn = snaps(closes, momentum)
    d1, d0 = closes.index[20], closes.index[10]
    ad.step(d1, sn[str(d1.date())], closes.loc[:d1], {}, [])
    with pytest.raises(A.GuardRailError):
        ad.step(d0, sn[str(d0.date())], closes.loc[:d0], {}, [])
    with pytest.raises(A.TimeFence):
        ad.step(d0, sn[str(d0.date())], closes.loc[:d1], {}, [])


# ---------------------------------------------------------------- a planted better neighbour
def planted_memory(ad, gain, weeks=12, arm=("knob", "k", 2, 3)):
    for w in range(weeks):
        ad.mem.record(arm, w, Z, gain + 0.0005 * ((-1) ** w))
    ad.weeks = weeks


def test_candidate_table_scores_a_planted_better_neighbour_and_blocks_a_thin_one():
    ad = A.Adapter(CFG)
    planted_memory(ad, 0.02)
    cand = {(c["knob"], c["to"]): c for c in ad._evaluate_candidates(Z, {})}
    assert cand[("k", 3)]["z"] > 2 and cand[("k", 3)]["blocked"] == []
    assert cand[("k", 1)]["blocked"] == ["thin evidence"]                      # never measured
    row = ad.knob_table().set_index("knob").loc["k"]
    assert row["neighbor"] == 3 and row["evidence"] > 0.01 and row["confidence"] > 2 and row["n_eff"] >= 4


def test_min_improve_floor_blocks_a_significant_but_tiny_gain():
    ad = A.Adapter(CFG, {"min_improve": 0.01})
    planted_memory(ad, 0.002)
    c = {(c["knob"], c["to"]): c for c in ad._evaluate_candidates(Z, {})}[("k", 3)]
    assert c["z"] > 2 and c["blocked"] == ["improvement below floor"]


def test_no_change_before_min_weeks_and_none_during_cooldown():
    ad = A.Adapter(CFG, {"min_weeks": 6, "cooldown": 4})
    ad.weeks = 3
    assert "minimum evidence" in ad._global_gate()
    ad.weeks, ad.since_switch = 8, 2
    assert "cooling off" in ad._global_gate()
    ad.since_switch = 4
    assert ad._global_gate() is None
    ad.meta["max_switches"], ad.n_switch = 1, 1
    assert ad._global_gate() == "switch budget spent"


def test_knob_cooldown_applies_only_to_the_knob_that_moved():
    ad = A.Adapter(CFG, {"knob_cooldown": 5})
    planted_memory(ad, 0.02)
    ad.knobs["k"].last_switch = ad.weeks - 1
    cand = {(c["knob"], c["to"]): c for c in ad._evaluate_candidates(Z, {})}
    assert "knob cooling off" in cand[("k", 3)]["blocked"] and ad.knobs["k"].cooldown == 4
    assert "knob cooling off" not in cand[("pool_q", 0.8)]["blocked"]


def test_every_switch_is_one_grid_step_and_at_least_switch_z_confident():
    sw = switches(replay())
    assert sw, "the planted world is known to adapt"
    for a in sw:
        assert A.one_step(a["knob"], a["from"], a["to"]) and a["z"] > 2.0


def test_illegal_switch_is_refused_by_the_rail_itself():
    """Bypass the candidate filter and hand the adapter a two-step jump: the last line of defence must stop it."""
    ad = A.Adapter(CFG, {"min_weeks": 1, "cooldown": 0})
    ad.weeks, ad.since_switch = 5, 99
    ad._evaluate_candidates = lambda ctx, gain: [{"knob": "k", "from": 2, "to": 4, "mean": 0.05, "se": 0.001, "n_eff": 9.0,
                                                  "z": 50.0, "blocked": []}]
    closes, _ = world(40)
    sn = snaps(closes, momentum)
    d0, d1 = closes.index[10], closes.index[15]
    ad.prev = (d0, sn[str(d0.date())])
    with pytest.raises(A.GuardRailError):
        ad._learn(d0, sn[str(d0.date())], d1, closes.loc[:d1], {}, [])
    assert ad.cfg["k"] == 2


def test_switch_budget_zero_freezes_the_settings_and_gate_records_why():
    S = replay(seed=3, meta={"max_switches": 0})
    assert switches(S) == [] and S.adapter.cfg_diff() == {}
    assert any(r["action"] == "hold" and "budget" in str(r["detail"]["why"]) for r in S.adapter.audit)


def test_huge_min_improve_means_no_switches_at_all():
    assert switches(replay(seed=3, meta={"min_improve": 1.0})) == []


def test_a_world_where_nothing_differs_produces_no_switch():
    """Constant signal + flat prices: every neighbour picks the same names, so no improvement can be 'significant'."""
    closes, opens = world(120, seed=2)
    flat = closes * 0 + 20.0
    S = A.replay(CFG, snaps(flat, lambda d, c: np.zeros(c.shape[1])), flat, 0.0, {}, adaptive=True, opens=flat)
    assert switches(S) == [] and S.adapter.log == []


# ---------------------------------------------------------------- rapid revert
def test_revert_returns_to_default_after_a_deviation_stops_working():
    S = replay()
    log = S.adapter.log
    assert any(a["action"] == "revert" for a in log)
    i = next(i for i, a in enumerate(log) if a["action"] == "revert")
    assert log[i - 1]["action"] == "switch"
    aud = [r for r in S.adapter.audit if r["action"] == "revert"][0]
    assert aud["detail"]["reverted"] and aud["cfg_diff"] == {}
    assert S.adapter.knobs[list(aud["detail"]["reverted"])[0]].reverts >= 1


def test_revert_fast_fires_on_one_bad_week_where_the_two_period_rule_would_wait():
    ad = A.Adapter(CFG, {"revert_fast": 0.02})
    ad.cfg["k"] = 3
    ad.recent = []                                            # no history: the two-period rule cannot fire
    closes, _ = world(40)
    sn = snaps(closes, momentum)
    d0, d1 = closes.index[10], closes.index[15]
    # force the deviation's excess over default to be a clear loss: make the default pick the winners, the deviation the losers
    ad.prev = (d0, sn[str(d0.date())])
    ad._period_return = lambda names, a, b, c: (-0.06 if len(names) == 3 else 0.0)
    ad._learn(d0, sn[str(d0.date())], d1, closes.loc[:d1], {}, [])
    assert ad.log and ad.log[-1]["action"] == "revert" and "one week" in ad.log[-1]["why"] and ad.cfg["k"] == 2


def test_revert_lockout_blocks_readopting_the_value_that_failed():
    S = replay(seed=3, meta={"revert_lockout": 10_000})
    pairs = [(a["knob"], a["to"]) for a in switches(S)]
    assert len(pairs) == len(set(pairs)), "a locked (knob, value) came back"
    fp = [(a["knob"], a["to"]) for a in switches(replay())]
    assert len(fp) != len(set(fp))                              # control: without the lockout the same value IS re-adopted


# ---------------------------------------------------------------- audit trail
def test_every_decision_is_audited_and_the_chain_verifies():
    S = replay(seed=3)
    ad = S.adapter
    assert len(ad.audit) == 52 and ad.verify_audit() == (True, None)
    assert {r["action"] for r in ad.audit} <= {"hold", "switch", "revert", "frozen_knobs", "preseason_set"}
    assert ad.audit_matches_cfg()
    assert S.result()["audit_digest"] == ad.audit_digest() and S.result()["audit_len"] == 52
    fr = ad.audit_frame()
    assert list(fr["i"]) == list(range(len(fr)))


def test_editing_a_record_is_detected_at_that_record():
    ad = replay(seed=3).adapter
    t = copy.deepcopy(ad)
    t.audit[10]["detail"]["why"] = "nothing to see"
    ok, bad = t.verify_audit()
    assert not ok and bad == 10


def test_dropping_or_reordering_records_is_detected():
    ad = replay(seed=3).adapter
    t = copy.deepcopy(ad)
    del t.audit[7]
    assert t.verify_audit()[0] is False
    r = copy.deepcopy(ad)
    r.audit[3], r.audit[4] = r.audit[4], r.audit[3]
    assert r.verify_audit()[0] is False


def test_a_hand_edit_of_the_settings_breaks_the_audit_match():
    ad = copy.deepcopy(replay().adapter)
    assert ad.audit_matches_cfg()
    ad.cfg["k"] = 12 if ad.cfg["k"] != 12 else 16                # changed outside step(), leaving no trace
    assert not ad.audit_matches_cfg()


def test_reconstruct_cfg_replays_switch_revert_and_preseason():
    default = {"k": 2, "pool_q": 0.7, "ew": {"a": 1}}
    audit = [{"action": "preseason_set", "detail": {"knob": "k", "value": 3}},
             {"action": "switch", "detail": {"knob": "pool_q", "to": 0.8}},
             {"action": "hold", "detail": {}},
             {"action": "switch", "detail": {"knob": "k", "to": 4}}]
    assert A.reconstruct_cfg(default, audit) == {"k": 4, "pool_q": 0.8}
    assert A.reconstruct_cfg(default, audit + [{"action": "revert", "detail": {}}]) == {"k": 3, "pool_q": 0.7}
    assert A.reconstruct_cfg(default, []) == {"k": 2, "pool_q": 0.7}


# ---------------------------------------------------------------- determinism is sacred
def test_replay_gives_identical_adaptations_audit_and_memory():
    a, b = replay(), replay(fresh=True)                     # b is a genuinely second run, not the cached object
    assert a.adapter.audit_digest() == b.adapter.audit_digest() and A.audits_equal(a.adapter.audit, b.adapter.audit)
    assert a.adapter.mem.fingerprint() == b.adapter.mem.fingerprint()
    assert a.adapter.det.fingerprint() == b.adapter.det.fingerprint()
    assert a.result() == b.result()


def test_a_different_world_gives_a_different_digest():
    assert replay(seed=3).adapter.audit_digest() != replay(seed=1).adapter.audit_digest()


def test_future_scramble_leaves_the_adapters_earlier_audit_unchanged():
    closes, opens = world(200, seed=4)
    sn = snaps(closes, momentum)
    cut = closes.index[100]
    S1 = A.replay(CFG, sn, closes, 5.0, {}, adaptive=True, opens=opens)
    S2 = A.replay(CFG, sn, closes, 5.0, {}, adaptive=True, opens=opens, scramble_after=cut, seed=9)
    before = lambda S: [(r["date"], r["action"], r["hash"]) for r in S.adapter.audit if r["date"] and pd.Timestamp(r["date"]) <= cut]
    a, b = before(S1), before(S2)
    assert a and a[:-1] == b[:-1]        # the record AT the cut day may differ: it scores the week ending on scrambled prices


def test_memory_uses_dates_only_when_asked():
    off = replay(seed=3, n=80)
    on = replay(seed=3, n=80, meta={"mem_use_dates": True})
    assert off.adapter.mem.date_now is None and on.adapter.mem.date_now is not None
    assert off.adapter.mem.p["mem_era_other"] == 1.0                            # neutral defaults: results unchanged by the new factors
    assert off.result()["year_return"] == on.result()["year_return"]


def test_lessons_from_a_real_replay_are_sanitised_and_typed():
    ad = replay(seed=3, n=120).adapter
    df = ad.mem.export_lessons(tickers=[f"T{i:02d}" for i in range(12)])
    assert len(df) > 100 and set(df["error_type"]) <= {"correct", "false_positive", "false_negative", "noise", "unscored"}
    assert df["source_experiment"].eq("adapter").all()
    assert "T00" not in df.to_json()


# ---------------------------------------------------------------- the detector inside the adapter
def test_detector_weight_is_zero_in_the_first_weeks_and_never_above_the_cap():
    S = replay(seed=3)
    rows = S.adapter.missed
    assert rows and all(r["detector_weight"] <= 0.5 for r in rows)
    assert all(r["detector_weight"] == 0.0 for r in rows[:5])
    assert set(rows[0]) >= {"date", "winners", "missed", "caught", "missed_minus_picked", "detector_skill", "detector_weight", "types"}


# ---------------------------------------------------------------- Session.fills (fill audit feed)
def test_every_fill_is_at_the_next_session_open_of_its_decision():
    closes, opens = world(80, seed=6)
    opens = closes * np.exp(np.random.default_rng(1).normal(0, 0.05, closes.shape))     # opens clearly unlike closes
    S = A.replay(CFG, snaps(closes, momentum), closes, 0.0, {}, opens=opens)
    days = [str(d.date()) for d in closes.index]
    assert S.fills and len(S.fills) == len(S.orders)
    for f in S.fills:
        assert days.index(f["fill_date"]) == days.index(f["decision_date"]) + 1
        assert f["fill_price"] == pytest.approx(float(opens.loc[f["fill_date"], f["ticker"]]))
        assert abs(f["fill_price"] - float(closes.loc[f["fill_date"], f["ticker"]])) > 1e-9
    assert {"order_id", "ticker", "decision_date", "fill_date", "fill_price", "dv", "reason"} == set(S.fills[0])


def test_order_ids_are_unique_and_match_the_orders_list():
    closes, opens = world(80, seed=6)
    S = A.replay(CFG, snaps(closes, momentum), closes, 0.0, {}, opens=opens)
    ids = [f["order_id"] for f in S.fills]
    assert len(ids) == len(set(ids))
    for f, o in zip(S.fills, S.orders):
        assert (f["fill_date"], f["ticker"], f["dv"], f["reason"]) == o
        assert f["order_id"].startswith(f"{f['decision_date']}:{f['ticker']}:")


def test_decision_dates_in_fills_are_dates_the_session_actually_decided():
    closes, opens = world(80, seed=6)
    S = A.replay(CFG, snaps(closes, momentum), closes, 0.0, {}, opens=opens)
    decided = {d for d, _ in S.decisions}
    assert {f["decision_date"] for f in S.fills} <= decided


def test_brake_orders_carry_the_day_the_brake_was_decided():
    cfg = {**CFG, "brake": 0.001}
    closes, opens = world(80, seed=8)
    S = A.replay(cfg, snaps(closes, momentum), closes, 0.0, {}, opens=opens)
    braked = [f for f in S.fills if f["reason"].startswith("weekly brake")]
    assert braked
    days = [str(d.date()) for d in closes.index]
    for f in braked:
        assert days.index(f["fill_date"]) == days.index(f["decision_date"]) + 1


def test_no_snapshots_means_no_fills_and_a_non_adaptive_session_has_no_audit():
    closes, opens = world(40)
    S = A.replay(CFG, {}, closes, 0.0, {}, opens=opens)
    assert S.fills == [] and S.result()["audit_digest"] is None and S.result()["audit_len"] == 0


def test_result_does_not_depend_on_the_interpreters_hash_seed():
    """Session._trade used to order tied names by set iteration, so the same inputs gave different adaptations in
    different processes. Run the same replay under two PYTHONHASHSEEDs: the audit digest must be identical."""
    import os, subprocess, sys
    code = ("import sys; sys.path.insert(0, 'tests'); sys.path.insert(0, '.'); import test_adapter as T; "
            "S = T.replay(seed=3, fresh=True); print(S.adapter.audit_digest(), round(S.result()['year_return'], 9))")
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONHASHSEED": h}) for h in ("0", "12345")]
    outs = [p.communicate(timeout=80)[0].strip() for p in procs]
    assert outs[0] == outs[1] and len(outs[0].split()[0]) == 64


# ---------------------------------------------------------------- timeline dial hook (off by default)
def _session(meta, n=110, seed=3):
    closes, opens = world(n, seed)
    return A.replay(CFG, snaps(closes, momentum), closes, 5.0, {}, adaptive=True, meta=meta, opens=opens)


def test_dial_off_leaves_every_decision_order_and_fill_identical():
    base, off = replay(n=260), replay(meta={"dial_on": False}, n=260, fresh=True)
    assert base.decisions == off.decisions and base.orders == off.orders and base.fills == off.fills
    assert base.result() == off.result() and off.dial_log == [] and off.exposure == 1.0


def test_dial_on_scales_rebalance_weights_by_the_exposure_and_logs_each_step():
    from engine import timeline
    fixed = timeline.DialParams(k_bounds=(2, 2), pool_bounds=(0.7, 0.7), exposure_bounds=(0.5, 0.5))
    off = _session({}, n=60)
    on = _session({"dial_on": True, "dial_params": fixed}, n=60)
    assert on.dial_log and all(x[1] == 2 and x[2] == 0.5 for x in on.dial_log)
    first = lambda S: sum(o[2] for o in S.orders if o[0] == S.orders[0][0] and o[2] > 0)
    assert first(on) == pytest.approx(0.5 * first(off), rel=1e-6)          # same names (k, pool_q pinned), half the money
    assert [d for d, _ in on.decisions][:3] == [d for d, _ in off.decisions][:3]


def test_dial_only_applies_to_an_adaptive_session():
    closes, opens = world(40)
    S = A.replay(CFG, snaps(closes, momentum), closes, 0.0, {}, adaptive=False, meta={"dial_on": True}, opens=opens)
    assert S.dial_on is False and S.dial_log == []


def test_meta_default_lists_the_detector_knobs_with_their_effective_values():
    assert A.META_DEFAULT["det_max"] == 0.5 and A.META_DEFAULT["det_min_weeks"] == 6 and A.META_DEFAULT["dial_on"] is False
    a = A.Adapter(CFG)
    assert a.det.meta["det_max"] == 0.5 and a.det.meta["det_min_weeks"] == 6


# ---------------------------------------------------------------- reporting and determinism gate
def test_adaptation_report_counts_actions_spells_and_integrity_on_a_real_run():
    rep = A.adaptation_report(replay().adapter)
    assert rep["actions"]["switch"] == 2 and rep["reverts"] == 2 and rep["revert_rate"] == 1.0
    assert rep["switches_by_knob"] == {"k": 2} and len(rep["spells"]) == 2 and all(s[2] == "revert" for s in rep["spells"])
    assert rep["chain_ok"] and rep["settings_match_audit"] and rep["weeks_off_default"] > 0
    assert 0 < rep["share_off_default"] < 1 and rep["mean_spell_weeks"] > 0
    assert "minimum evidence" in rep["hold_reasons"] and rep["knobs"].shape[0] == 6
    assert len(rep["digest"]) == 64 and len(rep["memory_fingerprint"]) == 64


def test_deviation_spells_and_hold_reasons_on_a_hand_built_trail():
    au = [{"week": 1, "cfg_diff": {}, "action": "hold", "detail": {"why": "cooling off: 1 < 3 weeks"}},
          {"week": 2, "cfg_diff": {"k": 3}, "action": "switch", "detail": {}},
          {"week": 3, "cfg_diff": {"k": 3}, "action": "hold", "detail": {"why": "no neighbour cleared the bar"}},
          {"week": 4, "cfg_diff": {}, "action": "revert", "detail": {}},
          {"week": 5, "cfg_diff": {"k": 2}, "action": "switch", "detail": {}}]
    assert A.deviation_spells(au) == [(2, 4, "revert"), (5, 5, "open")]
    assert A.hold_reasons(au) == {"cooling off": 1, "no neighbour cleared the bar": 1}
    assert A.deviation_spells([]) == [] and A.hold_reasons([]) == {}


def test_replay_check_passes_a_deterministic_pipeline_and_fails_a_nondeterministic_one(monkeypatch):
    closes, opens = world(70, seed=3)
    sn = snaps(closes, momentum)
    ok = A.replay_check(CFG, sn, closes, 5.0, {}, opens=opens)
    assert ok["identical"] and ok["differs"] == [] and len(ok["digest"]) == 64
    calls = {"n": 0}
    real_pick = A.pick

    def flaky(p, cfg, held, divs, det=None):
        out = real_pick(p, cfg, held, divs, det)
        calls["n"] += 1
        return out.iloc[1:] if calls["n"] % 40 == 0 else out              # drops a name once, differently on each run
    monkeypatch.setattr(A, "pick", flaky)
    bad = A.replay_check(CFG, sn, closes, 5.0, {}, opens=opens)
    assert not bad["identical"] and "decisions" in bad["differs"]


def test_rail_study_shows_what_each_rail_does_on_the_same_window():
    closes, opens = world(190, seed=3)
    sn = snaps(closes, momentum)
    t = A.rail_study(CFG, sn, closes, 5.0, {}, {"no_switches": {"max_switches": 0}, "lockout": {"revert_lockout": 10_000}},
                     opens=opens).set_index("variant")
    assert list(t.index) == ["base", "no_switches", "lockout"]
    assert t.loc["base", "switches"] >= 1 and t.loc["no_switches", "switches"] == 0 and t.loc["no_switches", "weeks_off_default"] == 0
    assert t.loc["lockout", "switches"] <= t.loc["base", "switches"]
    assert t["digest"].nunique() >= 2


# ---------------------------------------------------------------- checkpoint / resume
def _drive(ad, closes, sn, days, held=()):
    for d in days:
        ad.step(d, sn[str(d.date())], closes.loc[:d], {}, list(held))
    return ad


def _fridays(closes):
    idx = closes.index
    return [d for i, d in enumerate(idx[:-1]) if idx[i + 1].isocalendar().week != d.isocalendar().week]


def test_a_resumed_adapter_continues_exactly_like_an_uninterrupted_one():
    closes, _ = world(230, seed=3)
    sn = snaps(closes, momentum)
    days = _fridays(closes)
    whole = _drive(A.Adapter(CFG), closes, sn, days)
    assert whole.log, "the run must adapt for this test to mean anything"
    half = _drive(A.Adapter(CFG), closes, sn, days[:24])
    resumed = A.Adapter.from_snapshot(half.snapshot())
    _drive(resumed, closes, sn, days[24:])
    assert resumed.audit_digest() == whole.audit_digest() and resumed.log == whole.log
    assert resumed.mem.fingerprint() == whole.mem.fingerprint() and resumed.det.fingerprint() == whole.det.fingerprint()
    assert resumed.cfg == whole.cfg and resumed.verify_audit() == (True, None) and resumed.ledger.rows == whole.ledger.rows


def test_a_snapshot_is_a_copy_not_a_view():
    closes, _ = world(120, seed=3)
    sn = snaps(closes, momentum)
    ad = _drive(A.Adapter(CFG), closes, sn, _fridays(closes)[:10])
    snap = ad.snapshot()
    digest = ad.audit_digest()
    _drive(ad, closes, sn, _fridays(closes)[10:14])                      # the original moves on
    assert A.Adapter.from_snapshot(snap).audit_digest() == digest and ad.audit_digest() != digest


def test_resume_keeps_the_seal_locks_and_budget():
    closes, _ = world(230, seed=3)
    sn = snaps(closes, momentum)
    ad = _drive(A.Adapter(CFG, {"revert_lockout": 500}), closes, sn, _fridays(closes)[:40])
    r = A.Adapter.from_snapshot(ad.snapshot())
    assert r.sealed and r.locked == ad.locked and r.n_switch == ad.n_switch and r.weeks == ad.weeks
    with pytest.raises(A.ManualInterventionError):
        r.set_knob("k", 4)
    assert ad.locked, "a revert with lockout must have left a lock behind"


def test_detector_full_state_round_trips():
    from engine.missed_winners import MissedWinnerDetector
    import sys
    sys.path.insert(0, "tests")
    from test_missed_winners import weeks as mw_weeks
    d = MissedWinnerDetector({"det_step": 0.05})
    for _, p, f in mw_weeks(20, 1.0):
        d.learn(p, f)
    r = MissedWinnerDetector.from_full_state(d.full_state())
    assert r.fingerprint() == d.fingerprint() and r.weight() == d.weight() and r.history == d.history
    _, p, f = mw_weeks(1, 1.0, seed0=77)[0]
    d.learn(p, f); r.learn(p, f)
    assert r.fingerprint() == d.fingerprint()                             # and they keep learning identically
