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


def replay(seed=1, n=260, meta=None, cfg=CFG, fn=momentum, bps=5.0):
    closes, opens = world(n, seed)
    return A.replay(cfg, snaps(closes, fn), closes, bps, {}, adaptive=True, meta=meta, opens=opens)


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
    S = replay(seed=1, cfg={**CFG, "k": 5})
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
    for seed in (1, 3):
        S = replay(seed=seed)
        sw = switches(S)
        assert sw, "the planted worlds are known to adapt"
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
    S = replay(seed=1, meta={"max_switches": 0})
    assert switches(S) == [] and S.adapter.cfg_diff() == {}
    assert any(r["action"] == "hold" and "budget" in str(r["detail"]["why"]) for r in S.adapter.audit)


def test_huge_min_improve_means_no_switches_at_all():
    assert switches(replay(seed=1, meta={"min_improve": 1.0})) == []


def test_a_world_where_nothing_differs_produces_no_switch():
    """Constant signal + flat prices: every neighbour picks the same names, so no improvement can be 'significant'."""
    closes, opens = world(120, seed=2)
    flat = closes * 0 + 20.0
    S = A.replay(CFG, snaps(flat, lambda d, c: np.zeros(c.shape[1])), flat, 0.0, {}, adaptive=True, opens=flat)
    assert switches(S) == [] and S.adapter.log == []


# ---------------------------------------------------------------- rapid revert
def test_revert_returns_to_default_after_a_deviation_stops_working():
    S = replay(seed=1, meta={"revert_drop": -1.0})          # -1.0: 'any two periods' count as a failed deviation
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
    ad.recent = [-0.05]                                       # one period already lost 5%
    closes, _ = world(40)
    sn = snaps(closes, momentum)
    d0, d1 = closes.index[10], closes.index[15]
    # force the deviation's excess over default to be a clear loss: make the default pick the winners, the deviation the losers
    ad.prev = (d0, sn[str(d0.date())])
    ad._period_return = lambda names, a, b, c: (-0.06 if len(names) == 3 else 0.0)
    ad._learn(d0, sn[str(d0.date())], d1, closes.loc[:d1], {}, [])
    assert ad.log and ad.log[-1]["action"] == "revert" and "one week" in ad.log[-1]["why"] and ad.cfg["k"] == 2


def test_revert_lockout_blocks_readopting_the_value_that_failed():
    S = replay(seed=1, meta={"revert_drop": -1.0, "revert_lockout": 10_000})
    pairs = [(a["knob"], a["to"]) for a in switches(S)]
    assert len(pairs) == len(set(pairs)), "a locked (knob, value) came back"
    free = replay(seed=1, meta={"revert_drop": -1.0})
    fp = [(a["knob"], a["to"]) for a in switches(free)]
    assert len(fp) != len(set(fp))                              # control: without the lockout the same value IS re-adopted


# ---------------------------------------------------------------- audit trail
def test_every_decision_is_audited_and_the_chain_verifies():
    S = replay(seed=1)
    ad = S.adapter
    assert len(ad.audit) == 52 and ad.verify_audit() == (True, None)
    assert {r["action"] for r in ad.audit} <= {"hold", "switch", "revert", "frozen_knobs", "preseason_set"}
    assert ad.audit_matches_cfg()
    assert S.result()["audit_digest"] == ad.audit_digest() and S.result()["audit_len"] == 52
    fr = ad.audit_frame()
    assert list(fr["i"]) == list(range(len(fr)))


def test_editing_a_record_is_detected_at_that_record():
    ad = replay(seed=1).adapter
    t = copy.deepcopy(ad)
    t.audit[10]["detail"]["why"] = "nothing to see"
    ok, bad = t.verify_audit()
    assert not ok and bad == 10


def test_dropping_or_reordering_records_is_detected():
    ad = replay(seed=1).adapter
    t = copy.deepcopy(ad)
    del t.audit[7]
    assert t.verify_audit()[0] is False
    r = copy.deepcopy(ad)
    r.audit[3], r.audit[4] = r.audit[4], r.audit[3]
    assert r.verify_audit()[0] is False


def test_a_hand_edit_of_the_settings_breaks_the_audit_match():
    ad = replay(seed=1).adapter
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
    a, b = replay(seed=3), replay(seed=3)
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
    off = replay(seed=1, n=80)
    on = replay(seed=1, n=80, meta={"mem_use_dates": True})
    assert off.adapter.mem.date_now is None and on.adapter.mem.date_now is not None
    assert off.adapter.mem.p["mem_era_other"] == 1.0                            # neutral defaults: results unchanged by the new factors
    assert off.result()["year_return"] == on.result()["year_return"]


def test_lessons_from_a_real_replay_are_sanitised_and_typed():
    ad = replay(seed=1, n=120).adapter
    df = ad.mem.export_lessons(tickers=[f"T{i:02d}" for i in range(12)])
    assert len(df) > 100 and set(df["error_type"]) <= {"correct", "false_positive", "false_negative", "noise", "unscored"}
    assert df["source_experiment"].eq("adapter").all()
    assert "T00" not in df.to_json()


# ---------------------------------------------------------------- the detector inside the adapter
def test_detector_weight_is_zero_in_the_first_weeks_and_never_above_the_cap():
    S = replay(seed=1)
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
