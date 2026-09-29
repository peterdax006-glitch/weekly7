import os
import stat
import pytest
from engine import champion as CH

GOOD_GATES = {g: True for g in CH.REQUIRED_GATES}
EV = {"windows": ["w1", "w2", "w3"], "train_windows": ["t1"], "n_weeks": 150}
BLIND = {"passed": True, "windows": ["b1"]}


def rec(i, mean=0.07, inband=0.5, dd=-0.20, ww=-0.10, p05=-0.05, pos=0.6, **kw):
    m = {"mean_week": mean, "share_in_band": inband, "max_drawdown": dd, "worst_week": ww, "p05_week": p05,
         "positive_week_pct": pos, "catastrophic_losses": 0}
    return {"id": i, "metrics": m, "gates": dict(GOOD_GATES), "evidence": dict(EV), "blind": dict(BLIND),
            "reproduced": True, "basis": ["gates", "blind"], **kw}


def board(tmp_path, champion=None):
    b = CH.Board(tmp_path / "b.json")
    if champion:
        b.add_candidate(champion, "t")
        b.nominate(champion["id"], "t")
        b.s["champion"] = b.s["challengers"].pop(champion["id"])
    return b


def run(b, r):
    b.add_candidate(r, "t")
    b.nominate(r["id"], "t")
    return b.promote(r["id"], "t")


def test_first_champion_needs_gates_but_no_rival(tmp_path):
    b = board(tmp_path)
    assert run(b, rec("A"))["promote"] and b.roles()["champion"] == "A"


def test_better_challenger_replaces_and_champion_is_retired_not_deleted(tmp_path):
    b = board(tmp_path, rec("A", inband=0.4))
    res = run(b, rec("B", inband=0.6))
    assert res["promote"] and b.roles()["champion"] == "B"
    assert any(h["event"] == "retired" and h["id"] == "A" for h in b.s["history"])
    assert CH.Board(tmp_path / "b.json").roles()["champion"] == "B"


@pytest.mark.parametrize("mut,failed", [
    (lambda r: r["gates"].pop("parity"), "required_gates"),
    (lambda r: r["gates"].update(future_scramble=False), "required_gates"),
    (lambda r: r["evidence"].update(windows=["w1"]), "independent_evidence"),
    (lambda r: r["evidence"].update(train_windows=["w2"]), "independent_evidence"),
    (lambda r: r["evidence"].update(n_weeks=20), "independent_evidence"),
    (lambda r: r["metrics"].update(max_drawdown=-0.6), "risk_constraints"),
    (lambda r: r["metrics"].pop("worst_week"), "risk_constraints"),
    (lambda r: r.update(reproduced=None), "reproducibility"),
    (lambda r: r.update(blind={"passed": True, "windows": []}), "blind_validation"),
    (lambda r: r.pop("blind"), "blind_validation"),
])
def test_every_check_fails_closed(tmp_path, mut, failed):
    b = board(tmp_path, rec("A", inband=0.4))
    c = rec("B", inband=0.6)
    mut(c)
    res = run(b, c)
    assert not res["promote"] and not res["checks"][failed]["ok"]
    assert b.roles()["champion"] == "A" and "B" in b.roles()["challengers"]
    assert b.s["history"][-1]["event"] == "promotion_refused"


def test_newer_is_not_better(tmp_path):
    b = board(tmp_path, rec("A"))
    res = run(b, rec("B"))
    assert not res["promote"] and "no objective improved" in res["checks"]["priority_not_harmed"]["detail"]


def test_lower_tier_gain_cannot_buy_higher_tier_loss(tmp_path):
    b = board(tmp_path, rec("A", mean=0.07, pos=0.55))
    res = run(b, rec("B", mean=0.05, pos=0.80))
    assert not res["promote"] and "mean_week" in res["checks"]["priority_not_harmed"]["detail"]


def test_higher_tier_gain_may_cost_lower_tier(tmp_path):
    b = board(tmp_path, rec("A", inband=0.30, pos=0.70))
    assert run(b, rec("B", inband=0.60, pos=0.60))["promote"]


def test_challenger_missing_a_metric_is_harm(tmp_path):
    b = board(tmp_path, rec("A", inband=0.3))
    c = rec("B", inband=0.6)
    del c["metrics"]["p05_week"]
    assert not run(b, c)["promote"]


def test_check_exception_is_a_failure(tmp_path):
    b = board(tmp_path)
    c = rec("A")
    c["evidence"] = {"windows": 5}
    assert not run(b, c)["checks"]["independent_evidence"]["ok"]


def test_reject_needs_reason_and_is_final(tmp_path):
    b = board(tmp_path)
    b.add_candidate(rec("A"), "t")
    with pytest.raises(CH.ChampionError):
        b.reject("A", "", "t")
    b.reject("A", "overfit", "t")
    with pytest.raises(CH.ChampionError):
        b.nominate("A", "t")
    with pytest.raises(CH.ChampionError):
        b.add_candidate(rec("A"), "t")


def test_empty_board_and_unknown_ids(tmp_path):
    b = CH.Board(tmp_path / "x.json")
    assert b.roles()["champion"] is None
    with pytest.raises(CH.ChampionError):
        b.promote("ghost", "t")


BASE = {**{f: 1 for f in CH.BASELINE_FIELDS}, "direction_accuracy": None}


def test_baseline_frozen_once_and_tamper_detected(tmp_path):
    p = tmp_path / "base.json"
    h = CH.freeze_baseline(p, {**BASE, "max_drawdown": -0.4}, "2026-01-01", {"git": "abc"})
    assert CH.load_baseline(p)["sha256"] == h
    with pytest.raises(CH.ChampionError):
        CH.freeze_baseline(p, BASE, "later")
    os.chmod(p, stat.S_IWRITE)
    p.write_text(p.read_text().replace("-0.4", "-0.1"))
    with pytest.raises(CH.ChampionError):
        CH.load_baseline(p)


def test_baseline_incomplete_refused(tmp_path):
    with pytest.raises(CH.ChampionError) as e:
        CH.freeze_baseline(tmp_path / "b.json", {"max_drawdown": -0.3}, "t")
    assert "mover_accuracy" in str(e.value) and not (tmp_path / "b.json").exists()


def test_vs_baseline_delta(tmp_path):
    p = tmp_path / "b.json"
    CH.freeze_baseline(p, {**BASE, "max_drawdown": -0.4}, "t")
    d = CH.vs_baseline(CH.load_baseline(p), {"max_drawdown": -0.3, "unknown": 1.0, "costs": float("nan")})
    assert d["max_drawdown"] == pytest.approx(0.1) and d["unknown"] is None and d["costs"] is None


# ---- queue, tripwire, ledger ----
import numpy as np
from engine.experiment_memory import Space, TriedIndex


def two_champions(tmp_path):
    b = board(tmp_path, rec("A", inband=0.4))
    assert run(b, rec("B", inband=0.6))["promote"]
    return b


def feed(b, diffs, base=0.001):
    for i, d in enumerate(diffs):
        b.record_shadow_session(base + d, base, f"d{i}")


def test_pnl_alone_never_promotes(tmp_path):
    b = board(tmp_path)
    res = run(b, rec("A", basis=["pnl"]))
    assert not res["promote"] and not res["checks"]["not_pnl_alone"]["ok"] and b.roles()["champion"] is None
    assert not run(b, rec("B", basis=[]))["promote"]


def test_shadow_pending_confirm_and_rollback(tmp_path):
    b = two_champions(tmp_path)
    assert b.shadow_review("t")["status"] == "pending"
    feed(b, [0.001, 0.002, 0.0005, 0.0015, 0.002, 0.001, 0.0, 0.002, 0.001])
    assert b.shadow_review("t") == {"status": "pending", "sessions": 9, "needed": 10}
    feed(b, [0.001])
    r = b.shadow_review("t")
    assert r["status"] == "confirmed" and b.roles()["champion"] == "B"


def test_planted_underperformer_is_rolled_back(tmp_path):
    b = two_champions(tmp_path)
    feed(b, list(-0.004 + np.random.default_rng(1).normal(0, 0.001, 10)))
    r = b.shadow_review("t")
    assert r["status"] == "rolled_back" and r["restored"] == "A"
    assert b.roles()["champion"] == "A" and "B" in b.roles()["rejected"]
    assert "rolled back" in b.s["rejected"]["B"]["reason"]
    with pytest.raises(CH.ChampionError):
        b.rollback("t", "again")                                  # nothing left to restore


def test_noise_within_tolerance_does_not_roll_back(tmp_path):
    b = two_champions(tmp_path)
    feed(b, list(np.random.default_rng(2).normal(0, 0.003, 10)))
    assert b.shadow_review("t")["status"] == "confirmed" and b.roles()["champion"] == "B"


def test_shadow_rejects_missing_sessions_and_first_champion_has_none(tmp_path):
    b = two_champions(tmp_path)
    with pytest.raises(CH.ChampionError):
        b.record_shadow_session(float("nan"), 0.0, "d")
    (tmp_path / "solo").mkdir()
    b1 = board(tmp_path / "solo")
    run(b1, rec("A"))
    assert b1.shadow_review("t")["status"] == "no_predecessor"
    with pytest.raises(CH.ChampionError):
        b1.rollback("t", "why")


def test_ledger_chain_detects_edit_delete(tmp_path):
    b = two_champions(tmp_path)
    feed(b, [-0.005] * 10)
    b.shadow_review("t")
    led = b.ledger
    assert led.verify()["ok"] and [r["event"] for r in led.rows()][:1] == ["retired"]
    assert {"promoted", "rolled_back"} <= {r["event"] for r in led.rows()}
    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join(lines[:1] + [lines[1].replace("B", "Z")] + lines[2:]) + "\n")
    assert not led.verify()["ok"]
    led.path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert not led.verify()["ok"]


def test_refused_promotions_are_in_the_ledger(tmp_path):
    b = board(tmp_path)
    run(b, rec("A", basis=["pnl"]))
    assert [r["event"] for r in b.ledger.rows()] == ["promotion_refused"]
    assert b.ledger.rows()[0]["detail"]["failed"] == ["not_pnl_alone"]


def test_queue_order_dedupe_and_memory_block(tmp_path):
    sp = Space({"k": (2, 12, 2)})
    tried = TriedIndex(tmp_path / "t.jsonl", sp)
    tried.add("OLD", {"k": 8}, "reject", -1, "blew up")
    q = CH.ChallengerQueue(tmp_path / "q.json", tried)
    q.submit("Q1", {"k": 2}, 0.01, "t")
    q.submit("Q2", {"k": 4}, 0.05, "t")
    q.submit("Q3", {"k": 6}, 0.05, "t")
    with pytest.raises(CH.ChampionError):
        q.submit("Q4", {"k": 8}, 0.9, "t")                    # known failure
    with pytest.raises(CH.ChampionError):
        q.submit("Q5", {"k": 2.0}, 0.9, "t")                  # same config queued
    with pytest.raises(CH.ChampionError):
        q.submit("Q6", {"k": 10}, float("nan"), "t")
    assert [i["id"] for i in q.order()] == ["Q2", "Q3", "Q1"]  # gain desc, then arrival
    b = board(tmp_path)
    assert q.next_challenger(b, "t", lambda i: rec(i["id"])) == "Q2"
    assert q.next_challenger(b, "t", lambda i: rec(i["id"])) is None       # slot occupied
    assert [i["id"] for i in CH.ChallengerQueue(tmp_path / "q.json").order()] == ["Q3", "Q1"]   # persisted


def test_empty_queue(tmp_path):
    q = CH.ChallengerQueue(tmp_path / "q.json")
    assert q.order() == [] and q.next_challenger(board(tmp_path), "t", lambda i: rec(i["id"])) is None
