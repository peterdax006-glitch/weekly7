"""F7: thinking drills (before-the-fact predictions scored against what happened, strictly time-ordered), the trust gate, and the blueprint.
Synthetic state directories only (a planted history with a known pattern), so what the tests assert is the property, not Nupen's real numbers."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Optional

import pytest

from creator import registry as REG
from creator import sandbox as SB
from creator import thinking as T

T0 = 1_790_000_000.0


def iso(t: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def world(tmp: Path, n: int = 80, open_n: int = 0, seed: int = 1) -> Path:
    """n resolved packages: kind 'exists' is adopted 90% of the time (fast), kind 'size' 10% (slow); open_n further packages not yet resolved."""
    rnd = random.Random(seed)
    led, klog = [], []
    for i in range(n + open_n):
        kind = "exists" if i % 2 == 0 else "size"
        pkg = f"CP{i:04d}"
        created = T0 + i * 3600
        led.append({"rtype": "WorkPackage", "id": f"WP-{i}", "provenance": {"timestamp": iso(created)},
                    "data": {"package_id": pkg, "why_it_exists": f"gap ({'K07' if kind == 'exists' else 'EFF'}.{kind})", "outputs": ["a.py"] * (1 + i % 3)}})
        if i < n:
            adopted = rnd.random() < (0.9 if kind == "exists" else 0.1)
            secs = 900.0 if kind == "exists" else 3600.0
            klog.append({"cycle": 1, "package": pkg, "outcome": "ADOPTED" if adopted else "REJECTED", "requirement": f"{'K07' if kind == 'exists' else 'EFF'}.{kind}",
                         "seconds": secs, "usd": 0.0})
            led.append({"rtype": "StrategyOutcome", "id": f"SO-{i}", "provenance": {"timestamp": iso(created + secs)}, "data": {"subject_id": f"WP-{i}"}})
    (tmp / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in led) + "\n", encoding="utf-8")
    (tmp / "kernel_log.jsonl").write_text("\n".join(json.dumps(r) for r in klog) + "\n", encoding="utf-8")
    return tmp


def test_replay_learns_the_planted_pattern_and_beats_baselines(tmp_path: Path) -> None:
    st = world(tmp_path)
    sc = T.score(T.replay(T.load_items(st), "verdict"))
    assert sc["n"] == 80
    assert sc["brier"] < sc["brier_base_rate"] - 0.05 and sc["brier"] < sc["brier_last_value"] - 0.05
    assert sc["gain_ci95"][0] > 0


def test_no_leak_a_prediction_never_reads_a_later_record(tmp_path: Path) -> None:
    st = world(tmp_path)
    items = T.load_items(st)
    k = 40
    before = T.predict_item("verdict", items, items[k])
    cost_before = T.predict_item("cost", items, items[k])
    mutated = [T.Item(**{**i.__dict__}) for i in items]
    for i in mutated[k:]:                                   # rewrite everything from k on: outcome, time, seconds
        if i.resolved is not None:
            i.outcome, i.seconds = ("REJECTED" if i.outcome == "ADOPTED" else "ADOPTED"), i.seconds * 50
    after = T.predict_item("verdict", mutated, mutated[k])
    assert before.p == after.p and before.base == after.base and before.last == after.last
    assert cost_before.p == T.predict_item("cost", mutated, mutated[k]).p
    # a package resolved AFTER the prediction time is not visible even if it was created earlier
    assert all(i.resolved < items[k].created for i in T._train(items, items[k].created))


def test_the_model_changes_its_mind_when_outcomes_change(tmp_path: Path) -> None:
    items = T.load_items(world(tmp_path))
    it = T.Item("X", T0 + 1e7, "K07.exists")
    p_before = T.predict_item("verdict", items[:6], it).p
    p_after = T.predict_item("verdict", items, it).p
    assert p_after > 0.7 and p_after != p_before             # learned that 'exists' packages are adopted


def test_make_or_ask_threshold() -> None:
    import math
    assert T.make_or_ask(math.log(T.ASK_S * 0.9)) == "MAKE"
    assert T.make_or_ask(math.log(T.ASK_S * 1.1)) == "ASK"


def test_cost_estimate_uses_only_the_spec_and_is_in_the_right_range(tmp_path: Path) -> None:
    items = T.load_items(world(tmp_path))
    slow = T.predict_item("cost", items, T.Item("N1", T0 + 1e7, "EFF.size", spec_len=500))
    fast = T.predict_item("cost", items, T.Item("N2", T0 + 1e7, "K07.exists", spec_len=500))
    assert slow.extra["est_s"] > fast.extra["est_s"] and slow.p > fast.p


def _pred(p: float, o: int, base: float = 0.5, mode: str = "replay") -> T.Pred:
    return T.Pred("verdict", "x", 0.0, p, base, base, o, mode)


def test_trust_needs_n_baseline_win_and_calibration() -> None:
    rnd = random.Random(3)
    good = [_pred(0.9 if o else 0.1, o, 0.5, "live" if k < 12 else "replay") for k, o in enumerate(rnd.random() < 0.5 for _ in range(80))]
    assert T.trust_of(T.score(good)) == (True, [])
    few = good[:20]
    assert not T.trust_of(T.score(few))[0] and "scored predictions" in T.trust_of(T.score(few))[1][0]
    nolive = [_pred(p.p, int(p.outcome or 0), 0.5) for p in good]
    assert any("prospective" in w for w in T.trust_of(T.score(nolive))[1])
    coin = [_pred(0.5, o) for o in (rnd.random() < 0.5 for _ in range(80))]
    assert not T.trust_of(T.score(coin))[0]                 # equal to the baseline is not "beats the baseline"
    overconf = [_pred(0.95, 1 if rnd.random() < 0.6 else 0, 0.7, "live") for _ in range(80)]       # sharper than the truth
    ok, why = T.trust_of(T.score(overconf))
    assert not ok and any("calibrated" in w for w in why)


def test_ci_is_what_blocks_a_lucky_small_win() -> None:
    lucky = [_pred(0.9, 1, 0.5, "live") for _ in range(6)] + [_pred(0.9, 0, 0.5, "live")] * 1
    assert T.score(lucky)["gain_ci95"][0] is not None
    assert not T.trust_of(T.score(lucky))[0]


def test_live_pass_predicts_before_the_fact_is_idempotent_and_resolves(tmp_path: Path) -> None:
    st = world(tmp_path, n=30, open_n=3)
    now = T0 + 40 * 3600
    r1 = T.live_pass(st, now)
    assert r1["new"] == 9 and r1["resolved"] == 0           # 3 open packages x verdict/duration/cost
    assert T.live_pass(st, now + 60) == {"new": 0, "resolved": 0}
    rec = T.stored(st)["verdict:CP0030"]
    assert rec["outcome"] is None and rec["mode"] == "live" and rec["made_at"] == now
    # the outcome arrives: kernel_log gets a row and the ledger a StrategyOutcome
    with (st / "kernel_log.jsonl").open("a") as f:
        f.write(json.dumps({"package": "CP0030", "outcome": "ADOPTED", "requirement": "K07.exists", "seconds": 600.0}) + "\n")
    with (st / "ledger.jsonl").open("a") as f:
        f.write(json.dumps({"rtype": "StrategyOutcome", "id": "SO-30", "provenance": {"timestamp": iso(T0 + 31 * 3600)}, "data": {"subject_id": "WP-30"}}) + "\n")
    assert T.live_pass(st, now + 120)["resolved"] == 3
    live = [p for p in T.all_preds(st)["verdict"] if p.mode == "live"]
    assert len(live) == 1 and live[0].outcome == 1 and live[0].made_at == now


def test_independent_only_when_trusted(tmp_path: Path) -> None:
    st = world(tmp_path, n=120)
    assert T.independent(st, "verdict") is False            # no prospective predictions yet: replay alone never grants independence
    assert T.independent(st, "goal_value") is False and T.independent(st, "nonsense") is False
    for k in range(12):                                     # 12 resolved prospective predictions
        T._append(st, {"id": f"verdict:L{k}", "topic": "verdict", "subject": f"L{k}", "made_at": T0 + 1e8 + k, "p": 0.9 if k % 2 == 0 else 0.1,
                       "base": 0.5, "last": 0.5, "mode": "live", "outcome": None, "extra": {}})
        T._append(st, {"id": f"verdict:L{k}", "resolved": 1 if k % 2 == 0 else 0, "at": T0 + 1e8 + k + 5})
    assert T.independent(st, "verdict") is True
    rep = T.trust(st)
    assert json.loads((st / "trust.json").read_text())["topics"]["verdict"]["trusted"] is True and rep["topics"]["verdict"]["score"]["n_live"] == 12


def test_constraint_and_goal_value_drills(tmp_path: Path) -> None:
    snaps = [{"event": "snapshot", "at": f"2026-10-02T{9 + i:02d}:00:00", "ranked": [{"name": "a" if i < 4 else "b"}]} for i in range(6)]
    (tmp_path / "constraints.jsonl").write_text("\n".join(json.dumps(s) for s in snaps), encoding="utf-8")
    ps = T.replay_constraint(tmp_path)
    assert [p.outcome for p in ps] == [1, 1, 1, 0, 1] and all(0 < p.p < 1 for p in ps)
    g = [{"event": "proposal", "id": "G1", "created": "2026-10-02T08:00:00"}, {"event": "approve", "id": "G1", "at": "2026-10-02T09:00:00"}]
    (tmp_path / "goal_proposals.jsonl").write_text("\n".join(json.dumps(x) for x in g), encoding="utf-8")
    gp = T.replay_goal_value(tmp_path)
    assert len(gp) == 1 and gp[0].outcome == 1 and gp[0].p == 0.5


def test_trust_report_covers_every_topic_even_with_no_data(tmp_path: Path) -> None:
    rep = T.trust(tmp_path, write=False)
    assert set(rep["topics"]) == set(T.TOPICS) and not any(v["trusted"] for v in rep["topics"].values())


def test_registry_and_protected_paths() -> None:
    assert REG.get("thinking") is T and REG.get("focus").__name__ == "creator.focus"
    for p in ("state/creator/focus.json", "state/creator/trust.json", "state/creator/thinking/predictions.jsonl"):
        assert SB.is_protected(p), p


def test_blueprint_is_computed_from_state(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import nupen_blueprint as B
    st = world(tmp_path, n=60)
    (st / "selfmodel.json").write_text(json.dumps({"capabilities": [
        {"id": "K01", "name": "ledger", "state": "TESTED", "uncertainty": "LIKELY", "meaningful": 11, "present_modules": ["creator/a.py"], "last_results": {"t": "PASS"}, "why": "ok"},
        {"id": "K02", "name": "other", "state": "IMPLEMENTED", "uncertainty": "UNTESTED", "meaningful": 7, "present_modules": ["creator/b.py"], "last_results": {}, "why": "stale"}],
        "dependencies": {"creator/b.py": ["creator/a.py"]}, "limitations": [{"kind": "STUB"}], "active": {"ledger_head": "abc"}}), encoding="utf-8")
    (st / "constraints.jsonl").write_text(json.dumps({"event": "snapshot", "at": "2026-10-02T09:00:00", "headline": {"current": {"adopted": 3, "cycles": 9}, "window_h": 24},
                                                      "ranked": [{"name": "learning_signal", "value": 0.9, "unit": "u", "loss": 0.9, "score": 0.9, "remedy": "goal"}]}), encoding="utf-8")
    text = B.build(st)
    adopted = sum(1 for k in (json.loads(x) for x in (st / "kernel_log.jsonl").read_text().splitlines()) if k["outcome"] == "ADOPTED")
    assert f"{adopted} adopted" in text and "60 resolved cycles" in text
    for h in ("## 1. Vision", "## 2. Capability map", "## 3. Limiting factors", "## 4. Roadmap", "## 5. Trust scores"):
        assert h in text
    assert "learning_signal" in text and "K02 other: bring from IMPLEMENTED to TESTED" in text and "MAKE" in text
    out = tmp_path / "BP.md"
    assert B.main(["--state", str(st), "--out", str(out)]) == 0 and out.read_text(encoding="utf-8").startswith("# Nupen blueprint")
    assert not (st / "trust.json").exists()                  # the blueprint reads and reports; it does not write the trust file


def test_blueprint_ask_item_carries_a_spec(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import nupen_blueprint as B
    st = world(tmp_path, n=60)
    (st / "constraints.jsonl").write_text(json.dumps({"event": "snapshot", "at": "2026-10-02T09:00:00", "headline": {},
                                                      "ranked": [{"name": "huge", "value": 1, "unit": "u", "loss": 1.0, "score": 1.0, "remedy": "goal"}]}), encoding="utf-8")
    (st / "goal_proposals.jsonl").write_text(json.dumps({"event": "proposal", "id": "G", "source": "constraint", "key": "constraint:huge", "cost": {"packages": 400}}), encoding="utf-8")
    text = B.build(st)
    assert "**ASK**" in text and "Handoff specs (ASK items)" in text and "goals.propose" in text
