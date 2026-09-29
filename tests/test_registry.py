import json
import pytest
from engine.registry import Registry, ExperimentMemory, fingerprint, REQUIRED

FULL = dict(experiment_id="E1", t="2026-01-01T00:00:00", git_commit="abc", canon_sha="c", blueprint_version="4.0",
            config_hash="h", data_snapshot="d", seed=1, window_ids=["w1"], model_params={"a": 1}, train_range=["a", "b"],
            validation_range=["b", "c"], test_range=["c", "d"], metrics={"sharpe": 1.0}, gates={"parity": True},
            outcome="adopt", reason="ok")


def mk(tmp_path, recs, raw=()):
    p = tmp_path / "e.jsonl"
    p.write_text("\n".join([json.dumps(r) for r in recs] + list(raw)) + "\n")
    return Registry(p)


def test_complete_record_audits_clean(tmp_path):
    assert mk(tmp_path, [FULL]).audit()["ok"]


def test_planted_defects_are_caught(tmp_path):
    bad = {**FULL, "experiment_id": "E2"}
    del bad["seed"], bad["outcome"]
    dup = {**FULL}                                   # duplicate id
    r = mk(tmp_path, [FULL, bad, dup], raw=["{broken"])
    a = r.audit()
    assert not a["ok"]
    assert "seed" in a["missing_fields"] and a["orphans_no_outcome"] and a["duplicate_ids"] and a["bad_lines"]


def test_reproducibility_conflict_detected(tmp_path):
    k = dict(FULL, code_hash="x")
    a = {**k, "experiment_id": "A", "metrics": {"m": 1.0}}
    b = {**k, "experiment_id": "B", "metrics": {"m": 1.5}}
    c = {**k, "experiment_id": "C", "metrics": {"m": 1.0}}
    assert len(mk(tmp_path, [a, c]).reproducibility_conflicts()) == 0
    assert len(mk(tmp_path, [a, b]).reproducibility_conflicts()) == 1


def test_query_compare_and_missing_not_zero(tmp_path):
    recs = [{**FULL, "experiment_id": f"E{i}", "metrics": {"sharpe": i}} for i in range(1, 4)]
    recs.append({**FULL, "experiment_id": "E9", "metrics": {}})
    r = mk(tmp_path, recs)
    assert [x["experiment_id"] for x in r.query({"metrics.sharpe": {"ge": 2}})] == ["E2", "E3"]
    assert r.best("metrics.sharpe", 1)[0]["experiment_id"] == "E3"
    c = r.compare(["E1", "E3", "E9", "nope"], ["metrics.sharpe"])
    assert c["delta_vs_first"]["E3"]["metrics.sharpe"] == 2 and c["delta_vs_first"]["E9"]["metrics.sharpe"] is None
    assert c["missing"] == ["nope"]
    assert r.query(since="2027-01-01") == []


def test_empty_and_missing_log(tmp_path):
    r = Registry(tmp_path / "none.jsonl")
    assert len(r) == 0 and r.query() == [] and not r.audit()["ok"] and r.summary()["n"] == 0


def test_append_is_append_only(tmp_path):
    r = mk(tmp_path, [FULL])
    with pytest.raises(ValueError):
        r.append(FULL)
    r.append({**FULL, "experiment_id": "E2"})
    assert len(r) == 2


def test_reads_the_real_log_format(tmp_path):
    line = {"t": "2026-09-28T18:30:24", "event": "frontier", "canon_sha": "x", "seed": None}
    r = mk(tmp_path, [line])
    assert r.audit()["missing_fields"]["experiment_id"] and len(REQUIRED) == 17


ANS = dict(what_changed="w", why_changed="y", data_used="d", data_unseen="u", baseline="b", improved=False,
           worsened=False, statistically_meaningful=False, risk_changed=False, survived_another_window=False, adopted=False,
           if_rejected_why="no gain")


def test_memory_blocks_repeat_and_refuses_orphans(tmp_path):
    m = ExperimentMemory(tmp_path / "m.jsonl")
    change = {"k": 8, "q": 0.9}
    m.record("E1", change, ANS, "2026-01-01")
    assert m.already_tried({"q": 0.9, "k": 8})["blocked"]            # key order does not hide a repeat
    assert not m.already_tried({"k": 9, "q": 0.9})["tried"]
    with pytest.raises(ValueError):
        m.record("E2", {"z": 1}, {**ANS, "if_rejected_why": ""}, "t")  # rejected without a reason
    with pytest.raises(ValueError):
        m.record("E3", {"z": 2}, {k: v for k, v in ANS.items() if k != "baseline"}, "t")
    assert ExperimentMemory(tmp_path / "m.jsonl").already_tried(change)["ids"] == ["E1"]   # persisted
    reg = mk(tmp_path, [{**FULL, "experiment_id": "E1"}, {**FULL, "experiment_id": "E7"}])
    assert m.orphans(reg) == ["E7"]


def test_adopted_later_unblocks(tmp_path):
    m = ExperimentMemory(tmp_path / "m.jsonl")
    m.record("E1", {"a": 1}, ANS, "t")
    m.record("E2", {"a": 1}, {**ANS, "adopted": True, "improved": True, "if_rejected_why": None}, "t")
    assert not m.already_tried({"a": 1})["blocked"]


def test_fingerprint_stable():
    assert fingerprint({"a": 1.0000001, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1.0000002})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
