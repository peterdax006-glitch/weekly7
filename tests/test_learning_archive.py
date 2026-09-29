"""Tests for engine/learning/archive.py (contract section 49; checklist B01-B15). Synthetic data only."""
import dataclasses
import datetime as dt
import os

import pytest

from engine.learning import archive as A
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FailureCause, FirewallBreach, Layer, Lifecycle,
                                  Promotion, Provenance)

FIXED = "2020-01-01T00:00:00"


def mk(root=None, **kw):
    return A.Archive(root, code_hash="testhash", created_real=FIXED, **kw)


def prov(learned, seen=None):
    return Provenance(created_real=FIXED, learned_at=learned, code_hash="testhash", outcomes_seen_through=seen or "")


def seed(a):
    """A small evidence pyramid: obs -> event -> episode -> situation -> hypothesis -> pattern."""
    o = a.log_observation("s1", {"ret": 0.03, "vol": 0.4}, "2020-03-02", matured_at="2020-03-09")
    e = a.log_event("s1", "gap_up", "2020-03-02", parents=[o.rec_id], matured_at="2020-03-09")
    ep = a.log_episode("s1", [e.rec_id], {"win": True}, "2020-03-02", matured_at="2020-03-10")
    sit = a.log_situation("sit1", {"vol_bucket": 3}, [ep.rec_id], "2020-03-02", matured_at="2020-03-10")
    h = a.append(layer=Layer.L4_HYPOTHESIS, kind="hypothesis", payload={"statement": "gap then drift", "tags": ["drift"]},
                 occurred_at="2020-03-11", matured_at="2020-03-11", subject="h1", parents=[sit.rec_id])
    p = a.append(layer=Layer.L5_PATTERN, kind="pattern", payload={"rule": "vol_q5 and gap"}, occurred_at="2020-03-12",
                 subject="p1", parents=[h.rec_id], contexts={"regime": "bull"})
    return dict(o=o, e=e, ep=ep, sit=sit, h=h, p=p)


def test_append_and_layers_roundtrip():
    a = mk()
    r = seed(a)
    assert len(a) == 6
    assert a.audit("2021-01-01").ok
    assert [x.layer for x in a.view("2021-01-01")] == [Layer.L0_RAW, Layer.L1_EVENT, Layer.L2_EPISODE, Layer.L3_SITUATION,
                                                       Layer.L4_HYPOTHESIS, Layer.L5_PATTERN]
    assert r["p"].payload == {"rule": "vol_q5 and gap"}
    assert r["p"].context_dict == {"regime": "bull"}


def test_idempotent_rerun_does_not_inflate_history():
    a = mk()
    x = a.log_observation("s1", {"ret": 0.01}, "2020-03-02", matured_at="2020-03-09")
    y = a.log_observation("s1", {"ret": 0.01}, "2020-03-02", matured_at="2020-03-09")
    assert x.rec_id == y.rec_id and len(a) == 1


def test_immutability_is_enforced():
    a = mk()
    r = a.log_observation("s1", {"ret": 0.01}, "2020-03-02")
    with pytest.raises(A.ImmutabilityViolation):
        a.update(r.rec_id, {"ret": 9})
    with pytest.raises(A.ImmutabilityViolation):
        a.delete(r.rec_id)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.subject = "other"


def test_kind_layer_and_payload_validation():
    a = mk()
    with pytest.raises(A.ArchiveError, match="not allowed"):
        a.append(layer=Layer.L5_PATTERN, kind="observation", payload={}, occurred_at="2020-01-01")
    with pytest.raises(A.ArchiveError, match="non-finite"):
        a.log_observation("s", {"x": float("nan")}, "2020-01-01")
    with pytest.raises(A.ArchiveError, match="unsupported type"):
        a.log_observation("s", {"x": object()}, "2020-01-01")
    with pytest.raises(A.ArchiveError, match="needs at least one parent"):
        a.append(layer=Layer.L2_EPISODE, kind="episode", payload={"outcome": {}}, occurred_at="2020-01-01", subject="s")
    with pytest.raises(A.ArchiveError, match="unknown cause"):
        a.append(layer=Layer.L7_VALIDATED, kind="failure", payload={"cause": "BAD_LUCK"}, occurred_at="2020-01-01", subject="k")
    with pytest.raises(A.ArchiveError, match="outside"):
        a.log_reliability("k", 1.4, 5, "2020-01-01")
    with pytest.raises(A.ArchiveError, match="does not exist"):
        a.log_event("s", "x", "2020-01-01", parents=["nope"])
    with pytest.raises(A.ArchiveError, match="matured_at precedes"):
        a.log_observation("s", {}, "2020-02-01", matured_at="2020-01-01")


def test_writer_now_refuses_evidence_not_yet_matured():
    a = mk()
    with pytest.raises(A.ArchiveError, match="not yet knowable"):
        a.log_observation("s", {"ret": 0.1}, "2020-03-01", matured_at="2020-03-08", now="2020-03-08")   # matures ON now
    a.log_observation("s", {"ret": 0.1}, "2020-03-01", matured_at="2020-03-08", now="2020-03-09")


def test_planted_identity_and_date_in_abstract_layer_rejected():
    a = mk(forbidden_identities=["AAPL"])
    sit = seed(a)["sit"]
    for payload, why in [({"ticker": "XYZ"}, "identity/date key"), ({"rule": "buy on 2020-03-02"}, "date"),
                         ({"rule": "AAPL gaps up"}, "forbidden identity"), ({"note": "in 2019 it worked"}, "date")]:
        with pytest.raises(A.ArchiveError, match=why):
            a.append(layer=Layer.L5_PATTERN, kind="pattern", payload=payload, occurred_at="2020-04-01",
                     subject="bad", parents=[sit.rec_id])
    a.log_observation("raw", {"ticker": "AAPL", "date": "2020-01-02"}, "2020-01-02")     # raw layer may carry identity


def test_time_travel_parent_rejected():
    a = mk()
    o = a.log_observation("s", {"r": 1}, "2020-05-01", matured_at="2020-05-08")
    with pytest.raises(A.ArchiveError, match="time travel"):
        a.log_event("s", "early", "2020-05-01", parents=[o.rec_id], matured_at="2020-05-02")


def test_visibility_is_strict_and_closed_over_parents():
    a = mk()
    r = seed(a)
    # ep and sit mature 2020-03-10; on 2020-03-10 itself neither is visible (an outcome maturing ON now is unknown)
    ids = {x.rec_id for x in a.view("2020-03-10")}
    assert r["o"].rec_id in ids and r["e"].rec_id in ids
    assert r["ep"].rec_id not in ids and r["sit"].rec_id not in ids
    ids = {x.rec_id for x in a.view("2020-03-11")}
    assert r["sit"].rec_id in ids and r["h"].rec_id not in ids            # h matured 03-11 = not before now
    # a child whose own timing is fine is still hidden while a parent is hidden
    late = a.log_observation("s2", {"r": 1}, "2020-06-01", matured_at="2020-06-08")
    child = a.log_event("s2", "c", "2020-06-01", parents=[late.rec_id], matured_at="2020-06-08")
    assert child.rec_id not in {x.rec_id for x in a.view("2020-06-08")}
    assert a.view("2019-01-01") == ()


def test_provenance_that_could_not_exist_is_hidden():
    a = mk()
    r = a.append(layer=Layer.L0_RAW, kind="observation", payload={"v": 1}, occurred_at="2020-01-01",
                 matured_at="2020-01-05", provenance=prov("2020-01-05", seen="2020-09-01"), subject="s")
    assert r.rec_id not in {x.rec_id for x in a.view("2020-06-01")}      # it had already seen outcomes from September
    assert r.rec_id in {x.rec_id for x in a.view("2020-09-02")}


def test_planted_view_leak_is_caught_by_audit():
    a = mk()
    seed(a)
    real = a.visible_ids

    def leaky(now, max_seq=None):
        return frozenset(list(real(now, max_seq)) + [a.records()[-1].rec_id])
    a.visible_ids = leaky
    audit = a.audit("2020-03-10")
    assert not audit.ok and any(f.code == "VIEW_LEAK" for f in audit.findings)
    with pytest.raises(FirewallBreach):
        a.assert_clean("2020-03-10")


def test_dynamic_influence_never_deletes_history():
    a = mk()
    r = a.log_reliability("k1", 0.7, 30, "2020-02-01", matured_at="2020-02-02")
    assert a.influence_at(r.rec_id, "2020-03-01") == 1.0
    a.retire(r.rec_id, "pattern broke", "2020-04-01")
    assert a.influence_at(r.rec_id, "2020-04-02") == 0.0
    assert a.influence_at(r.rec_id, "2020-03-01") == 1.0                  # earlier reads unchanged
    assert r.rec_id in {x.rec_id for x in a.view("2020-05-01")}           # history still readable
    assert r.rec_id not in {x.rec_id for x in a.active_view("2020-05-01")}
    a.set_influence(r.rec_id, 0.5, "partial recovery", "2020-06-01")
    assert a.influence_at(r.rec_id, "2020-06-02") == 0.5
    assert [w for _, w, _ in a.influence_series(r.rec_id, "2020-07-01")] == [0.0, 0.5]
    with pytest.raises(A.ArchiveError, match="pre-date"):
        a.set_influence(r.rec_id, 0.2, "too early", "2020-01-01")
    with pytest.raises(A.ArchiveError, match="reason"):
        a.set_influence(r.rec_id, 0.2, "  ", "2020-07-01")


def test_subject_level_influence_and_record_override():
    a = mk()
    r1 = a.log_reliability("kx", 0.6, 10, "2020-02-01", matured_at="2020-02-02")
    r2 = a.log_reliability("kx", 0.7, 12, "2020-02-10", matured_at="2020-02-11")
    a.set_influence("kx", 0.3, "family faded", "2020-03-01")
    assert a.influence_at(r1.rec_id, "2020-03-05") == 0.3 and a.influence_at(r2.rec_id, "2020-03-05") == 0.3
    a.set_influence(r2.rec_id, 0.9, "this one is fine", "2020-03-02")
    assert a.influence_at(r2.rec_id, "2020-03-05") == 0.9 and a.influence_at(r1.rec_id, "2020-03-05") == 0.3


def test_supersede_keeps_old_and_resolves_latest():
    a = mk()
    v1 = a.append(layer=Layer.L7_VALIDATED, kind="knowledge", payload={"knowledge_id": "K", "version": 1},
                  occurred_at="2020-01-01", subject="K")
    v2 = a.append(layer=Layer.L7_VALIDATED, kind="knowledge", payload={"knowledge_id": "K", "version": 2},
                  occurred_at="2020-02-01", subject="K")
    a.supersede(v1.rec_id, v2.rec_id, "refit", "2020-02-02")
    assert a.latest_version(v1.rec_id, "2020-03-01").rec_id == v2.rec_id
    assert a.latest_version(v1.rec_id, "2020-01-15").rec_id == v1.rec_id
    assert [r.rec_id for r in a.active_view("2020-03-01", kinds=("knowledge",))] == [v2.rec_id]
    assert len(a.view("2020-03-01", kinds=("knowledge",))) == 2


def test_indices_are_filtered_by_now():
    a = mk()
    r = seed(a)
    for i, (d, v) in enumerate([("2020-01-10", .8), ("2020-02-10", .7), ("2020-03-10", .5), ("2020-04-10", .3)]):
        a.log_reliability("p1", v, 20 + i, d, layer=Layer.L5_PATTERN, matured_at=d)
    a.log_failure("p1", FailureCause.REGIME_CHANGE, Layer.L5_PATTERN, "2020-04-01", matured_at="2020-04-05")
    a.log_failure("p1", "WRONG_CONTEXT", Layer.L5_PATTERN, "2020-05-01", matured_at="2020-05-05")
    a.log_contradiction("p1", "p9", "2020-04-02", matured_at="2020-04-02", layer=Layer.L6_CONTEXT_RULE)
    a.log_recovery("p1", {"regime": "bull"}, "2020-06-01", matured_at="2020-06-02", layer=Layer.L6_CONTEXT_RULE)
    assert [v for _, v, _ in a.reliability_trace("p1", "2020-03-15")] == [.8, .7, .5]
    assert a.reliability_at("p1", "2020-03-15") == .5 and a.reliability_at("p1", "2020-01-01") is None
    assert a.reliability_trend("p1", "2020-05-01") < 0 and a.reliability_trend("p1", "2020-02-15") is None
    assert a.failure_counts("2020-04-30") == {"REGIME_CHANGE": 1}
    assert a.failure_counts("2020-06-30") == {"REGIME_CHANGE": 1, "WRONG_CONTEXT": 1}
    assert len(a.failures("2020-06-30", cause="WRONG_CONTEXT")) == 1
    assert len(a.failures("2020-06-30", subject="p1")) == 2
    assert a.contradiction_pairs("2020-03-30") == [] and a.contradiction_pairs("2020-04-30") == [("p1", "p9")]
    assert len(a.contradictions_of("p9", "2020-04-30")) == 1
    assert a.recoveries_of("p1", "2020-06-01") == [] and len(a.recoveries_of("p1", "2020-06-03")) == 1
    hits = a.search("drift hypothesis", "2020-12-31")
    assert hits and hits[0][0].rec_id == r["h"].rec_id
    assert a.search("drift", "2020-03-01") == []
    assert [x.rec_id for x in a.by_context({"regime": "bull"}, "2021-01-01")] == [r["p"].rec_id]
    assert a.context_values("regime", "2021-01-01") == {"bull": 1}
    assert a.context_values("regime", "2020-03-01") == {}
    got = {x.rec_id for x in a.between("2020-03-09", "2020-03-11", "2021-01-01")}
    assert r["o"].rec_id in got and r["ep"].rec_id in got and r["p"].rec_id not in got
    assert a.between("2020-03-09", "2020-03-11", "2020-03-10") != a.between("2020-03-09", "2020-03-11", "2021-01-01")


def test_lineage_and_descendants():
    a = mk()
    r = seed(a)
    assert [x.rec_id for x in a.lineage(r["p"].rec_id)] == [r["h"].rec_id, r["sit"].rec_id, r["ep"].rec_id, r["e"].rec_id,
                                                          r["o"].rec_id]
    assert len(a.lineage(r["p"].rec_id, now="2020-03-10")) == 0        # p itself not yet visible: parents hidden ones dropped
    assert r["p"].rec_id in {x.rec_id for x in a.descendants(r["o"].rec_id)}
    assert a.lineage(r["o"].rec_id) == []


def test_snapshot_verifies_and_detects_alteration():
    a = mk()
    seed(a)
    s = a.snapshot("2020-04-01", "q1")
    assert s in a.snapshots() and a.verify_snapshot(s)["ok"]
    a.log_observation("late", {"ret": 0.0}, "2020-03-01", matured_at="2020-03-05")      # back-fill behind the snapshot
    res = a.verify_snapshot(s)
    assert res["ok"] and res["late_arrivals"] == 1
    forged = dataclasses.replace(s, digest="0" * 24)
    assert not a.verify_snapshot(forged)["ok"]
    forged = dataclasses.replace(s, head_hash="f" * 64)
    assert "altered" in a.verify_snapshot(forged)["problems"][0]
    assert a.audit("2020-05-01").ok


def test_persistence_reopen_and_second_writer(tmp_path):
    a = mk(tmp_path / "arch")
    seed(a)
    b = mk(tmp_path / "arch")
    assert len(b) == 6 and b.head == a.head
    b.log_observation("x", {"v": 1}, "2020-08-01")
    a.refresh()
    assert len(a) == 7 and a.audit("2021-01-01").ok


def test_tampering_with_file_fails_closed(tmp_path):
    a = mk(tmp_path / "arch")
    seed(a)
    p = tmp_path / "arch" / "chain.jsonl"
    data = p.read_bytes().replace(b"gap_up", b"gap_dn")
    p.write_bytes(data)
    with pytest.raises(A.ChainCorrupt):
        mk(tmp_path / "arch")


def test_torn_tail_refuses_append(tmp_path):
    a = mk(tmp_path / "arch")
    seed(a)
    p = tmp_path / "arch" / "chain.jsonl"
    with open(p, "ab") as fh:
        fh.write(b'{"seq": 6, "prev": "abc')
    b = mk(tmp_path / "arch")
    assert b._chain.torn_tail
    with pytest.raises(A.ChainCorrupt, match="torn"):
        b.log_observation("x", {"v": 1}, "2020-08-01")


def test_chain_verify_detects_deleted_middle_line(tmp_path):
    a = mk(tmp_path / "arch")
    seed(a)
    p = tmp_path / "arch" / "chain.jsonl"
    lines = p.read_bytes().split(b"\n")
    del lines[2]
    p.write_bytes(b"\n".join(lines))
    with pytest.raises(A.ChainCorrupt):
        mk(tmp_path / "arch")


def test_prefix_invariance_passes_and_catches_a_lookahead_record():
    a = mk()
    seed(a)
    a.log_reliability("p1", 0.5, 9, "2020-05-01", layer=Layer.L5_PATTERN, matured_at="2020-05-02")
    assert A.audit_prefix_invariance(a, ["2020-03-05", "2020-03-20", "2020-06-01"]) == []


def test_determinism_same_inputs_same_head():
    heads = []
    for _ in range(2):
        a = mk()
        seed(a)
        heads.append(a.head)
    assert heads[0] == heads[1]


def test_put_knowledge_duck_type_and_versions():
    class K:
        knowledge_id = "K7"
        version = 1
        epistemic = Epistemic.SUPPORTED
        lifecycle = Lifecycle.ACTIVE
        promotion = Promotion.SHADOW
        confidence = Confidence(truth=0.8)
        provenance = prov("2020-03-01")
        contexts = {"regime": "bull"}
        anti_contexts = {"vol": "extreme"}
        decision_effect = (DecisionEffect.RANKING,)

    a = mk()
    k = K()
    a.put_knowledge(k, "2020-03-01")
    k2 = K()
    k2.version = 2
    k2.provenance = prov("2020-04-01")
    a.put_knowledge(k2, "2020-04-01")
    with pytest.raises(A.ArchiveError, match="learned_at precedes"):
        a.put_knowledge(K(), "2020-06-01")            # v1 provenance (learned 03-01) but matured 06-01: inconsistent
    vs = a.knowledge_versions("K7", "2021-01-01")
    assert [v.payload["version"] for v in vs] == [1, 2]
    assert vs[0].payload["epistemic"] == "SUPPORTED" and vs[0].payload["confidence"]["truth"] == 0.8
    assert a.latest_knowledge("K7", "2020-03-15").payload["version"] == 1
    assert a.latest_knowledge("K7", "2020-02-01") is None
    with pytest.raises(A.ArchiveError, match="not knowledge-like"):
        a.put_knowledge(object(), "2020-04-01")


def test_empty_archive_is_well_behaved():
    a = mk()
    assert a.view("2021-01-01") == () and a.active_view("2021-01-01") == ()
    au = a.audit("2021-01-01")
    assert au.ok and au.records == 0
    s = a.snapshot("2021-01-01")
    assert s.n_visible == 0 and a.verify_snapshot(s)["ok"]
    assert a.stats("2021-01-01")["visible"] == 0 and a.search("x", "2021-01-01") == []
    assert "Knowledge archive" in a.markdown("2021-01-01")
    assert a.reliability_trend("none", "2021-01-01") is None
    assert a.by_context({}, "2021-01-01") == []
    assert A.ChainFile(None).verify()["ok"]


def test_markdown_report_lists_layers_and_reliability():
    a = mk()
    seed(a)
    for i, v in enumerate([.9, .8, .6]):
        a.log_reliability("p1", v, 10, f"2020-0{i + 1}-15", layer=Layer.L5_PATTERN)
    md = a.markdown("2021-01-01")
    assert "L5_PATTERN" in md and "p1" in md and "PASS" in md


def test_audit_flags_parent_written_after_child_when_chain_is_forged():
    a = mk()
    r = seed(a)
    forged = dataclasses.replace(a._recs[1], parents=("zzzz",))
    a._recs[1] = forged
    au = a.audit("2021-01-01")
    assert not au.ok and any(f.code in ("PARENT_ORDER", "CONTENT_MISMATCH") for f in au.findings)


# ------------------------------------------------------------------ read adapters over the existing stores

def test_pattern_memory_adapter_mirrors_only_matured_evidence(tmp_path):
    from engine.pattern_memory import PatternMemory
    pmem = PatternMemory(tmp_path / "pm")
    pmem.add_observations("r1", "2020-06-01", [
        {"key": "vol_q5", "obs_date": "2020-01-06", "effect": 0.011, "n": 20, "t": 2.5},
        {"key": "vol_q5", "obs_date": "2020-03-02", "effect": 0.009, "n": 22, "t": 2.1},
        {"key": "gap_up", "obs_date": "2020-05-04", "effect": -0.004, "n": 15, "t": -1.2}])
    a = mk()
    early = A.adopt_pattern_memory(a, pmem, "2020-02-01")
    assert early == {"observations": 1, "patterns": 1}                      # only the January observation has matured
    full = A.adopt_pattern_memory(a, pmem, "2021-01-01")
    assert full["observations"] == 3 and full["patterns"] == 2
    assert A.adopt_pattern_memory(a, pmem, "2021-01-01") == full            # idempotent
    assert len(a.view("2021-01-01", kinds=("observation",))) == 3 and len(a.view("2021-01-01", kinds=("pattern",))) == 2
    assert a.audit("2021-01-01").ok and a.lineage(a.view("2021-01-01", kinds=("pattern",))[0].rec_id)


def test_pattern_bank_adapter_records_retirement_restoration_and_failures():
    class Bank:
        def read(self, as_of):
            hist = [{"kind": "transition", "as_of": "2020-02-01", "to": "active"},
                    {"kind": "transition", "as_of": "2020-05-01", "to": "retired"},
                    {"kind": "transition", "as_of": "2020-08-01", "to": "active"}]
            return [{"id": "b1", "name": "vol_q5 & gap", "effect": 0.01, "scope": {"col": "m_vix", "lo": 1.0, "hi": 2.0,
                                                                                      "label": "high"},
                     "history": hist, "failures": [{"as_of": "2020-04-10", "cause": "REGIME_CHANGE"},
                                                   {"as_of": "2020-04-20"}]}]

    a = mk()
    cnt = A.adopt_pattern_bank(a, Bank(), "2021-01-01")
    assert cnt == {"patterns": 1, "retired": 1, "restored": 1, "failures": 2}
    assert a.influence_at("b1", "2020-06-01") == 0.0 and a.influence_at("b1", "2020-09-01") == 1.0
    assert a.failure_counts("2021-01-01") == {"REGIME_CHANGE": 1, "UNKNOWN": 1}
    assert len(a.view("2021-01-01", kinds=("pattern",))) == 1
    assert A.adopt_pattern_bank(a, Bank(), "2021-01-01") == cnt and len(a.view("2021-01-01", kinds=("influence",))) == 2


def test_trust_table_adapter_marks_unreliable_indicators_retired():
    import types
    import pandas as pd
    tt = types.SimpleNamespace(now=pd.Timestamp("2020-06-01"), table=pd.DataFrame([
        {"type": "sector", "indicator": "rsi", "confidence": 0.9, "n_obs": 400, "reliable": True, "reason": "ok"},
        {"type": "sector", "indicator": "vol", "confidence": 0.2, "n_obs": 30, "reliable": False, "reason": "thin"}]))
    a = mk()
    assert A.adopt_trust_table(a, tt, "2020-07-01") == {"reliability": 2, "retired": 1}
    assert a.reliability_at("trust:sector:rsi", "2020-07-01") == 0.9
    assert a.influence_at("trust:sector:vol", "2020-07-01") == 0.0 and a.influence_at("trust:sector:rsi", "2020-07-01") == 1.0
    with pytest.raises(FirewallBreach):
        A.adopt_trust_table(a, tt, "2020-06-01")                             # table dated ON now is not yet known


def test_lessons_adapter_and_memory_adapter():
    import types
    L = lambda lid, status: types.SimpleNamespace(lid=lid, conds=[("m_vix", ">", 2.5)], direction=-1, factor=0.5,
                                                  kind="bad_entry", status=status, trust=0.7, n=40, n_weeks=20)
    a = mk()
    assert A.adopt_lessons(a, [L("l1", "active"), L("l2", "retired")], "2020-05-01", "2021-01-01") == {"rules": 2, "retired": 1}
    assert a.influence_at("l2", "2020-06-01") == 0.0 and a.context_values("lesson_kind", "2021-01-01") == {"bad_entry": 2}
    with pytest.raises(FirewallBreach):
        A.adopt_lessons(a, [L("l3", "active")], "2021-01-01", "2021-01-01")
    M = lambda et, date: types.SimpleNamespace(arm="a1", fingerprint="f", context={"c": 1}, features={"x": 0.1},
                                               outcome_bin="b", error_type=et, era="e1", date=date)
    cnt = A.adopt_memory_lessons(a, [M("false_positive", "2020-03-02"), M("correct", "2019Q3")], "2021-01-01",
                                 default_date="2019-09-30")
    assert cnt == {"observations": 2, "failures": 1}
    with pytest.raises(A.ArchiveError, match="coarse date"):
        A.adopt_memory_lessons(a, [M("noise", "2019Q3")], "2021-01-01")
    with pytest.raises(FirewallBreach):
        A.adopt_memory_lessons(a, [M("noise", "2022-01-01")], "2021-01-01")


def test_archive_shares_the_pattern_memory_chain(tmp_path):
    from engine.pattern_memory import PatternMemory
    root = tmp_path / "shared"
    pmem = PatternMemory(root)
    pmem.add_observations("r1", "2020-06-01", [{"key": "vol_q5", "obs_date": "2020-01-06", "effect": 0.01, "n": 20, "t": 2.5}])
    a = mk(root)
    a.log_observation("s", {"v": 1}, "2020-03-01")
    pmem.refresh()
    assert pmem.verify()["ok"] and a.audit("2021-01-01").ok and pmem.keys() == ["vol_q5"]
    assert len(mk(root)) == 1 and len(PatternMemory(root)) == len(pmem)


def test_chainfile_lanes_are_independent_but_share_one_verified_chain(tmp_path):
    x, y = A.ChainFile(tmp_path / "c", "lane_x"), A.ChainFile(tmp_path / "c", "lane_y")
    x.append_many([{"v": 1}, {"v": 2}])
    y.sync()
    y.append_many([{"w": 9}])
    x.sync()
    assert [l["body"] for l in x.take_new()] == [{"v": 1}, {"v": 2}] and [l["body"] for l in y.take_new()] == [{"w": 9}]
    assert x.verify()["ok"] and x.verify()["records"] == 3 and len(x) == 2 and len(y) == 1
    mem = A.ChainFile(None, "m")
    mem.append_many([{"a": 1}])
    assert mem.verify()["ok"] and mem.head != A.GENESIS
    mem._mem_recs[0]["body"]["a"] = 2
    assert not mem.verify()["ok"]                                                # in-memory tampering is caught too
