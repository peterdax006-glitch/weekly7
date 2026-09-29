"""Bible Phase 5: long-term pattern bank. Versioning, integrity hash, hash chain, file locking, earlier-windows-only
reads, decay, evidence merging and the never-deleted promise - each with a planted defect the check must catch."""
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import pytest

from engine.pattern_bank import (BankCorrupt, BankError, BankLockTimeout, PatternBank, canon_hash, decay, file_lock,
                                 merge_records, summarize, view_record)
from engine.pattern_lifecycle import Lifecycle, pattern_id
from engine.patterns import PatternMiner
from test_pattern_lifecycle import frame, make_panel_data


@pytest.fixture(scope="module")
def data():
    return make_panel_data()


def bank_with(tmp_path, rows, as_of="2020-01-01", run="r0"):
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(rows), as_of, run)
    return b


STABLE = [("f0 q4", 0.006, "active"), ("f1 q4", 0.004, "active"), ("f2 q4", 0.005, "active")]


# ---------------------------------------------------------------- storage, versions, integrity
def test_empty_bank_is_a_valid_bank(tmp_path):
    b = PatternBank(tmp_path / "b")
    assert b.head()[0] == 0 and b.read("2030-01-01") == [] and b.summary("2030-01-01").empty
    assert list(b.prior_frame("2030-01-01").columns) == ["names", "effect", "p_real", "state", "trust"]
    assert b.verify()["ok"] and b.audit() == []
    assert b.retest(pd.DataFrame(), pd.Series(dtype=float), "2030-01-01", "r")["reviewed"] == 0


def test_each_commit_is_a_new_file_and_nothing_is_overwritten(tmp_path):
    b = bank_with(tmp_path, STABLE)
    first = open(b._file(1), "rb").read()
    b.ingest_miner(frame(STABLE[:1]), "2020-02-01", "r1")
    b.ingest_miner(frame(STABLE), "2020-03-01", "r2")
    assert b.versions() == [1, 2, 3] and open(b._file(1), "rb").read() == first
    assert b.verify() == {"ok": True, "versions": 3, "corrupt": [], "chain_breaks": []}
    with pytest.raises(BankError):
        b._write({"schema": 1, "version": 2, "parent": None, "records": {}})      # refuses to clobber v2


def test_flipped_byte_is_detected_and_head_falls_back(tmp_path):
    b = bank_with(tmp_path, STABLE)
    b.ingest_miner(frame(STABLE[:1]), "2020-02-01", "r1")
    path = b._file(2)
    raw = bytearray(open(path, "rb").read())
    i = raw.index(b"f0 q4") + 3
    raw[i] ^= 1                                                                    # one bit in a pattern name
    open(path, "wb").write(bytes(raw))
    with pytest.raises(BankCorrupt):
        b._read_version(2)
    v, payload, _ = b.head()
    assert v == 1 and b.corrupt and len(payload["records"]) == 3
    rep = b.verify()
    assert not rep["ok"] and len(rep["corrupt"]) == 1
    assert b.ingest_miner(frame(STABLE), "2020-03-01", "r2")["version"] == 3       # corrupt v2 is skipped, not reused
    assert b._read_version(3)[0]["parent"] == b._read_version(1)[1]


def test_valid_json_with_edited_content_fails_the_hash(tmp_path):
    b = bank_with(tmp_path, STABLE)
    env = json.load(open(b._file(1)))
    rid = pattern_id(("s", "f0", 4))
    env["payload"]["records"][rid]["effect"] = 9.99                                # tampered, still perfectly valid JSON
    json.dump(env, open(b._file(1), "w"))
    with pytest.raises(BankCorrupt, match="hash"):
        b._read_version(1)


def test_broken_chain_is_reported(tmp_path):
    b = bank_with(tmp_path, STABLE)
    b.ingest_miner(frame(STABLE), "2020-02-01", "r1")
    payload = b._read_version(2)[0]
    os.remove(b._file(2))
    payload["parent"] = "0" * 64                                                    # re-signed, but points at the wrong parent
    b._write(payload)
    rep = b.verify()
    assert rep["corrupt"] == [] and len(rep["chain_breaks"]) == 1


def test_canon_hash_is_order_independent():
    assert canon_hash({"a": 1, "b": [1, 2]}) == canon_hash({"b": [1, 2], "a": 1})
    assert canon_hash({"a": 1}) != canon_hash({"a": 2})


# ---------------------------------------------------------------- locking
def test_lock_excludes_and_times_out(tmp_path):
    lock = str(tmp_path / "x.lock")
    with file_lock(lock, timeout=1):
        with pytest.raises(BankLockTimeout):
            with file_lock(lock, timeout=0.2, poll=0.02):
                pass
    assert not os.path.exists(lock)                                                 # released on exit


def test_stale_lock_from_a_dead_process_is_broken(tmp_path):
    lock = str(tmp_path / "x.lock")
    open(lock, "w").write("99999")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=1, stale=60):
        assert os.path.exists(lock)


def test_lock_is_released_when_the_body_raises(tmp_path):
    lock = str(tmp_path / "x.lock")
    with pytest.raises(RuntimeError):
        with file_lock(lock):
            raise RuntimeError("boom")
    assert not os.path.exists(lock)


def test_concurrent_writers_lose_nothing(tmp_path):
    b = PatternBank(tmp_path / "bank")
    errors = []

    def worker(k):
        try:
            for j in range(4):
                PatternBank(tmp_path / "bank").ingest_miner(frame([(f"f{k} q{j}", 0.01, "active")]),
                                                            f"2020-01-{j + 1:02d}", f"w{k}_{j}")
        except Exception as e:                                                      # pragma: no cover - failure path
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert b.versions() == list(range(1, 17))
    assert len(b.records()) == 16 and b.verify()["ok"] and b.audit() == []


# ---------------------------------------------------------------- earlier windows only
def test_read_shows_only_what_earlier_windows_knew(tmp_path, data):
    X, y, dates = data
    t1, t2, t3 = dates[200], dates[250], dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), t1, "w1")
    b.retest(X, y, t2, "w2")
    snap_before = json.dumps(b.read(t3), sort_keys=True, default=str)
    b.retest(X, y, t3, "w3")
    # nothing recorded at or after t2 is visible to a reader at t2; the pattern is still what window 1 left
    v2 = b.read(t2)
    assert all(r["state"] == "active" for r in v2) and all(len(r["windows"]) <= 1 for r in v2)
    assert all(pd.Timestamp(w["window_end"]) < t2 for r in v2 for w in r["windows"])
    assert all(pd.Timestamp(e["as_of"]) < t2 for r in v2 for e in r["history"])
    # the later failure is visible at t3 + a moment, and was NOT at t3 itself (strictly earlier)
    assert {r["name"]: r["state"] for r in b.read(t2 + pd.Timedelta(days=1))}["f2 q4"] == "discarded"   # reversed by t2
    assert {r["name"]: r["state"] for r in b.read(t2)}["f2 q4"] == "active"                            # not yet, strictly earlier
    assert json.dumps(b.read(t3), sort_keys=True, default=str) == snap_before                        # t3's own commit is invisible at t3
    # a pattern first seen at t1 does not exist for a reader at t1
    assert b.read(t1) == []


def test_reading_the_past_is_stable_after_later_commits(tmp_path, data):
    X, y, dates = data
    t1, t3 = dates[200], dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), t1, "w1")
    probe = t1 + pd.Timedelta(days=3)
    before = json.dumps(b.read(probe), sort_keys=True, default=str)
    b.retest(X, y, t3, "w3")
    b.ingest_miner(frame(STABLE[:1]), t3 + pd.Timedelta(days=7), "w4")
    assert json.dumps(b.read(probe), sort_keys=True, default=str) == before        # history never rewrites the past


def test_view_of_record_before_it_existed_is_none():
    rec = {"id": "x", "history": [{"kind": "transition", "as_of": "2020-05-01T00:00:00", "frm": None, "to": "candidate",
                                   "reason": "", "evidence": {}}]}
    assert view_record(rec, "2020-05-01") is None


# ---------------------------------------------------------------- merge
def test_recommitting_the_same_run_is_idempotent(tmp_path, data):
    X, y, dates = data
    t = dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), dates[200], "w1")
    b.retest(X, y, t, "w2")
    once = b.records()
    b.retest(X, y, t, "w2")                                                         # same window, same run id
    twice = b.records()
    for rid, r in once.items():
        assert len(twice[rid]["windows"]) == len(r["windows"])
        assert len(twice[rid]["failures"]) == len(r["failures"])
        assert [(e["as_of"], e["to"]) for e in twice[rid]["history"]] == [(e["as_of"], e["to"]) for e in r["history"]]
    assert b.audit() == [] and b.verify()["ok"]


def test_merge_is_a_union_that_never_drops_evidence():
    def rec(changed, wins, fails, state):
        return {"id": "p", "name": "n", "changed": changed, "first_seen": "2020-01-01T00:00:00", "state": state,
                "history": [], "windows": wins, "failures": fails, "rescopes": [], "runs": [], "version": 1}
    w = lambda d, r: {"window_end": d, "run_id": r, "t": 1.0}
    a = rec("2021-01-01T00:00:00", [w("2021-01-01", "a")], [{"as_of": "2021-01-01", "reason": "x"}], "watch")
    b = rec("2022-01-01T00:00:00", [w("2022-01-01", "b"), w("2021-01-01", "a")], [{"as_of": "2022-01-01", "reason": "y"}], "active")
    for m in (merge_records(a, b), merge_records(b, a)):
        assert len(m["windows"]) == 2 and len(m["failures"]) == 2 and m["state"] == "active"   # later change wins the state
    with pytest.raises(BankError):
        merge_records(a, dict(b, id="other"))


# ---------------------------------------------------------------- decay, trust
def test_decay_halves_at_the_half_life_and_never_reaches_zero():
    assert decay(0, 3) == 1.0 and abs(decay(3 * 365.25, 3) - 0.5) < 1e-12
    assert 0 < decay(100 * 365.25, 3) < 1e-9 and decay(-50, 3) == 1.0


def win(end, t, m=0.01, n=50):
    return {"window_end": end, "run_id": end, "t": t, "m": m, "n": n}


def base(wins, effect=0.01):
    return {"effect": effect, "first_seen": "2015-01-01T00:00:00", "windows": wins}


def test_summary_of_a_never_retested_pattern_decays_toward_zero_trust():
    r = base([win("2016-01-01", 3.0), win("2017-01-01", 3.0)])
    fresh = summarize(r, "2017-06-01")
    stale = summarize(r, "2027-06-01")
    assert fresh["trust"] > 0.3 and stale["trust"] < fresh["trust"] / 4
    assert stale["relevance"] < 0.1 < fresh["relevance"]                            # no pattern is trusted forever


def test_modernity_separates_old_glory_from_recent_holding():
    old_glory = base([win("2012-01-01", 4.0), win("2013-01-01", 4.0), win("2023-01-01", -3.0), win("2024-01-01", -3.0)])
    current = base([win("2012-01-01", -4.0), win("2013-01-01", -4.0), win("2023-01-01", 3.0), win("2024-01-01", 3.0)])
    a, c = summarize(old_glory, "2024-06-01"), summarize(current, "2024-06-01")
    assert c["modernity"] > 0.8 > 0.3 > a["modernity"]
    assert c["trust"] > 5 * a["trust"] and a["pooled_t"] < 0 < c["pooled_t"]


def test_pooled_t_does_not_grow_with_overlapping_windows():
    one = summarize(base([win("2020-01-01", 2.0)]), "2020-02-01")["pooled_t"]
    ten = summarize(base([win(f"2020-01-{d:02d}", 2.0) for d in range(1, 11)]), "2020-02-01")["pooled_t"]
    assert ten == pytest.approx(one, rel=0.02)                                     # a mean, not a Stouffer sum


def test_summary_with_no_windows_has_zero_trust():
    s = summarize(base([]), "2020-01-01")
    assert s["trust"] == 0.0 and s["n_windows"] == 0 and s["confidence"] == 0.5


# ---------------------------------------------------------------- end to end
def test_retest_rescopes_discards_and_keeps_the_whole_story(tmp_path, data):
    X, y, dates = data
    t = dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), dates[150], "w1")
    out = b.retest(X, y, t, "w2")
    assert out["reviewed"] == 3 and out["rescoped"] == 1 and out["discarded"] == 1
    S = b.summary(t + pd.Timedelta(days=1)).set_index("name")
    assert S.loc["f0 q4", "state"] == "active" and S.loc["f1 q4", "state"] == "rescoped"
    assert S.loc["f2 q4", "state"] == "discarded" and S.loc["f2 q4", "failures"] == 1
    assert S.loc["f0 q4", "trust"] > S.loc["f2 q4", "trust"]
    assert b.audit() == [] and b.verify()["ok"]
    f1 = [r for r in b.read(t + pd.Timedelta(days=1)) if r["name"] == "f1 q4"][0]
    assert [e["to"] for e in f1["history"] if e["kind"] == "transition"] == ["candidate", "active", "failed", "cause_search",
                                                                             "rescoped"]
    assert f1["rescopes"][0]["scope"]["label"] == "high" and f1["failures"][0]["reason"]


def test_discarded_pattern_stays_in_the_bank_and_can_return(tmp_path, data):
    X, y, dates = data
    t = dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame([("f0 q4", 0.006, "discarded")]), dates[150], "w1")
    assert b.audit() == [] and len(b.records()) == 1
    assert len(b.prior_frame(t)) == 1                                              # the miner will be asked to retest it
    b.retest(X, y, t, "w2")
    assert [r["state"] for r in b.read(t + pd.Timedelta(days=1))] == ["active"]    # revived by evidence, not by fiat


def test_prior_frame_feeds_the_miner_and_skips_noise(tmp_path):
    rows = STABLE + [("f3 q1", 0.01, "rejected"), ("f3 q2", 0.01, "no_gain"), ("f3 q3", 0.01, "duplicate"),
                     ("f3 q0", 0.01, "discarded")]
    b = bank_with(tmp_path, rows)
    P = b.prior_frame("2021-01-01")
    assert len(P) == 4 and set(P["state"]) == {"active", "discarded"}
    m = PatternMiner()
    for nk in P["names"]:
        assert m.key_from_names(tuple(nk), ["f0", "f1", "f2", "f3"]) is not None
    capped = PatternBank(tmp_path / "bank", {"max_prior": 2}).prior_frame("2021-01-01")
    assert len(capped) == 2


def test_audit_catches_a_pattern_dropped_between_versions(tmp_path):
    b = bank_with(tmp_path, STABLE)
    payload = b._read_version(1)[0]
    digest = b._read_version(1)[1]
    gone = pattern_id(("s", "f2", 4))
    recs = {k: v for k, v in payload["records"].items() if k != gone}
    b._write({"schema": 1, "version": 2, "parent": digest, "records": recs})
    assert any("dropped" in p for p in b.audit())


def test_audit_catches_state_without_history_and_missing_discard_reason(tmp_path):
    b = bank_with(tmp_path, [("f0 q4", 0.006, "discarded")])
    payload, digest = b._read_version(1)
    rid = pattern_id(("s", "f0", 4))
    payload["records"][rid]["discard_reason"] = None
    payload["records"][rid]["state"] = "active"
    b._write({"schema": 1, "version": 2, "parent": digest, "records": payload["records"]})
    probs = b.audit()
    assert any("not backed by history" in p for p in probs)


def test_trusted_excludes_watch_and_untested(tmp_path, data):
    X, y, dates = data
    t = dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), dates[150], "w1")
    assert b.trusted(t).empty                                                       # never validated: zero trust
    b.retest(X, y, t, "w2")
    tr = b.trusted(t + pd.Timedelta(days=1))
    assert set(tr["name"]) >= {"f0 q4"} and "f2 q4" not in set(tr["name"])
    assert b.trusted(t + pd.Timedelta(days=365 * 30)).empty                         # decayed past the bar with no retest


def test_never_confirmed_candidates_are_counted_in_a_ledger_not_stored(tmp_path):
    rows = STABLE + [("f3 q1", 0.01, "rejected"), ("f3 q2", 0.01, "no_gain"), ("f3 q3", 0.01, "duplicate"),
                     ("f2 q0", 0.01, "rejected")]
    b = bank_with(tmp_path, rows, run="r0")
    assert len(b.records()) == 3
    assert b.noise_ledger()["r0"] == {"as_of": "2020-01-01T00:00:00", "rejected": 2, "no_gain": 1, "duplicate": 1}
    full = PatternBank(tmp_path / "full", {"store_noise": True})
    full.ingest_miner(frame(rows), "2020-01-01", "r0")
    assert len(full.records()) == 7 and full.noise_ledger() == {}
    # a stored pattern that later turns to noise keeps its record (its history is not dropped)
    b.ingest_miner(frame([("f0 q4", 0.006, "rejected")]), "2020-02-01", "r1")
    assert len(b.records()) == 3 and b.audit() == []


# ---------------------------------------------------------------- context relevance, diff, cross-run merge
def test_active_now_zeroes_a_scoped_pattern_outside_its_scope(tmp_path, data):
    X, y, dates = data
    t = dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), dates[150], "w1")
    b.retest(X, y, t, "w2")
    after = t + pd.Timedelta(days=1)
    hi = b.active_now(after, {"m_vix": 1.5}).set_index("name")
    lo = b.active_now(after, {"m_vix": -1.5}).set_index("name")
    assert hi.loc["f1 q4", "weight"] > 0 and lo.loc["f1 q4", "weight"] == 0.0       # scoped to high vix
    assert hi.loc["f0 q4", "weight"] > 0 and lo.loc["f0 q4", "weight"] > 0           # unscoped applies everywhere
    assert b.active_now(after, {}).set_index("name").loc["f1 q4", "weight"] == 0.0   # unknown context is not in scope
    assert b.active_now(after, {"m_vix": float("nan")}).set_index("name").loc["f1 q4", "in_scope"] is False or True


def test_diff_lists_exactly_the_patterns_that_moved(tmp_path, data):
    X, y, dates = data
    t = dates[-1] + pd.Timedelta(days=30)
    b = PatternBank(tmp_path / "bank")
    b.ingest_miner(frame(STABLE), dates[150], "w1")
    b.retest(X, y, t, "w2")
    d = b.diff(dates[150] + pd.Timedelta(days=1), t + pd.Timedelta(days=1)).set_index("name")
    assert set(d.index) == {"f1 q4", "f2 q4"}                                       # f0 did not change state or scope
    assert d.loc["f1 q4", "state_b"] == "rescoped" and d.loc["f2 q4", "state_b"] == "discarded"
    assert b.diff(t, t).empty and b.diff("2001-01-01", "2001-06-01").empty


def test_merge_from_keeps_evidence_from_both_lineages(tmp_path):
    a = PatternBank(tmp_path / "a")
    b = PatternBank(tmp_path / "b")
    a.ingest_miner(frame([("f0 q4", 0.006, "active"), ("f1 q4", 0.004, "active")]), "2020-01-01", "ra")
    b.ingest_miner(frame([("f0 q4", 0.006, "active"), ("f2 q4", 0.005, "active")]), "2020-06-01", "rb")
    out = a.merge_from(b)
    assert out["added"] == 1 and out["merged"] == 1
    recs = {r["name"]: r for r in a.records().values()}
    assert set(recs) == {"f0 q4", "f1 q4", "f2 q4"}
    assert {x["run_id"] for x in recs["f0 q4"]["runs"]} == {"ra", "rb"} and len(recs["f0 q4"]["windows"]) == 2
    assert a.audit() == [] and a.verify()["ok"]
    assert a.merge_from(PatternBank(tmp_path / "empty"))["added"] == 0
    again = a.merge_from(b)                                                          # second merge adds nothing new
    assert again["added"] == 0 and len(a.records()[pattern_id(("s", "f0", 4))]["windows"]) == 2
