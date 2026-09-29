"""Tests for engine.learning.compute (section 57) and engine.learning.checkpoints (sections 58, 59, 84).
Synthetic data only; no real processes are started (liveness and memory are injected)."""
import json
import os
import stat

import numpy as np
import pytest

from engine.learning import checkpoints as CP
from engine.learning import compute as CM
from engine.learning.core import BuildStatus, FirewallBreach, ValidationLabel, current_code_hash

HASH = "code_v1"
NEWHASH = "code_v2"


def spec(name="exp", seed=1, as_of="2022-01-03", **kw):
    kw.setdefault("params", {"k": 3, "lr": 0.1})
    return CM.ExperimentSpec(name, seed=seed, as_of=as_of, **kw)


def fn_ok(sp, ctx):
    r = ctx.rng("main").normal(size=5)
    return {"mean": float(r.mean()), "vals": [float(x) for x in r], "np": np.float64(1.5)}


def make(tmp_path, **kw):
    return CM.ExperimentLedger(tmp_path / "ledger.json", **kw), tmp_path / "out"


# ================================================================================================ specs and determinism
def test_experiment_key_is_stable_and_sensitive():
    a = spec()
    assert a.key == spec().key and a.run_key(HASH) != a.run_key(NEWHASH)
    assert spec(seed=2).key != a.key and spec(params={"k": 4, "lr": 0.1}).key != a.key and spec(as_of="2022-01-04").key != a.key
    assert spec(params={"lr": 0.1, "k": 3}).key == a.key                       # dict order is irrelevant
    assert spec(params={"k": np.int64(3), "lr": np.float64(0.1)}).key == a.key   # numpy scalars do not change identity


def test_spec_validation_catches_bad_input():
    assert spec(name="a/b").validate()
    assert spec(seed=-1).validate()
    assert spec(as_of="not-a-date").validate()
    assert spec(est_gb=0).validate()
    assert spec(params={"x": object()}).validate()
    assert not spec().validate()


def test_worker_streams_are_deterministic_and_independent():
    ctx = lambda sp: CM.WorkerContext(sp, None, {}, HASH, "w", 1)
    a1, a2 = ctx(spec()).rng("x").normal(size=4), ctx(spec()).rng("x").normal(size=4)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, ctx(spec()).rng("y").normal(size=4))
    assert not np.array_equal(a1, ctx(spec(seed=2)).rng("x").normal(size=4))
    assert CM.worker_seed_for(spec()) != CM.worker_seed_for(spec(name="other"))


def test_two_workers_in_different_folders_produce_identical_results(tmp_path):
    outs = []
    for i in range(2):
        led = CM.ExperimentLedger(tmp_path / f"l{i}.json")
        led.submit(spec(), 0, HASH)
        outs.append(CM.run_worker(spec(), fn_ok, led, None, tmp_path / f"o{i}", f"w{i}", 1.0, HASH))
    assert outs[0]["state"] == CM.DONE and outs[0]["result_hash"] == outs[1]["result_hash"]


# ================================================================================================ snapshots
def test_snapshot_is_content_addressed_and_tamper_evident(tmp_path):
    st = CM.SnapshotStore(tmp_path / "snap")
    sid = st.create({"patterns": [1, 2, np.float64(0.5)]}, "2021-12-31", HASH)
    assert st.create({"patterns": [1, 2, 0.5]}, "2021-12-31", HASH) == sid
    assert st.load(sid)["state"]["patterns"] == [1, 2, 0.5]
    f = tmp_path / "snap" / sid / "snapshot.json"
    os.chmod(f, stat.S_IWRITE)
    f.write_text(f.read_text(encoding="utf-8").replace("0.5", "0.9"), encoding="utf-8")
    with pytest.raises(CM.SnapshotError):
        st.load(sid)
    with pytest.raises(CM.SnapshotError):
        st.load("nope")


def test_snapshot_from_the_future_is_a_firewall_breach(tmp_path):
    st = CM.SnapshotStore(tmp_path / "snap")
    late = st.create({"x": 1}, "2022-01-03", HASH)                            # contains info through the experiment's as_of
    with pytest.raises(FirewallBreach):
        st.for_experiment(spec(snapshot_id=late))
    early = st.create({"x": 1}, "2022-01-02", HASH)
    assert st.for_experiment(spec(snapshot_id=early))["state"] == {"x": 1}
    assert st.for_experiment(spec())["state"] == {}
    out = st.materialize(early, tmp_path / "w" / "in")
    assert out.exists()


# ================================================================================================ duplicate prevention
def test_duplicate_active_and_done_experiments_are_refused(tmp_path):
    led, out = make(tmp_path)
    assert led.submit(spec(), 0, HASH).action == "queued"
    with pytest.raises(CM.DuplicateExperiment):
        led.submit(spec(), 1, HASH)                                            # already PENDING
    CM.run_worker(spec(), fn_ok, led, None, out, "w", 2.0, HASH)
    with pytest.raises(CM.DuplicateExperiment, match="already finished"):
        led.submit(spec(), 3, HASH)
    r = led.submit(spec(), 4, NEWHASH)                                         # same question, new code: allowed, marked superseding
    assert r.action == "superseding" and led.get(spec().key)["runs"][0]["code_hash"] == HASH


def test_only_one_worker_can_claim(tmp_path):
    led, _ = make(tmp_path)
    led.submit(spec(), 0, HASH)
    led.claim(spec().key, "w1", 1, HASH)
    with pytest.raises(CM.ClaimError):
        led.claim(spec().key, "w2", 1, HASH)
    led.submit(spec(name="b"), 0, HASH)
    with pytest.raises(CM.ClaimError, match="code"):
        led.claim(spec(name="b").key, "w1", 1, NEWHASH)
    with pytest.raises(CM.ClaimError):
        led.claim("missing", "w1", 1, HASH)


def test_attempt_directories_are_never_shared(tmp_path):
    led, out = make(tmp_path)
    led.submit(spec(), 0, HASH)
    (CM.attempt_dir(out, spec().key, 1)).mkdir(parents=True)                   # a leftover folder from an earlier crash
    with pytest.raises(CM.ComputeError, match="never share"):
        CM.run_worker(spec(), fn_ok, led, None, out, "w", 1.0, HASH)


# ================================================================================================ failure handling
def test_oom_grows_memory_estimate_shrinks_batch_and_retries(tmp_path):
    led, out = make(tmp_path, max_attempts=3)
    sp = spec(est_gb=2.0)
    calls = []

    def fn(s, ctx):
        calls.append((ctx.attempt, ctx.batch_scale))
        if ctx.attempt < 3:
            raise MemoryError("Unable to allocate 4 GiB")
        return {"ok": True, "scale": ctx.batch_scale}

    led.submit(sp, 0, HASH)
    for _ in range(3):
        res = CM.run_worker(sp, fn, led, None, out, "w", 1.0, HASH)
        if res["state"] == CM.DONE:
            break
        assert res["state"] == CM.OOM and res["next"] == CM.PENDING
    e = led.get(sp.key)
    assert e["state"] == CM.DONE and calls == [(1, 1.0), (2, 0.5), (3, 0.25)]
    assert e["est_gb"] == pytest.approx(2.0 * 1.5 * 1.5)


def test_attempt_cap_gives_up_and_refuses_resubmission(tmp_path):
    led, out = make(tmp_path, max_attempts=2)
    sp = spec()

    def bad(s, ctx):
        raise ValueError("boom")

    led.submit(sp, 0, HASH)
    assert CM.run_worker(sp, bad, led, None, out, "w", 1.0, HASH)["state"] == CM.FAILED
    assert led.get(sp.key)["state"] == CM.PENDING
    assert CM.run_worker(sp, bad, led, None, out, "w", 2.0, HASH)["next"] == CM.GAVE_UP
    with pytest.raises(CM.DuplicateExperiment, match="exhausted"):
        led.submit(sp, 3, HASH)
    assert (CM.attempt_dir(out, sp.key, 1) / "error.txt").read_text(encoding="utf-8").count("boom") >= 1


def test_firewall_breach_is_recorded_and_reraised(tmp_path):
    led, out = make(tmp_path)

    def leak(s, ctx):
        raise FirewallBreach("future row")

    led.submit(spec(), 0, HASH)
    with pytest.raises(FirewallBreach):
        CM.run_worker(spec(), leak, led, None, out, "w", 1.0, HASH)
    assert "FirewallBreach" in json.dumps(led.get(spec().key)["history"])


def test_unserialisable_result_fails_the_attempt_not_the_reconcile(tmp_path):
    led, out = make(tmp_path)
    led.submit(spec(), 0, HASH)
    res = CM.run_worker(spec(), lambda s, c: {"x": object.__new__(type("Odd", (), {}))}, led, None, out, "w", 1.0, HASH)
    assert res["state"] != CM.DONE


def test_crash_recovery_requeues_dead_and_silent_workers(tmp_path):
    led, _ = make(tmp_path, max_attempts=2)
    for n in ("a", "b", "c"):
        led.submit(spec(n), 0, HASH)
        led.claim(spec(n).key, f"w_{n}", 100.0, HASH)
    led.heartbeat(spec("c").key, 190.0)
    found = led.recover(200.0, alive=lambda w: w != "w_a", heartbeat_s=60.0)
    by = {f["key"]: f for f in found}
    assert by[spec("a").key]["why"] == "worker gone" and by[spec("b").key]["why"].startswith("silent")
    assert spec("c").key not in by                                              # alive and beating
    assert led.get(spec("a").key)["state"] == CM.PENDING
    led.claim(spec("a").key, "w2", 210.0, HASH)
    led.recover(211.0, alive=lambda w: False)
    assert led.get(spec("a").key)["state"] == CM.GAVE_UP                        # second crash hits the cap


def test_stale_code_is_detected_and_marked(tmp_path):
    led, out = make(tmp_path)
    led.submit(spec("a"), 0, HASH)
    led.submit(spec("b"), 0, HASH)
    CM.run_worker(spec("b"), fn_ok, led, None, out, "w", 1.0, HASH)
    assert led.stale(HASH) == [] and sorted(led.stale(NEWHASH)) == sorted([spec("a").key, spec("b").key])
    assert sorted(led.mark_stale(NEWHASH, 5.0)) == sorted([spec("a").key, spec("b").key])
    assert led.counts() == {CM.STALE: 2}
    assert led.submit(spec("a"), 6.0, NEWHASH).action == "requeued"             # STALE may be retried under the new code


def test_classify_failure():
    assert CM.classify_failure(MemoryError()) == CM.OOM
    assert CM.classify_failure(RuntimeError("std::bad_alloc")) == CM.OOM
    assert CM.classify_failure(None, returncode=137) == CM.OOM
    assert CM.classify_failure(None, 1, "MemoryError: Unable to allocate") == CM.OOM
    assert CM.classify_failure(None, returncode=1) == CM.CRASHED
    assert CM.classify_failure(ValueError("x")) == CM.FAILED
    assert CM.classify_failure(None, timed_out=True) == CM.CRASHED


# ================================================================================================ reconciliation
def rewrite(path, obj):
    """Edit a sealed (read-only) result the way a careless or hostile process would."""
    os.chmod(path, stat.S_IWRITE)
    path.write_text(json.dumps(obj), encoding="utf-8")


def run_all(tmp_path, names=("a", "b", "c"), code=HASH):
    led, out = make(tmp_path)
    for n in names:
        led.submit(spec(n), 0, code)
        CM.run_worker(spec(n), fn_ok, led, None, out, "w", 1.0, code)
    return led, out


def test_clean_reconcile_accepts_everything_in_key_order(tmp_path):
    led, out = run_all(tmp_path)
    rec = CM.reconcile(led, out, HASH)
    assert rec.clean and [k for k, _ in rec.accepted] == sorted(k for k, _ in rec.accepted) and len(rec.accepted) == 3


def test_reconcile_catches_tampered_result_missing_marker_and_orphans(tmp_path):
    led, out = run_all(tmp_path)
    ka, kb = spec("a").key, spec("b").key
    rf = CM.attempt_dir(out, ka, 1) / "result.json"
    d = json.loads(rf.read_text(encoding="utf-8"))
    d["result"]["mean"] = 99.0
    rewrite(rf, d)
    os.chmod(CM.attempt_dir(out, kb, 1) / "DONE", stat.S_IWRITE)
    os.remove(CM.attempt_dir(out, kb, 1) / "DONE")
    (out / "deadbeef").mkdir()
    rec = CM.reconcile(led, out, HASH)
    assert rec.reasons() == {"hash_mismatch": 1, "missing_result": 1} and rec.orphans == ("deadbeef",)
    assert [k for k, _ in rec.accepted] == [spec("c").key] and not rec.clean


def test_reconcile_rejects_stale_code_wrong_seed_and_wrong_spec(tmp_path):
    led, out = run_all(tmp_path)
    assert CM.reconcile(led, out, NEWHASH).reasons() == {"stale_code": 3}
    kc = spec("c").key
    rf = CM.attempt_dir(out, kc, 1) / "result.json"
    d = json.loads(rf.read_text(encoding="utf-8"))
    d["seed"] = 12345
    rewrite(rf, d)
    ka = spec("a").key
    rf = CM.attempt_dir(out, ka, 1) / "result.json"
    d = json.loads(rf.read_text(encoding="utf-8"))
    d["spec"]["params"] = {"k": 99}
    rewrite(rf, d)
    assert CM.reconcile(led, out, HASH).reasons() == {"seed_mismatch": 1, "spec_mismatch": 1}


def test_duplicate_attempts_that_disagree_prove_nondeterminism(tmp_path):
    led, out = run_all(tmp_path, ("a",))
    ka = spec("a").key
    two = CM.attempt_dir(out, ka, 2)
    two.mkdir()
    env = json.loads((CM.attempt_dir(out, ka, 1) / "result.json").read_text(encoding="utf-8"))
    env["result"] = {"mean": 0.123}
    env["result_hash"] = CM.phash(env["result"], 20)
    (two / "result.json").write_text(json.dumps(env), encoding="utf-8")
    (two / "DONE").write_text(env["result_hash"], encoding="utf-8")
    assert CM.reconcile(led, out, HASH).reasons() == {"nondeterministic": 1}


def test_merge_does_not_depend_on_completion_order(tmp_path):
    led, out = run_all(tmp_path, ("a", "b", "c", "d"))
    rec = CM.reconcile(led, out, HASH)
    shuffled = CM.Reconciliation(tuple(reversed(rec.accepted)), (), ())
    reducer = lambda acc, key, res: acc + [(key, round(res["mean"], 9))]          # order-sensitive on purpose
    assert CM.merge_results(rec, reducer, []) == CM.merge_results(shuffled, reducer, [])
    assert CM.merge_results(CM.Reconciliation((), (), ()), reducer, ["init"]) == ["init"]


def test_empty_ledger_reconciles_cleanly(tmp_path):
    led, out = make(tmp_path)
    rec = CM.reconcile(led, out, HASH)
    assert rec.clean and rec.accepted == () and led.counts() == {} and led.stale(HASH) == []
    assert "no experiments" in CM.compute_report(led, rec)


def test_report_lists_problem_experiments(tmp_path):
    led, out = run_all(tmp_path, ("a",))
    led.submit(spec("z"), 0, HASH)
    led.claim(spec("z").key, "w", 1, HASH)
    led.fail(spec("z").key, CM.FAILED, "bad input", 2)
    rec = CM.reconcile(led, out, NEWHASH)
    txt = CM.compute_report(led, rec)
    assert "stale_code" in txt and "DONE=1" in txt


# ================================================================================================ scheduling
def test_plan_launches_respects_memory_priority_and_slots(tmp_path):
    led, _ = make(tmp_path)
    for i, (n, pri, gb) in enumerate((("hi", 1, 2.0), ("mid", 5, 2.0), ("lo", 9, 2.0), ("big", 3, 6.0))):
        led.submit(spec(n, est_gb=gb, priority=pri), i, HASH)
    plan = CM.plan_launches(led, free_gb=7.0, running=0, cores=8, reserve_gb=2.5)
    assert plan.workers == 2 and list(plan.launch) == [spec("hi").key, spec("mid").key]
    held = dict(plan.held)
    assert "GB" in held[spec("big").key] and "slot" in held[spec("lo").key]
    again = CM.plan_launches(led, free_gb=7.0, running=0, cores=8, reserve_gb=2.5)
    assert again == plan                                                             # deterministic
    assert CM.plan_launches(led, free_gb=9.0, running=7, cores=8).launch == ()       # no free slot


def test_plan_launches_fails_closed_when_memory_unreadable_or_low(tmp_path):
    led, _ = make(tmp_path)
    led.submit(spec(), 0, HASH)
    assert CM.plan_launches(led, free_gb=None).launch == ()
    assert CM.plan_launches(led, free_gb=1.0).launch == ()
    assert CM.plan_launches(CM.ExperimentLedger(tmp_path / "none.json"), 16.0).launch == ()


def test_job_for_and_subprocess_entry(tmp_path):
    led, out = make(tmp_path)
    sp = spec("entry")
    job = CM.job_for(sp, tmp_path / "specs", out, led.path, python="python")
    assert job.job_id == sp.key and job.cmd[2:4] == ["engine.learning.compute", "worker"] and job.est_gb == sp.est_gb
    assert CM.main([]) == 2
    assert CM.main(["worker", job.cmd[-3], str(out), str(led.path)]) == 3           # not registered
    CM.WORKERS.pop("entry", None)
    CM.register_worker("entry")(fn_ok)
    with pytest.raises(CM.ComputeError):
        CM.register_worker("entry")(lambda s, c: 1)
    led.submit(sp, 0, current_code_hash())
    assert CM.main(["worker", job.cmd[-3], str(out), str(led.path)]) == 0
    assert led.get(sp.key)["state"] == CM.DONE
    CM.WORKERS.pop("entry")


# ================================================================================================ checkpoints
def save(store, n=0, **kw):
    kw.setdefault("phase", "p")
    kw.setdefault("next_action", f"do step {n}")
    kw.setdefault("code_hash", HASH)
    return store.save(f"t{n}", **kw)


def test_save_read_roundtrip_and_sequence(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    a = save(s, 0, current_experiment="expkey", completed={"t1": HASH}, pending=("t2",),
             failures=[CP.FailureNote("t0", "t2", "ERROR", "bad")], notes={"why": "x"})
    b = save(s, 1)
    assert (a.sequence, b.sequence) == (0, 1) and s.sequences() == [0, 1]
    assert s.read(0) == a and s.read(0).digest == a.digest
    assert s.latest_valid()[0].sequence == 1 and not s.latest_valid()[1]


def test_invalid_state_is_refused(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    with pytest.raises(CP.CheckpointCorrupt):
        s.save("t", "p", "", HASH)                                                   # no next action
    with pytest.raises(CP.CheckpointCorrupt):
        s.save("t", "p", "go", "")                                                   # no code hash
    with pytest.raises(CP.CheckpointCorrupt):
        s.save("t", "p", "go", HASH, completed={"a": HASH}, pending=("a",))
    with pytest.raises(ValueError):
        CP.CheckpointStore(tmp_path, "../evil")
    with pytest.raises(ValueError):
        CP.CheckpointStore(tmp_path, "r", keep=1)
    assert s.sequences() == []


def test_torn_newest_checkpoint_falls_back_to_the_last_good_one(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    for i in range(4):
        save(s, i)
    p3 = s._path(3)
    p3.write_text(p3.read_text(encoding="utf-8")[:40], encoding="utf-8")            # crash mid-write
    p2 = s._path(2)
    p2.write_text(p2.read_text(encoding="utf-8").replace("do step 2", "do step 9"), encoding="utf-8")   # silent corruption
    st, skipped = s.latest_valid()
    assert st.sequence == 1 and skipped == [3, 2]
    plan = CP.resume(s, HASH)
    assert plan.skipped_corrupt == (3, 2) and "corrupt" in plan.reasons[0]
    for q in s.sequences():
        s._path(q).write_text("garbage", encoding="utf-8")
    assert CP.resume(s, HASH).action == "START_FRESH" and s.latest_valid() == (None, [3, 2, 1, 0])


def test_pruning_keeps_the_newest(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1", keep=4)
    for i in range(9):
        save(s, i)
    assert s.sequences() == [5, 6, 7, 8] and s.age_s(10**10) > 0
    assert CP.CheckpointStore(tmp_path, "empty").age_s(1.0) is None


def test_resume_decisions(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    assert CP.resume(s, HASH).action == "START_FRESH"
    save(s, 0, current_experiment="ek", completed={"a": HASH, "b": "oldcode"}, pending=("c",))
    assert CP.resume(s, HASH, lambda k: "RUNNING").action == "RECOVER_EXPERIMENT"
    assert CP.resume(s, HASH, lambda k: "DONE").action == "RECONCILE_EXPERIMENT"
    p = CP.resume(s, NEWHASH, lambda k: "RUNNING")
    assert p.action == "REDO_EXPERIMENT" and p.revalidate == ("a", "b") and p.code_changed
    save(s, 1, current_experiment="", completed={"a": HASH})
    assert CP.resume(s, HASH).action == "CONTINUE"


def test_artifact_hashes_are_recorded(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    s = CP.CheckpointStore(tmp_path, "run1")
    st = save(s, 0, artifacts=[f, tmp_path / "gone.txt"])
    assert st.artifacts[str(f)] != "MISSING" and st.artifacts[str(tmp_path / "gone.txt")] == "MISSING"


def test_milestone_bundle_is_write_once(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    st = save(s, 0)
    p = s.seal_milestone("phaseA", st, {"tests": 12}, 7, "2022-01-03")
    assert (p / "MANIFEST.json").exists()
    with pytest.raises(Exception):
        s.seal_milestone("phaseA", st, {"tests": 13}, 7, "2022-01-04")


# ================================================================================================ the loop
def test_loop_runs_in_dependency_order_and_continues_past_failure(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    order = []

    def ok(n):
        return lambda: order.append(n)

    def bad():
        raise RuntimeError("nope")

    tasks = [CP.Task("c", ok("c"), depends=("b",)), CP.Task("a", ok("a")), CP.Task("b", bad, depends=("a",), retries=1),
             CP.Task("d", ok("d"), depends=("a",))]
    res = CP.ExecutionLoop(s, tasks, HASH, clock=lambda: 1.0).run()
    assert order == ["a", "d"] and res.failed == ("b",) and res.blocked == ("c",) and set(res.done) == {"a", "d"}
    st = res.state
    assert st.pending == ("b", "c") and len(st.open_failures()) == 2 and st.completed == {"a": HASH, "d": HASH}
    assert "next: b" in st.next_action


def test_loop_retries_then_succeeds_and_marks_failures_resolved(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    n = {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] < 3:
            raise MemoryError("oom")

    res = CP.ExecutionLoop(s, [CP.Task("t", flaky, retries=2)], HASH, clock=lambda: 1.0).run()
    assert res.done == ("t",) and n["i"] == 3 and not res.state.open_failures()
    assert {f.kind for f in res.state.failures} == {"OOM"} and all(f.resolved for f in res.state.failures)


def test_loop_resumes_and_skips_completed_work_unless_code_changed(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    ran = []
    tasks = [CP.Task("a", lambda: ran.append("a")), CP.Task("b", lambda: ran.append("b"), depends=("a",))]
    CP.ExecutionLoop(s, tasks, HASH, clock=lambda: 1.0).run()
    res = CP.ExecutionLoop(s, tasks, HASH, clock=lambda: 2.0).run()
    assert ran == ["a", "b"] and res.skipped == ("a", "b") and "all tasks complete" in res.state.next_action
    res = CP.ExecutionLoop(s, tasks, NEWHASH, clock=lambda: 3.0).run()
    assert ran == ["a", "b", "a", "b"] and res.done == ("a", "b")


def test_loop_stops_on_firewall_breach_after_saving_state(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")

    def leak():
        raise FirewallBreach("label in feature")

    with pytest.raises(FirewallBreach):
        CP.ExecutionLoop(s, [CP.Task("x", leak, retries=5)], HASH, clock=lambda: 1.0).run()
    st = s.latest_valid()[0]
    assert st.failures[-1].kind == "FIREWALL" and st.failures[-1].escalated and "INVESTIGATE" in st.next_action
    assert len([f for f in st.failures if f.kind == "FIREWALL"]) == 1               # not retried


def test_topological_order_rejects_cycles_and_unknown_dependencies():
    f = lambda: None
    with pytest.raises(ValueError, match="cycle"):
        CP.topological_order([CP.Task("a", f, ("b",)), CP.Task("b", f, ("a",))])
    with pytest.raises(ValueError, match="unknown"):
        CP.topological_order([CP.Task("a", f, ("zzz",))])
    with pytest.raises(ValueError, match="duplicate"):
        CP.topological_order([CP.Task("a", f), CP.Task("a", f)])
    assert CP.topological_order([]) == []
    assert [t.name for t in CP.topological_order([CP.Task("b", f), CP.Task("a", f)])] == ["a", "b"]


# ================================================================================================ checklist and stop
def test_checklist_never_validates_without_evidence(tmp_path):
    t = CP.ChecklistTracker(tmp_path / "cl.json")
    t.add("J01", "champion board")
    t.add("J02", "promotion gate", critical=False)
    with pytest.raises(ValueError):
        t.mark("J01", BuildStatus.VALIDATED)
    with pytest.raises(ValueError):
        t.mark("J01", BuildStatus.IMPLEMENTED, label=ValidationLabel.VALIDATED)
    it = t.mark("J01", BuildStatus.IMPLEMENTED)
    assert it.label == ValidationLabel.NOT_VALIDATED
    assert t.next_incomplete().item_id == "J01"
    t.mark("J01", BuildStatus.VALIDATED, ["run 2026-10-01: learning delta +0.4%"])
    assert t.next_incomplete() is None and t.next_incomplete(critical_only=False).item_id == "J02"
    t2 = CP.ChecklistTracker(tmp_path / "cl.json")
    assert t2.items["J01"].status == BuildStatus.VALIDATED and t2.summary() == {"VALIDATED": 1, "NOT_STARTED": 1}
    with pytest.raises(ValueError):
        t2.add("J01", "dup")


def test_false_completion_is_detected_in_a_hand_edited_checklist(tmp_path):
    t = CP.ChecklistTracker(tmp_path / "cl.json")
    t.add("J01", "x")
    t.add("J02", "y")
    raw = json.loads((tmp_path / "cl.json").read_text(encoding="utf-8"))
    raw["J01"]["status"] = "VALIDATED"                                              # claimed with no evidence
    raw["J02"]["label"] = "VALIDATED"                                               # label with no status
    (tmp_path / "cl.json").write_text(json.dumps(raw), encoding="utf-8")
    assert CP.ChecklistTracker(tmp_path / "cl.json").falsely_complete() == ["J01", "J02"]


def test_stop_conditions_all_required():
    assert not CP.evaluate_stop(CP.CompletionEvidence()).can_stop                    # empty evidence is not completion
    full = CP.CompletionEvidence({"44": True}, {"44": (720, 700)}, {"t": True}, {"f": True}, {"r": True}, True, 0, ())
    assert CP.evaluate_stop(full).can_stop
    for change, frag in ((dict(sections_complete={"44": False}), "1."), (dict(line_depth={"44": (650, 700)}), "2."),
                         (dict(tests_exist={"t": False}), "3."), (dict(firewalls_exist={}), "4."),
                         (dict(reports_exist={"r": False}), "5."), (dict(checklist_complete=False), "6."),
                         (dict(open_failures=2), "7."), (dict(falsely_marked=("J01",)), "8.")):
        v = CP.evaluate_stop(CP.CompletionEvidence(**{**{f.name: getattr(full, f.name) for f in __import__("dataclasses").fields(full)}, **change}))
        assert not v.can_stop and any(u.startswith(frag) for u in v.unmet), (frag, v)


def test_false_completion_scan():
    bad = "Section 44 is complete.\nThe engine is production ready\nlearning works now"
    assert [i for i, _ in CP.scan_false_completion(bad)] == [1, 2, 3]
    fine = ("Section 44: IMPLEMENTED — NOT VALIDATED\nnot yet validated on real data\nthe run is incomplete\n"
            "validated by E-2026-10-01-17 (learning delta +0.4%)\nnothing else to say")
    assert CP.scan_false_completion(fine, ["E-2026-10-01-17"]) == []
    assert [i for i, _ in CP.scan_false_completion("done", [])] == [1] and CP.scan_false_completion("") == []


def test_handoff_contains_the_seven_items(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    st = save(s, 0, current_experiment="ek99", failures=[CP.FailureNote("t0", "job", "OOM", "alloc")])
    txt = CP.render_handoff(st, CP.resume(s, HASH))
    for tag in ("SAVE STATE", "WRITE CHECKPOINT", "WRITE NEXT ACTION", "WRITE CURRENT FAILURES", "WRITE CURRENT EXPERIMENT",
                "WRITE CURRENT CODE HASH", "CONTINUE FROM CHECKPOINT"):
        assert tag in txt
    assert "ek99" in txt and "[OOM] job" in txt and HASH in txt
    p = CP.write_handoff(tmp_path / "h" / "handoff.txt", st)
    assert p.read_text(encoding="utf-8").startswith("SAVE STATE")


# ================================================================================================ additions
def test_finished_attempts_are_sealed_read_only(tmp_path):
    led, out = run_all(tmp_path, ("a",))
    rf = CM.attempt_dir(out, spec("a").key, 1) / "result.json"
    with pytest.raises(PermissionError):
        rf.write_text("{}", encoding="utf-8")


def test_isolation_audit_finds_strays_and_phantoms(tmp_path):
    led, out = run_all(tmp_path, ("a", "b"))
    assert CM.verify_isolation(led, out) == []
    ka = spec("a").key
    (out / ka / "notes.txt").write_text("shared scratch", encoding="utf-8")
    (out / ka / "attempt_07").mkdir()
    kb = spec("b").key
    marker = CM.attempt_dir(out, kb, 1) / "DONE"
    os.chmod(marker, stat.S_IWRITE)
    os.remove(marker)
    kinds = sorted(f["kind"] for f in CM.verify_isolation(led, out))
    assert kinds == ["done_without_marker", "phantom_attempt", "stray_file"]
    assert CM.verify_isolation(led, tmp_path / "nowhere") == []


def test_deadline_turns_a_hang_into_a_retryable_crash(tmp_path):
    led, out = make(tmp_path)
    t = {"now": 0.0}

    def slow(s, ctx):
        ctx.deadline = CM.Deadline(25.0, clock=lambda: t["now"])
        for _ in range(5):
            t["now"] += 10.0
            ctx.beat()
        return {"never": True}

    led.submit(spec(), 0, HASH)
    res = CM.run_worker(spec(), slow, led, None, out, "w", 1.0, HASH, clock=lambda: 1.0)
    assert res["state"] == CM.CRASHED and res["next"] == CM.PENDING
    with pytest.raises(ValueError):
        CM.Deadline(0)
    assert CM.classify_failure(TimeoutError("late")) == CM.CRASHED


def test_memory_calibrator_learns_from_peaks_and_ooms(tmp_path):
    cal = CM.MemoryCalibrator(tmp_path / "cal.json")
    assert cal.estimate("grid", 2.0) == 2.0
    cal.observe("grid", 2.0, 3.0)
    cal.observe("grid", 2.0, 3.2)
    assert cal.estimate("grid", 2.0) == 2.0                      # below min_obs: still the caller's default
    cal.observe("grid", 2.0, 3.6)
    assert cal.estimate("grid", 2.0) == pytest.approx(3.6 * 1.25)
    cal.observe("grid", 2.0, None, oom=True)
    assert cal.estimate("grid", 2.0) == pytest.approx(4.5)
    for _ in range(3):
        cal.observe("big", 6.0, 7.5)
    assert cal.estimate("big", 6.0) == 8.0                       # capped
    assert cal.report()["grid"]["n_oom"] == 1
    with pytest.raises(ValueError):
        cal.observe("x", -1.0)
    (tmp_path / "cal.json").write_text("{corrupt", encoding="utf-8")
    assert cal.estimate("grid", 2.0) == 2.0                      # corrupt calibration is advice lost, not a crash
    with pytest.raises(ValueError):
        CM.MemoryCalibrator(tmp_path / "c2.json", margin=0.5)


def test_expand_grid_is_deterministic_and_deduplicated():
    a = CM.expand_grid("g", {"depth": 3}, {"lr": [0.1, 0.2], "k": [1, 2, 3]}, [2, 1, 1], "2022-01-03")
    b = CM.expand_grid("g", {"depth": 3}, {"k": [1, 2, 3], "lr": [0.1, 0.2]}, [1, 2], "2022-01-03")
    assert len(a) == 12 and [s.key for s in a] == [s.key for s in b] and len({s.key for s in a}) == 12
    assert a[0].params == {"depth": 3, "k": 1, "lr": 0.1}
    with pytest.raises(ValueError):
        CM.expand_grid("g", {}, {"lr": []}, [1], "2022-01-03")
    with pytest.raises(ValueError):
        CM.expand_grid("g", {}, {"lr": [1]}, [], "2022-01-03")
    assert len(CM.expand_grid("g", {}, {}, [1], "2022-01-03")) == 1


def test_shard_balances_memory_and_ignores_input_order():
    specs = [spec(f"e{i}", est_gb=g) for i, g in enumerate((4.0, 1.0, 3.0, 2.0, 2.0, 1.0))]
    bins = CM.shard(specs, 3)
    loads = sorted(sum(s.est_gb for s in b) for b in bins)
    assert loads == [4.0, 4.0, 5.0] and sum(len(b) for b in bins) == 6
    assert [[s.key for s in b] for b in bins] == [[s.key for s in b] for b in CM.shard(list(reversed(specs)), 3)]
    assert CM.shard([], 2) == [[], []]
    with pytest.raises(ValueError):
        CM.shard(specs, 0)


def test_merge_log_is_idempotent_and_flags_changed_results(tmp_path):
    led, out = run_all(tmp_path, ("a", "b"))
    rec = CM.reconcile(led, out, HASH)
    log = CM.MergeLog(tmp_path / "merged.json")
    add = lambda acc, key, res: acc + [round(res["mean"], 9)]
    state, plan = CM.merge_new(rec, log, add, [])
    assert len(state) == 2 and len(plan["new"]) == 2
    assert log.commit(plan) == 2
    state2, plan2 = CM.merge_new(rec, log, add, state)
    assert state2 == state and plan2["new"] == [] and len(plan2["already_merged"]) == 2
    altered = CM.Reconciliation(tuple((k, {**r, "mean": r["mean"] + 1}) for k, r in rec.accepted), (), ())
    _, plan3 = CM.merge_new(altered, log, add, state)
    assert len(plan3["conflicts"]) == 2
    with pytest.raises(CM.ComputeError):
        log.commit(plan3)


def test_crash_between_state_save_and_commit_repeats_cleanly(tmp_path):
    led, out = run_all(tmp_path, ("a", "b", "c"))
    rec = CM.reconcile(led, out, HASH)
    log = CM.MergeLog(tmp_path / "merged.json")
    add = lambda acc, key, res: acc + [key]
    state, plan = CM.merge_new(rec, log, add, [])
    # crash: state persisted by the caller, commit never ran -> next run reloads persisted state and sees the same plan
    again, plan_again = CM.merge_new(rec, log, add, [])
    assert again == state and [k for k, _, _ in plan_again["new"]] == [k for k, _, _ in plan["new"]]


class FakeFleet:
    def __init__(self):
        self.started, self.alive = [], set()

    def launch(self, sp):
        self.started.append(sp.key)

    def is_alive(self, worker):
        return worker in self.alive


def test_supervisor_launches_and_never_double_launches_within_grace(tmp_path):
    led, out = make(tmp_path)
    fleet = FakeFleet()
    free = {"gb": 9.0}
    sup = CM.Supervisor(led, lambda: free["gb"], fleet.launch, fleet.is_alive, HASH, cores=8, grace_s=60.0)
    for n in ("a", "b", "c"):
        led.submit(spec(n), 0, HASH)
    r1 = sup.tick(100.0)
    assert len(r1["launched"]) == 3
    r2 = sup.tick(110.0)                                          # all still PENDING but launched within grace
    assert r2["launched"] == [] and len(fleet.started) == 3
    r3 = sup.tick(400.0)                                          # grace over and nobody claimed: relaunch is allowed
    assert len(r3["launched"]) == 3 and len(fleet.started) == 6


def test_supervisor_recovers_a_silent_worker(tmp_path):
    led, out = make(tmp_path)
    fleet = FakeFleet()
    sup = CM.Supervisor(led, lambda: 16.0, fleet.launch, fleet.is_alive, HASH, cores=8, heartbeat_s=60.0)
    led.submit(spec("a"), 0, HASH)
    led.claim(spec("a").key, "w_a", 100.0, HASH)
    fleet.alive.add("w_a")
    assert sup.tick(120.0)["recovered"] == []                     # alive and inside the heartbeat window
    r = sup.tick(500.0)
    assert [x["key"] for x in r["recovered"]] == [spec("a").key] and r["launched"] == [spec("a").key]


def test_supervisor_marks_stale_and_reports_launch_failures(tmp_path):
    led, _ = make(tmp_path)

    def boom(sp):
        raise OSError("cannot spawn")

    sup = CM.Supervisor(led, lambda: 16.0, boom, lambda w: True, NEWHASH, cores=8)
    led.submit(spec("old"), 0, HASH)
    led.submit(spec("new"), 0, NEWHASH)
    r = sup.tick(10.0)
    assert r["stale"] == [spec("old").key] and r["launched"] == []
    assert r["launch_failed"][0]["key"] == spec("new").key and "cannot spawn" in r["launch_failed"][0]["error"]
    assert led.get(spec("new").key)["state"] == CM.PENDING        # still queued, not dropped


def test_worker_feeds_the_calibrator_and_calibrated_specs_keep_their_identity(tmp_path):
    led, out = make(tmp_path)
    cal = CM.MemoryCalibrator(tmp_path / "cal.json", min_obs=1)
    sp = spec("hog", est_gb=2.0)
    led.submit(sp, 0, HASH)

    def oom(s, ctx):
        raise MemoryError("Unable to allocate")

    assert CM.run_worker(sp, oom, led, None, out, "w", 1.0, HASH, calibrator=cal)["state"] == CM.OOM
    assert cal.estimate("hog", 2.0) == pytest.approx(3.0)         # 2.0 GB died -> next estimate 2.0 * 1.5
    assert led.get(sp.key)["est_gb"] == pytest.approx(3.0) and led.get(sp.key)["batch_scale"] == 0.5
    assert CM.run_worker(sp, fn_ok, led, None, out, "w", 2.0, HASH, calibrator=cal)["state"] == CM.DONE
    assert cal.report()["hog"]["n_oom"] == 1
    cs = CM.calibrated(sp, cal)
    assert cs.est_gb >= 3.0 and cs.key == sp.key                  # memory estimate is not part of experiment identity


def test_experiment_stats_prices_experiments(tmp_path):
    led, out = run_all(tmp_path, ("a", "b"))
    led.submit(spec("z"), 0, HASH)
    led.claim(spec("z").key, "w", 5.0, HASH)
    led.fail(spec("z").key, CM.FAILED, "x", 6.0)
    st = CM.experiment_stats(led)
    assert st["a"]["done"] == 1 and st["a"]["success_share"] == 1.0 and st["z"]["success_share"] == 0.0
    assert st["a"]["mean_seconds"] is not None and st["z"]["mean_seconds"] is None
    assert CM.experiment_stats(CM.ExperimentLedger(tmp_path / "none.json")) == {}


def test_plan_launches_accounts_for_recently_launched(tmp_path):
    led, _ = make(tmp_path)
    for n in ("a", "b", "c"):
        led.submit(spec(n, est_gb=2.0), 0, HASH)
    led.mark_launched(spec("a").key, 100.0)
    p = CM.plan_launches(led, free_gb=9.0, running=0, cores=8, now=110.0, grace_s=60.0)
    assert spec("a").key not in p.launch and len(p.launch) == 2   # a's 2 GB and slot are already spoken for
    p = CM.plan_launches(led, free_gb=9.0, running=0, cores=8, now=200.0, grace_s=60.0)
    assert len(p.launch) == 3                                     # grace expired: a is launchable again
    led.claim(spec("a").key, "w", 1.0, HASH)
    with pytest.raises(CM.ComputeError):
        led.mark_launched(spec("a").key, 1.0)                     # only PENDING work can be launched


# ================================================================================================ additions
def test_failure_summary_finds_recurrence_and_escalates():
    fs = [CP.FailureNote(f"t{i}", "fit", "OOM", "alloc", i, resolved=(i < 2)) for i in range(3)]
    fs += [CP.FailureNote("t9", "other", "ERROR", "x")]
    s = CP.summarise_failures(fs)
    assert s["n"] == 4 and s["open"] == 2 and s["recurrence"] == {"fit/OOM": 3}
    assert [e["task"] for e in s["escalate"]] == ["fit"] and "3 occurrences" in s["escalate"][0]["reasons"][0]
    out = CP.apply_escalations(fs)
    assert [f.escalated for f in out] == [False, False, True, False]         # only the still-open one is newly escalated
    assert CP.apply_escalations(out) == out
    assert CP.summarise_failures([])["escalate"] == []


def test_firewall_failures_always_escalate_and_policy_is_validated():
    s = CP.summarise_failures([CP.FailureNote("t", "x", "FIREWALL", "leak")])
    assert s["escalate"][0]["reasons"] == ["FIREWALL always escalates"]
    assert CP.EscalationPolicy(max_same_failure=1).validate()
    assert not CP.EscalationPolicy().validate()


def test_spinning_loop_is_detected_but_progress_is_not_flagged(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    for i in range(4):
        save(s, i, next_action="same step", completed={"a": HASH})
    assert CP.detect_spinning(s)["spinning"] and CP.stall_report(s, 10**10, 60.0)["stalled"]
    save(s, 4, next_action="new step", completed={"a": HASH, "b": HASH})
    assert not CP.detect_spinning(s)["spinning"]
    assert CP.detect_spinning(CP.CheckpointStore(tmp_path, "fresh")) == {"spinning": False, "checked": 0}
    assert CP.stall_report(CP.CheckpointStore(tmp_path, "fresh"), 1.0, 60.0)["stalled"]


def test_resume_reads_experiment_state_from_the_compute_ledger(tmp_path):
    from engine.learning import compute as CM
    led = CM.ExperimentLedger(tmp_path / "l.json")
    sp = CM.ExperimentSpec("e", {"a": 1}, 1, "2022-01-03")
    led.submit(sp, 0, HASH)
    look = CP.experiment_state_from_ledger(led)
    assert look(sp.key) == "PENDING" and look("nope") is None
    s = CP.CheckpointStore(tmp_path, "run1")
    save(s, 0, current_experiment=sp.key)
    assert CP.resume(s, HASH, look).action == "RECOVER_EXPERIMENT"
    led.claim(sp.key, "w", 1, HASH)
    led.complete(sp.key, "h", 2)
    assert CP.resume(s, HASH, look).action == "RECONCILE_EXPERIMENT"


def test_progress_ledger_is_chained_and_tamper_evident(tmp_path):
    pl = CP.ProgressLedger(tmp_path / "progress.jsonl")
    pl.note("saved", "run1", "2022-01-03", seq=1, path=tmp_path)
    pl.note("resumed", "run1", "2022-01-04", action="CONTINUE")
    assert pl.digest() == {"n": 2, "by_event": {"resumed": 1, "saved": 1}, "chain_ok": True}
    assert len(pl.events("saved")) == 1
    p = tmp_path / "progress.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace("CONTINUE", "REDO"), encoding="utf-8")
    assert not pl.verify()["ok"] and not pl.digest()["chain_ok"]


def test_meaningful_line_counter_handles_strings_comments_and_docstrings(tmp_path):
    src = tmp_path / "m.py"
    src.write_text('"""module doc\nspans lines"""\n# a comment\n\nX = 1  # trailing\ndef f():\n    """doc"""\n    s = """a\n# not a comment\nb"""\n    return s\n', encoding="utf-8")
    assert CP.count_meaningful_lines(src) == 9
    assert CP.count_meaningful_lines(src, include_docstrings=False) == 6
    empty = tmp_path / "e.py"
    empty.write_text("", encoding="utf-8")
    assert CP.count_meaningful_lines(empty) == 0


def test_depth_report_counts_missing_files_as_a_shortfall(tmp_path):
    src = tmp_path / "m.py"
    src.write_text("a = 1\nb = 2\n", encoding="utf-8")
    rep = CP.depth_report({"44": ([src, tmp_path / "gone.py"], 700), "57": ([], 900)})
    assert rep == {"44": (2, 700), "57": (0, 900)}
    ev = CP.CompletionEvidence(line_depth=rep)
    assert "2. line-depth minimums: ['44', '57']" in CP.evaluate_stop(ev).unmet[1]


def test_completion_evidence_is_derived_from_facts(tmp_path):
    t = CP.ChecklistTracker(tmp_path / "cl.json")
    t.add("J01", "board")
    t.add("J02", "gate")
    t.mark("J01", BuildStatus.VALIDATED, ["run 7"])
    f = tmp_path / "t.py"
    f.write_text("x = 1\n", encoding="utf-8")
    s = CP.CheckpointStore(tmp_path, "run1")
    st = save(s, 0, failures=[CP.FailureNote("t", "ERROR", "ERROR", "boom")])
    ev = CP.derive_completion_evidence(t, {"44": (800, 700)}, st, {"test:44": [f], "test:57": [tmp_path / "nope.py"], "report:x": [f]})
    assert ev.sections_complete == {"J01": True, "J02": False} and ev.tests_exist == {"test:44": True, "test:57": False}
    assert ev.reports_exist == {"report:x": True} and not ev.checklist_complete and ev.open_failures == 1
    v = CP.evaluate_stop(ev)
    assert not v.can_stop and len(v.unmet) >= 4
    t.mark("J02", BuildStatus.VALIDATED, ["run 8"])
    ev2 = CP.derive_completion_evidence(t, {"44": (800, 700)}, None, {"test:44": [f], "firewall:1": [f], "report:x": [f]})
    assert CP.evaluate_stop(ev2).can_stop


def test_resume_report_is_plain_language(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    save(s, 0, completed={"a": "oldcode"}, pending=("b",))
    txt = CP.resume_report(CP.resume(s, HASH))
    assert "resume action: CONTINUE" in txt and "revalidate before trusting: a" in txt and "checkpoint #0" in txt
    assert CP.resume_report(CP.resume(CP.CheckpointStore(tmp_path, "none"), HASH)).startswith("resume action: START_FRESH")
