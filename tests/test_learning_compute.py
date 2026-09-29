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
    env["result_hash"] = CM.stable_hash(env["result"], 20)
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


# ================================================================================================ additions: backoff, dependencies, manifests
def test_backoff_is_deterministic_bounded_and_growing():
    b = CM.BackoffPolicy(base_s=10, factor=2, cap_s=100, jitter=0.25)
    d = [b.delay("k", n) for n in (1, 2, 3, 4, 5, 6)]
    assert d == [b.delay("k", n) for n in (1, 2, 3, 4, 5, 6)]
    assert 7.5 <= d[0] <= 12.5
    assert d[1] > d[0] * 1.2 and max(d) <= 125.0 and b.delay("k", 1) != b.delay("other", 1)
    assert CM.BackoffPolicy(jitter=1.5).validate() and CM.BackoffPolicy(cap_s=1, base_s=5).validate() and not b.validate()


def test_failed_experiments_wait_before_retrying(tmp_path):
    led = CM.ExperimentLedger(tmp_path / "l.json", backoff=CM.BackoffPolicy(base_s=60, jitter=0.0))
    out = tmp_path / "o"
    sp = spec()
    led.submit(sp, 0, HASH)

    def bad(s, c):
        raise ValueError("x")

    CM.run_worker(sp, bad, led, None, out, "w", 100.0, HASH, clock=lambda: 100.0)
    assert led.get(sp.key)["not_before"] == 160.0
    plan = CM.plan_launches(led, 16.0, cores=8, now=120.0)
    assert plan.launch == () and "backing off" in dict(plan.held)[sp.key]
    with pytest.raises(CM.ClaimError, match="backing off"):
        led.claim(sp.key, "w", 130.0, HASH)
    assert CM.plan_launches(led, 16.0, cores=8, now=161.0).launch == (sp.key,)
    led.claim(sp.key, "w", 161.0, HASH)
    assert led.get(sp.key)["not_before"] is None
    with pytest.raises(ValueError):
        CM.ExperimentLedger(tmp_path / "l2.json", backoff=CM.BackoffPolicy(factor=0.5))


def test_dependency_order_and_cycle_detection():
    a = spec("a")
    b = spec("b", after=(a.key,))
    c = spec("c", after=(a.key, b.key))
    order = [s.name for s in CM.dependency_order([c, b, a])]
    assert order == ["a", "b", "c"]
    x = spec("x")
    y = dataclasses_replace(spec("y"), after=(x.key,))
    x2 = dataclasses_replace(x, after=(y.key,))
    with pytest.raises(CM.ComputeError, match="cycle"):
        CM.dependency_order([x2, y])
    assert spec("a").key == dataclasses_replace(spec("a"), after=("zzz",)).key         # dependencies are not identity
    assert CM.ExperimentSpec.from_dict(b.to_dict()) == b


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_dependent_experiments_wait_for_prerequisites(tmp_path):
    led, out = make(tmp_path)
    a = spec("a")
    b = spec("b", after=(a.key,))
    res = CM.submit_batch(led, [b, a], 0, HASH)
    assert res["queued"] == [a.key, b.key] and res["duplicates"] == []
    plan = CM.plan_launches(led, 16.0, cores=8)
    assert plan.launch == (a.key,) and "waiting for prerequisites" in dict(plan.held)[b.key]
    CM.run_worker(a, fn_ok, led, None, out, "w", 1.0, HASH)
    assert CM.plan_launches(led, 16.0, cores=8).launch == (b.key,)


def test_submit_batch_is_safe_to_repeat_and_rejects_unknown_prerequisites(tmp_path):
    led, out = make(tmp_path)
    a, b = spec("a"), spec("b", after=(spec("a").key,))
    CM.submit_batch(led, [a, b], 0, HASH)
    again = CM.submit_batch(led, [a, b], 1, HASH)
    assert again["queued"] == [] and len(again["duplicates"]) == 2
    with pytest.raises(CM.ComputeError, match="unknown"):
        CM.submit_batch(led, [spec("c", after=("feedfacefeedface0000",))], 2, HASH)
    assert CM.submit_batch(led, [], 3, HASH) == {"queued": [], "duplicates": []}


def test_dependents_of_a_given_up_experiment_are_reported_blocked(tmp_path):
    led, out = make(tmp_path, max_attempts=1)
    a = spec("a")
    b = spec("b", after=(a.key,))
    c = spec("c", after=(b.key,))
    d = spec("d")
    CM.submit_batch(led, [a, b, c, d], 0, HASH)

    def bad(s, ctx):
        raise ValueError("x")

    assert CM.run_worker(a, bad, led, None, out, "w", 1.0, HASH)["next"] == CM.GAVE_UP
    assert sorted(CM.blocked_by_failure(led), key=lambda r: r["name"]) == [{"key": b.key, "name": "b", "blocked_by": a.key},
                                                                             {"key": c.key, "name": "c", "blocked_by": b.key}]
    assert d.key not in [x["key"] for x in CM.blocked_by_failure(led)]
    assert CM.blocked_by_failure(CM.ExperimentLedger(tmp_path / "none.json")) == []


def test_run_manifest_detects_tampering_and_compares_reruns(tmp_path):
    led1, out1 = run_all(tmp_path / "r1")
    led2, out2 = run_all(tmp_path / "r2")
    rec1, rec2 = CM.reconcile(led1, out1, HASH), CM.reconcile(led2, out2, HASH)
    m1 = CM.write_run_manifest(tmp_path / "m1.json", led1, rec1, HASH, ["snapA"])
    m2 = CM.write_run_manifest(tmp_path / "m2.json", led2, rec2, HASH)
    assert CM.verify_manifest(tmp_path / "m1.json") == []
    cmp_ = CM.compare_manifests(m1, m2)
    assert cmp_["reproducible"] and cmp_["compared"] == 3 and cmp_["differ"] == []
    doc = json.loads((tmp_path / "m1.json").read_text(encoding="utf-8"))
    doc["body"]["code_hash"] = "other"
    (tmp_path / "m1.json").write_text(json.dumps(doc), encoding="utf-8")
    assert CM.verify_manifest(tmp_path / "m1.json") == ["manifest body altered after writing"]
    assert CM.verify_manifest(tmp_path / "missing.json") == ["manifest missing or unreadable"]


def test_manifest_comparison_exposes_a_nondeterministic_experiment(tmp_path):
    def fn_a(s, c):
        return {"v": 1}

    def fn_b(s, c):
        return {"v": 2}

    docs = []
    for i, fn in enumerate((fn_a, fn_b)):
        led, out = make(tmp_path / f"r{i}")
        led.submit(spec(), 0, HASH)
        CM.run_worker(spec(), fn, led, None, out, "w", 1.0, HASH)
        docs.append(CM.write_run_manifest(tmp_path / f"m{i}.json", led, CM.reconcile(led, out, HASH), HASH))
    c = CM.compare_manifests(*docs)
    assert c["differ"] == [spec().key] and not c["reproducible"]
    docs[1]["body"]["code_hash"] = "other"
    assert not CM.compare_manifests(*docs)["same_code"]


# ================================================================================================ additions: audit, repair, schema, hand-off
def test_audit_store_finds_corruption_gaps_and_bad_pointer(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    for i in range(5):
        save(s, i)
    a = CP.audit_store(s)
    assert a["healthy"] and a["valid"] == [0, 1, 2, 3, 4] and a["latest_pointer"] == 4
    os.remove(s._path(1))
    s._path(3).write_text("torn", encoding="utf-8")
    a = CP.audit_store(s)
    assert a["gaps"] == [1] and a["corrupt"] == [3] and a["newest_valid"] == 4 and a["latest_pointer_ok"] and not a["healthy"]
    (s.dir / "LATEST.json").write_text("{not json", encoding="utf-8")
    a = CP.audit_store(s)
    assert a["latest_pointer"] is None and not a["latest_pointer_ok"]
    assert CP.repair_latest(s) == 4 and CP.audit_store(s)["latest_pointer_ok"]
    assert CP.audit_store(CP.CheckpointStore(tmp_path, "empty")) == {
        "sequences": 0, "valid": [], "corrupt": [], "gaps": [], "newest_valid": None, "latest_pointer": None,
        "latest_pointer_ok": True, "newer_schema": [], "healthy": True}
    assert CP.repair_latest(CP.CheckpointStore(tmp_path, "empty")) is None


def test_states_from_newer_and_older_code_still_load(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    save(s, 0)
    p = s._path(0)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["body"]["added_in_future"] = "x"
    doc["body"]["schema"] = CP.SCHEMA_VERSION + 1
    doc["body"]["failures"] = [{"at": "t", "task": "a", "kind": "ERROR", "message": "m", "new_field": 1}]
    del doc["body"]["notes"]                                                          # an older writer did not have this field
    from engine import checkpoint as bundle
    doc["sha256"] = bundle._body_hash(doc["body"])
    p.write_text(json.dumps(doc), encoding="utf-8")
    st = s.read(0)
    assert st.notes == {} and st.failures[0].task == "a"
    assert CP.audit_store(s)["newer_schema"] == [0]


def test_timeline_and_progress(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    assert CP.progress_made(s)["checkpoints"] == 0
    save(s, 0, completed={}, pending=("a", "b"), failures=[CP.FailureNote("t", "a", "ERROR", "x")])
    save(s, 1, completed={"a": HASH}, pending=("b",))
    save(s, 2, completed={"a": HASH, "b": NEWHASH}, code_hash=NEWHASH)
    s._path(1).write_text("torn", encoding="utf-8")
    tl = CP.timeline(s)
    assert [r.get("corrupt", False) for r in tl] == [False, True, False]
    pm = CP.progress_made(s)
    assert pm == {"checkpoints": 2, "tasks_completed": 2, "code_changes": 1, "failures_delta": -1}


def test_masterstock_block_is_complete_and_honest(tmp_path):
    t = CP.ChecklistTracker(tmp_path / "cl.json")
    t.add("J01", "champion board")
    t.mark("J01", BuildStatus.IMPLEMENTED)
    s = CP.CheckpointStore(tmp_path, "run1")
    fails = [CP.FailureNote(f"t{i}", "fit", "OOM", "alloc") for i in range(3)]
    st = save(s, 0, current_experiment="ek1", failures=fails, next_action="rerun fit with half batch")
    txt = CP.render_masterstock_block(st, CP.resume(s, HASH), t)
    for frag in ("NEXT SESSION: START HERE", "rerun fit with half batch", "experiment in flight: ek1", "ESCALATE: fit/OOM x3",
                 "first incomplete critical item: J01 champion board", "IMPLEMENTED — NOT VALIDATED", HASH):
        assert frag in txt
    raw = json.loads((tmp_path / "cl.json").read_text(encoding="utf-8"))
    raw["J01"]["status"] = "VALIDATED"
    (tmp_path / "cl.json").write_text(json.dumps(raw), encoding="utf-8")
    assert "FALSELY MARKED COMPLETE" in CP.render_masterstock_block(st, None, CP.ChecklistTracker(tmp_path / "cl.json"))
    assert "checklist" not in CP.render_masterstock_block(st)


# ================================================================================================ additions: guarded steps
def test_guarded_step_records_success_and_marks_earlier_failures_resolved(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    done = {}
    old = [CP.FailureNote("t0", "fit", "ERROR", "earlier")]
    with CP.guarded_step(s, "fit", HASH, clock=lambda: 5.0, completed=done, pending=("report",), failures=old, next_action="write report"):
        pass
    st = s.latest_valid()[0]
    assert done == {"fit": HASH} and st.completed == {"fit": HASH} and st.next_action == "write report"
    assert st.failures[0].resolved and s.read(st.sequence - 1).next_action == "running fit"


def test_guarded_step_records_failure_and_reraises(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    with pytest.raises(ValueError):
        with CP.guarded_step(s, "fit", HASH, clock=lambda: 5.0, pending=("report",)):
            raise ValueError("bad input")
    st = s.latest_valid()[0]
    assert st.failures[-1].kind == "ERROR" and "bad input" in st.failures[-1].message
    assert st.pending == ("fit", "report") and st.next_action == "retry or fix fit" and st.completed == {}
    with pytest.raises(MemoryError):
        with CP.guarded_step(s, "big", HASH, clock=lambda: 6.0):
            raise MemoryError("alloc")
    assert s.latest_valid()[0].failures[-1].kind == "OOM"
    with pytest.raises(KeyboardInterrupt):
        with CP.guarded_step(s, "long", HASH, clock=lambda: 7.0):
            raise KeyboardInterrupt()
    assert s.latest_valid()[0].failures[-1].kind == "INTERRUPTED"


def test_guarded_step_escalates_a_firewall_breach(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    with pytest.raises(FirewallBreach):
        with CP.guarded_step(s, "audit", HASH, clock=lambda: 5.0):
            raise FirewallBreach("future rows")
    st = s.latest_valid()[0]
    assert st.failures[-1].kind == "FIREWALL" and st.failures[-1].escalated and "INVESTIGATE" in st.next_action
    assert "audit" not in st.completed


def test_new_run_id_is_deterministic_and_safe():
    a = CP.new_run_id("s11", "2026-09-29")
    assert a == CP.new_run_id("s11", "2026-09-29") and a != CP.new_run_id("s11", "2026-09-29", "b") and a.startswith("s11_20260929_")
    CP.CheckpointStore("x", a)                                                       # accepted as a run id


def test_diff_states(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    a = save(s, 0, completed={"x": HASH}, failures=[CP.FailureNote("t", "y", "ERROR", "m")], current_experiment="e1")
    b = save(s, 1, completed={"x": HASH, "y": HASH}, failures=[CP.FailureNote("t", "y", "ERROR", "m", resolved=True)], code_hash=NEWHASH)
    d = CP.diff_states(a, b)
    assert d["completed_added"] == ["y"] and d["completed_lost"] == [] and d["failures_added"] == 0 and d["failures_resolved"] == 1
    assert d["code_changed"] and d["next_action"] == ("do step 0", "do step 1") and d["experiment"] == ("e1", "")
    same = CP.diff_states(a, a)
    assert not same["code_changed"] and same["next_action"] is None and same["completed_added"] == []


# ================================================================================================ additions: admission, real child processes, adjudication
PY = __import__("sys").executable


def test_admission_follows_the_2_5_gb_rule_and_fails_closed():
    assert CM.admit(2.0, lambda: 8.0) == (True, "ok")
    ok, why = CM.admit(1.0, lambda: 2.4)
    assert not ok and "2.5 GB rule" in why
    ok, why = CM.admit(5.0, lambda: 6.0)
    assert not ok and "reserve" in why
    assert CM.admit(1.0, lambda: None)[0] is False and CM.admit(1.0, lambda: float("nan"))[0] is False
    assert CM.admit(1.0, lambda: 2.5 + 1.0 + 2.5)[0]
    real = CM.admit(0.01)                                             # the real machine reading is at least well-formed
    assert isinstance(real[0], bool) and isinstance(real[1], str)


def test_subprocess_success_and_output_files(tmp_path):
    r = CM.run_subprocess([PY, "-c", "print('hello'); import sys; sys.stderr.write('note')"], tmp_path / "w", 30)
    assert r.kind == CM.DONE and r.returncode == 0 and not r.timed_out and r.elapsed_s < 30
    assert (tmp_path / "w" / "stdout.txt").read_text(encoding="utf-8").strip() == "hello" and r.stderr_tail == "note"


def test_subprocess_timeout_kills_only_the_child_and_is_retryable(tmp_path):
    r = CM.run_subprocess([PY, "-c", "import time; time.sleep(60)"], tmp_path / "w", 0.5)
    assert r.timed_out and r.kind == CM.CRASHED and r.elapsed_s < 20


def test_subprocess_memory_error_and_exit_codes_are_classified(tmp_path):
    r = CM.run_subprocess([PY, "-c", "raise MemoryError('Unable to allocate 9 GiB')"], tmp_path / "a", 30)
    assert r.kind == CM.OOM and r.returncode == 1 and "MemoryError" in r.stderr_tail
    r = CM.run_subprocess([PY, "-c", "import os; os._exit(137)"], tmp_path / "b", 30)
    assert r.kind == CM.OOM and r.returncode == 137
    r = CM.run_subprocess([PY, "-c", "raise ValueError('plain bug')"], tmp_path / "c", 30)
    assert r.kind == CM.FAILED and "plain bug" in r.stderr_tail
    r = CM.run_subprocess([PY, "-c", "import sys; sys.exit(3)"], tmp_path / "d", 30)
    assert r.kind == CM.FAILED and r.returncode == 3


def test_subprocess_over_memory_ceiling_is_killed_as_oom(tmp_path):
    fake = iter([0.1, 0.2, 5.0, 5.0, 5.0])
    r = CM.run_subprocess([PY, "-c", "import time; time.sleep(60)"], tmp_path / "w", 30, mem_limit_gb=1.0,
                          rss_fn=lambda pid: next(fake, 5.0))
    assert r.killed_for_memory and r.kind == CM.OOM and r.peak_gb == 5.0 and r.elapsed_s < 20


def test_settle_fails_a_child_that_died_without_recording(tmp_path):
    led, out = make(tmp_path, max_attempts=2, oom_growth=2.0)
    sp = spec(est_gb=1.0)
    led.submit(sp, 0, HASH)
    led.claim(sp.key, "child", 1.0, HASH)
    res = CM.run_subprocess([PY, "-c", "raise MemoryError('Unable to allocate')"], tmp_path / "w", 30)
    assert CM.settle(led, sp.key, res, 2.0) == CM.PENDING
    e = led.get(sp.key)
    assert e["est_gb"] == 2.0 and e["batch_scale"] == 0.5 and "MemoryError" in json.dumps(e["history"])
    led.claim(sp.key, "child2", 3.0, HASH)
    clean = CM.run_subprocess([PY, "-c", "pass"], tmp_path / "w2", 30)
    assert CM.settle(led, sp.key, clean, 4.0) == CM.GAVE_UP                          # exit 0 but nothing recorded is a failure
    assert CM.settle(led, "unknown", clean, 5.0) == "UNKNOWN"


def test_settle_leaves_a_child_that_recorded_its_own_result(tmp_path):
    led, out = make(tmp_path)
    led.submit(spec(), 0, HASH)
    CM.run_worker(spec(), fn_ok, led, None, out, "w", 1.0, HASH)
    res = CM.run_subprocess([PY, "-c", "pass"], tmp_path / "w", 30)
    assert CM.settle(led, spec().key, res, 9.0) == CM.DONE


def write_attempt(out, key, n, spec_, result, code=HASH, seed=None, worker="w"):
    d = CM.attempt_dir(out, key, n)
    d.mkdir(parents=True)
    env = CM._envelope(spec_, result, code, worker, n, 0.0, 1.0)
    if seed is not None:
        env["seed"] = seed
    (d / "result.json").write_text(json.dumps(env), encoding="utf-8")
    (d / "DONE").write_text(env["result_hash"], encoding="utf-8")


def test_adjudicate_explains_a_nondeterministic_pair_and_asks_for_a_third(tmp_path):
    led, out = make(tmp_path)
    sp = spec()
    led.submit(sp, 0, HASH)
    write_attempt(out, sp.key, 1, sp, {"mean": 0.010, "v": [1.0, 2.0]}, worker="w1")
    write_attempt(out, sp.key, 2, sp, {"mean": 0.011, "v": [1.0, 2.0]}, worker="w2")
    a = CM.adjudicate(led, out, sp.key)
    assert not a["agree"] and a["cause"] == "nondeterministic" and a["n_distinct"] == 2 and a["majority_hash"] is None
    assert a["action"] == "run a third attempt" and "value_change" in " ".join(a["difference"]["causes"])
    write_attempt(out, sp.key, 3, sp, {"mean": 0.010, "v": [1.0, 2.0]}, worker="w3")
    a = CM.adjudicate(led, out, sp.key)
    assert a["majority_hash"] and a["action"].startswith("majority reported; still flagged")
    write_attempt(out, sp.key, 4, sp, {"mean": 0.012, "v": [1.0, 2.0]}, worker="w4")
    assert CM.adjudicate(led, out, sp.key)["action"] == "escalate: fix the source of non-determinism"


def test_adjudicate_separates_code_and_seed_differences_from_nondeterminism(tmp_path):
    led, out = make(tmp_path)
    sp = spec()
    write_attempt(out, sp.key, 1, sp, {"m": 1.0})
    write_attempt(out, sp.key, 2, sp, {"m": 2.0}, code=NEWHASH)
    assert CM.adjudicate(led, out, sp.key)["cause"] == "code_difference"
    sp2 = spec("other")
    write_attempt(out, sp2.key, 1, sp2, {"m": 1.0})
    write_attempt(out, sp2.key, 2, sp2, {"m": 2.0}, seed=99)
    assert CM.adjudicate(led, out, sp2.key)["cause"] == "seed_difference"
    sp3 = spec("same")
    write_attempt(out, sp3.key, 1, sp3, {"m": 1.0})
    write_attempt(out, sp3.key, 2, sp3, {"m": 1.0})
    assert CM.adjudicate(led, out, sp3.key)["agree"]
    assert CM.adjudicate(led, out, "nokey") == {"key": "nokey", "agree": True, "attempts": 0, "action": "nothing to adjudicate"}


WORKER_MODULE = """
from engine.learning import compute as CM

CM.current_code_hash = lambda: "test_code"      # parallel builders edit engine/ files, so pin the hash in this child

@CM.register_worker("child_exp")
def run(spec, ctx):
    if spec.params.get("boom") == "oom":
        raise MemoryError("Unable to allocate 64 GiB")
    return {"draw": float(ctx.rng("x").normal()), "attempt": ctx.attempt}
"""


def test_experiment_runs_end_to_end_in_a_real_child_process(tmp_path):
    (tmp_path / "w7_child_worker.py").write_text(WORKER_MODULE, encoding="utf-8")
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(tmp_path) + (os.pathsep + old if old else "")
    try:
        led = CM.ExperimentLedger(tmp_path / "ledger.json", max_attempts=2)
        sp = spec("child_exp", params={"boom": "no"})
        code = "test_code"
        led.submit(sp, 0, code)
        out = CM.run_experiment_process(sp, led, tmp_path / "out", tmp_path / "work", 5.0, 120, ["w7_child_worker"], free_fn=lambda: 16.0)
        assert out["launched"] and out["result"].kind == CM.DONE and out["state"] == CM.DONE, out["result"].stderr_tail
        rec = CM.reconcile(led, tmp_path / "out", code)
        assert rec.clean and len(rec.accepted) == 1 and CM.verify_isolation(led, tmp_path / "out") == []
        again = spec("child_exp", params={"boom": "no"})
        assert again.key == sp.key
        oom = spec("child_exp", params={"boom": "oom"})
        led.submit(oom, 6.0, code)
        r2 = CM.run_experiment_process(oom, led, tmp_path / "out", tmp_path / "work", 7.0, 120, ["w7_child_worker"], free_fn=lambda: 16.0)
        assert r2["result"].returncode == 1 and led.get(oom.key)["state"] == CM.PENDING       # the child recorded its OOM itself
        assert led.get(oom.key)["batch_scale"] == 0.5
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old


def test_no_child_is_started_when_memory_is_short(tmp_path):
    led = CM.ExperimentLedger(tmp_path / "ledger.json")
    led.submit(spec(), 0, HASH)
    out = CM.run_experiment_process(spec(), led, tmp_path / "out", tmp_path / "work", 1.0, 30, free_fn=lambda: 2.0)
    assert not out["launched"] and "2.5 GB rule" in out["why"] and out["state"] == CM.PENDING
    assert not (tmp_path / "work").exists()


# ================================================================================================ additions: stale-state refusal, interruption record, pruning
def test_resume_verified_refuses_state_from_other_code(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    assert CP.resume_verified(s, HASH).action == "START_FRESH"                      # nothing saved is not an error
    save(s, 0, completed={"a": HASH})
    assert CP.resume_verified(s, HASH).action == "CONTINUE"
    with pytest.raises(CP.StaleState, match="saved under code"):
        CP.resume_verified(s, NEWHASH)
    plan = CP.resume_verified(s, NEWHASH, allow_code_change=True)
    assert plan.revalidate == ("a",) and plan.code_changed


def test_resume_verified_names_tasks_done_under_other_code_even_if_the_checkpoint_code_matches(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    save(s, 0, completed={"a": "older", "b": HASH})
    with pytest.raises(CP.StaleState, match=r"\['a'\]"):
        CP.resume_verified(s, HASH)
    assert CP.resume_verified(s, HASH, allow_code_change=True).revalidate == ("a",)


def test_interruption_record_roundtrip_and_tamper_detection(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    fails = [CP.FailureNote("t0", "fit", "OOM", "alloc"), CP.FailureNote("t1", "old", "ERROR", "x", resolved=True)]
    st = save(s, 0, current_experiment="ek9", failures=fails, next_action="rerun fit with half batch")
    rec = CP.InterruptionRecord.from_state(st, "power loss")
    assert rec.next_action == "rerun fit with half batch" and rec.current_experiment == "ek9" and rec.code_hash == HASH
    assert [f.task for f in rec.failures] == ["fit"]                                # only OPEN failures are carried
    p = CP.write_interruption(tmp_path / "hand" / "interrupt.json", rec)
    assert CP.read_interruption(p) == rec
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["body"]["next_action"] = "delete everything"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CP.CheckpointCorrupt, match="hash mismatch"):
        CP.read_interruption(p)
    p.write_text("{torn", encoding="utf-8")
    with pytest.raises(CP.CheckpointCorrupt, match="unreadable"):
        CP.read_interruption(p)
    with pytest.raises(CP.CheckpointCorrupt, match="unreadable"):
        CP.read_interruption(tmp_path / "missing.json")


def test_invalid_interruption_record_is_never_written(tmp_path):
    bad = CP.InterruptionRecord("run1", 0, "t", "", "", "")
    assert len(bad.validate()) == 2
    with pytest.raises(CP.CheckpointCorrupt):
        CP.write_interruption(tmp_path / "x.json", bad)
    assert not (tmp_path / "x.json").exists()


def test_prune_keeps_the_newest_valid_checkpoint_and_protected_ones(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1", keep=60)
    for i in range(40):
        save(s, i)
    pol = CP.PrunePolicy(keep_last=5, keep_every=10, keep_sequences=(7,))
    assert CP.prune(s, pol, dry_run=True) == [q for q in range(35) if q not in (0, 7, 10, 20, 30)]
    assert len(s.sequences()) == 40                                                 # dry run deleted nothing
    gone = CP.prune(s, pol)
    assert s.sequences() == [0, 7, 10, 20, 30, 35, 36, 37, 38, 39] and len(gone) == 30
    assert s.latest_valid()[0].sequence == 39


def test_prune_never_touches_corrupt_files_or_anything_newer_than_the_newest_valid(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1", keep=60)
    for i in range(12):
        save(s, i)
    s._path(11).write_text("torn", encoding="utf-8")                                # newest file is corrupt
    s._path(3).write_text("torn", encoding="utf-8")                                 # and an old one
    gone = CP.prune(s, CP.PrunePolicy(keep_last=3, keep_every=0))
    assert 11 not in gone and 3 not in gone and 10 not in gone                      # 10 is the newest VALID one and is kept
    assert set(gone) == {0, 1, 2, 4, 5, 6, 7, 8} and s.latest_valid()[0].sequence == 10
    assert 3 in s.sequences() and 11 in s.sequences()


def test_prune_edge_cases(tmp_path):
    s = CP.CheckpointStore(tmp_path, "run1")
    assert CP.prune(s) == []                                                        # empty store
    for i in range(2):
        save(s, i)
    assert CP.prune(s) == []                                                        # fewer than keep_last
    with pytest.raises(ValueError):
        CP.prune(s, CP.PrunePolicy(keep_last=1))
    for q in s.sequences():
        s._path(q).write_text("garbage", encoding="utf-8")
    assert CP.prune(s) == [] and len(s.sequences()) == 2                            # nothing valid: delete nothing
