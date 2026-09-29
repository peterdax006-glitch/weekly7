"""Phase 44 resource manager. Each mechanism has a planted defect and a control that must stay clean."""
import os
import subprocess
import sys
import time

import pytest

from engine import resources as R

NOW = 1_000_000.0


def alive_info(table):
    """Fake process_info: table maps pid -> (create_time, ppid); anything else is dead."""
    def info(pid):
        if pid in table:
            ct, pp = table[pid]
            return {"alive": True, "create_time": ct, "ppid": pp, "rss_gb": 1.0}
        return {"alive": False, "create_time": None, "ppid": None, "rss_gb": None}
    return info


def reg(tmp_path):
    return R.ProcRegistry(tmp_path / "procs.json")


def add(r, jid, pid, cmd="python w.py", out=None, hb=NOW, ppid=1, ct=100.0, est=2.0):
    r.register(jid, pid, cmd, out, est, now=hb, ppid=ppid, create_time=ct)


# ---------------------------------------------------------------- worker count
def test_worker_count_scales_with_free_memory_and_stays_in_3_to_7():
    assert R.worker_count(4.6, 2.0, cores=8)["workers"] == 1         # (4.6-2.5)/2
    assert R.worker_count(8.5, 2.0, cores=8)["workers"] == 3
    assert R.worker_count(12.5, 2.0, cores=8)["workers"] == 5
    assert R.worker_count(40.0, 2.0, cores=8)["workers"] == 7        # ceiling, and cores-1 = 7
    assert R.worker_count(40.0, 2.0, cores=4)["workers"] == 3        # cores-1
    r = R.worker_count(40.0, 2.0, cores=8)
    assert r["limit"] in ("ceiling", "cores")


def test_memory_outranks_the_floor_and_unreadable_memory_is_zero_workers():
    low = R.worker_count(3.0, 2.0, cores=8)
    assert low["workers"] == 0 and "memory" in low["limit"]
    assert R.worker_count(None)["workers"] == 0
    assert R.worker_count(float("nan"))["workers"] == 0
    assert R.worker_count(10.0, per_worker_gb=0)["workers"] == 0


def test_worker_seed_is_deterministic_and_distinct():
    assert R.worker_seed(7, "a") == R.worker_seed(7, "a")
    assert len({R.worker_seed(7, f"job{i}") for i in range(50)}) == 50
    assert R.worker_seed(7, "a") != R.worker_seed(8, "a")


def test_memory_gb_reads_something_real():
    free, total = R.memory_gb()
    assert free is not None and 0 < free <= total


# ---------------------------------------------------------------- registry
def test_registry_refuses_duplicate_live_job_and_shared_output_folder(tmp_path):
    r = reg(tmp_path)
    me = os.getpid()
    ct = R.process_info(me)["create_time"]
    r.register("J1", me, "python a.py", tmp_path / "out", now=NOW, create_time=ct)
    with pytest.raises(ValueError, match="already running"):
        r.register("J1", me, "python a.py", tmp_path / "other", now=NOW, create_time=ct)
    with pytest.raises(ValueError, match="already owned"):       # the real incident: two workers, one folder
        r.register("J2", me, "python b.py", tmp_path / "out", now=NOW, create_time=ct)
    r.register("J3", me, "python c.py", tmp_path / "out3", now=NOW, create_time=ct)   # control: distinct id and folder is fine
    assert set(r.running()) == {"J1", "J3"}


def test_registry_survives_missing_and_corrupt_file(tmp_path):
    r = reg(tmp_path)
    assert r.load() == {} and r.running() == {}
    r.path.write_text("{not json", encoding="utf-8")
    assert r.load() == {}


def test_stop_kills_only_the_recorded_pid_and_leaves_a_same_named_bystander(tmp_path):
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    mine = subprocess.Popen(sleeper)
    bystander = subprocess.Popen(sleeper)      # same image name, NOT registered: must survive
    try:
        r = reg(tmp_path)
        r.register("mine", mine.pid, "sleeper", now=NOW)
        assert R.is_same_process(mine.pid, r.load()["mine"]["create_time"])
        res = r.stop("mine", grace_s=5)
        assert res["stopped"] and mine.pid in res["pids"]
        mine.wait(timeout=10)
        assert mine.poll() is not None
        assert bystander.poll() is None, "stop() must never touch a process it did not start"
        assert r.load()["mine"]["state"] == "stopped"
    finally:
        for p in (mine, bystander):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=10)


def test_stop_refuses_unknown_job_and_a_recycled_pid(tmp_path):
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        r = reg(tmp_path)
        assert r.stop("nobody")["stopped"] is False
        r.register("stale", victim.pid, "x", now=NOW, create_time=12345.0)   # create-time says a different, older process
        out = r.stop("stale")
        assert out["stopped"] is False and "reused" in out["reason"]
        assert victim.poll() is None, "a PID whose create-time does not match must not be signalled"
        assert r.load()["stale"]["state"] == "failed"
    finally:
        victim.kill()
        victim.wait(timeout=10)


def test_stop_does_nothing_twice(tmp_path):
    r = reg(tmp_path)
    add(r, "J", 4242)
    r.finish("J", "done", NOW)
    assert r.stop("J")["stopped"] is False
    with pytest.raises(ValueError):
        r.finish("J", "vanished")
    with pytest.raises(KeyError):
        r.heartbeat("missing")


def test_prune_keeps_running_rows(tmp_path):
    r = reg(tmp_path)
    for i in range(6):
        add(r, f"d{i}", 100 + i)
        r.finish(f"d{i}", "done", NOW + i)
    add(r, "live", 200)
    dropped = r.prune(keep_finished=2)
    assert len(dropped) == 4 and "live" in r.load() and dropped[0] == "d0"


def test_lock_breaks_a_crashed_writers_stale_lock(tmp_path):
    r = reg(tmp_path)
    lock = tmp_path / "procs.json.lock"
    lock.write_text("999999")
    os.utime(lock, (time.time() - 3600, time.time() - 3600))
    add(r, "J", 1)                        # would hang without the stale-lock rule
    assert "J" in r.load() and not lock.exists()


# ---------------------------------------------------------------- staleness
def test_stale_report_finds_every_planted_kind_and_only_those(tmp_path):
    r = reg(tmp_path)
    add(r, "healthy", 10, "python h.py", tmp_path / "h", hb=NOW - 30, ppid=1)
    add(r, "dead", 11, "python d.py", tmp_path / "d", hb=NOW - 30)
    add(r, "orphan", 12, "python o.py", tmp_path / "o", hb=NOW - 30, ppid=99)          # parent 99 is not alive
    add(r, "silent", 13, "python s.py", tmp_path / "s", hb=NOW - 4000)
    add(r, "twin_a", 14, "python  t.py", tmp_path / "ta", hb=NOW - 5)
    add(r, "twin_b", 15, "python t.py", tmp_path / "tb", hb=NOW - 5)                    # same command, whitespace differs
    add(r, "share_a", 16, "python x1.py", tmp_path / "shared", hb=NOW - 5)
    add(r, "share_b", 17, "python x2.py", tmp_path / "shared", hb=NOW - 5)
    info = alive_info({10: (100.0, 1), 12: (100.0, 99), 13: (100.0, 1), 14: (100.0, 1), 15: (100.0, 1), 16: (100.0, 1), 17: (100.0, 1), 1: (1.0, 0)})
    rep = R.stale_report(r, NOW, heartbeat_s=300, info=info)
    got = {(f["job"], f["kind"]) for f in rep["findings"]}
    assert ("dead", "dead") in got and ("orphan", "orphaned") in got and ("silent", "silent") in got
    assert {("twin_a", "duplicate_job"), ("twin_b", "duplicate_job")} <= got
    assert {("share_a", "shared_output"), ("share_b", "shared_output")} <= got
    assert not [f for f in rep["findings"] if f["job"] == "healthy"], "the control job must stay clean"
    assert not rep["healthy"]


def test_recycled_pid_counts_as_dead_and_reap_marks_failed_not_done(tmp_path):
    r = reg(tmp_path)
    add(r, "J", 10, ct=100.0)
    info = alive_info({10: (999.0, 1)})              # alive, but a different process instance
    assert [f["kind"] for f in R.stale_report(r, NOW, info=info)["findings"]] == ["dead"]
    assert R.reap_dead(r, NOW, info=info) == ["J"]
    assert r.load()["J"]["state"] == "failed"


def test_stale_report_on_empty_registry_is_healthy(tmp_path):
    rep = R.stale_report(reg(tmp_path), NOW)
    assert rep["healthy"] and rep["n_running"] == 0 and rep["findings"] == []


# ---------------------------------------------------------------- queue
class FakeProc:
    def __init__(self, pid):
        self.pid = pid


def make_queue(tmp_path, free, launched, **kw):
    pids = iter(range(5000, 6000))
    table = {}

    def launcher(job):
        pid = next(pids)
        table[pid] = (100.0, os.getpid())
        launched.append(job.job_id)
        return FakeProc(pid)
    q = R.JobQueue(reg(tmp_path), free_fn=lambda: free[0], launcher=launcher, rss_fn=lambda pid: 0.1, cores=8,
                   info=lambda pid: {"alive": True, "create_time": None, "ppid": None, "rss_gb": 0.1}, **kw)
    return q


def test_queue_launches_only_what_memory_allows_and_says_why_it_held(tmp_path):
    launched, free = [], [6.0]
    q = make_queue(tmp_path, free, launched)
    q.submit(R.Job("a", "python a.py", est_gb=2.0, out_dir=tmp_path / "a"))
    q.submit(R.Job("b", "python b.py", est_gb=2.0, out_dir=tmp_path / "b"))
    log = q.tick(NOW)
    # 6 GB free, 2.5 reserved: "a" fits. It is still ramping (owes 1.9 GB), so headroom is 4.1 and 4.1 - 2.5 < 2: "b" waits.
    assert launched == ["a"]
    assert log[-1]["event"] == "hold" and log[-1]["job"] == "b" and "needs 2.0 GB" in log[-1]["reason"]
    assert [j[2].job_id for j in q.pending] == ["b"]


def test_queue_counts_recently_launched_jobs_as_owed_memory(tmp_path):
    """Planted defect: 5 GB free is enough for ONE 2-GB job but not two - the second must wait until the first has
    grown into its estimate. A queue that ignores the ramp launches both and thrashes."""
    launched, free = [], [5.0]
    q = make_queue(tmp_path, free, launched)
    q.submit(R.Job("a", "python a.py", est_gb=2.0))
    q.submit(R.Job("b", "python b.py", est_gb=2.0))
    q.tick(NOW)
    assert launched == ["a"]
    assert q.headroom_gb(NOW + 1) == pytest.approx(5.0 - (2.0 - 0.1))
    q.tick(NOW + 1)
    assert launched == ["a"], "second job must be held while the first is still ramping"
    free[0] = 2.8                                   # the first job has now really consumed its memory
    q.tick(NOW + 500)                               # past the ramp window
    assert launched == ["a"]                        # 2.8 - reserve 2.5 = 0.3 < 2.0: still held, correctly
    free[0] = 5.0
    q.tick(NOW + 501)
    assert launched == ["a", "b"]


def test_queue_respects_priority_and_worker_ceiling(tmp_path):
    launched, free = [], [100.0]
    q = make_queue(tmp_path, free, launched, max_workers=2)
    q.submit(R.Job("low", "x", priority=9))
    q.submit(R.Job("high", "x", priority=1))
    q.submit(R.Job("mid", "x", priority=5))
    log = q.tick(NOW)
    assert launched == ["high", "mid"]
    assert any("ceiling" in x.get("reason", "") for x in log)
    assert [j[2].job_id for j in q.pending] == ["low"]


def test_queue_refuses_duplicates_and_bad_jobs(tmp_path):
    launched, free = [], [100.0]
    q = make_queue(tmp_path, free, launched)
    q.submit(R.Job("a", "x", out_dir=tmp_path / "o"))
    with pytest.raises(ValueError, match="already queued"):
        q.submit(R.Job("a", "x"))
    with pytest.raises(ValueError, match="claimed"):
        q.submit(R.Job("b", "x", out_dir=tmp_path / "o"))
    q.tick(NOW)
    with pytest.raises(ValueError, match="already running"):
        q.submit(R.Job("a", "x"))
    with pytest.raises(ValueError, match="owned by running"):
        q.submit(R.Job("c", "x", out_dir=tmp_path / "o"))
    for bad in (dict(job_id="", cmd="x"), dict(job_id="a/b", cmd="x"), dict(job_id="z", cmd=""), dict(job_id="z", cmd="x", est_gb=-1),
                dict(job_id="z", cmd="x", est_gb=float("nan"))):
        with pytest.raises(ValueError):
            R.Job(**bad)


def test_queue_holds_everything_when_memory_is_unreadable_and_when_empty(tmp_path):
    launched, free = [], [None]
    q = make_queue(tmp_path, free, launched)
    assert q.tick(NOW) == []                          # empty queue: nothing to do, no error
    q.submit(R.Job("a", "x"))
    log = q.tick(NOW)
    assert launched == [] and log[0]["event"] == "hold" and "unreadable" in log[0]["reason"]


def test_queue_records_a_failed_launch_instead_of_losing_the_job(tmp_path):
    def boom(job):
        raise OSError("no such file")
    q = R.JobQueue(reg(tmp_path), free_fn=lambda: 50.0, launcher=boom, cores=8)
    q.submit(R.Job("a", "nonexistent"))
    log = q.tick(NOW)
    assert log[0]["event"] == "launch_failed" and "no such file" in log[0]["reason"]


def test_queue_really_launches_and_registers_a_process(tmp_path):
    q = R.JobQueue(reg(tmp_path), free_fn=lambda: 50.0, cores=8, log_dir=tmp_path)
    q.submit(R.Job("real", [sys.executable, "-c", "import time; time.sleep(60)"], est_gb=0.1, out_dir=tmp_path / "o"))
    log = q.tick(time.time())
    pid = log[0]["pid"]
    try:
        assert log[0]["event"] == "launch" and R.process_info(pid)["alive"]
        assert q.registry.running()["real"]["pid"] == pid
    finally:
        q.registry.stop("real", grace_s=5)
    assert not R.process_info(pid)["alive"]


# ---------------------------------------------------------------- cleanup
def touch(p, age_days, now=NOW, size=10):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    t = now - age_days * 86400
    os.utime(p, (t, t))
    return p


def test_cleanup_dry_run_plans_but_never_deletes(tmp_path):
    old = touch(tmp_path / "scratch" / "a.tmp", 10)
    res = R.cleanup([tmp_path], 7, NOW)
    assert res["dry_run"] and res["would_delete"] == 1 and old.exists() and res["deleted"] == []


def test_cleanup_deletes_only_old_temp_files_and_spares_the_rest(tmp_path):
    old_tmp = touch(tmp_path / "run1" / "x.tmp", 10)
    new_tmp = touch(tmp_path / "run1" / "y.tmp", 1)
    old_result = touch(tmp_path / "run1" / "result.json", 400)         # old but not a temp file: real output
    sealed = touch(tmp_path / "livesim" / "z.tmp", 400)               # protected tree
    cache = touch(tmp_path / "cache" / "c.partial", 400)
    live_dir = tmp_path / "running"
    inside = touch(live_dir / "w.tmp", 400)
    r = reg(tmp_path)
    me = os.getpid()
    r.register("live", me, "x", live_dir, now=NOW, create_time=R.process_info(me)["create_time"])
    res = R.cleanup([tmp_path], 7, NOW, dry_run=False, registry=r)
    assert not old_tmp.exists()
    for keep in (new_tmp, old_result, sealed, cache, inside):
        assert keep.exists(), keep
    assert {s["reason"] for s in res["skipped"]} >= {"sealed or cache tree"}
    assert any("live job folder" in s["reason"] for s in res["skipped"])
    assert res["bytes"] == 10


def test_cleanup_refuses_too_broad_roots_and_future_files_are_not_old(tmp_path):
    from engine import config as K
    with pytest.raises(ValueError, match="too broad"):
        R.cleanup([K.ROOT], 1, NOW)
    with pytest.raises(ValueError, match="too broad"):
        R.cleanup([os.path.abspath(os.sep)], 1, NOW)
    with pytest.raises(ValueError):
        R.cleanup([tmp_path], -1, NOW)
    fut = touch(tmp_path / "f.tmp", -5)                               # mtime in the future
    assert R.cleanup([tmp_path], 0, NOW)["would_delete"] == 0 and fut.exists()
    with pytest.raises(ValueError, match="narrow"):
        for i in range(3):
            touch(tmp_path / f"m{i}.tmp", 30)
        R.cleanup([tmp_path], 1, NOW, max_delete=2)


def test_cleanup_on_missing_root_is_empty(tmp_path):
    res = R.cleanup([tmp_path / "nope"], 1, NOW)
    assert res["would_delete"] == 0 and res["plan"] == []


# ---------------------------------------------------------------- health + waiting
def test_worker_health_reports_jobs_with_age_and_flags_memory_overrun(tmp_path):
    r = reg(tmp_path)
    add(r, "big", 10, hb=NOW - 60, est=1.0)
    info = lambda pid: {"alive": True, "create_time": 100.0, "ppid": 1, "rss_gb": 3.0} if pid == 10 else {"alive": pid == 1, "create_time": 1.0, "ppid": 0, "rss_gb": None}
    h = R.worker_health(r, NOW, free_fn=lambda: 9.0, info=info)
    assert h["n_running"] == 1 and h["running"][0]["heartbeat_age_s"] == 60 and h["over_estimate"] == ["big"]
    assert h["recommended"]["workers"] == 3
    assert R.worker_health(reg(tmp_path / "x"), NOW, free_fn=lambda: None)["recommended"]["workers"] == 0


def test_wait_for_memory_gives_up_and_succeeds_on_a_fake_clock():
    t = [0.0]
    sleeps = []
    ok, f = R.wait_for_memory(3.0, free_fn=lambda: 1.0, poll_s=60, give_up_s=300, sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)),
                              clock=lambda: t[0])
    assert not ok and f == 1.0 and len(sleeps) == 5
    seq = iter([1.0, 1.0, 4.0])
    ok, f = R.wait_for_memory(3.0, free_fn=lambda: next(seq), sleep=lambda s: None, clock=lambda: 0.0)
    assert ok and f == 4.0
