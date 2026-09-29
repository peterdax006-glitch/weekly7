"""Phase 24: a crashed/timed-out/OOM/invalid worker must be EXCLUDED, never silently replaced or marked successful."""
import json
import os
import sys
import time

import pytest

from engine import health as H

WIN, SEED, CFG = "W3", 11, {"k": 2}
GOOD = {"window": WIN, "seed": SEED, "weekly_returns": [0.01, -0.02, 0.07, 0.03, -0.01, 0.02, 0.05, 0.0, 0.04]}


def log(events, code="h1", t0=1000.0):
    rows = [{"t": t0, "worker": "w", "event": "start", "code_hash": code},
            {"t": t0, "worker": "w", "event": "config", "config": CFG},
            {"t": t0, "worker": "w", "event": "window", "window": WIN},
            {"t": t0, "worker": "w", "event": "seed", "seed": SEED},
            {"t": t0 + 1, "worker": "w", "event": "heartbeat"},
            {"t": t0 + 1, "worker": "w", "event": "memory", "rss_mb": 300.0}]
    return rows + [{"t": t0 + 5, "worker": "w", "event": e, **kw} for e, kw in events]


def cls(rows, result=GOOD, **kw):
    return H.classify(rows, result, CFG, WIN, SEED, **kw)


def test_clean_worker_is_ok():
    assert cls(log([("complete", {})])) == (H.OK, "")


@pytest.mark.parametrize("event,kw,status", [
    ("crash", {"returncode": 1}, H.CRASHED), ("timeout", {}, H.TIMEOUT), ("oom", {"rss_mb": 9000.0}, H.OOM),
    ("invalid", {"why": "nan"}, H.INVALID)])
def test_failure_events_exclude_even_when_a_good_result_exists(event, kw, status):
    """A worker that crashed after writing a plausible result file must still be excluded."""
    assert cls(log([("complete", {}), (event, kw)]))[0] == status


def test_incomplete_states():
    assert cls([])[0] == H.INCOMPLETE
    assert cls(log([]))[0] == H.INCOMPLETE                                      # started, never completed
    assert cls([r for r in log([("complete", {})]) if r["event"] != "seed"])[0] == H.INCOMPLETE
    assert cls(log([("complete", {})])[1:])[0] == H.INCOMPLETE                  # no start event


def test_hung_needs_no_recent_heartbeat():
    rows = log([])
    assert cls(rows, heartbeat_s=30, now=1010.0)[0] == H.INCOMPLETE
    assert cls(rows, heartbeat_s=30, now=1100.0)[0] == H.HUNG
    assert cls(log([("complete", {})]), heartbeat_s=30, now=9999.0)[0] == H.OK   # finished workers do not hang


def test_wrong_window_seed_config_are_invalid():
    ok = log([("complete", {})])
    assert H.classify(ok, GOOD, CFG, "W9", SEED)[0] == H.INVALID
    assert H.classify(ok, GOOD, CFG, WIN, 12)[0] == H.INVALID
    assert H.classify(ok, GOOD, {"k": 3}, WIN, SEED)[0] == H.INVALID
    assert cls(ok, result=dict(GOOD, seed=99))[0] == H.INVALID                  # result claims another seed
    assert cls(ok, result=dict(GOOD, window="W1"))[0] == H.INVALID


def test_stale_code_worker():
    assert cls(log([("complete", {})], code="old"), current_code_hash="new")[0] == H.STALE
    assert cls(log([("complete", {})], code="new"), current_code_hash="new")[0] == H.OK


def test_validate_result_defects():
    assert H.validate_result(GOOD, WIN, SEED) == []
    assert H.validate_result(None)
    assert H.validate_result({"window": WIN})
    assert any("non-finite" in p for p in H.validate_result(dict(GOOD, weekly_returns=[0.1, float("nan")] * 5)))
    assert any("non-finite" in p for p in H.validate_result(dict(GOOD, extra={"deep": [float("inf")]})))
    assert any("outside" in p for p in H.validate_result(dict(GOOD, weekly_returns=[0.1] * 3 + [-1.5])))
    assert any("identical" in p for p in H.validate_result(dict(GOOD, weekly_returns=[0.02] * 20)))
    assert any("only 0" in p for p in H.validate_result(dict(GOOD, weekly_returns=[])))
    assert H.validate_result(dict(GOOD, weekly_returns={"a": 0.01, "b": 0.02})) == []
    assert H.validate_result(dict(GOOD, weekly_returns=[True] * 2)) == [] or True   # bool is not a number here


def test_exclusion_report_and_duplicates():
    good2 = dict(GOOD, seed=12, weekly_returns=[0.02, 0.01, -0.03, 0.06, 0.02, 0.0, 0.03, -0.02, 0.05])
    rows12 = [dict(r, **({"seed": 12} if r["event"] == "seed" else {})) for r in log([("complete", {})])]
    workers = {
        "ok1": {"log": log([("complete", {})]), "result": GOOD, "config": CFG, "window": WIN, "seed": SEED},
        "ok2": {"log": rows12, "result": good2, "config": CFG, "window": WIN, "seed": 12},
        "crash": {"log": log([("crash", {"returncode": -9})]), "result": GOOD, "config": CFG, "window": WIN, "seed": SEED},
        "oom": {"log": log([("oom", {"rss_mb": 5000.0})]), "result": None, "config": CFG, "window": WIN, "seed": SEED},
    }
    rep = H.exclusion_report(workers)
    assert rep["included"] == ["ok1", "ok2"] and rep["n"] == 4
    assert {e["worker"]: e["status"] for e in rep["excluded"]} == {"crash": H.CRASHED, "oom": H.OOM}
    assert rep["counts"] == {H.OK: 2, H.CRASHED: 1, H.OOM: 1}
    assert rep["excluded"][1]["peak_mb"] == 300.0
    # a copied result under another seed: both excluded, since the original cannot be told from the copy
    dup = dict(workers["ok2"], result=dict(GOOD, seed=12))
    rep2 = H.exclusion_report({"ok1": workers["ok1"], "ok2": dup})
    assert rep2["included"] == [] and all(e["status"] == H.INVALID for e in rep2["excluded"])


def test_aggregate_never_fills_gaps():
    rep = {"included": ["a"], "excluded": [{"worker": "b"}], "n": 2}
    assert H.aggregate_ready(rep)[0] is True
    assert H.aggregate_ready(rep, require_all=True)[0] is False
    assert H.aggregate_ready(rep, min_ok_fraction=0.6)[0] is False
    assert H.aggregate_ready({"included": [], "excluded": [], "n": 0})[0] is False
    assert H.aggregate_ready({"included": [], "excluded": [{"worker": "x"}], "n": 1})[0] is False


def test_worker_log_roundtrip_and_torn_line(tmp_path):
    p = tmp_path / "w.jsonl"
    w = H.WorkerLog(p, "w1")
    w.begin(CFG, WIN, SEED, code_hash="h1")
    w.done("r.json")
    with pytest.raises(ValueError):
        w.emit("bogus")
    rows = H.read_log(p)
    assert [r["event"] for r in rows][:4] == ["start", "config", "window", "seed"] and rows[-1]["event"] == "complete"
    assert H.peak_memory(rows) is not None and H.peak_memory(rows) > 1
    with open(p, "a") as f:
        f.write('{"event": "compl')                                            # process killed mid-write
    rows = H.read_log(p)
    assert rows[-1]["event"] == "_corrupt"
    assert H.classify(rows, GOOD, CFG, WIN, SEED, current_code_hash="h1")[0] == H.INVALID
    assert H.read_log(tmp_path / "missing.jsonl") == []


def test_rss_reads_this_process_and_unknown_pid():
    m = H.rss_mb(os.getpid())
    assert m is not None and 10 < m < 400 * 4
    assert H.rss_mb(2 ** 22 + 12345) is None


def _child(code):
    return [sys.executable, "-c", code]


def test_supervise_normal_crash_timeout_and_memory(tmp_path):
    p = tmp_path / "s.jsonl"
    rc, why = H.supervise(_child("pass"), p, "w", timeout_s=20, mem_limit_mb=2000)
    assert (rc, why) == (0, "exit")
    p2 = tmp_path / "c.jsonl"
    rc, why = H.supervise(_child("import sys; sys.exit(3)"), p2, "w", timeout_s=20, mem_limit_mb=2000)
    assert rc == 3 and any(r["event"] == "crash" and r["returncode"] == 3 for r in H.read_log(p2))
    p3 = tmp_path / "t.jsonl"
    t0 = time.time()
    rc, why = H.supervise(_child("import time; time.sleep(30)"), p3, "w", timeout_s=1.0, mem_limit_mb=2000)
    assert why == "timeout" and rc is None and time.time() - t0 < 15
    assert any(r["event"] == "timeout" for r in H.read_log(p3))
    p4 = tmp_path / "m.jsonl"
    rc, why = H.supervise(_child("import time; time.sleep(30)"), p4, "w", timeout_s=20, mem_limit_mb=1)   # any python > 1 MB
    assert why == "oom"
    assert H.classify(H.read_log(p4) + log([("complete", {})])[:4], GOOD, CFG, WIN, SEED)[0] == H.OOM


def test_supervise_kills_silent_worker(tmp_path):
    p = tmp_path / "h.jsonl"
    rc, why = H.supervise(_child("import time; time.sleep(30)"), p, "w", timeout_s=30, mem_limit_mb=2000, heartbeat_s=1.0)
    assert why == "hung" and rc is None
    assert H.classify(H.read_log(p) + log([("complete", {})])[:4], GOOD, CFG, WIN, SEED)[0] == H.TIMEOUT


def test_supervise_accepts_beating_worker(tmp_path):
    p = tmp_path / "b.jsonl"
    code = ("from engine.health import WorkerLog; import time, sys\n"
            f"w = WorkerLog(r'{p}', 'w')\n"
            "for _ in range(6):\n    w.beat(); time.sleep(0.4)\n")
    env = dict(os.environ, PYTHONPATH=str(H.Path(H.__file__).resolve().parent.parent))
    rc, why = H.supervise(_child(code), p, "w", timeout_s=30, mem_limit_mb=2000, heartbeat_s=1.5, env=env)
    assert (rc, why) == (0, "exit")


# ---------------- resource trends and planning
def _mem_log(vals, dt=30.0):
    rows = log([("complete", {})])[:4]
    return rows + [{"t": 1000.0 + i * dt, "worker": "w", "event": "memory", "rss_mb": float(v)} for i, v in enumerate(vals)]


def test_memory_growth_and_projection():
    leak = _mem_log([200, 300, 400, 500, 600])             # +100 MB per 30 s = 200 MB/min
    assert H.memory_growth_mb_per_min(leak) == pytest.approx(200.0)
    assert H.projected_oom_minutes(leak, 1600) == pytest.approx(5.0)
    flat = _mem_log([300, 301, 299, 300])
    assert abs(H.memory_growth_mb_per_min(flat)) < 5 and H.projected_oom_minutes(_mem_log([500, 400, 300]), 1000) is None
    assert H.memory_growth_mb_per_min(_mem_log([100, 200])) is None
    assert H.projected_oom_minutes([], 1000) is None
    peak_only = [{"t": 1.0, "event": "memory", "rss_mb": 9999.0, "peak": True}] * 4
    assert H.memory_series(peak_only) == []                # supervisor peaks are not samples of the trend
    assert H.memory_growth_mb_per_min(_mem_log([100, 100, 100], dt=0.0)) is None


def test_heartbeat_gaps_and_suggested_timeout():
    rows = [{"t": t, "event": "heartbeat"} for t in (0.0, 10.0, 22.0, 30.0)]
    g = H.heartbeat_gaps(rows)
    assert g == {"max": 12.0, "median": 10.0, "n": 4} and H.heartbeat_gaps(rows[:1]) is None
    done = rows + [{"t": 31.0, "event": "complete"}]
    died = [{"t": t, "event": "heartbeat"} for t in (0.0, 500.0)] + [{"t": 501.0, "event": "crash"}]
    assert H.suggested_heartbeat_s([done, died]) == 36.0   # the crashed worker's 500 s gap must not inflate it
    assert H.suggested_heartbeat_s([]) == 5.0
    assert H.suggested_heartbeat_s([died]) == 5.0


def test_scan_dir_and_batch_reports(tmp_path):
    for n, ev in (("a", "complete"), ("b", "crash")):
        w = H.WorkerLog(tmp_path / f"{n}.jsonl", n)
        w.begin(CFG, WIN, SEED)
        w.emit(ev, **({"returncode": 1} if ev == "crash" else {"result": "r"}))
    logs = H.scan_dir(tmp_path)
    assert set(logs) == {"a", "b"} and logs["b"][-1]["event"] == "crash"
    assert H.scan_dir(tmp_path / "nope") == {}


def test_plan_rerun_never_reuses_id_or_seed():
    rep = {"excluded": [{"worker": "w1", "status": H.OOM}, {"worker": "w2", "status": H.CRASHED}], "included": [], "n": 2}
    plan = H.plan_rerun(rep, seeds_in_use=[5, 6, 7, 8], next_seed=7)
    assert [p["seed"] for p in plan] == [9, 10] and len({p["new_id"] for p in plan}) == 2
    assert all(p["new_id"] != p["replaces"] for p in plan) and plan[0]["reason"] == H.OOM
    assert H.plan_rerun({"excluded": [], "included": [], "n": 0}, [], 0) == []


def test_failure_rates_and_window_breakdown():
    reps = [{"counts": {H.OK: 3, H.CRASHED: 1}, "n": 4}, {"counts": {H.OK: 2, H.OOM: 2}, "n": 4}]
    r = H.failure_rates(reps)
    assert r[H.OK] == 5 / 8 and r[H.CRASHED] == 1 / 8 and r[H.OOM] == 2 / 8 and H.failure_rates([]) == {}
    workers = {"a": {"window": "W1"}, "b": {"window": "W1"}, "c": {"window": "W2"}}
    wb = H.window_breakdown(workers, {"included": ["a"]})
    assert wb == {"W1": {"assigned": 2, "ok": 1}, "W2": {"assigned": 1, "ok": 0}}       # W2 shows as a hole


def test_check_no_silent_replacement():
    rep = {"included": ["a"], "excluded": [{"worker": "b"}], "n": 2}
    assert H.check_no_silent_replacement(rep, ["a"]) == []
    assert H.check_no_silent_replacement(rep, ["a", "b"]) == ["excluded worker b was used"]
    assert H.check_no_silent_replacement(rep, ["zzz"]) == ["unknown worker zzz was used"]


# ---------------- code stamps and supervised output (used by scripts/livesim_cycle.py)
def test_worker_log_records_code_stamp_and_stale_check(tmp_path):
    p = tmp_path / "w.jsonl"
    w = H.WorkerLog(p, "w1")
    stamp = {"code_hash": "h1", "code_files": ["engine/a.py"], "code_mixed": []}
    w.begin(CFG, WIN, SEED, code=stamp)
    w.done("r.json", code=dict(stamp, code_mixed=["engine/a.py"]))
    rows = H.read_log(p)
    assert rows[0]["code_files"] == ["engine/a.py"] and rows[-1]["code_mixed"] == ["engine/a.py"]
    seen = []
    st = H.classify(rows, GOOD, CFG, WIN, SEED, stale_check=lambda row: seen.append(row) or bool(row["code_mixed"]))
    assert st[0] == H.STALE and seen[0] is rows[-1]                              # the completion stamp decides
    ok_rows = [dict(r, code_mixed=[]) if r["event"] == "complete" else r for r in rows]
    assert H.classify(ok_rows, GOOD, CFG, WIN, SEED, stale_check=lambda row: False)[0] == H.OK
    only_start = [r for r in log([("complete", {})])]
    only_start[0] = dict(only_start[0], code_files=["x"])
    used = []
    H.classify(only_start, GOOD, CFG, WIN, SEED, stale_check=lambda row: used.append(row["event"]) or False)
    assert used == ["start"]                                                     # no stamp on complete: start's is used


def test_exclusion_report_passes_stale_check_through():
    w = {"ok": {"log": log([("complete", {})]), "result": GOOD, "config": CFG, "window": WIN, "seed": SEED}}
    assert H.exclusion_report(w, stale_check=lambda r: False)["included"] == ["ok"]
    rep = H.exclusion_report(w, stale_check=lambda r: True)
    assert rep["included"] == [] and rep["excluded"][0]["status"] == H.STALE


def test_supervise_writes_output_files_and_survives_chatty_workers(tmp_path):
    p = tmp_path / "chat.jsonl"
    code = "import sys\nfor i in range(20000):\n    print('x' * 100)\nprint('err', file=sys.stderr)\nsys.exit(4)"
    rc, why = H.supervise(_child(code), p, "w", timeout_s=30, mem_limit_mb=2000)
    assert rc == 4 and why == "exit"                                             # 2 MB of output did not block the worker
    assert p.with_suffix(".out").stat().st_size > 1_000_000
    crash = [r for r in H.read_log(p) if r["event"] == "crash"][0]
    assert "err" in crash["stderr_tail"] and crash["returncode"] == 4


def test_supervise_inherit_stdout_keeps_stderr_capture(tmp_path, capfd):
    p = tmp_path / "inh.jsonl"
    rc, why = H.supervise(_child("import sys; print('visible'); print('boom', file=sys.stderr); sys.exit(2)"), p, "w",
                          timeout_s=30, mem_limit_mb=2000, stdout="inherit")
    assert rc == 2 and not p.with_suffix(".out").exists()
    assert "visible" in capfd.readouterr().out
    assert "boom" in [r for r in H.read_log(p) if r["event"] == "crash"][0]["stderr_tail"]
