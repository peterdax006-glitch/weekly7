"""Pure functions of scripts/livesim_loop2.py: objective choice, window selection, basis adoption, gates still wired."""
import importlib.util
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from engine import objective as O, basis_search as B

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "livesim_loop2.py"


@pytest.fixture(scope="module")
def L():
    argv, sys.argv = sys.argv, ["livesim_loop2.py"]
    try:
        spec = importlib.util.spec_from_file_location("livesim_loop2_under_test", SRC)
        m = importlib.util.module_from_spec(spec)
        before = m_state_stamp()
        spec.loader.exec_module(m)
        assert m_state_stamp() == before, "importing the loop wrote its state file"
        return m
    finally:
        sys.argv = argv


def m_state_stamp():
    p = ROOT / "state" / "livesim" / "loop2.json"
    return (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None


def row(inb, over=0.0, w5=-0.05, dd=-0.1, pos=0.5):
    return {"in_band": inb, "over_band": over, "worst5": w5, "max_dd": dd, "pos_in_band": pos, "mean_week": 0.01, "sd_week": 0.04}


def test_import_does_not_start_the_loop(L):
    assert callable(L.main) and callable(L.train_basis) and callable(L.archive_dirs)


def test_tiered_is_the_library_objective(L):
    rows = [row(0.6), row(0.4, 0.1)]
    s = O.evaluate(rows)
    assert L.tiered(rows) == (s.scalar, s.t1, s.risk, s.t3)
    assert L.objective(rows, "volatility")[0] == s.scalar


def test_objective_is_lexicographic_through_the_loop_api(L):
    """Planted: a basis with worse tier 1 but a far better tail and direction must NOT win (old blended score allowed it)."""
    a = [row(0.20, 0.0, -0.01, -0.02, 1.0)]
    b = [row(0.30, 0.0, -0.30, -0.60, 0.0)]
    assert L.objective(b, "x")[0] > L.objective(a, "x")[0]


def test_objective_floor_and_empty(L):
    assert L.objective([row(0.9, dd=-0.995)], "x")[0] == -9.0
    assert L.objective([], "x") == (-9.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        L.objective([row(float("nan"))], "x")


def test_meta_space_is_the_bible_list_and_cfg_space_covers_the_start_cfg(L):
    assert L.META_SPACE is B.META_SPACE
    assert set(L.st["cfg"]) <= set(L.CFG_SPACE)
    rng = np.random.default_rng(0)
    for _ in range(30):
        c, m = B.sample_candidate(rng, L.st["cfg"], L.st["meta"], L.CFG_SPACE, L.META_SPACE)
        assert set(c) >= set(L.st["cfg"]) and not B.validate_meta(m)


def test_archive_dirs_skips_backups_empties_and_files(L, tmp_path):
    for name, snap in (("r01a", True), ("w02b", True), ("_w01c_original", True), ("r03c", False)):
        d = tmp_path / name
        d.mkdir()
        if snap:
            (d / "wsnap_2190-06-14.parquet").write_bytes(b"x")
    (tmp_path / "loop2.json").write_text("{}")
    assert [a.name for a in L.archive_dirs(tmp_path)] == ["r01a", "w02b"]
    assert L.archive_dirs(tmp_path / "missing") == []


def test_run_window_builds_tier_row_from_the_session(L, monkeypatch):
    class FakeSession:
        weeks = [0.07, -0.06, 0.02, 0.12, -0.30, 0.08]
        def result(self):
            return {"mean_week": 0.0, "max_dd": -0.42, "year_return": 0.1}
    monkeypatch.setattr(L.A, "replay", lambda *a, **k: FakeSession())
    w = {"snaps": {}, "closes": None, "bps": 5, "divs": {}, "opens": None, "ltm": None}
    r = L.run_window(w, {}, {})
    assert r["in_band"] == pytest.approx(0.5) and r["max_dd"] == -0.42          # daily drawdown kept, not the weekly one
    assert r["worst5"] == pytest.approx(np.quantile(FakeSession.weeks, 0.05)) and r["cat_rate"] == pytest.approx(1 / 6)
    assert O.evaluate([r]).key[0] > 0


def synth_windows(n):
    return [{"id": f"w{i}", "closes": pd.DataFrame({"a": [1.0, 2.0]}, index=pd.to_datetime([f"2190-01-{i + 1:02d}", f"2190-02-{i + 1:02d}"]))}
            for i in range(n)]


def planted_eval(w, cfg, meta):
    rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
    return O.week_row(0.005 + 0.08 * cfg["k"] ** -0.5 * rng.standard_normal(52) * 0.45)


def test_train_basis_adopts_a_planted_improvement_and_tags_windows(L):
    start = {**L.st["cfg"], "k": 4}
    r = L.train_basis(synth_windows(20), start, L.st["meta"], seed=1, evaluate=planted_eval)
    assert r.adopted and r.cfg["k"] < 4 and r.n_windows == 20 and r.n_screen == L.SCREEN_N
    assert r.winner.confirm.t1 > r.incumbent.confirm.t1


def test_train_basis_keeps_incumbent_on_noise_and_is_seeded(L):
    def noise(w, c, m):
        return O.week_row(0.07 * np.random.default_rng(zlib.crc32((w["id"] + B.fingerprint(c, m)).encode())).standard_normal(52))
    a = L.train_basis(synth_windows(26), L.st["cfg"], L.st["meta"], seed=3, evaluate=noise)
    b = L.train_basis(synth_windows(26), L.st["cfg"], L.st["meta"], seed=3, evaluate=noise)
    assert (a.adopted, a.cfg) == (b.adopted, b.cfg)
    c = L.train_basis(synth_windows(2), L.st["cfg"], L.st["meta"], seed=3, evaluate=noise)
    assert not c.adopted and c.n_evals == 0                                      # too few windows: keep the basis


def test_train_basis_honours_as_of(L):
    seen = set()
    def ev(w, c, m):
        seen.add(w["id"])
        return planted_eval(w, c, m)
    ws = synth_windows(20)
    cut = ws[9]["closes"].index[-1]
    L.train_basis(ws, {**L.st["cfg"], "k": 4}, L.st["meta"], seed=1, evaluate=ev, as_of=cut)
    assert seen and max(int(i[1:]) for i in seen) <= 9


def test_safety_gates_are_still_wired_into_the_round():
    src = SRC.read_text(encoding="utf-8")
    for needle in ("provenance.stale(provenance.code_stamp())", "STALE CODE", "fill_audit.gate(S1", "future-scramble",
                   "gate_ok &= rep and scram and fill_ok", "ANTI-CHEAT GATE FAILED", "scramble_after=cut"):
        assert needle in src, f"loop lost a gate: {needle}"
    assert src.index("ANTI-CHEAT GATE FAILED") < src.index("res = train_basis(wins")    # gates precede any retraining


# ---------------------------------------------------------------- Phase 21-24 machinery (worker health, supervision, reveal)
import json
import time
from types import SimpleNamespace
from engine import blind_gates as BG, health

CFG = {"k": 2, "pool_q": 0.7, "brake": None}
META = {"half_life": 6}


def fake_run(weeks, findings, mem_rows=0):
    """A stand-in for livesim.run: returns (feed, trader, sealed, wall) with just what the worker touches."""
    idx = pd.date_range("2190-01-01", periods=4)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    mem = pd.DataFrame({"x": range(mem_rows)})
    feed = SimpleNamespace(_stocks={"Close": frame, "Open": frame}, first_live=idx[0], cost_bps=5, real_end=lambda: "2020-01-01",
                           sic=pd.DataFrame({"ticker": ["a"], "sic": [1]}), audit=lambda: findings)
    trader = SimpleNamespace(long_term=None, snaps={}, warm_snaps={}, preseason={}, days=[1, 2, 3], cfg=CFG,
                             session=SimpleNamespace(adapter=SimpleNamespace(mem=SimpleNamespace(export=lambda: mem)),
                                                     result=lambda: {"mean_week": 0.01, "max_dd": -0.1, "year_return": 0.1,
                                                                     "adaptations": [], "missed_winners": [], "weeks_ge_7": 1},
                                                     weeks=weeks, days=[1, 2, 3]))
    return lambda cfg, run_id, log=None, adaptive=False, meta=None: (feed, trader, None, 1.5)


@pytest.fixture
def sandbox(L, tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DIR", tmp_path)
    monkeypatch.setattr(L, "BEAT_EVERY_S", 0.05)
    return tmp_path


GOOD_WEEKS = [0.07, -0.06, 0.02, 0.09, -0.03, 0.05, 0.08, -0.04, 0.06, 0.01]


def run_fake_worker(L, monkeypatch, weeks=GOOD_WEEKS, findings=(), rid="w99a"):
    monkeypatch.setattr(L.livesim, "run", fake_run(weeks, list(findings)))
    L.worker(rid, CFG, META)
    return rid


def test_worker_logs_start_config_window_seed_heartbeat_complete(L, sandbox, monkeypatch):
    monkeypatch.setattr(L.livesim, "run", lambda *a, **k: (time.sleep(0.3), fake_run(GOOD_WEEKS, [])(*a, **k))[1])
    L.worker("w99a", CFG, META)
    ev = [r["event"] for r in health.read_log(sandbox / "w99a" / "health2.jsonl")]
    for needed in ("start", "config", "window", "seed", "complete"):
        assert needed in ev
    assert ev.count("heartbeat") >= 3                                  # begin() beat + the background thread's 30 s beats (shrunk here)
    start = [r for r in health.read_log(sandbox / "w99a" / "health2.jsonl") if r["event"] == "start"][0]
    assert "code_files" in start and "code_hash" in start
    res = json.loads((sandbox / "w99a" / "result2.json").read_text())
    assert res["window"] == "w99a" and res["seed"] == L.MODEL_SEED


def test_failed_audit_writes_no_result_and_records_invalid(L, sandbox, monkeypatch):
    bad = [BG.Finding("seal", "fail", "window overlaps a played window")]
    run_fake_worker(L, monkeypatch, findings=bad, rid="w98a")
    assert not (sandbox / "w98a" / "result2.json").exists()
    rows = health.read_log(sandbox / "w98a" / "health2.jsonl")
    assert any(r["event"] == "invalid" and "blind gates failed" in r["why"] for r in rows)
    assert not any(r["event"] == "complete" for r in rows)
    assert not (sandbox / "memory_bank.parquet").exists()               # nothing archived, so no bank pollution
    rep = L.classify_round(["w98a"], CFG, META, root=sandbox, stale_check=lambda r: False)
    assert rep["included"] == [] and rep["excluded"][0]["status"] == health.INVALID


def test_warnings_do_not_block_but_nan_weeks_do(L, sandbox, monkeypatch):
    run_fake_worker(L, monkeypatch, findings=[BG.Finding("seal", "warn", "legacy seal")], rid="w97a")
    assert (sandbox / "w97a" / "result2.json").exists()
    run_fake_worker(L, monkeypatch, weeks=GOOD_WEEKS[:-1] + [float("nan")], rid="w97b")
    assert not (sandbox / "w97b" / "result2.json").exists() and not (sandbox / "memory_bank.parquet").exists()
    rows = health.read_log(sandbox / "w97b" / "health2.jsonl")
    assert any(r["event"] == "invalid" for r in rows)


def test_memory_bank_written_only_for_a_valid_window(L, sandbox, monkeypatch):
    monkeypatch.setattr(L.livesim, "run", fake_run(GOOD_WEEKS, [], mem_rows=3))
    L.worker("w96a", CFG, META)
    assert len(pd.read_parquet(sandbox / "memory_bank.parquet")) == 3


def test_a_stale_result_is_removed_when_a_worker_restarts(L, sandbox, monkeypatch):
    d = sandbox / "w95a"
    d.mkdir()
    (d / "result2.json").write_text('{"old": true}')
    run_fake_worker(L, monkeypatch, findings=[BG.Finding("seal", "fail", "x")], rid="w95a")
    assert not (d / "result2.json").exists()


def test_classify_round_includes_only_healthy_workers(L, sandbox, monkeypatch):
    for rid in ("w90a", "w90b"):
        run_fake_worker(L, monkeypatch, weeks=GOOD_WEEKS if rid == "w90a" else GOOD_WEEKS[::-1], rid=rid)
    run_fake_worker(L, monkeypatch, findings=[BG.Finding("seal", "fail", "x")], rid="w90c")
    rep = L.classify_round(["w90a", "w90b", "w90c"], CFG, META, rnd=9, root=sandbox, stale_check=lambda r: False)
    assert rep["included"] == ["w90a", "w90b"] and [e["worker"] for e in rep["excluded"]] == ["w90c"]
    assert (sandbox / "round_09_loop2_health.json").exists()
    assert not health.check_no_silent_replacement(rep, ["w90a", "w90b"])
    assert health.check_no_silent_replacement(rep, ["w90a", "w90c"])           # using the excluded one is caught


def test_stale_code_worker_is_excluded_and_default_check_fails_closed(L, sandbox, monkeypatch):
    run_fake_worker(L, monkeypatch, rid="w91a")
    rep = L.classify_round(["w91a"], CFG, META, root=sandbox, stale_check=lambda r: True)
    assert rep["excluded"][0]["status"] == health.STALE
    # planted: a worker whose log carries no code stamp at all is stale under the default check
    d = sandbox / "w91b"
    d.mkdir()
    wl = health.WorkerLog(d / "health2.jsonl", "w91b")
    wl.begin(L.expected_config(CFG, META), "w91b", L.MODEL_SEED)
    (d / "result2.json").write_text(json.dumps({"window": "w91b", "seed": L.MODEL_SEED, "weekly_returns": GOOD_WEEKS}))
    wl.done(d / "result2.json")
    assert L.classify_round(["w91b"], CFG, META, root=sandbox)["excluded"][0]["status"] == health.STALE


def test_wrong_config_wrong_seed_torn_and_duplicate_results_are_excluded(L, sandbox, monkeypatch):
    run_fake_worker(L, monkeypatch, rid="w92a")
    assert L.classify_round(["w92a"], {**CFG, "k": 9}, META, root=sandbox, stale_check=lambda r: False)["excluded"][0]["status"] == "INVALID"
    (sandbox / "w92a" / "result2.json").write_text('{"window": "w92a", "seed": ')                  # torn write
    rep = L.classify_round(["w92a"], CFG, META, root=sandbox, stale_check=lambda r: False)
    assert rep["included"] == []
    run_fake_worker(L, monkeypatch, rid="w92b")
    run_fake_worker(L, monkeypatch, rid="w92c")
    body = json.loads((sandbox / "w92b" / "result2.json").read_text())      # planted: c's result is a copy of b's body
    (sandbox / "w92c" / "result2.json").write_text(json.dumps({**body, "window": "w92c"}))
    rep = L.classify_round(["w92b", "w92c"], CFG, META, root=sandbox, stale_check=lambda r: False)
    assert rep["included"] == [] and {e["worker"] for e in rep["excluded"]} == {"w92b", "w92c"}


def test_missing_worker_is_incomplete_not_silently_dropped(L, sandbox):
    rep = L.classify_round(["w93z"], CFG, META, root=sandbox, stale_check=lambda r: False)
    assert rep["n"] == 1 and rep["excluded"][0]["status"] == health.INCOMPLETE


def test_run_workers_supervises_each_window_in_parallel_with_limits(L, sandbox):
    calls, active, peak = [], [0], [0]
    import threading
    lock = threading.Lock()
    def fake_supervise(cmd, log_path, wid, timeout_s, mem_mb, heartbeat_s=None, stdout=None, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.15)
        with lock:
            active[0] -= 1
        calls.append((cmd, str(log_path), wid, timeout_s, mem_mb, heartbeat_s, stdout))
        return 0, "exit"
    out = L.run_workers(["w1a", "w1b", "w1c"], CFG, META, supervise=fake_supervise)
    assert set(out) == {"w1a", "w1b", "w1c"} and peak[0] == 3
    for cmd, log, wid, t, mem, hb, so in calls:
        assert "--worker" in cmd and wid in cmd and log.endswith("health2.jsonl")
        assert (t, mem, hb, so) == (L.WORKER_TIMEOUT_S, L.WORKER_MEM_MB, L.WORKER_HEARTBEAT_S, "inherit")
    assert L.WORKER_HEARTBEAT_S > L.BEAT_EVERY_S * 10                  # the supervisor's patience is far above the beat period


def test_supervise_kills_a_silent_worker_and_the_round_excludes_it(L, sandbox):
    """End to end with the real supervisor: a child that never heartbeats is killed as hung, then classified excluded."""
    d = sandbox / "w94a"
    d.mkdir()
    rc, reason = health.supervise([sys.executable, "-c", "import time; time.sleep(30)"], d / "health2.jsonl", "w94a",
                                  20, 6000, heartbeat_s=1, poll_s=0.1, stdout="inherit")
    assert reason == "hung" and rc is None
    rep = L.classify_round(["w94a"], CFG, META, root=sandbox, stale_check=lambda r: False)
    assert rep["included"] == [] and rep["excluded"][0]["status"] in (health.TIMEOUT, health.INCOMPLETE)


def test_reset_worker_files_clears_old_results_logs_and_audits(L, tmp_path):
    for f in L.WORKER_FILES:
        (tmp_path / f).write_text("x")
    (tmp_path / "keep.txt").write_text("x")
    L.reset_worker_files(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]
    L.reset_worker_files(tmp_path / "nowhere")                          # missing dir is fine


def test_reveal_is_refused_until_adjustments_are_locked(L):
    seen = []
    class Sealed:
        def __init__(self, rid): self.rid = rid
        def reveal(self, gate=None):
            gate.reveal()                                                # the real SealedYear.reveal does exactly this
            seen.append(self.rid)
            return f"period-{self.rid}"
    with pytest.raises(PermissionError):
        L.reveal_round(["w1a", "w1b"], False, sealed=Sealed)
    assert seen == []                                                    # nothing leaked before the lock
    assert L.reveal_round(["w1a", "w1b"], True, sealed=Sealed) == {"w1a": "period-w1a", "w1b": "period-w1b"}


def test_main_orders_supervision_gates_basis_and_reveal():
    src = SRC.read_text(encoding="utf-8")
    order = ["run_workers(ids", "classify_round(ids", "check_no_silent_replacement(report, done)", "ANTI-CHEAT GATE FAILED",
             "res = train_basis(wins", 'reveal_round([w["']
    pos = [src.index(x) for x in order]
    assert pos == sorted(pos), dict(zip(order, pos))
    assert "subprocess.Popen" not in src and "p.wait()" not in src        # no unsupervised children
    assert 'livesim.SealedYear(w["run_id"]).reveal()' not in src        # no ungated reveal
