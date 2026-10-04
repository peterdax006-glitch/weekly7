"""GPU PULSE RUNNER (creator.gpupulse, scripts/gpu_pulse.py) with local fakes only: a local bash stands in for the pod (fake nvidia-smi, fake
llama-server, models served from a file:// 'Hugging Face'), an in-process fake server stands in for the tunnel. Nothing is rented or contacted."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest

from creator import device as DEV
from creator import gpupulse as GP
from creator import judgment as J
from creator import reasondrills as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpupulse_fake_server as FAKE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else (shutil.which("bash") or "")
FAKE_MODEL = "Fake-1B.gguf"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _posix(p: Path) -> str:
    return p.resolve().as_posix()


@pytest.fixture()
def rt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Runtime dir (config, ledger, tunnel, monitor log) in tmp: outside the repository, as in real use."""
    d = tmp_path / "runtime"
    monkeypatch.setenv("NUPEN_RUNTIME", str(d))
    monkeypatch.delenv("NUPEN_GPU_PULSE", raising=False)
    monkeypatch.delenv("NUPEN_PULSE_CONFIG", raising=False)
    return d


@pytest.fixture()
def fake_server() -> Iterator[int]:
    FAKE.STATE.update(tokens=0, answer="A", bare=False, requests=[])
    srv, port = FAKE.start()
    yield port
    srv.shutdown()


def _pod_cfg(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, content: bytes = b"gguf" * 250, sha: Optional[str] = None) -> dict[str, Any]:
    """A fake pod: a local bash with a fake nvidia-smi and llama-server on PATH; the model is served by a file:// 'Hugging Face' and pinned in
    the runtime's MODELS.json (sha256 = the real hash unless `sha` says otherwise)."""
    if not BASH:
        pytest.skip("no bash")
    binp = tmp_path / "bin"
    binp.mkdir()
    fake = Path(FAKE.__file__).resolve()
    (binp / "llama-server").write_text(f'#!/bin/bash\nexec "{_posix(Path(sys.executable))}" "{_posix(fake)}" "$@"\n', encoding="utf-8", newline="\n")
    util = tmp_path / "util.txt"
    util.write_text("95", encoding="utf-8")
    (binp / "nvidia-smi").write_text(
        '#!/bin/bash\ncase "$*" in\n'
        f'  *utilization*) echo "$(cat "{_posix(util)}"), 9000";;\n'
        '  *query-gpu=name*) echo "NVIDIA GeForce RTX 4090, 24564 MiB, 1 MiB, 580.82";;\n'
        '  *compute-apps*) ;;\n'
        '  *) echo "| NVIDIA-SMI 580.82   Driver Version: 580.82   CUDA Version: 13.1 |";;\nesac\n', encoding="utf-8", newline="\n")
    monkeypatch.setenv("PATH", str(binp) + os.pathsep + os.environ.get("PATH", ""))
    hf = tmp_path / "hf" / "fake" / "repo" / "resolve" / "main"
    hf.mkdir(parents=True)
    (hf / FAKE_MODEL).write_bytes(content)
    (rt / "models").mkdir(parents=True, exist_ok=True)
    (rt / "models" / "MODELS.json").write_text(json.dumps({FAKE_MODEL: {"repo": "fake/repo", "bytes": len(content),
                                                                        "sha256": sha or hashlib.sha256(content).hexdigest()}}), encoding="utf-8")
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    cfg.update(shell_argv=[BASH, "-s"], remote_dir=_posix(tmp_path / "pod"), models=[FAKE_MODEL], hf_base="file:///" + _posix(tmp_path / "hf"),
               slots=2, ctx_per_slot=512, health_timeout_s=60, remote_port_base=_free_port(), deadman="stop", _util=str(util))
    return cfg


def _kill_pod(cfg: dict[str, Any]) -> None:
    GP.shell_for(cfg).run(GP.stop_servers_script(str(cfg["remote_dir"])), timeout=60, check=False)


# ------------------------------------------------------------------------------------------------ setup on a fake pod
def test_setup_is_idempotent_and_teardown_stops_the_servers(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch)
    said: list[str] = []
    try:
        first = GP.setup(cfg, say=said.append)
        assert first["fetched"] == [FAKE_MODEL] and first["cached"] == []
        assert "CUDA0" in first["facts"]["devices"] and "4090" in first["facts"]["gpu"]
        port = first["plan"][FAKE_MODEL][0]
        assert GP._healthy(port)                                                   # the fake llama-server answers on the planned port
        second = GP.setup(cfg, say=said.append)
        assert second["fetched"] == [] and second["cached"] == [FAKE_MODEL]          # a verified model is never fetched twice
        assert any("kept (healthy, same arguments)" in s for s in said)              # a healthy server with the same arguments is kept
        args = (tmp_path / "pod" / "run" / f"{port}.sig").read_text(encoding="utf-8")
        assert args.strip() == f"{FAKE_MODEL}|2|512"
        assert GP.budget_for(cfg).open_pulse() is not None                           # billing is counted from setup on
        out = GP.teardown(cfg, say=said.append)
        assert out["stopped"] == [str(port)] and out["destroy_command"].startswith("vastai destroy instance")
        assert GP.budget_for(cfg).open_pulse() is None
        time.sleep(1.0)
        assert not GP._healthy(port)
    finally:
        _kill_pod(cfg)


def test_a_model_whose_sha256_differs_is_deleted_and_refused(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch, sha="0" * 64)
    try:
        with pytest.raises(GP.PulseError, match="sha256 mismatch"):
            GP.setup(cfg, say=lambda s: None)
        left = list((tmp_path / "pod" / "models").iterdir())
        assert left == []                                                            # neither the .part nor a 'verified' marker stays
    finally:
        _kill_pod(cfg)


def test_a_model_without_a_pinned_hash_is_never_downloaded(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch)
    cfg["models"] = ["Unpinned-7B.gguf"]
    with pytest.raises(GP.PulseError, match="no pinned sha256"):
        GP.setup(cfg, say=lambda s: None)


def test_the_monitor_reads_gpu_busy_and_tokens_from_the_pod(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch)
    FAKE.STATE["tokens"] = 1234
    util, toks = GP.sample_pod(GP.shell_for(cfg), [fake_server])
    assert util == 95.0 and toks == 1234.0


# ------------------------------------------------------------------------------------------------ the tunnel: off unless switched on
def test_the_tunnel_is_used_only_when_the_pulse_is_switched_on(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    from creator import generator as G
    model = rt / "models" / "Qwen3-14B-Q4_K_M.gguf"                               # NOT on this disk: only a pulse can serve it
    pf = GP.write_tunnel({model.name: [fake_server]}, "P1", path=rt / "gpu" / "tunnel.json", slots={model.name: 8})
    base = dict(DEV.derive(DEV.get()), think_model=model.name, gpu_pulse=False)
    monkeypatch.setattr(DEV, "settings", lambda refresh=False: dict(base))
    assert not DEV.pulse_on(base)
    assert DEV.think_model_path(base) is None                                     # off: a model that is not on disk is not available
    lm = G.LocalModel(model=model, pidfile=tmp_path / "srv" / "llama_server.pid")
    assert lm._pulse_attach() is False and lm.pulse == ""
    monkeypatch.setenv("NUPEN_GPU_PULSE", str(pf))                                # on (as run_jobs sets it for its job processes)
    on = dict(base)
    assert DEV.pulse_on(on) and DEV.think_model_path(on) == model
    assert lm._pulse_attach() is True and lm.port == fake_server and lm.pulse == "P1"
    assert "ANSWER: A" in lm.chat([{"role": "user", "content": "Pick one. ANSWER: <letter>"}], max_tokens=20)
    with pytest.raises(GP.PulseError, match="private"):
        lm.chat([{"role": "user", "content": "Latest owner directives: ..."}])      # the private-marker guard runs before any send
    lm._stop()
    assert lm.pulse == ""                                                         # the pod's server is not ours to stop
    monkeypatch.setenv("NUPEN_PULSE_MODEL", model.name)
    ov = GP.settings_overlay(pf)
    assert ov == {"think_model": model.name, "think_servers": 8}                   # as many thinkers as the pod has slots
    monkeypatch.delenv("NUPEN_GPU_PULSE")
    assert GP.pulse_file(dict(base, gpu_pulse=True)) == rt / "gpu" / "tunnel.json"  # device setting 'gpu_pulse' -> the runtime's tunnel file


def test_a_served_model_that_does_not_answer_is_an_error_not_a_cpu_fallback(rt: Path) -> None:
    pf = GP.write_tunnel({"M.gguf": [_free_port()]}, "P1", path=rt / "gpu" / "tunnel.json")
    with pytest.raises(GP.PulseError, match="do not answer"):
        GP.attach(pf, "M.gguf")
    assert GP.attach(pf, "Other.gguf") is None


# ------------------------------------------------------------------------------------------------ gpu_pulse marker; CPU statistics exclude it
class _Fake:
    def __init__(self, pulse: str = "", model: str = "m.gguf", reply: str = "ANSWER: A") -> None:
        self.pulse, self.model, self.reply = pulse, model, reply
        self.sent: list[str] = []

    def __enter__(self) -> "_Fake":
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def chat(self, messages: Any, **kw: Any) -> str:
        self.sent.append(" ".join(m["content"] for m in messages))
        return self.reply


def _q(i: int, source: str = "repo", t: float = 1000.0, answer: int = 0) -> R.Question:
    return R.Question(qid=f"q{i}", kind="files_changed", source=source, subject=f"c{i}", t=t + i, stem=f"Which file did commit {i} change in parser module?",
                      options=["src/parser.py", "docs/a.md", "setup.py", "tests/x.py"], answer=answer, difficulty="easy", keys=(f"c{i}",))


def test_gpu_pulse_rows_are_marked_and_cpu_seconds_per_item_excludes_them(tmp_path: Path) -> None:
    st = tmp_path / "state"
    R.run_batch(st, [(_q(1), "plain", 0)], lambda: _Fake(pulse="20261004T000000Z"), [])
    R.run_batch(st, [(_q(2), "plain", 0)], lambda: _Fake(), [])
    rows = [json.loads(x) for x in R.path(st).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["gpu_pulse"] == "20261004T000000Z" and "gpu_pulse" not in rows[1]
    rows[0]["seconds"], rows[1]["seconds"] = 0.1, 30.0
    assert R._cpu_per_item(rows) == 30.0                                          # the GPU row's 0.1 s never lowers the CPU's cost
    assert R._cpu_per_item(rows[:1]) is None
    assert J._per_item([r for r in rows if not r.get("gpu_pulse")]) == 30.0
    R.path(st).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = R.report_section(st, tag="m.gguf")
    assert rep["strategies"]["plain"]["seconds_per_item"] == 30.0


# ------------------------------------------------------------------------------------------------ the budget and the monitor stop runs
def _sleeper(monkeypatch: pytest.MonkeyPatch, secs: float = 60.0) -> None:
    real = subprocess.Popen

    def popen(argv: Any, **kw: Any) -> Any:
        return real([sys.executable, "-c", f"import time; time.sleep({secs})"], stdout=kw.get("stdout"), stderr=kw.get("stderr"))
    monkeypatch.setattr(GP.subprocess, "Popen", popen)


def test_the_budget_cap_stops_a_run(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    cfg.update(budget_usd=1.0, usd_per_hr=1.0, reserve_minutes=0.0, monitor_s=0.0, shell_argv=["unused"])
    monkeypatch.setattr(GP, "serve", lambda c, models, sh=None, exe="", say=print: {models[0]: [fake_server]})
    tunnel = lambda plan: GP.write_tunnel(plan, "P", path=rt / "gpu" / "tunnel.json")  # noqa: E731
    _sleeper(monkeypatch)
    now = {"t": 1_000_000.0}
    clock = lambda: now["t"]  # noqa: E731
    calls = {"n": 0}

    def ticking() -> float:                                                       # 30 paid minutes pass at every look at the clock
        calls["n"] += 1
        now["t"] += 1800.0 if calls["n"] > 3 else 0.0
        return now["t"]
    t0 = time.monotonic()
    out = GP.run_jobs(cfg, [f"drills:{FAKE_MODEL}:5", f"drills:{FAKE_MODEL}:5"], tmp_path / "state", ROOT, tmp_path / "owner",
                      tunnel=tunnel, say=lambda s: None, clock=ticking, poll_s=0.05)
    assert time.monotonic() - t0 < 40                                             # the 60 s child was stopped, not waited out
    assert out[0]["stopped"] == "budget" and len(out) == 1                        # ...and the second job never started
    assert GP.budget_for(cfg, clock).spent() >= 1.0
    with pytest.raises(GP.BudgetExceeded):                                       # a new run refuses at once
        GP.budget_for(cfg, clock).check(1.0)
    recorded = [json.loads(x) for x in (tmp_path / "state" / "thinking" / "gpu_pulse_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recorded[0]["stopped"] == "budget" and recorded[0]["gpu_pulse"]


def test_a_job_that_leaves_the_gpu_idle_is_stopped_with_the_reason(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    cfg.update(budget_usd=100.0, monitor_s=0.01, low_util_abort_minutes=1, shell_argv=["unused"])
    monkeypatch.setattr(GP, "serve", lambda c, models, sh=None, exe="", say=print: {models[0]: [fake_server]})
    monkeypatch.setattr(GP, "sample_pod", lambda sh, ports: (12.0, 100.0))       # 12% busy: the PC side is the bottleneck
    _sleeper(monkeypatch)
    now = {"t": 1_000_000.0}

    def clock() -> float:
        now["t"] += 20.0
        return now["t"]
    said: list[str] = []
    out = GP.run_jobs(cfg, [f"judgment:{FAKE_MODEL}:3", f"judgment:{FAKE_MODEL}:3"], tmp_path / "state", ROOT, tmp_path / "owner",
                      tunnel=lambda plan: GP.write_tunnel(plan, "P", path=rt / "gpu" / "tunnel.json"), say=said.append, clock=clock, poll_s=0.02)
    assert [o.get("stopped") for o in out] == ["low_gpu_utilisation"] * 2          # stopped, and the next job still runs
    assert out[0]["monitor"]["flag_low"] and any("LOW: the PC side is the bottleneck" in s for s in said)
    log = (rt / "gpu" / "monitor.jsonl").read_text(encoding="utf-8").splitlines()
    assert log and json.loads(log[0])["util"] == 12.0


def test_monitor_tok_s_low_flag_and_abort() -> None:
    m = GP.Monitor(low_pct=70.0, abort_minutes=3)
    assert m.add(0, 90.0, 0.0)["tok_s"] is None
    assert m.add(60, 95.0, 6000.0)["tok_s"] == 100.0
    s = m.add(120, 40.0, 6600.0)
    assert s["low"] and s["tok_s"] == 10.0 and not m.should_abort(120)
    m.add(180, 30.0, 6700.0)
    assert not m.should_abort(299) and m.should_abort(300)                       # 3 minutes low in a row
    m.add(360, 99.0, 9000.0)
    assert m.low_since is None and not m.should_abort(1000)                      # busy again: the streak is over
    sm = m.summary()
    assert sm["low_minutes"] == 2 and sm["flag_low"] and sm["samples"] == 5


# ------------------------------------------------------------------------------------------------ security: nothing private leaves, nothing secret in the repo
def test_no_pulse_file_is_ever_written_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    inside = ROOT / "state" / "creator" / "gpu_test_runtime"
    monkeypatch.setenv("NUPEN_RUNTIME", str(inside))
    for p in (GP.config_path(), GP.ledger_path(), GP.tunnel_path(), GP.gpu_dir() / "monitor.jsonl", GP.gpu_dir() / "prepared.json"):
        with pytest.raises(GP.PulseError, match="inside the repository"):
            GP._write_json(p, {"host": "1.2.3.4"})
        with pytest.raises(GP.PulseError):
            GP.outside_repo(p)
    with pytest.raises(GP.PulseError):
        GP.load_config()                                                         # a config in the public repo is refused even for reading
    with pytest.raises(GP.PulseError):
        GP._log("monitor.jsonl", {"x": 1})
    with pytest.raises(GP.PulseError):
        GP.ssh_base({"host": "h", "port": 1})                                    # known_hosts would land in the repo
    assert not inside.exists()
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert not [f for f in tracked if Path(f).name in ("pulse.json", "tunnel.json", "known_hosts", "nupen_vast", "nupen_vast.pub", "prepared.json")
                or f.endswith("gpu/ledger.json") or f.endswith("gpu/monitor.jsonl")]


def test_the_upload_list_and_the_outbound_guard_carry_nothing_private(rt: Path) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    m = GP.manifest(cfg)
    assert m["files_uploaded"] == []
    assert all(u.startswith(("https://huggingface.co/", "https://github.com/ggml-org/")) for u in m["downloaded_by_the_pod"])
    assert GP.check_upload([Path("creator/gpupulse.py"), Path("data/public/x.txt")])
    for bad in ("state/livesim/x.json", "C:/Users/p/oldpc/proj/a.py", "C:/Users/p/Masterstock/MASTERSTOCK.md", "C:/Users/p/.ssh/nupen_vast",
                "C:/Users/p/creator_runtime/gpu/pulse.json", "repo/.env"):
        with pytest.raises(GP.PulseError):
            GP.check_upload([Path("creator/ok.py"), Path(bad)])
    for k in GP.PRIVATE_MARKERS:
        with pytest.raises(GP.PulseError):
            GP.outbound_ok([{"role": "user", "content": f"x {k} y"}])
    GP.outbound_ok([{"role": "user", "content": _q(1).prompt()}])
    secrets = json.dumps(GP.DEFAULTS) + Path(GP.__file__).read_text(encoding="utf-8")
    assert "BEGIN OPENSSH PRIVATE KEY\n" not in secrets and '"host": ""' in json.dumps(GP.DEFAULTS)


# ------------------------------------------------------------------------------------------------ slots, plan, smoke, probe, traces, prepare
def test_slots_default_to_8_16_by_model_size_and_vram() -> None:
    got = {m: GP.auto_slots(m) for m in GP.CATALOG}
    assert got == {GP.M17: (16, 8192), GP.M4: (16, 8192), GP.M8: (12, 8192), GP.M14: (8, 8192)}
    assert all(GP.vram_need_gb(m, *got[m]) <= GP.VRAM_GB - 1.0 for m in got)
    assert GP.auto_slots(GP.M14, 8192, vram_gb=16.0) == (8, 4096)               # less VRAM: shorter slots before fewer than 8
    assert GP.served_slots({"slots": 4, "ctx_per_slot": 2048}, [GP.M8]) == {GP.M8: (4, 2048)}
    two = GP.served_slots({"slots": "auto", "ctx_per_slot": 8192}, [GP.M4, GP.M8])
    assert sum(GP.vram_need_gb(m, *two[m]) for m in two) <= GP.VRAM_GB


def test_plan_gives_minutes_and_dollars_per_job(tmp_path: Path) -> None:
    cfg = dict(GP.DEFAULTS)
    jobs = GP.pulse_jobs(cfg, 1)
    assert jobs[0].startswith("smoke:") and {GP.parse_job(j)["model"] for j in jobs} == {GP.M17, GP.M4, GP.M8, GP.M14}
    p = GP.plan(cfg, jobs, tmp_path / "state")
    assert len(p["jobs"]) == len(jobs) + 2 and all(r["minutes"] > 0 for r in p["jobs"])
    assert abs(sum(r["minutes"] for r in p["jobs"]) - p["minutes"]) < 0.2
    assert abs(p["usd"] - p["minutes"] / 60 * 0.343) < 0.01 and p["usd"] < 1.0
    p4 = GP.plan(dict(cfg, best_model=GP.M14), GP.pulse_jobs(dict(cfg, best_model=GP.M14), 4), include_setup=False)
    assert GP.M14 in p4["jobs"][1]["job"] and p4["jobs"][1]["minutes"] == pytest.approx(350.0)   # time-capped (served since the smoke)
    th = tmp_path / "state" / "thinking"
    th.mkdir(parents=True)
    (th / "gpu_probe.jsonl").write_text(json.dumps({"model": GP.M8, "best_tok_s": 2000.0}) + "\n", encoding="utf-8")
    p8 = GP.plan(cfg, [f"drills:{GP.M8}:100"], tmp_path / "state", include_setup=False)
    assert "measured" in p8["jobs"][0]["basis"]
    assert "TOTAL" in GP.format_plan(p8)
    with pytest.raises(GP.PulseError):
        GP.parse_job("mine:bitcoin")


def test_smoke_and_probe_against_a_fake_server(tmp_path: Path, fake_server: int) -> None:
    llm = GP.PodLLM(fake_server, GP.M8, "P1")
    s = GP.smoke(llm)
    assert s["ok"] and s["guard_refuses_private"] and s["answer"] == "42"
    r = GP.probe(llm, 8, tmp_path / "state")
    assert set(r["tok_s_by_concurrency"]) == {"1", "4", "8"} and r["best_tok_s"] > 0 and r["gpu_pulse"] == "P1"
    rows = (tmp_path / "state" / "thinking" / "gpu_probe.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["best_concurrency"] in (1, 4, 8)
    assert GP.measured_tps(tmp_path / "state", GP.M8) == r["best_tok_s"]


def test_trace_bank_keeps_only_correct_outcome_blind_traces_and_feeds_retrieval(tmp_path: Path, fake_server: int) -> None:
    st = tmp_path / "state"
    qs = [_q(1, answer=0), _q(2, answer=1), _q(3, answer=0)]                     # the fake answers A: q1 and q3 right, q2 wrong
    llm = GP.PodLLM(fake_server, GP.M8, "P4")
    stats = GP.traces(llm, st, ROOT, 0, 2, time.monotonic() + 60, questions=lambda: qs)
    assert stats["asked"] == 3 and stats["correct"] == 2 and stats["kept"] == 2
    bank = R.trace_bank(st)
    assert set(bank) == {"q1", "q3"} and all(b["gpu_pulse"] == "P4" and b["model"] == GP.M8 for b in bank.values())
    assert not any("Correct answer" in t for t in FAKE.STATE["requests"])        # the prompts never carried an answer or an example
    again = GP.traces(llm, st, ROOT, 0, 2, time.monotonic() + 60, questions=lambda: qs)
    assert again["asked"] == 1                                                   # banked questions are not asked again
    FAKE.STATE["bare"] = True                                                    # 'ANSWER: A' with no reasoning is not a worked example
    assert GP.traces(llm, st, ROOT, 0, 1, time.monotonic() + 60, questions=lambda: [_q(4)])["kept"] == 0
    assert not R.trace_ok("ANSWER: A") and R.trace_ok("Option A names the parser module.\nANSWER: A")
    # the CPU model's retrieval: a later question of another repository sees the banked trace as a worked example
    new = _q(9, source="other", t=5000.0)
    solved = [(q, q.answer, 1.0) for q in qs if q.qid in bank]
    ex = R.solved_examples(new, solved, 4, time.time())
    msgs = R.build_messages(R.BY_NAME["retrieve4"], new, ex, {k: v["trace"] for k, v in bank.items()})
    assert "Worked reasoning: Option A" in msgs[1]["content"] and "checked against the known answer" in msgs[1]["content"]
    assert "Worked reasoning" not in R.build_messages(R.BY_NAME["retrieve4"], new, ex)[1]["content"]   # no bank: today's prompt unchanged
    cpu = _Fake()
    R.run_batch(st, [(new, "retrieve4", 0)], lambda: cpu, solved, {k: v["trace"] for k, v in bank.items()})
    row = json.loads(R.path(st).read_text(encoding="utf-8").splitlines()[-1])
    assert row["bank_examples"] == 2 and "Worked reasoning" in cpu.sent[0]


def test_reasoning_filler_hands_banked_traces_to_retrieval(tmp_path: Path) -> None:
    st = tmp_path / "state"
    old = _q(1, source="a", t=10.0)
    R.bank_add(st, old, "The subject names the parser.\nANSWER: A", GP.M8, "P4")
    new = [_q(i, source="b", t=1e6) for i in range(2, 60)]
    seen: list[str] = []

    class L(_Fake):
        def chat(self, messages: Any, **kw: Any) -> str:
            seen.append(" ".join(m["content"] for m in messages))
            return "x\nANSWER: A"
    nj = R.reasoning_filler(st, ROOT, max_servers=1, llm_factory=lambda: L(), questions=lambda: [old] + new, tag="m.gguf")
    for _ in range(40):
        job = nj()
        if job is None:
            break
        job()
        if any("Worked reasoning: The subject names the parser." in s for s in seen):
            break
    assert any("Worked reasoning: The subject names the parser." in s for s in seen)


def test_prepare_stages_and_checks_before_renting(tmp_path: Path, rt: Path) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    cfg["key_path"] = str(tmp_path / "nokey")
    qs = [_q(1), _q(2)]
    leak = _q(3)
    leak.stem = "Masterstock says: which file?"
    r = GP.prepare(cfg, GP.pulse_jobs(cfg, 4), tmp_path / "state", ROOT, questions=lambda: qs + [leak])
    assert r["checks"]["reasoning_questions"] == {"total": 3, "not_yet_in_bank": 3, "private_refused": 1}
    assert r["checks"]["models_pinned"] == {GP.M8: True} and r["checks"]["ssh_key"] is False
    assert any("ssh-keygen" in t for t in r["owner_todo"]) and any("host" in t for t in r["owner_todo"])
    assert (rt / "gpu" / "prepared.json").is_file() and r["plan"]["usd"] > 0
    r2 = GP.prepare(dict(cfg, models=[GP.M4]), [f"smoke:{GP.M8}"], tmp_path / "state", ROOT)
    assert not r2["ready"] and r2["missing_from_config"] == [GP.M8]


def test_the_cli_plan_and_checklist_run_without_a_pod(rt: Path) -> None:
    env = dict(os.environ, NUPEN_RUNTIME=str(rt))
    py = [sys.executable, str(ROOT / "scripts" / "gpu_pulse.py")]
    p = subprocess.run(py + ["plan", "--pulse", "1"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "TOTAL" in p.stdout and "probe:Qwen3-14B" in p.stdout, p.stdout + p.stderr
    c = subprocess.run(py + ["checklist"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert "nupen_vast" in c.stdout and "cuda-12.9" in c.stdout and "LLAMA_MODEL" in c.stdout
    i = subprocess.run(py + ["init"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert i.returncode == 0 and (rt / "gpu" / "pulse.json").is_file()


# ------------------------------------------------------------------------------------------------ the interface for other modules' GPU work
def test_external_jobs_run_from_a_job_list_and_bring_outputs_home(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch)
    (tmp_path / "pod").mkdir()
    cfg.update(monitor_s=0.0)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent) + os.pathsep + str(ROOT))
    monkeypatch.setattr(GP, "serve", lambda c, models, sh=None, exe="", say=print: {models[0]: [fake_server]})
    jobs = GP.job_list(cfg, jobs_from="gpupulse_fake_server:gpuday")
    assert GP.parse_job(jobs[1])["spec"] == "ext:x"
    remote = {"name": "train", "remote": "mkdir -p out && echo weights > out/lora.gguf && echo '@@result={\"loss\": 0.5}'", "minutes": 30,
              "free_gpu": True, "outputs": ["out/lora.gguf"]}
    call = {"name": "embed", "call": "gpupulse_fake_server:ext_call", "model": FAKE_MODEL, "minutes": 10}
    out = GP.run_jobs(cfg, [remote, call], tmp_path / "state", ROOT, tmp_path / "owner", say=lambda s: None, poll_s=0.05,
                      tunnel=lambda plan: GP.write_tunnel(plan, "P9", path=rt / "gpu" / "tunnel.json", slots={FAKE_MODEL: 2}))
    assert out[0]["rc"] == 0 and out[0]["result"] == {"loss": 0.5}
    home = Path(out[0]["outputs_home"])
    assert (home / "out" / "lora.gguf").read_text(encoding="utf-8").strip() == "weights" and rt in home.parents
    assert out[1]["rc"] == 0 and out[1]["result"]["models"] == [FAKE_MODEL] and out[1]["result"]["workers"] == GP.inflight(cfg, 2) == 6
    p = GP.plan(cfg, [remote, call], include_setup=False)
    assert [r["minutes"] for r in p["jobs"][:2]] == [30.0, 10.6]
    for bad in ({"name": "a", "minutes": 1}, {"name": "a", "call": "nomodule", "minutes": 1}, {"call": "m:f", "minutes": 1},
                {"name": "a", "call": "m:f", "remote": "x", "minutes": 1}, {"name": "a", "call": "m:f"}):
        with pytest.raises(GP.PulseError):
            GP.parse_job(bad)


def test_setup_steps_are_pluggable_and_idempotent(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch)
    calls: list[str] = []
    cfg["setup_steps"] = [{"name": "trainstack", "script": f"echo installed >> {_posix(tmp_path)}/installs.txt"}]
    monkeypatch.setattr(GP, "SETUP_HOOKS", [lambda c, sh, say: calls.append("hook")])
    sh = GP.shell_for(cfg)
    assert GP.run_setup_steps(cfg, sh, lambda s: None) == {"trainstack": "done"}
    assert GP.run_setup_steps(cfg, sh, lambda s: None) == {"trainstack": "kept"}
    assert (tmp_path / "installs.txt").read_text(encoding="utf-8").count("installed") == 1 and calls == ["hook", "hook"]
    cfg["setup_steps"] = [{"name": "broken", "script": "false"}]
    with pytest.raises(GP.PulseError, match="setup step broken failed"):
        GP.run_setup_steps(cfg, sh, lambda s: None)


def test_endpoints_document_the_tunnel_for_external_scripts(rt: Path) -> None:
    pf = GP.write_tunnel({GP.M8: [18120]}, "P2", path=rt / "gpu" / "tunnel.json", slots={GP.M8: 12}, extra={"tensorboard": 16006})
    e = GP.endpoints(pf)
    assert e == {"pulse": "P2", "models": {GP.M8: {"urls": ["http://127.0.0.1:18120/v1"], "slots": 12}}, "extra": {"tensorboard": "http://127.0.0.1:16006"}}
    assert GP.endpoints(rt / "gpu" / "none.json") == {}


# ------------------------------------------------------------------------------------------------ pod-local models (extra_models, hook H3)
def test_extra_models_are_validated_and_the_registration_file_is_laid_over(tmp_path: Path, rt: Path) -> None:
    sha = "ab" * 32
    cfg = {"extra_models": {"Tuned.gguf": {"bytes": 10, "sha256": sha.upper()}}, "extra_models_file": str(tmp_path / "reg.json")}
    assert GP.extra_models(cfg) == {"Tuned.gguf": {"bytes": 10, "sha256": sha, "pod_local": True}}
    (tmp_path / "reg.json").write_text(json.dumps({"Later.gguf": {"bytes": 5, "sha256": "cd" * 32}}), encoding="utf-8")
    assert set(GP.extra_models(cfg)) == {"Tuned.gguf", "Later.gguf"}                 # read fresh on every call (mid-run registration)
    assert GP.extra_models_file({}) == rt / "gpu" / "extra_models.json"               # default: the runtime dir, never pulse.json
    assert GP.extra_models_file({}, env={"NUPEN_GPU_EXTRA_MODELS": str(tmp_path / "e.json")}) == tmp_path / "e.json"
    assert GP.vram_need_gb("Later.gguf", 1, 1024) > GP.COMPUTE_BUFFER_GB                # its size counts in the VRAM fit
    for bad in ({"x.gguf": {"bytes": 0, "sha256": sha}}, {"x.gguf": {"bytes": 1, "sha256": "nothex"}}, {"../x.gguf": {"bytes": 1, "sha256": sha}},
                {"x.bin": {"bytes": 1, "sha256": sha}}, {"x.gguf": "sha"}):
        with pytest.raises(GP.PulseError):
            GP.extra_models({"extra_models": bad, "extra_models_file": str(tmp_path / "none.json")})


def test_pod_local_models_are_hashed_on_the_pod_before_serving(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _pod_cfg(tmp_path, rt, monkeypatch)
    models = tmp_path / "pod" / "models"
    models.mkdir(parents=True)
    data = b"tuned" * 100
    (models / "Tuned.gguf").write_bytes(data)
    good = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    cfg.update(extra_models={"Tuned.gguf": good}, extra_models_file=str(tmp_path / "none.json"))
    sh = GP.shell_for(cfg)
    assert GP.verify_pod_models(cfg, [FAKE_MODEL, "Tuned.gguf"], sh) == ["Tuned.gguf"]      # catalog models are not re-checked here
    assert (models / "Tuned.gguf.ok").read_text(encoding="utf-8").strip() == good["sha256"]
    cfg["extra_models"] = {"Tuned.gguf": dict(good, sha256="0" * 64)}
    with pytest.raises(GP.PulseError, match="sha256 mismatch"):
        GP.verify_pod_models(cfg, ["Tuned.gguf"], sh)
    assert not (models / "Tuned.gguf.ok").exists() and (models / "Tuned.gguf").is_file()   # marker dropped, the job's output kept
    cfg["extra_models"] = {"Tuned.gguf": dict(good, bytes=len(data) + 1)}
    with pytest.raises(GP.PulseError, match="size mismatch"):
        GP.verify_pod_models(cfg, ["Tuned.gguf"], sh)
    cfg["extra_models"] = {"Gone.gguf": good}
    with pytest.raises(GP.PulseError, match="not on the pod"):
        GP.verify_pod_models(cfg, ["Gone.gguf"], sh)


def test_a_job_whose_model_cannot_be_served_is_skipped_and_the_day_goes_on(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    cfg.update(budget_usd=100.0, monitor_s=0.0, shell_argv=["unused"])

    def serve(c: Any, models: Any, sh: Any = None, exe: str = "", say: Any = print) -> dict[str, list[int]]:
        if models[0] == "Never-made.gguf":
            raise GP.PulseError("pod-local model Never-made.gguf is not on the pod")
        return {models[0]: [fake_server]}
    monkeypatch.setattr(GP, "serve", serve)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent) + os.pathsep + str(ROOT))
    call = {"name": "embed", "call": "gpupulse_fake_server:ext_call", "model": FAKE_MODEL, "minutes": 10}
    out = GP.run_jobs(cfg, ["thinkbench:Never-made.gguf", call], tmp_path / "state", ROOT, tmp_path / "owner", say=lambda s: None, poll_s=0.05,
                      tunnel=lambda plan: GP.write_tunnel(plan, "P7", path=rt / "gpu" / "tunnel.json", slots={FAKE_MODEL: 2}))
    assert "not on the pod" in out[0]["skipped"] and out[1]["rc"] == 0 and out[1]["result"]["models"] == [FAKE_MODEL]
    rows = (tmp_path / "state" / "thinking" / "gpu_pulse_runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and "skipped" in json.loads(rows[0])


def test_prepare_counts_a_model_registered_mid_run_as_pinned(tmp_path: Path, rt: Path) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    jobs = [{"name": "register_ft", "call": "creator.gpuday:register_tuned_job", "args": {"source": "ft", "serve_as": "T.gguf"}, "minutes": 1},
            "thinkbench:T.gguf"]
    r = GP.prepare(cfg, jobs, tmp_path / "state", ROOT)
    assert r["checks"]["models_pinned"] == {"T.gguf": True} and r["checks"]["models_registered_mid_run"] == ["T.gguf"]
    assert r["missing_from_config"] == []
    with pytest.raises(GP.PulseError, match="args must be a dict"):
        GP.parse_job({"name": "x", "call": "m:f", "args": ["no"], "minutes": 1})


def test_server_flags_pass_through_and_change_the_signature() -> None:
    """h51 (3 Oct): extra llama-server flags (e.g. a bigger prefill micro-batch) from config; none = the old command and signature."""
    from creator import gpupulse as GP
    assert GP.server_flags({}) == ()
    assert GP.server_flags({"server_flags": "-ub 2048 -fa on"}) == ("-ub", "2048", "-fa", "on")
    assert GP.server_flags({"server_flags": ["-ctk", "q8_0"]}) == ("-ctk", "q8_0")
    for bad in ("-ub 2048; rm -rf /", "$(id)", "--x=`y`"):
        with pytest.raises(GP.PulseError):
            GP.server_flags({"server_flags": bad})
    plain = GP.server_script("/w", "/opt/llama-server", "M.gguf", 18100, 16, 8192)
    tuned = GP.server_script("/w", "/opt/llama-server", "M.gguf", 18100, 16, 8192, flags=("-ub", "2048"))
    assert "--no-webui > 18100.log" in plain and 'echo "M.gguf|16|8192" > 18100.sig' in plain
    assert "--no-webui -ub 2048 > 18100.log" in tuned and 'echo "M.gguf|16|8192|-ub 2048" > 18100.sig' in tuned


def test_stop_servers_never_stops_the_deadman_switch(tmp_path):
    """Server pid files are <port>.pid; deadman.pid (the pod's budget guard) must survive every server stop / free_gpu job."""
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    run = tmp_path / "run"
    run.mkdir()
    (run / "18120.pid").write_text("999999")
    (run / "18120.sig").write_text("x")
    (run / "deadman.pid").write_text("999998")
    out = subprocess.run([bash, "-c", GP.stop_servers_script(tmp_path.as_posix())], capture_output=True, text=True, timeout=60).stdout
    assert "@@stopped=18120" in out
    assert not (run / "18120.pid").exists()
    assert (run / "deadman.pid").exists() and "deadman" not in out


# ------------------------------------------------------------------------------------------------ requests in flight (h52, 3 Oct 2026)
def test_inflight_default_is_factor_times_slots() -> None:
    assert GP.DEFAULTS["inflight_factor"] == 3
    assert GP.inflight(GP.DEFAULTS, 16) == 48 and GP.inflight({}, 16) == 48            # the measured fix: 48 in flight kept all 16 slots busy
    assert GP.inflight({"inflight_factor": 1}, 16) == 16 and GP.inflight({"inflight_factor": 2.5}, 4) == 10
    assert GP.inflight({"inflight_factor": "x"}, 2) == 6 and GP.inflight({"inflight_factor": 0}, 2) == 2 and GP.inflight({}, 0) == 3


def test_run_jobs_keeps_factor_x_slots_in_flight_and_explicit_workers_win(tmp_path: Path, rt: Path, monkeypatch: pytest.MonkeyPatch,
                                                                           fake_server: int) -> None:
    cfg = GP.load_config(rt / "gpu" / "pulse.json")
    cfg.update(budget_usd=100.0, monitor_s=0.0, shell_argv=["unused"], slots=4)
    monkeypatch.setattr(GP, "serve", lambda c, models, sh=None, exe="", say=print: {models[0]: [fake_server]})
    job = {"name": "noop", "command": [sys.executable, "-c", "print('@@result={}')"], "model": FAKE_MODEL, "minutes": 1}
    said: list[str] = []
    tunnel = lambda plan: GP.write_tunnel(plan, "P", path=rt / "gpu" / "tunnel.json")  # noqa: E731
    out = GP.run_jobs(cfg, [job], tmp_path / "state", ROOT, tmp_path / "owner", tunnel=tunnel, say=said.append, poll_s=0.02)
    assert out[0]["rc"] == 0 and out[0]["workers"] == 12 and any("(12 requests in flight)" in s for s in said)
    cfg["inflight_factor"] = 1
    assert GP.run_jobs(cfg, [job], tmp_path / "state", ROOT, tmp_path / "owner", tunnel=tunnel, say=said.append, poll_s=0.02)[0]["workers"] == 4
    out = GP.run_jobs(cfg, [job], tmp_path / "state", ROOT, tmp_path / "owner", workers=5, tunnel=tunnel, say=said.append, poll_s=0.02)
    assert out[0]["workers"] == 5                                                     # an explicit --workers still wins


def test_traces_start_on_the_first_questions_while_the_rest_are_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_server: int) -> None:
    """h52: the GPU no longer waits for every git history: the cached piece is asked while a cold repository is still being read."""
    st = tmp_path / "state"
    gate = threading.Event()
    asked_before_second: list[int] = []

    def stream(sources: Any, state: Any, cache: Any = None) -> Iterator[list[R.Question]]:
        yield [_q(1, answer=0), _q(2, answer=0)]
        assert gate.wait(30)                                                     # the 'cold repository' finishes only after work started
        asked_before_second.append(len(FAKE.STATE["requests"]))
        yield [_q(3, answer=0)]
    monkeypatch.setattr(R, "generate_stream", stream)
    real = GP.PodLLM.request

    def request(self: Any, *a: Any, **k: Any) -> dict[str, Any]:
        r = real(self, *a, **k)
        gate.set()
        return r
    monkeypatch.setattr(GP.PodLLM, "request", request)
    FAKE.STATE["requests"].clear()
    FAKE.STATE["bare"] = False
    stats = GP.traces(GP.PodLLM(fake_server, GP.M8, "P5"), st, ROOT, 0, 2, time.monotonic() + 60)
    assert stats["asked"] == 3 and stats["kept"] == 3 and asked_before_second and asked_before_second[0] >= 1
    assert set(R.trace_bank(st)) == {"q1", "q2", "q3"} and "fresh_left_partial" not in stats


def test_generate_stream_is_generate_in_pieces(tmp_path: Path) -> None:
    import dataclasses
    repo = tmp_path / "tiny"
    repo.mkdir()
    for i in range(40):
        (repo / f"m{i % 6}.py").write_text(f"x = {i}\n", encoding="utf-8")
        d = f"2020-01-{i % 28 + 1:02d}T{i % 24:02d}:00:00"
        env = dict(os.environ, GIT_AUTHOR_DATE=d, GIT_COMMITTER_DATE=d, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
        if i == 0:
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", f"Tune the parser step {i}"], check=True, capture_output=True, env=env)
    src = [("tiny", repo)]
    st = tmp_path / "state"
    whole = R.generate(src, st, cache=tmp_path / "qc")
    parts = list(R.generate_stream(src, st, cache=tmp_path / "qc"))
    assert parts[0] and len(parts) == 1                                             # warm: everything in the first piece
    dump = lambda qs: sorted(json.dumps(dataclasses.asdict(q), sort_keys=True) for q in qs)  # noqa: E731
    assert dump(parts[0]) == dump(whole) and R.order(parts[0]) == R.order(whole)
    cold = list(R.generate_stream(src, st, cache=tmp_path / "qc2"))
    assert cold[0] == [] and dump([q for p in cold for q in p]) == dump(whole)


# ------------------------------------------------------------------------------------------------ kept-alive connections (h52, 3 Oct 2026)
def test_podllm_keeps_one_connection_per_thread_and_reconnects(fake_server: int) -> None:
    import http.server
    import urllib.error
    conns: list[Any] = []

    class KeepAlive(FAKE.Handler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            conns.append(self.connection)

        def do_POST(self) -> None:
            if "boom" in self.headers.get("X-Test", "") or self.headers.get("Content-Length") == "0":
                self._send(500, b"{}")
                return
            super().do_POST()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), KeepAlive)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = GP.PodLLM(srv.server_address[1], GP.M8, "P6")
        q = [{"role": "user", "content": "What is 17 + 25? Reply with only the number."}]
        for _ in range(5):
            r = llm.request(q, max_tokens=8)
            assert r["text"] == "42" and r["tokens"] == 8 and r["seconds"] >= 0
        assert len(conns) == 1                                                    # five requests, one connection
        conns[0].shutdown(2)                                                      # the server drops the idle connection
        assert llm.chat(q, max_tokens=8) == "42" and len(conns) == 2               # reopened once, same answer
        out: list[str] = []
        ts = [threading.Thread(target=lambda: out.append(llm.chat(q, max_tokens=8))) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert out == ["42"] * 3 and len(conns) == 5                              # one connection per thread
        n = len(FAKE.STATE["requests"])
        with pytest.raises(GP.PulseError):                                        # the private-marker guard still runs before any send
            llm.request([{"role": "user", "content": "Masterstock notes"}])
        assert len(FAKE.STATE["requests"]) == n and len(conns) == 5
        with pytest.raises(urllib.error.HTTPError) as e:                          # an HTTP error is still an HTTPError (as with urlopen)
            llm._post("/v1/chat/completions", b"", 10.0)
        assert e.value.code == 500
        assert llm.chat(q, max_tokens=8) == "42"                                  # and the connection still works afterwards
    finally:
        srv.shutdown()


def test_short_ctx_slots_proposal_doubles_slots_for_short_requests_at_the_same_kv() -> None:
    assert GP.auto_slots(GP.M14, 8192, 32.0) == (16, 8192)                                      # today's default on the 5090
    assert GP.short_ctx_slots(GP.M14, 1400, 260, 32.0) == (32, 4096)                           # traces: ~1.4k prompt + 260 reply cap
    assert GP.vram_need_gb(GP.M14, 32, 4096) == GP.vram_need_gb(GP.M14, 16, 8192)               # the same KV cache
    assert GP.fits([GP.M14], 32, 4096, 32.0) and not GP.fits([GP.M14], 48, 4096, 32.0)
    assert GP.short_ctx_slots(GP.M14, 1400, 260, 24.0) == (16, 4096)                           # a 24 GB card: 16 (today 8 x 8192)
    assert GP.short_ctx_slots(GP.M14, 3500, 400, 32.0) == (16, 8192)                           # long prompts keep 8k
    assert GP.short_ctx_slots(GP.M17, 100, 50, 32.0)[1] == 2048 and GP.short_ctx_slots(GP.M17, 100, 50, 32.0)[0] == 32
