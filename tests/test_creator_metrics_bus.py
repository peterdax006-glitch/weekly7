"""creator.slowpath metrics bus (P0.2): one event per step, every local-model call accounted through model_call()."""
from __future__ import annotations

import http.server
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from creator import effladder as EF
from creator import generator as G
from creator import gpuday as GD
from creator import slowpath as SP


class _Fake(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:                           # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        body = json.dumps({"choices": [{"message": {"content": "hi", "reasoning_content": ""}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a: Any) -> None:
        pass


@pytest.fixture()
def server() -> Iterator[int]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture()
def bus(tmp_path: Path) -> Iterator[Path]:
    SP.set_state(tmp_path)
    yield tmp_path
    SP.set_state(None)


def _events(state: Path) -> list[dict[str, Any]]:
    SP.flush()
    out: list[dict[str, Any]] = []
    for f in sorted((state / SP.METRICS_DIR).glob("events-*.jsonl")):
        out += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    return out


def test_every_model_call_path_logs_exactly_one_event(bus: Path, server: int) -> None:
    lm = G.LocalModel(model=Path("qwen3-test.gguf"))
    lm.port = server
    msgs = [{"role": "user", "content": "x"}]
    n = 0
    with SP.step_context(goal_id="g1", step="code", actor="coder"):
        for _ in range(5):
            assert lm.chat(msgs) == "hi"
            n += 1
        ep = EF.Endpoint(server, "qwen3-test.gguf")
        for _ in range(4):
            assert ep.call(msgs, 16, False)["tok_in"] == 11
            n += 1
        call = GD.chat_http(f"http://127.0.0.1:{server}")
        for _ in range(3):
            assert call(msgs, 0.2, 16, 0) == "hi"
            n += 1
    ev = _events(bus)
    assert len(ev) == n == 12                            # 100% of the calls, none doubled
    assert all(e["model"] and e["goal_id"] == "g1" and e["step"] == "code" and e["actor"] == "coder" for e in ev)
    assert all(e["in_tok"] == 11 and e["out_tok"] == 3 and e["outcome"] == "ok" and e["wall_s"] >= 0 and "cpu_s" in e for e in ev)
    assert {e["backend_kind"] for e in ev if "backend_kind" in e} == {"effladder", "gpuday"}
    assert all(e["ram_peak_mb"] is None or e["ram_peak_mb"] > 0 for e in ev)


def test_failed_call_logged_with_outcome(bus: Path) -> None:
    ep = EF.Endpoint(1, "m")                              # nothing listens on port 1
    with pytest.raises(Exception):
        ep.call([{"role": "user", "content": "x"}], 8, False)
    ev = _events(bus)
    assert len(ev) == 1 and ev[0]["outcome"] != "ok" and ev[0]["actor"] == "model:m"


def test_no_state_means_no_write_under_pytest() -> None:
    assert SP.event("x") is False                         # PYTEST_CURRENT_TEST set, no bus configured


def test_daily_rotation_and_report(bus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = time.time()
    monkeypatch.setattr(SP, "_now", lambda: t0 - 86400 * 2)
    SP.event("old")
    monkeypatch.setattr(SP, "_now", lambda: t0)
    SP.event("planner", wall_s=0.5)
    SP.event("planner", wall_s=0.5)
    SP.event("coder", model=True, in_tok=100, out_tok=20, wall_s=2.0, cpu_s=1.0, cache_hit=True)
    SP.event("coder", model=True, in_tok=50, out_tok=5, outcome="TimeoutError")
    SP.flush()
    assert len(list((bus / SP.METRICS_DIR).glob("events-*.jsonl"))) == 2
    rep = SP.metrics_report(bus, days=1, now=t0)
    assert rep["steps"] == 4 and rep["model_steps"] == 2 and rep["qwen_share"] == 0.5
    c = rep["actors"]["coder"]
    assert c["in_tok"] == 150 and c["out_tok"] == 25 and c["cache_hits"] == 1 and c["errors"] == 1 and rep["actors"]["planner"]["steps"] == 2


def test_overhead_per_event(bus: Path) -> None:
    n = 10000
    t0 = time.perf_counter()
    for i in range(n):
        SP.event("bench", goal_id="g", step="s", in_tok=10, out_tok=5, wall_s=0.1, cpu_s=0.05, model=True)
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"METRICS_US_PER_EVENT={us:.1f}")
    assert len(_events(bus)) == n
    assert us < 20                                        # target <= 10 us; slack for a busy PC


def test_kernel_stages_emit_non_model_events(bus: Path) -> None:
    from creator import kernel as K
    st = K._Stages("pkg1")
    with st("sandbox_open"):
        pass
    with pytest.raises(ValueError):
        with st("worker"):
            raise ValueError("x")
    SP.flush()
    ev = _events(bus)
    assert [(e["actor"], e["step"], e["goal_id"], e["outcome"], e["model"]) for e in ev] == [
        ("kernel", "sandbox_open", "pkg1", "ok", False), ("worker", "worker", "pkg1", "ValueError", False)]
    rep = SP.metrics_report(bus)
    assert rep["steps"] == 2 and rep["qwen_share"] == 0.0
