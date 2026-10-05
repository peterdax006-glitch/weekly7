"""The improvement engine is live but GATED (blueprint 7, 9.1): the swarm round starts an engine cycle only with state/NUPEN_STOP absent AND
state/ENGINE_ON present; a class at A0 only proposes (diff + benchmark + verdict artifacts, nothing merged). Producers (every live model-call
path) write events carrying cls / sig / form_in / form_out; the writers feed metrics/skills.jsonl and metrics/shadow.jsonl."""
from __future__ import annotations

import http.server
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

import test_creator_engineloop as H                      # the dry-run fixtures (temp git repo, seeded metrics, scripted team)
from creator import adopt as A
from creator import constraints as CON
from creator import effladder as EF
from creator import engineloop as EL
from creator import generator as G
from creator import gpuday as GD
from creator import kernel as K
from creator import slowpath as SP
from creator import swarm as W
from creator import talk as T
from creator import team as TM


class _Stop(Exception):
    """Ends the round right after the engine hook (the hook sits before the kernel's prepare)."""


def _run_round(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise _Stop()
    monkeypatch.setattr(K, "prepare", boom)
    with pytest.raises(_Stop):
        W.run_round(cfg, lambda: None, None, max_packages=0, poll_s=0.05)


def _setup(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, K.KernelConfig, list[int]]:
    state, cfg = H.setup(tmp, monkeypatch)
    calls: list[int] = []
    real = EL.engine_step

    def step(st: Path, c: Any, team: Any, by: str, **kw: Any) -> Any:
        calls.append(1)
        return real(st, c, team, by, now=H.NOW, bench=H.bench, warm=False, **{k: v for k, v in kw.items() if k not in ("now", "bench", "warm")})
    monkeypatch.setattr(EL, "engine_step", step)
    monkeypatch.setattr(EL, "slot_team", lambda st: H.make_team(tmp))
    return state, cfg, calls


def _wait(state: Path, key: str = "outcome", timeout: float = 180.0) -> dict[str, Any]:
    f = state / EL.LAST_STEP
    end = time.time() + timeout
    while time.time() < end:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if key in d or "error" in d or "skipped" in d:
                return d
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise AssertionError("engine cycle did not finish")


def test_nupen_stop_means_no_engine_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg, calls = _setup(tmp_path, monkeypatch)
    (state / EL.ENGINE_ON).write_text("on", encoding="utf-8")
    (state / EL.STOP_FILE).write_text("stop", encoding="utf-8")
    _run_round(cfg, monkeypatch)
    assert EL.maybe_step(state, cfg, spawn=False) == f"{EL.STOP_FILE} exists"
    assert not calls and not (state / EL.LAST_STEP).exists() and not CON.in_flight(state, H.NOW) and not (state / "adopt").exists()


def test_without_engine_on_no_engine_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg, calls = _setup(tmp_path, monkeypatch)
    _run_round(cfg, monkeypatch)
    assert EL.maybe_step(state, cfg, spawn=False).startswith("no ENGINE_ON")
    assert not calls and not (state / EL.LAST_STEP).exists() and not CON.in_flight(state, H.NOW) and not (state / "adopt").exists()


def test_engine_on_at_a0_proposes_and_benchmarks_but_merges_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg, calls = _setup(tmp_path, monkeypatch)
    (state / EL.ENGINE_ON).write_text("on", encoding="utf-8")
    assert A.level(state, "tool") == 0                                           # nothing was granted: A0, the strict default
    _run_round(cfg, monkeypatch)                                                 # the swarm's own entry point starts the (idle-priority) cycle
    last = _wait(state)
    assert last.get("outcome") == "PROPOSED", last
    assert len(calls) == 1
    assert EL.maybe_step(state, cfg, spawn=False).startswith("cadence")           # once per interval
    dec = CON._jsonl(state / "engine" / "decisions.jsonl")
    assert dec and dec[-1].get("key") == "hot_path:indexer"                      # a decision
    art = state / "adopt" / "hot_path_indexer"
    assert (art / "candidate.diff").is_file() and (art / "benchmark.json").is_file() and (art / "verdict.json").is_file()
    bench = json.loads((art / "benchmark.json").read_text(encoding="utf-8"))
    assert bench["measure"] and len(bench["before"]) == 5
    assert "sorted(xs)[0]" in (cfg.repo / "creator" / "fastfoo.py").read_text(encoding="utf-8")      # nothing merged
    assert A.history(state)[-1]["outcome"] == "PROPOSED" and A.level(state, "tool") == 0
    assert not EL._WIP.locked()                                                  # WIP 1 released


def test_only_one_cycle_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg, calls = _setup(tmp_path, monkeypatch)
    (state / EL.ENGINE_ON).write_text("on", encoding="utf-8")
    assert EL._WIP.acquire(blocking=False)
    try:
        assert EL.maybe_step(state, cfg, spawn=False).startswith("WIP 1")
    finally:
        EL._WIP.release()
    assert not calls


# ------------------------------------------------------------------------------------------------ producers (fake model server)
class _Fake(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:                           # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({"choices": [{"message": {"content": "0.5", "reasoning_content": ""}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 40, "completion_tokens": 3}}).encode()
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
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def _events(state: Path) -> list[dict[str, Any]]:
    SP.flush()
    out: list[dict[str, Any]] = []
    for f in sorted((state / SP.METRICS_DIR).glob("events-*.jsonl")):
        out += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    return out


def test_live_producers_write_cls_sig_and_form_fields(tmp_path: Path, server: int) -> None:
    SP.set_state(tmp_path)
    try:
        msgs = [{"role": "system", "content": "You are a judge. " * 20}, {"role": "user", "content": "Is it 7 or 8?"}]
        lm = G.LocalModel(model=Path("qwen3-test.gguf"))
        lm.port = server
        lm.chat(msgs)                                                              # LocalModel._post, no class given -> "chat"
        with SP.step_context(cls="judgment"):                                      # judgment.py is protected: the label comes from the caller context
            lm.chat(msgs)
        EF.Endpoint(server, "qwen3-test.gguf").call(msgs, 16, False)               # effladder
        GD.chat_http(f"http://127.0.0.1:{server}")(msgs, 0.2, 16, 0)               # gpuday
        v = T.Voice.__new__(T.Voice)                                               # talk's direct path (no model start)
        v.timeout_s = 30.0
        v._post(server, msgs, 16, 0.2)
        team = H.make_team(tmp_path, "x")
        team.log = TM.slowpath_log()
        team.set_goal("g1")
        team.dispatch(TM.Envelope("g1", "SPEC", inputs=[team.board.put("g1", "task", "{}")]))        # team dispatch
        ev = [e for e in _events(tmp_path) if e.get("model")]
    finally:
        SP.set_state(None)
    assert len(ev) == 6
    assert {e["cls"] for e in ev} == {"chat", "judgment", "effladder", "gpuday", "talk", "SPEC"}
    assert all(len(e["sig"]) == 12 and e["form_in"] > 0 and e["form_out"] >= 0 for e in ev)
    chat = [e for e in ev if e["cls"] == "chat"][0]
    assert chat["form_in"] < chat["in_tok"] and chat["form_out"] == chat["out_tok"] == 3       # the system boilerplate is the waste


def test_identical_prompts_share_one_signature() -> None:
    a = SP.model_call("m", **SP.chat_fields([{"role": "user", "content": "Locate file 12"}]))
    b = SP.model_call("m", **SP.chat_fields([{"role": "user", "content": "Locate file 99"}]))
    assert a.extra["sig"] == b.extra["sig"]


# ------------------------------------------------------------------------------------------------ writers
def test_pipeline_eval_compare_writes_skills_and_shadow_policy_writes_metrics(tmp_path: Path) -> None:
    from creator import pipelinemix as PM
    from creator import shadow as SH
    cmp = {"roles": {"LOCATE": {"n": 40, "tuned_acc": 0.9, "base_acc": 0.7}, "CODE": {"n": 40, "tuned_acc": None, "base_acc": 0.5}}}
    assert PM.export_skills(tmp_path, cmp) == 2
    rows = [json.loads(x) for x in (tmp_path / "metrics" / "skills.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["role"]: r["score"] for r in rows} == {"LOCATE": 0.9, "CODE": 0.5}
    st = {"n": 40, "default_acc": 0.6, "chooser_acc": 0.7}
    import creator.shadow as shadow_mod
    orig = shadow_mod.stats
    shadow_mod.stats = lambda state, lessons: st                                   # type: ignore[assignment]
    try:
        assert SH.export_metrics(tmp_path, [], None, None) == 2
        assert SH.export_metrics(tmp_path, [], None, None) == 0                    # unchanged -> not appended again
    finally:
        shadow_mod.stats = orig                                                    # type: ignore[assignment]
    sh = [json.loads(x) for x in (tmp_path / "metrics" / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sh) == 2 and all(r["cost"] is None for r in sh)
    assert CON.detect_shadow(tmp_path, [{"model": True, "cls": "candidate_pick"}], 1.0) == []   # an unmeasured cost claims nothing
