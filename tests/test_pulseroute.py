"""creator.pulseroute: the live swarm's home -> GPU overflow. Off unless 'pulse_route' is 'auto'; a missing, stale or silent tunnel means local, instantly."""
from __future__ import annotations

import json
import os
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from creator import generator as G
from creator import pulseroute as PR

AUTO = {"pulse_route": "auto"}
MODEL = "Qwen3-8B-Q4_K_M.gguf"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("NUPEN_GPU_PULSE", raising=False)
    PR.invalidate()
    probes: list[int] = []
    up: set[int] = set()
    monkeypatch.setitem(PR.HEALTH, "fn", lambda p: (probes.append(p), p in up)[1])
    clock = {"t": 1000.0}
    monkeypatch.setitem(PR.CLOCK, "fn", lambda: clock["t"])
    yield type("Env", (), {"probes": probes, "up": up, "clock": clock})
    PR.invalidate()


def _tunnel(tmp_path: Path, models: Any = None, age_h: float = 0.0) -> Path:
    p = tmp_path / "tunnel.json"
    p.write_text(json.dumps({"pulse": "pz1", "models": models if models is not None else {MODEL: [18100, 18101]},
                             "slots": {MODEL: 4}, "created": time.time()}), encoding="utf-8")
    if age_h:
        t = time.time() - age_h * 3600
        os.utime(p, (t, t))
    return p


def test_off_by_default_never_looks(tmp_path: Path, _clean: Any) -> None:
    pf = _tunnel(tmp_path)
    _clean.up.update({18100, 18101})
    for cfg in (None, {}, {"pulse_route": "off"}):
        s = PR.status(cfg, pf)
        assert s["up"] is False and s["why"] == "off"
    assert _clean.probes == []
    assert PR.attach(MODEL, {}) is None and not PR.serves(MODEL, {})


def test_explicit_pulse_job_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUPEN_GPU_PULSE", str(tmp_path / "x.json"))
    assert PR.enabled(AUTO) is False
    monkeypatch.delenv("NUPEN_GPU_PULSE")
    assert PR.enabled({**AUTO, "gpu_pulse": True}) is False
    assert PR.enabled(AUTO) is True


def test_auto_healthy_routes_round_robin(tmp_path: Path, _clean: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pf = _tunnel(tmp_path)
    _clean.up.update({18100, 18101})
    s = PR.status(AUTO, pf)
    assert s["up"] and s["pulse"] == "pz1" and s["models"] == {MODEL: [18100, 18101]}
    monkeypatch.setattr(PR, "tunnel_file", lambda: pf)
    got = [PR.attach(MODEL, AUTO) for _ in range(4)]
    assert all(g is not None and g[1] == "pz1" for g in got)
    assert {g[0] for g in got if g} == {18100, 18101}
    assert PR.attach("other.gguf", AUTO) is None                # a model the pod does not serve stays home
    assert PR.pod_slots(MODEL, AUTO) == 8


def test_only_healthy_ports_are_used(tmp_path: Path, _clean: Any) -> None:
    pf = _tunnel(tmp_path)
    _clean.up.add(18101)
    assert PR.status(AUTO, pf)["models"] == {MODEL: [18101]}


def test_stale_tunnel_is_local_without_probing(tmp_path: Path, _clean: Any) -> None:
    pf = _tunnel(tmp_path, age_h=73)
    _clean.up.update({18100, 18101})
    s = PR.status(AUTO, pf)
    assert s["up"] is False and s["why"] == "stale tunnel file" and _clean.probes == []
    fresh = _tunnel(tmp_path, age_h=71)
    PR.invalidate()
    assert PR.status(AUTO, fresh)["up"] is True


def test_missing_or_garbage_tunnel_is_local(tmp_path: Path) -> None:
    assert PR.status(AUTO, tmp_path / "nope.json")["why"] == "no tunnel"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert PR.status(AUTO, bad)["up"] is False


def test_health_fail_is_local(tmp_path: Path) -> None:
    s = PR.status(AUTO, _tunnel(tmp_path))                      # nothing answers
    assert s["up"] is False and s["why"] == "no forwarded server answers"


def test_health_result_is_cached_not_probed_per_call(tmp_path: Path, _clean: Any) -> None:
    pf = _tunnel(tmp_path)
    _clean.up.update({18100, 18101})
    for _ in range(50):
        assert PR.status(AUTO, pf)["up"]
    assert len(_clean.probes) == 2                              # one /health per forwarded port
    _clean.clock["t"] += PR.TTL_S + 1                           # the verdict expires
    PR.status(AUTO, pf)
    assert len(_clean.probes) == 4
    PR.invalidate()
    PR.status(AUTO, pf)
    assert len(_clean.probes) == 6


def test_failed_verdict_is_cached_too(tmp_path: Path, _clean: Any) -> None:
    pf = _tunnel(tmp_path)
    for _ in range(20):
        PR.status(AUTO, pf)
    assert len(_clean.probes) == 2


def test_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pf = _tunnel(tmp_path, models={MODEL: ["x"]})
    monkeypatch.setattr(PR, "tunnel_file", lambda: pf)
    assert PR.serves(MODEL, AUTO) is False
    assert PR.attach(MODEL, AUTO) is None
    assert PR.settings_overlay(AUTO) == {}
    assert PR.may_leave([object()]) is False                    # an odd message is answered here


def test_private_prompt_does_not_leave() -> None:
    from creator import gpupulse as GP
    assert PR.may_leave([{"role": "user", "content": "what is 2+2"}]) is True
    marker = sorted(GP.PRIVATE_MARKERS)[0]
    assert PR.may_leave([{"role": "user", "content": "see " + marker}]) is False


# ---- LocalModel wiring ------------------------------------------------------------------------------------------------------------
def _lm(tmp_path: Path) -> G.LocalModel:
    return G.LocalModel(model=tmp_path / MODEL, pidfile=tmp_path / "srv" / "llama_server.pid")


def test_localmodel_off_does_not_attach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _lm(tmp_path)
    monkeypatch.setattr(G.DEV, "settings", lambda refresh=False: {"pulse_route": "off"})
    monkeypatch.setattr(PR, "attach", lambda *a, **k: pytest.fail("route consulted while off"))
    assert lm._pulse_attach() is False and lm.routed is False


def test_localmodel_auto_attaches_and_marks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _lm(tmp_path)
    monkeypatch.setattr(G.DEV, "settings", lambda refresh=False: dict(AUTO))
    monkeypatch.setattr(PR, "attach", lambda m, c=None: (18100, "pz1"))
    assert lm._pulse_attach() is True
    assert (lm.port, lm.pulse, lm.routed) == (18100, "pz1", True)


def test_localmodel_auto_tunnel_down_is_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _lm(tmp_path)
    monkeypatch.setattr(G.DEV, "settings", lambda refresh=False: dict(AUTO))
    monkeypatch.setattr(PR, "attach", lambda m, c=None: None)
    assert lm._pulse_attach() is False and lm.pulse == "" and lm.routed is False


def _stub_local(lm: G.LocalModel, monkeypatch: pytest.MonkeyPatch) -> None:
    def enter_local() -> G.LocalModel:
        lm.port = 9999
        return lm
    monkeypatch.setattr(lm, "_enter_local", enter_local)
    monkeypatch.setattr(G.DEV, "get", lambda refresh=False: {})
    monkeypatch.setattr(G.DEV, "derive", lambda d, **k: {"think_servers": 1})
    monkeypatch.setattr(G.DEV, "server_threads", lambda *a, **k: 2)


def test_fallback_mid_call_goes_local_and_clears_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _lm(tmp_path)
    lm.port, lm.pulse, lm.routed = 18100, "pz1", True
    seen: list[tuple[int, float]] = []

    def post(messages: Any, mt: int, temp: float, seed: int, timeout: float) -> str:
        seen.append((lm.port, timeout))
        if lm.routed:
            raise urllib.error.URLError("tunnel reset")
        return "local answer"

    monkeypatch.setattr(lm, "_post", post)
    _stub_local(lm, monkeypatch)
    out = lm.chat([{"role": "user", "content": "hi"}], timeout=900.0)
    assert out == "local answer"
    assert seen[0] == (18100, PR.CALL_TIMEOUT_S)                # the pod gets the short leash
    assert seen[1] == (9999, 900.0)                             # then this PC, full timeout
    assert lm.pulse == "" and lm.routed is False                # no pod marker on a local answer


def test_private_prompt_is_answered_locally_without_touching_pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import gpupulse as GP
    lm = _lm(tmp_path)
    lm.port, lm.pulse, lm.routed = 18100, "pz1", True
    ports: list[int] = []
    monkeypatch.setattr(lm, "_post", lambda *a: (ports.append(lm.port), "ok")[1])
    _stub_local(lm, monkeypatch)
    assert lm.chat([{"role": "user", "content": sorted(GP.PRIVATE_MARKERS)[0]}]) == "ok"
    assert ports == [9999]


def test_route_is_off_in_derived_defaults() -> None:
    from creator import device as DEV
    assert DEV.derive(DEV.get(), overrides={})["pulse_route"] == "off"


def _backfill(tmp_path: Path, model: str = MODEL, port: int = 18300, slots: int = 8, age_h: float = 0.0) -> Path:
    p = tmp_path / "tunnel.backfill.json"
    p.write_text(json.dumps({"pulse": "bf1", "models": {model: [port]}, "slots": {model: slots}, "created": time.time()}), encoding="utf-8")
    if age_h:
        t = time.time() - age_h * 3600
        os.utime(p, (t, t))
    return p


def test_backfill_tunnel_merges_with_runner_tunnel(tmp_path: Path, _clean: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pf = _tunnel(tmp_path, models={"Qwen3-14B-Q4_K_M.gguf": [18130]})
    _backfill(tmp_path)
    _clean.up.update({18130, 18300})
    s = PR.status(AUTO, pf)
    assert s["up"] and s["models"] == {"Qwen3-14B-Q4_K_M.gguf": [18130], MODEL: [18300]}
    monkeypatch.setattr(PR, "tunnel_file", lambda: pf)
    assert PR.attach(MODEL, AUTO) == (18300, "bf1")             # the answer carries the backfill's own pulse id
    assert PR.attach("Qwen3-14B-Q4_K_M.gguf", AUTO) == (18130, "pz1")
    assert PR.pod_slots(MODEL, AUTO) == 8


def test_backfill_alone_routes_when_runner_tunnel_is_gone(tmp_path: Path, _clean: Any) -> None:
    _backfill(tmp_path)
    _clean.up.add(18300)
    s = PR.status(AUTO, tmp_path / "tunnel.json")
    assert s["up"] and s["models"] == {MODEL: [18300]} and s["pulse"] == "bf1"


def test_same_model_in_both_files_sums_capacity(tmp_path: Path, _clean: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pf = _tunnel(tmp_path)                                      # MODEL on 18100, 18101 at 4 slots each
    _backfill(tmp_path, slots=8)
    _clean.up.update({18100, 18101, 18300})
    monkeypatch.setattr(PR, "tunnel_file", lambda: pf)
    assert PR.status(AUTO, pf)["models"][MODEL] == [18100, 18101, 18300]
    assert PR.pod_slots(MODEL, AUTO) == 16


def test_stale_or_silent_backfill_is_ignored(tmp_path: Path, _clean: Any) -> None:
    pf = _tunnel(tmp_path, models={"Qwen3-14B-Q4_K_M.gguf": [18130]})
    _backfill(tmp_path, age_h=PR.MAX_AGE_H + 1)
    _clean.up.update({18130, 18300})
    assert MODEL not in PR.status(AUTO, pf)["models"]
    PR.invalidate()
    _backfill(tmp_path)
    _clean.up.discard(18300)
    assert MODEL not in PR.status(AUTO, pf)["models"]
    assert PR.status(AUTO, pf)["up"]                            # the runner's server still routes
