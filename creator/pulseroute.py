"""Cross-machine overflow, home -> GPU (owner, 3 Oct 2026: "if we use all the RAM on one we swap to the other so then we fully utilize and RAM
shortage doesnt become an issue"). Job-level, not memory pooling: model weights never cross the network, the CALLS do.

While device setting 'pulse_route' is 'auto' and a GPU pulse tunnel (creator.gpupulse, <runtime>/gpu/tunnel.json) is fresh and its forwarded
servers answer /health, the LIVE swarm's model calls (judgment, reasoning drills, thinking: every creator.generator.LocalModel of a model the
pod serves) go to the pod, and the local warm pool of that model drains to 0 (creator.modelpool.tick) - ~10 GB of home RAM for drills and tests.

Unlike a pulse RUNNER job (creator.gpupulse.attach: a dead served model is an error, CPU and GPU timings must not mix), the live swarm FALLS
BACK: a tunnel that is missing, stale, or not answering means local servers, and the pool regrows under its normal RAM gate. Records carry
the pulse id in 'gpu_pulse' (LocalModel.pulse) only for answers that really came from the pod. Every prompt passes gpupulse.outbound_ok()
before it leaves; a prompt that does not is answered on this PC instead.

Health is cached TTL_S seconds per tunnel file (one /health per forwarded port), so the hot path is a stat and a dict lookup. Off by default;
the explicit pulse switch (env NUPEN_GPU_PULSE / setting 'gpu_pulse', runner jobs) takes precedence. Loaded on demand only while switched on."""
from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

KEY = "pulse_route"             # device setting: 'auto' (use a healthy tunnel) | 'off' (default)
MAX_AGE_KEY = "pulse_route_max_age_h"
TTL_S = 5.0                     # a health verdict is reused this long
CALL_TIMEOUT_S = 300.0          # one routed call; longer = the pod is stuck, answer locally
MAX_AGE_H = 72.0                # a tunnel file older than this is stale (a pulse is hours, a rental days at most)


def _healthy(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return bool(r.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


HEALTH: dict[str, Callable[[int], bool]] = {"fn": _healthy}      # injectable: tests fake the probe
CLOCK: dict[str, Callable[[], float]] = {"fn": time.monotonic}
_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()
_RR = {"i": 0}


def enabled(cfg: Optional[Mapping[str, Any]] = None) -> bool:
    """Setting 'pulse_route' == 'auto' and no explicit pulse job (env NUPEN_GPU_PULSE / 'gpu_pulse': the runner's strict mode governs)."""
    c = cfg or {}
    return str(c.get(KEY) or "off").lower() == "auto" and not os.environ.get("NUPEN_GPU_PULSE") and not c.get("gpu_pulse")


def tunnel_file() -> Path:
    from creator import gpupulse as GP
    return GP.tunnel_path()


def invalidate() -> None:
    """Forget the cached verdict (a routed call just failed: the next look probes again)."""
    with _LOCK:
        _CACHE.clear()


def status(cfg: Optional[Mapping[str, Any]] = None, pf: Optional[Path] = None) -> dict[str, Any]:
    """{"up": bool, "pulse": id, "models": {model file name: [healthy local ports]}, "slots": {model: n}, "why": reason when down}.
    Down when switched off, no tunnel file, a stale one, or no forwarded server answers."""
    if not enabled(cfg):
        return {"up": False, "pulse": "", "models": {}, "slots": {}, "why": "off"}
    p = Path(pf) if pf else tunnel_file()
    now = CLOCK["fn"]()
    try:
        st = p.stat()
    except OSError:
        return {"up": False, "pulse": "", "models": {}, "slots": {}, "why": "no tunnel"}
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _LOCK:
        hit = _CACHE.get("v")
        if hit is not None and _CACHE.get("key") == key and now - float(_CACHE.get("t", -1e18)) < TTL_S:
            return dict(hit)
    max_age = float((cfg or {}).get(MAX_AGE_KEY) or MAX_AGE_H)
    from creator import gpupulse as GP
    t = GP._tunnel(p)
    if time.time() - st.st_mtime > max_age * 3600:
        v: dict[str, Any] = {"up": False, "pulse": "", "models": {}, "slots": {}, "why": "stale tunnel file"}
    else:
        good = {str(m): [int(x) for x in ps if HEALTH["fn"](int(x))] for m, ps in (t.get("models") or {}).items()}
        good = {m: ps for m, ps in good.items() if ps}
        v = {"up": bool(good), "pulse": str(t.get("pulse") or "pulse"), "models": good, "slots": dict(t.get("slots") or {}),
             "why": "" if good else "no forwarded server answers"}
    with _LOCK:
        _CACHE.update(key=key, t=now, v=v)
    return dict(v)


def serves(model: Any, cfg: Optional[Mapping[str, Any]] = None) -> bool:
    """Is this model (by file name) answered by a healthy pod server right now? Never raises."""
    try:
        return bool(status(cfg)["models"].get(Path(str(model)).name))
    except Exception:                                                  # noqa: BLE001 - a broken route is no route
        return False


def attach(model: Any, cfg: Optional[Mapping[str, Any]] = None) -> Optional[tuple[int, str]]:
    """(local port, pulse id) of a healthy forwarded server of this model, round robin; None = answer locally. Never raises."""
    try:
        s = status(cfg)
        ports = list(s["models"].get(Path(str(model)).name) or [])
    except Exception:                                                  # noqa: BLE001
        return None
    if not ports:
        return None
    with _LOCK:
        _RR["i"] += 1
        i = _RR["i"]
    return int(ports[i % len(ports)]), str(s["pulse"])


def pod_slots(model: Any, cfg: Optional[Mapping[str, Any]] = None) -> int:
    """Parallel requests the pod takes for this model (slots per server x healthy servers); 0 when not routed."""
    s = status(cfg)
    ports = s["models"].get(Path(str(model)).name) or []
    return int(s["slots"].get(Path(str(model)).name) or 8) * len(ports) if ports else 0


def may_leave(messages: Any) -> bool:
    """False when a prompt must not leave this PC (gpupulse.outbound_ok refuses it)."""
    from creator import gpupulse as GP
    try:
        GP.outbound_ok(messages)
        return True
    except GP.PulseError:
        return False


def settings_overlay(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Device settings while the route is up for the thinking model: as many thinkers at once as the pod takes (never fewer than locally)."""
    from creator import device as DEV
    think = DEV.think_model_path(cfg)
    n = pod_slots(think, cfg) if think is not None else 0
    return {"think_servers": max(int(cfg.get("think_servers") or 0), n)} if n else {}
