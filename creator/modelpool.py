"""Warm model-server pool (owner, 2 Oct 2026: "as long as my computer is on Nupen should be maxing my computer out"; CPU is full while RAM
sits half empty). A student's LocalModel used to pay the llama-server start-up (process, weights from disk, warm-up: seconds to tens of
seconds) on every attempt. While RAM has room the swarm keeps N servers LOADED AND IDLE; LocalModel attaches to one (a lease) and the
server stays loaded when the student is done.

INVARIANTS (the one-server-per-slot rule of creator.generator still holds):
* A pooled server occupies a normal slot: the pool holds that slot's MachineLock and writes the slot's pidfile. LocalModel never starts a
  second server in an occupied slot; it leases the pooled one instead (a per-slot lease lock: one student per server).
* Nothing leaks: every pooled server belongs to the pool process's kill-on-close Job Object (a hard kill of the swarm takes them down),
  `close()` runs at exit (atexit), when the swarm round has nothing to do, and a later pool or LocalModel reaps a server whose recorded
  owner is dead (reap_stale_server). NUPEN_STOP ends the swarm process, which ends the pool.
* The RAM floor is honoured: a server starts only while free RAM - claimed - SERVER_GB stays above floor + headroom; when free RAM falls
  under floor + SHRINK_MARGIN idle servers are stopped (never a leased one). Never grows while a server is still loading.
Loaded on demand through creator.registry ('modelpool')."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from creator import device as DEV
from creator import generator as G

HEADROOM_GB = 3.0               # free RAM kept beyond the floor and the new server: sandboxes and tests need room too
SHRINK_MARGIN_GB = 0.5          # idle servers are stopped when free RAM falls under floor + this
TICK_S = 5.0
STATE_GLOB = "llama_pool*.json"


def _sfx(i: int) -> str:
    return "" if i == 0 else f".{i}"


def slot_paths(base_pidfile: Path, i: int) -> tuple[Path, Path, Path, Path]:
    """(slot lock, slot pidfile, pool state file, lease lock) of slot i - slot 0/lock/pid names are generator's own."""
    d = base_pidfile.parent
    return (d / f"llama_server{_sfx(i)}.lock", d / (base_pidfile.stem + _sfx(i) + base_pidfile.suffix),
            d / f"llama_pool{_sfx(i)}.json", d / f"llama_pool{_sfx(i)}.lease")


def _alive(pid: int) -> bool:
    return pid > 0 and G._pid_image(pid) is not None


def _healthy(port: int, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return bool(r.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def lease(base_pidfile: Path, model: Path, ctx: int, health: Callable[[int], bool] = _healthy,
          alive: Callable[[int], bool] = _alive) -> Optional[tuple[int, Any, int]]:
    """Attach to an idle warm server of this machine (any process): (port, held lease lock, slot) or None. The caller releases the
    lock when done; the server itself stays loaded."""
    try:
        states = sorted(base_pidfile.parent.glob(STATE_GLOB))
    except OSError:
        return None
    for sf in states:
        try:
            st = json.loads(sf.read_text(encoding="utf-8"))
            slot, port, owner, pid = int(st["slot"]), int(st["port"]), int(st["owner"]), int(st["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if os.path.normcase(str(st.get("model", ""))) != os.path.normcase(str(model)) or int(st.get("ctx", 0)) != int(ctx):
            continue
        if not (alive(owner) and alive(pid)):
            continue
        lk = G.MachineLock(slot_paths(base_pidfile, slot)[3])
        try:
            lk.path.parent.mkdir(parents=True, exist_ok=True)
            lk.fh = open(lk.path, "a+b")
        except OSError:
            continue
        if not lk._try():
            lk.fh.close()
            lk.fh = None
            continue
        if health(port):
            return port, lk, slot
        lk.release()
    return None


class ModelPool:
    """Owns the warm servers of THIS process (see the module doc). `tick` is cheap and never blocks: a start runs in a thread."""

    def __init__(self, base_pidfile: Path = G.PIDFILE, model: Path = G.DEFAULT_MODEL, exe: Path = G.SERVER_EXE, ctx: int = 8192,
                 slots: Optional[int] = None, floor_gb: Optional[float] = None, server_gb: float = DEV.SERVER_GB,
                 headroom_gb: float = HEADROOM_GB, max_servers: Optional[int] = None,
                 free_gb: Callable[[], Optional[float]] = G._free_ram_gb,
                 spawn: Optional[Callable[[list[str]], Any]] = None, health: Callable[[int], bool] = _healthy,
                 clock: Callable[[], float] = time.monotonic) -> None:
        cfg = DEV.settings()
        self.base, self.model, self.exe, self.ctx = Path(base_pidfile), Path(model), Path(exe), ctx
        self.slots = max(1, int(cfg.get("llama_servers", 1)) if slots is None else slots)
        self.max_servers = self.slots if max_servers is None else min(self.slots, max_servers)
        # threads for the servers that may REALLY run at once (a 2-server thinking pool of 7 slots: 4 each, not 3)
        self.threads = DEV.server_threads(cfg, self.max_servers)
        self.gpu_layers = int(cfg["gpu_layers"])
        self.floor_gb = floor_gb if floor_gb is not None else self._floor()
        self.server_gb, self.headroom_gb = server_gb, headroom_gb
        self.free_gb, self.health, self.clock = free_gb, health, clock
        self.spawn = spawn or (lambda cmd: subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.job: Any = None
        self.servers: dict[int, dict[str, Any]] = {}            # slot -> {proc, port, lock, lease, ready_at}
        self.loading: Optional[threading.Thread] = None
        self.guard = threading.Lock()
        self.last_tick = -1e9
        self.closed = False
        self.log: list[dict[str, Any]] = []
        atexit.register(self.close)

    @staticmethod
    def _floor() -> float:
        from creator import testslots
        return float(testslots.reserve_gb(DEV.get().ram_gb))

    # ------------------------------------------------------------------ policy

    def can_grow(self) -> bool:
        free = self.free_gb()
        if free is None or self.closed or len(self.servers) >= self.max_servers or self.loading is not None:
            return False
        return free - self.server_gb > self.floor_gb + self.headroom_gb

    def must_shrink(self) -> bool:
        free = self.free_gb()
        return free is not None and free < self.floor_gb + SHRINK_MARGIN_GB

    def tick(self, allowed: bool = True) -> dict[str, Any]:
        """One step: drop dead servers, shrink when RAM tightens, otherwise start one more while RAM has room. `allowed` False (the
        governor says tight, or a stop) only ever shrinks. Rate-limited to TICK_S."""
        now = self.clock()
        if self.closed or now - self.last_tick < TICK_S:
            return self.status()
        self.last_tick = now
        with self.guard:
            for slot in [s for s, v in self.servers.items() if v["proc"].poll() is not None]:
                self._drop(slot)
            if self.loading is not None and not self.loading.is_alive():
                self.loading = None
            if self.must_shrink() or not allowed:
                if self.must_shrink():
                    self._stop_one_idle()
            elif self.can_grow() and self.exe.is_file() and self.model.is_file():
                self.loading = threading.Thread(target=self._start_one, name="model-pool-start", daemon=True)
                self.loading.start()
        return self.status()

    def status(self) -> dict[str, Any]:
        return {"servers": len(self.servers), "loading": self.loading is not None, "ports": sorted(v["port"] for v in self.servers.values())}

    # ------------------------------------------------------------------ servers

    def _command(self, port: int) -> list[str]:
        return [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port), "-c", str(self.ctx),
                "-t", str(self.threads), "--log-disable"] + (["-ngl", str(self.gpu_layers)] if self.gpu_layers > 0 else [])

    def _take_slot(self) -> Optional[tuple[int, Any]]:
        for i in range(self.slots):
            lock_p, _, _, _ = slot_paths(self.base, i)
            lk = G.MachineLock(lock_p)
            try:
                lock_p.parent.mkdir(parents=True, exist_ok=True)
                lk.fh = open(lock_p, "a+b")
            except OSError:
                continue
            if lk._try():
                return i, lk
            lk.fh.close()
        return None

    def _start_one(self) -> None:
        got = self._take_slot()
        if got is None:
            return
        slot, lock = got
        _, pidfile, state, _ = slot_paths(self.base, slot)
        marker = pidfile.with_suffix(".starting")
        proc: Any = None
        t0 = self.clock()
        try:
            try:
                G.reap_stale_server(pidfile, self.exe)
            except OSError:
                pass
            port = G.free_port()
            marker.write_text(str(time.time()), encoding="utf-8")
            if self.job is None:
                self.job = G._KillOnCloseJob()
            proc = self.spawn(self._command(port))
            if hasattr(proc, "_handle"):
                self.job.adopt(proc)
            pidfile.write_text(json.dumps({"pid": proc.pid, "parent": os.getpid()}), encoding="utf-8")
            last = self.clock()
            limit = DEV.call_timeout_s(self.model, 240.0)              # a bigger model loads longer
            while not self.closed and self.clock() - t0 < limit:
                if self.clock() - last > 2.0:
                    os.utime(marker)
                    last = self.clock()
                if self.health(port):
                    break
                if proc.poll() is not None:
                    raise RuntimeError("pooled server exited during start-up")
                time.sleep(0.2)
            else:
                raise TimeoutError("pooled server did not become healthy")
            state.write_text(json.dumps({"slot": slot, "port": port, "pid": proc.pid, "owner": os.getpid(), "model": str(self.model),
                                         "ctx": self.ctx}), encoding="utf-8")
            with self.guard:
                self.servers[slot] = {"proc": proc, "port": port, "lock": lock, "ready_s": self.clock() - t0}
            self.log.append({"event": "ready", "slot": slot, "port": port, "pid": proc.pid, "seconds": round(self.clock() - t0, 2)})
            lock = None
        except Exception as e:                                       # noqa: BLE001 - a failed start leaves nothing behind
            self.log.append({"event": "failed", "error": f"{type(e).__name__}: {e}"[:200]})
            if proc is not None:
                self._kill(proc)
            pidfile.unlink(missing_ok=True)
        finally:
            marker.unlink(missing_ok=True)
            if lock is not None:
                lock.release()

    @staticmethod
    def _kill(proc: Any) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        except Exception:                                            # noqa: BLE001
            pass

    def _drop(self, slot: int) -> None:
        v = self.servers.pop(slot, None)
        if v is None:
            return
        _, pidfile, state, _ = slot_paths(self.base, slot)
        state.unlink(missing_ok=True)                                # first: nobody may lease it any more
        self._kill(v["proc"])
        pidfile.unlink(missing_ok=True)
        v["lock"].release()

    def _stop_one_idle(self) -> bool:
        """Stop one server nobody has leased (the lease lock is ours to take only when it is idle)."""
        for slot in sorted(self.servers, reverse=True):
            lk = G.MachineLock(slot_paths(self.base, slot)[3])
            try:
                lk.fh = open(lk.path, "a+b")
            except OSError:
                continue
            if lk._try():
                try:
                    self._drop(slot)
                    self.log.append({"event": "shrunk", "slot": slot})
                finally:
                    lk.release()
                return True
            lk.fh.close()
        return False

    def close(self) -> None:
        """Stop every pooled server (leased ones too: the swarm is ending)."""
        self.closed = True
        t = self.loading
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        with self.guard:
            for slot in list(self.servers):
                self._drop(slot)
        if self.job is not None:
            self.job.close()
            self.job = None


_POOL: Optional[ModelPool] = None


def pool_model(state: Optional[Path] = None, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """What the warm servers load. 3 Oct 2026: in THINKING focus no code student runs, yet the pool kept 6 of 7 slots loaded with the fast code
    model while every judgment batch queued for the one slot left (and cold-started the thinking model each time). In thinking focus the pool
    keeps the THINKING model (generator.thinker() leases it: same model and context) and at most 'think_servers' of them, so the other slots
    stay free for a fast-model student. Otherwise (or with no thinking model on disk) the fast model, as before."""
    try:
        from creator import focus as F
        c = DEV.settings() if cfg is None else cfg
        think = DEV.think_model_path(c)
        if think is not None and int(c.get("think_servers", 0)) > 0 and F.current(state or F.DEFAULT_STATE) == "thinking":
            return {"model": think, "max_servers": int(c["think_servers"]), "server_gb": DEV.server_gb_for(think)}
    except Exception:                                                  # noqa: BLE001 - an unreadable focus keeps the old pool
        pass
    return {}


def tick(gov: Any, free_ram: Callable[[], float]) -> Optional[dict[str, Any]]:
    """run_round hook: a REAL-machine governor (admit set) only. The pool is one per process and ends with it."""
    global _POOL
    if gov.admit is None:
        return None
    if _POOL is None or _POOL.closed:
        _POOL = ModelPool(free_gb=lambda: float(free_ram()), **pool_model())
    return _POOL.tick(allowed=not gov.too_tight())


def close_if_idle() -> None:
    """Called when a round ended with nothing to do: give the RAM back until work returns."""
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None
