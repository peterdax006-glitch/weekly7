"""Warm model-server pool: grows with free RAM above the floor, shrinks when RAM tightens, hands idle servers to LocalModel (one lessee per
server), never leaks a server. Fakes stand in for llama-server (the real one is measured in scripts/ - see the commit message)."""
from __future__ import annotations

import http.server
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from creator import generator as G
from creator import modelpool as MP


class FakeProc:
    _n = 40000

    def __init__(self) -> None:
        FakeProc._n += 1
        self.pid, self.dead = FakeProc._n, False

    def poll(self) -> Any:
        return 0 if self.dead else None

    def terminate(self) -> None:
        self.dead = True

    kill = terminate

    def wait(self, timeout: float = 0) -> int:
        return 0


class Rig:
    def __init__(self, tmp: Path, free: float, slots: int = 3) -> None:
        self.free = free
        self.procs: list[FakeProc] = []
        (tmp / "exe").write_text("x")
        (tmp / "m.gguf").write_text("x")
        self.base = tmp / "llama_server.pid"
        self.pool = MP.ModelPool(base_pidfile=self.base, model=tmp / "m.gguf", exe=tmp / "exe", ctx=4096, slots=slots, floor_gb=1.0,
                                 server_gb=2.0, headroom_gb=1.0, free_gb=lambda: self.free, spawn=self._spawn, health=lambda p: True)

    def _spawn(self, cmd: list[str]) -> FakeProc:
        p = FakeProc()
        self.procs.append(p)
        return p

    def step(self) -> dict[str, Any]:
        self.pool.last_tick = -1e9
        out = self.pool.tick()
        if self.pool.loading is not None:
            self.pool.loading.join(5)
        return out


@pytest.fixture()
def rig(tmp_path: Path) -> Iterator[Rig]:
    r = Rig(tmp_path, free=20.0)
    yield r
    r.pool.close()


def test_grows_one_at_a_time_while_ram_stays_above_floor(rig: Rig) -> None:
    for _ in range(6):
        rig.step()
    assert rig.pool.status()["servers"] == 3 and len(rig.procs) == 3         # capped by the slots


def test_does_not_grow_when_ram_is_tight(rig: Rig) -> None:
    rig.free = 3.5                      # 3.5 - 2.0 server = 1.5 < floor 1.0 + headroom 1.0
    rig.step()
    assert rig.procs == [] and rig.pool.status()["servers"] == 0
    rig.free = 5.0
    rig.step()
    assert len(rig.procs) == 1


def test_shrinks_idle_servers_when_ram_tightens_but_never_a_leased_one(rig: Rig) -> None:
    for _ in range(3):
        rig.step()
    got = MP.lease(rig.base, rig.pool.model, 4096, health=lambda p: True, alive=lambda p: True)
    assert got is not None
    port, lk, slot = got
    rig.free = 1.2                                                           # under floor + margin
    for _ in range(4):
        rig.step()
    assert rig.pool.status()["servers"] == 1 and slot in rig.pool.servers    # only the leased one survives
    lk.release()
    rig.step()
    assert rig.pool.status()["servers"] == 0


def test_close_leaves_nothing_behind(rig: Rig, tmp_path: Path) -> None:
    for _ in range(3):
        rig.step()
    rig.pool.close()
    assert all(p.dead for p in rig.procs)
    assert list(tmp_path.glob("llama_pool*.json")) == [] and list(tmp_path.glob("llama_server*.pid")) == []
    lk = G.MachineLock(tmp_path / "llama_server.lock")                       # the slot locks are free again
    lk.fh = open(lk.path, "a+b")
    assert lk._try()
    lk.release()


def test_one_lessee_per_server_and_matching_model_only(rig: Rig, tmp_path: Path) -> None:
    rig.step()
    ok = dict(health=lambda p: True, alive=lambda p: True)
    a = MP.lease(rig.base, rig.pool.model, 4096, **ok)
    assert a is not None
    assert MP.lease(rig.base, rig.pool.model, 4096, **ok) is None            # leased: nobody else may attach
    a[1].release()
    assert MP.lease(rig.base, rig.pool.model, 8192, **ok) is None            # another context size is another server
    assert MP.lease(rig.base, tmp_path / "other.gguf", 4096, **ok) is None
    assert MP.lease(rig.base, rig.pool.model, 4096, health=lambda p: False, alive=lambda p: True) is None
    assert MP.lease(rig.base, rig.pool.model, 4096, health=lambda p: True, alive=lambda p: False) is None   # dead owner


def test_a_failed_start_leaves_no_slot_lock_or_pidfile(tmp_path: Path) -> None:
    r = Rig(tmp_path, free=20.0)
    r.pool.health = lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
    r.step()
    assert r.pool.status()["servers"] == 0 and r.procs[0].dead
    assert list(tmp_path.glob("llama_server*.pid")) == []
    r.pool.health = lambda p: True
    r.step()
    assert r.pool.status()["servers"] == 1
    r.pool.close()


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a: Any) -> None:
        pass


def test_local_model_attaches_to_the_warm_server_and_leaves_it_running(tmp_path: Path) -> None:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    exe, model = tmp_path / "exe", tmp_path / "m.gguf"
    exe.write_text("x")
    model.write_text("x")
    me = os.getpid()                                                         # a live pid stands in for the pool process and its server
    (tmp_path / "llama_pool.json").write_text(json.dumps({"slot": 0, "port": port, "pid": me, "owner": me, "model": str(model), "ctx": 8192}))
    m = G.LocalModel(model=model, exe=exe, ctx=8192, pidfile=tmp_path / "llama_server.pid", servers=2)
    try:
        with m as llm:
            assert llm.leased and llm.port == port and llm.proc is None      # no process was started
            assert MP.lease(tmp_path / "llama_server.pid", model, 8192) is None     # a second student cannot take the same server
        assert not m.leased
        again = MP.lease(tmp_path / "llama_server.pid", model, 8192)         # the lease is free again, the server was not touched
        assert again is not None
        again[1].release()
    finally:
        srv.shutdown()


def test_the_hook_does_nothing_for_a_fake_governor(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gov:
        admit = None

    monkeypatch.setattr(MP, "_POOL", None)
    assert MP.tick(Gov(), lambda: 20.0) is None and MP._POOL is None


def test_in_thinking_focus_the_pool_keeps_the_thinking_model(tmp_path: Path) -> None:
    # 3 Oct: the pool held 6 of 7 slots with the fast code model (no code student runs in thinking focus) while judgment queued for one slot
    think = tmp_path / "Thinky-Q4.gguf"
    think.write_bytes(b"x" * 2_000_000)
    cfg = {"think_model": str(think), "think_servers": 4}
    st = tmp_path / "state"
    st.mkdir()
    assert MP.pool_model(st, cfg) == {}                                  # no focus: the fast model, as before
    (st / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    got = MP.pool_model(st, cfg)
    assert got["model"] == think and got["max_servers"] * got.get("share", 1) == 4 and got["server_gb"] > 0    # 4 lessees at once
    assert got["share"] == 4 and got["max_servers"] == 1 and 1 <= got["threads"] <= 8                    # h59: on one shared server
    old = MP.pool_model(st, {**cfg, "think_share": 1})                                                  # the one-per-server pool
    assert old["max_servers"] == 4 and "share" not in old and got["server_gb"] < 4 * old["server_gb"]
    assert MP.pool_model(st, {"think_model": str(think), "think_servers": 0}) == {}         # thinking servers not allowed: fast model
    assert MP.pool_model(st, {"think_model": str(tmp_path / "missing.gguf"), "think_servers": 4}) == {}


def test_a_shared_server_takes_one_lessee_per_parallel_slot_and_shrinks_only_when_all_are_free(tmp_path: Path) -> None:
    r = Rig(tmp_path, free=20.0, slots=1)
    r.pool.share = 3
    r.step()
    st = json.loads((tmp_path / "llama_pool.json").read_text(encoding="utf-8"))
    assert st["share"] == 3
    cmd = r.pool._command(1234)
    assert cmd[cmd.index("-np") + 1] == "3" and cmd[cmd.index("--kv-unified-per-slot") + 1] == "4096" and cmd[cmd.index("-c") + 1] == "12288"
    assert cmd[cmd.index("--cache-ram") + 1] == str(MP.SHARED_CACHE_MIB)
    ok = dict(health=lambda p: True, alive=lambda p: True)
    got = [MP.lease(r.base, r.pool.model, 4096, **ok) for _ in range(3)]
    assert all(g is not None for g in got) and len({g[1].path for g in got if g is not None}) == 3   # three distinct leases, one server
    assert MP.lease(r.base, r.pool.model, 4096, **ok) is None                # all parallel slots taken
    r.free = 1.2                                                              # RAM tight: a server with any lessee is never stopped
    r.step()
    assert r.pool.status()["servers"] == 1
    for g in got[:2]:
        assert g is not None
        g[1].release()
    r.step()
    assert r.pool.status()["servers"] == 1                                    # one lessee left
    assert got[2] is not None
    got[2][1].release()
    r.step()
    assert r.pool.status()["servers"] == 0
    r.pool.close()


def test_an_unshared_server_keeps_the_old_command_and_one_lessee(rig: Rig) -> None:
    cmd = rig.pool._command(1234)
    assert "-np" not in cmd and cmd[cmd.index("-c") + 1] == "4096"
    rig.step()
    ok = dict(health=lambda p: True, alive=lambda p: True)
    first = MP.lease(rig.base, rig.pool.model, 4096, **ok)
    assert first is not None
    assert MP.lease(rig.base, rig.pool.model, 4096, **ok) is None
    first[1].release()
