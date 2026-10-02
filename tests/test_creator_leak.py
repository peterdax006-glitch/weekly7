"""A local-model server must never outlive its owner (a llama-server was found running an hour after its parent died).
A tiny python http server stands in for llama-server."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from creator import generator as G

SERVER = textwrap.dedent("""
    import sys, http.server
    mode, port = sys.argv[1], int(sys.argv[2])
    if mode == "die":
        sys.exit(3)
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if mode == "ok" else 503); self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a): pass
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
""")


def fake_model(tmp: Path, mode: str, startup_s: float = 20.0) -> G.LocalModel:
    script = tmp / "fake_server.py"
    script.write_text(SERVER, encoding="utf-8")
    model = tmp / "m.gguf"
    model.write_text("x", encoding="utf-8")

    class Fake(G.LocalModel):
        def _command(self) -> list[str]:
            return [sys.executable, str(script), mode, str(self.port)]

    return Fake(model=model, exe=Path(sys.executable), startup_s=startup_s, pidfile=tmp / "srv.pid")


def gone(pid: int, within: float = 15.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < within:
        if G._pid_image(pid) is None:
            return True
        time.sleep(0.2)
    return False


def test_a_clean_run_stops_the_server_and_removes_the_pidfile(tmp_path: Path) -> None:
    llm = fake_model(tmp_path, "ok")
    with llm:
        pid = llm.proc.pid  # type: ignore[union-attr]
        assert json.loads((tmp_path / "srv.pid").read_text())["pid"] == pid
    assert gone(pid) and not (tmp_path / "srv.pid").exists()


def test_a_server_that_dies_during_startup_leaves_nothing(tmp_path: Path) -> None:
    llm = fake_model(tmp_path, "die")
    with pytest.raises(RuntimeError):
        llm.__enter__()
    assert llm.proc is None and not (tmp_path / "srv.pid").exists()


def test_a_server_that_never_gets_healthy_is_killed(tmp_path: Path) -> None:
    llm = fake_model(tmp_path, "sick", startup_s=1.5)
    pids: list[int] = []
    real = G.LocalModel._stop

    def spy(self: G.LocalModel) -> None:
        if self.proc is not None:
            pids.append(self.proc.pid)
        real(self)

    G.LocalModel._stop = spy  # type: ignore[method-assign]
    try:
        with pytest.raises(TimeoutError):
            llm.__enter__()
    finally:
        G.LocalModel._stop = real  # type: ignore[method-assign]
    assert pids and gone(pids[0])


def test_an_interrupt_during_startup_kills_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = fake_model(tmp_path, "sick")
    seen: list[int] = []

    def boom(self: G.LocalModel) -> None:
        seen.append(self.proc.pid)  # type: ignore[union-attr]
        raise KeyboardInterrupt

    monkeypatch.setattr(G.LocalModel, "_wait_healthy", boom)
    with pytest.raises(KeyboardInterrupt):
        llm.__enter__()
    assert gone(seen[0])


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_a_hard_killed_owner_leaves_no_orphan_server(tmp_path: Path) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(SERVER, encoding="utf-8")
    (tmp_path / "m.gguf").write_text("x", encoding="utf-8")
    owner = tmp_path / "owner.py"
    owner.write_text(textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(G.__file__).resolve().parents[1])!r})
        from creator import generator as G
        class Fake(G.LocalModel):
            def _command(self):
                return [sys.executable, {str(script)!r}, "ok", str(self.port)]
        with Fake(model=Path({str(tmp_path / 'm.gguf')!r}), exe=Path(sys.executable), pidfile=Path({str(tmp_path / 'o.pid')!r})) as m:
            print(m.proc.pid, flush=True)
            time.sleep(300)
    """), encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(owner)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout is not None
        pid = int(p.stdout.readline())
        assert G._pid_image(pid) is not None
        p.kill()                                    # hard kill: no __exit__, no finally
        p.wait(timeout=10)
        assert gone(pid), "the server outlived its killed owner"
    finally:
        if p.poll() is None:
            p.kill()


def test_a_stale_server_whose_owner_is_dead_is_reaped_by_pid_only(tmp_path: Path) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(SERVER, encoding="utf-8")
    port = G.free_port()
    orphan = subprocess.Popen([sys.executable, str(script), "ok", str(port)])
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    bystander = subprocess.Popen([sys.executable, str(script), "ok", str(G.free_port())])
    try:
        time.sleep(0.5)
        pidfile = tmp_path / "srv.pid"
        img = G._pid_image(orphan.pid)
        assert img
        pidfile.write_text(json.dumps({"pid": orphan.pid, "parent": dead.pid}), encoding="utf-8")
        # a live parent: not reaped
        pidfile.write_text(json.dumps({"pid": orphan.pid, "parent": bystander.pid}), encoding="utf-8")
        assert G.reap_stale_server(pidfile, Path(img)) is None and orphan.poll() is None
        # a dead parent but a different image: not reaped
        pidfile.write_text(json.dumps({"pid": orphan.pid, "parent": dead.pid}), encoding="utf-8")
        assert G.reap_stale_server(pidfile, tmp_path / "other.exe") is None and orphan.poll() is None
        # a dead parent and the recorded server image: reaped, and nothing else is touched
        assert G.reap_stale_server(pidfile, Path(img)) == orphan.pid
        assert orphan.wait(timeout=10) is not None and bystander.poll() is None
        assert not pidfile.exists()
    finally:
        for q in (orphan, bystander):
            if q.poll() is None:
                q.kill()


def test_one_local_model_per_machine_lock_waits_times_out_and_survives_a_killed_holder(tmp_path: Path) -> None:
    """2 Oct: three llama servers (~1 GB each) ran at once. MachineLock lets one holder per machine, across processes."""
    import subprocess
    import sys
    import time
    from creator import generator as G
    lock = tmp_path / "llama_server.lock"
    holder = [sys.executable, "-c",
              "import sys, time; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from creator import generator as G; "
              "m = G.MachineLock(Path(sys.argv[2])); m.acquire(); print('held', flush=True); time.sleep(float(sys.argv[3])); m.release()",
              str(Path(__file__).resolve().parents[1]), str(lock)]
    p = subprocess.Popen(holder + ["2.5"], stdout=subprocess.PIPE, text=True)
    assert p.stdout is not None and p.stdout.readline().strip() == "held"
    with pytest.raises(TimeoutError):
        G.MachineLock(lock, wait_s=0.3, poll_s=0.05).acquire()              # held elsewhere: no second server
    t0 = time.monotonic()
    m = G.MachineLock(lock, wait_s=20, poll_s=0.05)
    m.acquire()                                                              # waits for the holder to release
    assert time.monotonic() - t0 > 0.5
    m.release()
    p.wait(10)
    q = subprocess.Popen(holder + ["60"], stdout=subprocess.PIPE, text=True)
    assert q.stdout is not None and q.stdout.readline().strip() == "held"
    q.kill()                                                                 # a hard-killed holder must not leave it locked
    q.wait(10)
    m2 = G.MachineLock(lock, wait_s=5, poll_s=0.05)
    m2.acquire()
    m2.release()
