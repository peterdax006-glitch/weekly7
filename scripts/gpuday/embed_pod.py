"""Build the embedding index ON THE POD (GPU) - and the same code builds/queries it on the PC's CPU.

Pod (an external 'remote' job, after gpuday_upload put gpuday/gpuday_lib.py (= creator/gpuday.py) and gpuday/data/embed_docs.jsonl there):
    python3 gpuday/embed_pod.py build --docs gpuday/data/embed_docs.jsonl --out gpuday/index --model Qwen3-Embedding-0.6B-Q8_0.gguf
PC (CPU, IDLE priority; same GGUF, llama.cpp CPU build):
    python scripts/gpuday/embed_pod.py query --index <home>/index --model <path to the same gguf> "how does the sandbox evaluate a change"
    python scripts/gpuday/embed_pod.py build --docs docs.jsonl --out <dir> --model <gguf> --exe <llama-server>      (the CPU test)
The llama-server is started here with --embedding and the model's pooling (EMBED_MODELS), on 127.0.0.1 only, and always stopped."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1]):                    # pod: gpuday_lib.py next to this file; PC: the repo's creator package
    sys.path.insert(0, str(p))
try:
    import gpuday_lib as GD                          # type: ignore[import-not-found]
except ImportError:
    from creator import gpuday as GD                 # type: ignore[no-redef]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_exe(given: str) -> str:
    if given:
        return given
    for c in (Path("run/exe"),):
        if c.is_file() and Path(c.read_text().strip()).is_file():
            return c.read_text().strip()
    return "llama-server"


def fetch_model(name: str, dest: Path) -> Path:
    """Download a catalogued embedding GGUF from Hugging Face (sha256 checked; a mismatching file is deleted)."""
    e = GD.EMBED_MODELS[name]
    p = dest / name
    if not (p.is_file() and p.stat().st_size == e["bytes"]):
        dest.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{e['repo']}/resolve/main/{name}", str(p))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != e["sha256"]:
        p.unlink()
        raise SystemExit(f"sha256 mismatch for {name}: {h}")
    return p


class Server:
    def __init__(self, exe: str, model: Path, pooling: str, gpu: bool, threads: int = 0) -> None:
        self.port = free_port()
        argv = [exe, "-m", str(model), "--embedding", "--pooling", pooling, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "4"]
        argv += ["-ngl", "99"] if gpu else ["-ngl", "0"] + (["-t", str(threads)] if threads else [])
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://127.0.0.1:{self.port}"
        t0 = time.monotonic()
        while time.monotonic() - t0 < 180:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited at once (rc {self.proc.returncode}): {' '.join(argv)}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        self.stop()
        raise SystemExit("embedding server did not become healthy in 180 s")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def model_path(a: argparse.Namespace) -> Path:
    p = Path(a.model)
    if p.is_file():
        return p
    return fetch_model(p.name, Path(a.models_dir))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "query"):
        p = sub.add_parser(n)
        p.add_argument("--model", required=True, help="GGUF path, or a name of gpuday.EMBED_MODELS to download")
        p.add_argument("--models-dir", default="models")
        p.add_argument("--exe", default="")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--threads", type=int, default=0)
        if n == "build":
            p.add_argument("--docs", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--batch", type=int, default=64)
        else:
            p.add_argument("--index", required=True)
            p.add_argument("-k", type=int, default=5)
            p.add_argument("text")
    a = ap.parse_args(argv)
    mp = model_path(a)
    pooling = GD.EMBED_MODELS.get(mp.name, {}).get("pooling", "mean")
    gpu = not a.cpu and bool(os.environ.get("CUDA_VISIBLE_DEVICES", "x")) and Path("/proc/driver/nvidia").exists()
    srv = Server(find_exe(a.exe), mp, pooling, gpu, a.threads)
    try:
        emb = GD.embed_http(srv.url)
        if a.cmd == "build":
            docs = GD.jsonl_rows(Path(a.docs))
            meta = GD.build_index(docs, emb, Path(a.out), mp.name, batch=a.batch, model_sha256=GD.EMBED_MODELS.get(mp.name, {}).get("sha256", ""),
                                  allow_repo=True)
            meta["gpu"] = gpu
            print("@@result=" + json.dumps(meta), flush=True)
        else:
            for h in GD.query_index(Path(a.index), emb, a.text, k=a.k):
                print(json.dumps(h, ensure_ascii=False))
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
