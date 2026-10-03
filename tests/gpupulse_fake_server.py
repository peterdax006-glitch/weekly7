"""A stand-in for llama-server in the GPU pulse tests (no model, no GPU): /health, /metrics, /v1/chat/completions.

Run as a script it behaves like the binary on the pod (`--port N`, `--list-devices`, `--version`); imported, `start()` serves on a free
loopback port in a thread. Replies: '42' to the smoke question; for a multiple-choice question 'Option ... ANSWER: <letter>' where the letter is
FAKE_ANSWER (env / attribute, default A) - so tests decide which traces come out correct."""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
from typing import Any

STATE: dict[str, Any] = {"tokens": 0, "answer": os.environ.get("FAKE_ANSWER", "A"), "bare": False, "requests": []}
LOCK = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        return None

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, b'{"status":"ok"}')
        elif self.path == "/metrics":
            self._send(200, f"# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total {STATE['tokens']}\n".encode(), "text/plain")
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        d = json.loads(self.rfile.read(n) or b"{}")
        text = " ".join(str(m.get("content", "")) for m in d.get("messages", []))
        with LOCK:
            STATE["requests"].append(text)
        if "17 + 25" in text:
            out = "42"
        elif "ANSWER" in text:
            out = f"ANSWER: {STATE['answer']}" if STATE["bare"] else f"Option {STATE['answer']} matches the files the subject names.\nANSWER: {STATE['answer']}"
        else:
            out = "1. Rivers flow downhill."
        toks = min(int(d.get("max_tokens") or 50), 50)
        with LOCK:
            STATE["tokens"] += toks
        self._send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": out}}],
                                    "usage": {"completion_tokens": toks}, "timings": {"predicted_per_second": 500.0}}).encode())


def ext_call(ctx: dict[str, Any]) -> dict[str, Any]:
    """An external job's function (creator.gpupulse.ext_job 'call'): reports what the runner handed it."""
    return {"pulse": ctx["pulse"], "models": sorted(ctx["endpoints"].get("models", {})), "workers": ctx["workers"]}


def gpuday(cfg: Any) -> list[Any]:
    return ["smoke:Fake-1B.gguf", {"name": "x", "call": "gpupulse_fake_server:ext_call", "minutes": 5}]


def start() -> tuple[http.server.ThreadingHTTPServer, int]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, int(srv.server_address[1])


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list-devices" in args:
        print("Available devices:\n  CUDA0: NVIDIA GeForce RTX 4090 (24080 MiB, 23500 MiB free)")
        raise SystemExit(0)
    if "--version" in args:
        print("version: 9999 (fake)")
        raise SystemExit(0)
    port = int(args[args.index("--port") + 1])
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
