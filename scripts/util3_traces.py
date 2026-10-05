"""h64 util3: the PC side of the GPU filler - drives creator.gpupulse.traces against the util3 filler servers (pod ports 18350-18353, watch3.sh).

The pod watcher decides how many filler instances run and which model (4B Q4 beside the runner's work, 14B Q4 when the runner is idle, nothing
while the runner serves an 8B+ model). This driver keeps one SSH tunnel to those ports, and one thread per port feeds whatever instance answers
there with worked-example questions (reasondrills.generate_stream, cached per repository; frozen items already dropped), 3x its slots in
flight. Only correct, outcome-blind traces reach the bank (gpupulse.traces -> reasondrills.bank_add; each row records the model). A question a
model already answered (right or wrong) is not asked again by that model (asked_<model>.txt); threads never share a question in flight.
Runs at BELOW_NORMAL priority (it feeds the paid GPU).
    python scripts/util3_traces.py --host <pod> --sshport <port> [--main $HOME/weekly7]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

RT = Path.home() / "creator_runtime/util3"
PORTS = (18350, 18351, 18352, 18353)
SLOTS = 16                                   # -np of every filler instance (watch3.sh)
BATCH_Q = SLOTS * 60


def log(msg: str) -> None:
    print(f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def tunnel(host: str, sshport: str, key: str) -> subprocess.Popen:
    fw = [x for p in PORTS for x in ("-L", f"{p}:127.0.0.1:{p}")]
    return subprocess.Popen(["ssh", "-i", key, "-p", sshport, "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ExitOnForwardFailure=yes",
                             "-o", "ServerAliveInterval=15", "-N", *fw, f"root@{host}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def served_model(port: int) -> str:
    """The model file the filler on `port` serves, or '' when none answers."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
            d = json.loads(r.read())
        return Path(str(d["data"][0]["id"]).replace("\\", "/")).name
    except Exception:                                                    # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(Path.home() / "weekly7"))
    ap.add_argument("--host", required=True)
    ap.add_argument("--sshport", required=True)
    ap.add_argument("--relay", default="", help="fallback host:port (vast ssh relay) used when the direct tunnel keeps failing")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "nupen_vast"))
    ap.add_argument("--batch-minutes", type=float, default=6.0)
    a = ap.parse_args()
    sys.path.insert(0, a.main)
    sys.path.insert(0, str(Path(a.main) / "scripts"))
    os.chdir(a.main)
    from lowprio import lower_own_priority
    lower_own_priority()
    from creator import gpupulse as GP
    from creator import reasondrills as R
    state, repo = Path(a.main) / "state" / "creator", Path(a.main)
    pulse = "util3-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    qs: list = []
    pool = threading.Lock()
    inflight: set[str] = set()
    asked: dict[str, set[str]] = {}
    loaded = {"t": 0.0}
    totals = {"kept": 0}

    def load_questions() -> None:
        t0 = time.monotonic()
        got: list = []
        for part in R.generate_stream(R.repos(repo), state):
            got += [q for q in R.order(part) if R.REVISIT not in q.qid]
        qs[:] = got
        loaded["t"] = time.monotonic()
        log(f"questions: {len(got)} in {time.monotonic() - t0:.0f}s")

    def asked_of(model: str) -> set[str]:
        if model not in asked:
            f = RT / f"asked_{model}.txt"
            asked[model] = set(f.read_text(encoding="utf-8").split()) if f.exists() else set()
        return asked[model]

    def claim(model: str) -> list:
        with pool:
            if time.monotonic() - loaded["t"] > R.REFRESH_S:
                load_questions()
            have, done = set(R.trace_bank(state)), asked_of(model)
            out = []
            for q in qs:
                if q.qid not in done and q.qid not in have and q.qid not in inflight:
                    out.append(q)
                    if len(out) >= BATCH_Q:
                        break
            inflight.update(q.qid for q in out)
            return out

    def release(model: str, batch: list, answered: list[str]) -> None:
        with pool:
            inflight.difference_update(q.qid for q in batch)
            asked_of(model).update(answered)
            with (RT / f"asked_{model}.txt").open("a", encoding="utf-8") as f:
                f.write("".join(x + "\n" for x in answered))

    def port_worker(port: int) -> None:
        while not (RT / "STOP").exists():
            model = served_model(port)
            if not model:
                time.sleep(2)
                continue
            batch = claim(model)
            if not batch:
                log(f"{port}: no pending questions for {model}")
                time.sleep(60)
                continue
            by_prompt = {q.prompt(): q.qid for q in batch}
            answered: list[str] = []
            flag = {"down": False}
            lk = threading.Lock()

            class Filler(GP.PodLLM):
                def request(self, messages, max_tokens=400, temperature=0.2, seed=0, timeout=300.0):  # type: ignore[override]
                    if flag["down"]:
                        raise GP.PulseError("filler down")
                    GP.outbound_ok(messages)
                    body = json.dumps({"messages": [dict(m) for m in messages], "max_tokens": max_tokens, "temperature": temperature,
                                       "seed": seed}).encode()
                    t0 = time.monotonic()
                    try:
                        d = json.loads(self._post("/v1/chat/completions", body, timeout))
                    except Exception:
                        flag["down"] = True
                        raise
                    got = Path(str(d.get("model") or model).replace("\\", "/")).name
                    if got != model:                                      # the watcher swapped models: the trace is not this model's
                        flag["down"] = True
                        raise GP.PulseError(f"model changed to {got}")
                    u = messages[-1]["content"]
                    for p, qid in by_prompt.items():
                        if p in u:
                            with lk:
                                answered.append(qid)
                            break
                    return {"text": str(d["choices"][0]["message"].get("content") or ""),
                            "tokens": int((d.get("usage") or {}).get("completion_tokens") or 0), "seconds": time.monotonic() - t0}

            llm = Filler(port, model, pulse)
            t0 = time.monotonic()
            st: dict = {}
            try:
                st = GP.traces(llm, state, repo, len(batch), SLOTS * 3, time.monotonic() + a.batch_minutes * 60, questions=lambda: batch)
            except Exception as e:                                       # noqa: BLE001
                log(f"{port}: traces error: {e!r}")
            finally:
                llm._drop()
                release(model, batch, answered)
            with pool:
                totals["kept"] += int(st.get("kept") or 0)
                row = {"at": dt.datetime.now().isoformat(timespec="seconds"), "port": port, "model": model, "s": round(time.monotonic() - t0, 1),
                       "down": flag["down"], **{k: st.get(k) for k in ("asked", "correct", "kept", "errors")}, "tot_kept": totals["kept"]}
                with (RT / "driver_batches.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            log(json.dumps(row))
            if flag["down"]:
                time.sleep(2)

    load_questions()
    routes = [(a.host, a.sshport)] + ([tuple(a.relay.split(":"))] if a.relay else [])
    ri = 0
    tun = tunnel(*routes[ri], a.key)
    ts = [threading.Thread(target=port_worker, args=(p,), daemon=True, name=f"port{p}") for p in PORTS]
    for t in ts:
        t.start()
    while not (RT / "STOP").exists():
        if tun.poll() is not None:                                       # the tunnel died: next route (direct <-> relay)
            time.sleep(3)
            ri = (ri + 1) % len(routes)
            log(f"tunnel down; reconnecting via route {ri}")
            tun = tunnel(*routes[ri], a.key)
        time.sleep(5)
    for t in ts:
        t.join(timeout=a.batch_minutes * 60 + 30)
    tun.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
