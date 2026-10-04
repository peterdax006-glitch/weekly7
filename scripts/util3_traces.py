"""h64 util3: the PC side of the GPU filler - drives creator.gpupulse.traces against the util3 filler server (pod port 18350, watch3.sh).

The pod watcher decides whether a filler runs and which model (4B Q4 beside the runner's small servers, 14B Q4 when the runner is idle, nothing
while the runner serves an 8B+ model). This driver keeps an SSH tunnel to that port, and while a filler answers it feeds it worked-example
questions (reasondrills.generate_stream, cached per repository; frozen items already dropped) with 3x its slots in flight. Only correct,
outcome-blind traces reach the bank (gpupulse.traces -> reasondrills.bank_add; each row records the model). A question this model already
answered (right or wrong) is not asked again by the same model (asked_<model>.txt). Runs at BELOW_NORMAL priority (it feeds the paid GPU).
    python scripts/util3_traces.py --main C:/Users/peter/weekly7
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

RT = Path("C:/Users/peter/creator_runtime/util3")
PORT = 18350
SLOTS = {"4b": 24, "14b": 32}


def log(msg: str) -> None:
    line = f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def tunnel(host: str, sshport: str, key: str) -> subprocess.Popen:
    return subprocess.Popen(["ssh", "-i", key, "-p", sshport, "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ExitOnForwardFailure=yes",
                             "-o", "ServerAliveInterval=15", "-N", "-L", f"{PORT}:127.0.0.1:{PORT}", f"root@{host}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def served_model() -> str:
    """The model file the filler serves, or '' when none answers."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
            d = json.loads(r.read())
        return Path(str(d["data"][0]["id"]).replace("\\", "/")).name
    except Exception:                                                    # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="C:/Users/peter/weekly7")
    ap.add_argument("--host", required=True)
    ap.add_argument("--sshport", required=True)
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "nupen_vast"))
    ap.add_argument("--batch-minutes", type=float, default=8.0)
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
    loaded = {"t": 0.0}

    def load_questions() -> None:
        t0 = time.monotonic()
        got: list = []
        for part in R.generate_stream(R.repos(repo), state):
            got += [q for q in R.order(part) if R.REVISIT not in q.qid]
        qs[:] = got
        loaded["t"] = time.monotonic()
        log(f"questions: {len(got)} in {time.monotonic() - t0:.0f}s")

    load_questions()
    tun = tunnel(a.host, a.sshport, a.key)
    totals = {"kept": 0, "asked": 0}
    while not (RT / "STOP").exists():
        if tun.poll() is not None:
            time.sleep(3)
            tun = tunnel(a.host, a.sshport, a.key)
        model = served_model()
        if not model:
            time.sleep(2)
            continue
        kind = "14b" if "14B" in model else "4b"
        slots = SLOTS[kind]
        askedf = RT / f"asked_{model}.txt"
        asked = set(askedf.read_text(encoding="utf-8").split()) if askedf.exists() else set()
        have = set(R.trace_bank(state))
        if time.monotonic() - loaded["t"] > R.REFRESH_S:
            load_questions()
        pending = [q for q in qs if q.qid not in asked and q.qid not in have]
        if not pending:
            log(f"no pending questions for {model}; sleeping")
            time.sleep(60)
            continue
        by_prompt = {q.prompt(): q.qid for q in pending[: slots * 60]}
        done: list[str] = []
        flag = {"down": False}
        lock = threading.Lock()

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
                if got != model:                                          # the watcher swapped models mid-batch: the trace is not this model's
                    flag["down"] = True
                    raise GP.PulseError(f"model changed to {got}")
                u = messages[-1]["content"]
                for p, qid in by_prompt.items():
                    if p in u:
                        with lock:
                            done.append(qid)
                        break
                return {"text": str(d["choices"][0]["message"].get("content") or ""),
                        "tokens": int((d.get("usage") or {}).get("completion_tokens") or 0), "seconds": time.monotonic() - t0}

        batch = pending[: slots * 60]
        llm = Filler(PORT, model, pulse)
        t0 = time.monotonic()
        try:
            st = GP.traces(llm, state, repo, len(batch), slots * 3, time.monotonic() + a.batch_minutes * 60, questions=lambda: batch)
        except Exception as e:                                           # noqa: BLE001
            log(f"traces error: {e!r}")
            st = {}
        finally:
            llm._drop()
        with askedf.open("a", encoding="utf-8") as f:
            f.write("".join(q + "\n" for q in done))
        totals["kept"] += int(st.get("kept") or 0)
        totals["asked"] += int(st.get("asked") or 0)
        row = {"at": dt.datetime.now().isoformat(timespec="seconds"), "model": model, "s": round(time.monotonic() - t0, 1),
               "down": flag["down"], **{k: st.get(k) for k in ("asked", "correct", "kept", "errors")}, "tot_kept": totals["kept"]}
        with (RT / "driver_batches.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log(json.dumps(row))
    tun.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
