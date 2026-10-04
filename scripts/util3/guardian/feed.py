#!/usr/bin/env python3
"""GPU guardian feeder (runs ON THE POD, stdlib only): keeps every filler llama-server on ports 18350-18353 busy with staged reasoning
questions (q/questions.jsonl.gz, staged from the PC by guardian_sync.py stage) and logs every answer to out/<model>.jsonl, which the PC pulls
and re-checks before anything reaches the trace bank. 3x slots in flight per server.
Order per model: questions never answered correctly by anyone and never asked of this model (first pass); then the still-unsolved ones again
at temperature 0.8 with a new seed (up to MAX_TRIES per model) - a correct trace for a hard question is the most useful distillation row."""
import gzip, json, os, re, threading, time, urllib.request

D = "/root/guardian"
QF = f"{D}/q/questions.jsonl.gz"
PORTS = (18350, 18351, 18352, 18353)
INFLIGHT = 48                       # 3 x the 16 slots of a filler server
MAX_TRIES = 4
ANSWER = re.compile(r"ANSWER\s*[:=]\s*\(?\**\s*([A-D])\b", re.I)
THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)

lock = threading.Lock()
Q = {"mtime": 0, "list": [], "by": {}}
solved = set()                      # qid answered correctly by any model (or banked on the PC)
tries = {}                          # (model, qid) -> answers so far
inflight = set()                    # (model, qid)
cursor = {}                         # model -> scan position
live = {}                           # port -> model file name ('' = down)


def log(msg):
    with open(f"{D}/feed.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def load():
    st = os.stat(QF)
    if st.st_mtime == Q["mtime"]:
        return
    qs = []
    with gzip.open(QF, "rt", encoding="utf-8") as f:
        for ln in f:
            qs.append(json.loads(ln))
    with lock:
        Q["list"], Q["by"], Q["mtime"] = qs, {q["qid"]: q for q in qs}, st.st_mtime
        for q in qs:
            if q.get("banked"):
                solved.add(q["qid"])
            for m in q.get("asked_by") or ():
                tries[(m, q["qid"])] = max(tries.get((m, q["qid"]), 0), 1)
        cursor.clear()
    log(f"questions loaded: {len(qs)}")


def replay():
    os.makedirs(f"{D}/out", exist_ok=True)
    for name in os.listdir(f"{D}/out"):
        if not name.endswith(".jsonl"):
            continue
        with open(f"{D}/out/{name}", encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                k = (r["model"], r["qid"])
                tries[k] = tries.get(k, 0) + 1
                if r.get("correct"):
                    solved.add(r["qid"])


def claim(model):
    """Next question for this model (first pass, then retries), or None."""
    with lock:
        qs = Q["list"]
        for rnd in (0, 1):
            start = cursor.get((model, rnd), 0)
            for i in range(start, len(qs)):
                qid = qs[i]["qid"]
                k = (model, qid)
                if qid in solved or k in inflight:
                    continue
                n = tries.get(k, 0)
                if (rnd == 0 and n == 0) or (rnd == 1 and 0 < n < MAX_TRIES):
                    inflight.add(k)
                    cursor[(model, rnd)] = i + 1 if rnd == 0 else i
                    return qs[i], n
            cursor[(model, rnd)] = len(qs)
        # the retry cursor only moves forward; reset it once a full sweep found nothing
        cursor[(model, 1)] = 0
    return None


def served(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return os.path.basename(str(json.loads(r.read())["data"][0]["id"]))
    except Exception:
        return ""


def worker(port, wi):
    while not os.path.exists(f"{D}/STOP_FEED"):
        model = live.get(port, "")
        if not model:
            time.sleep(2)
            continue
        got = claim(model)
        if got is None:
            time.sleep(30)
            continue
        q, n = got
        temp, seed = (0.2, 0) if n == 0 else (0.8, n)
        body = json.dumps({"messages": q["messages"], "max_tokens": 260, "temperature": temp, "seed": seed}).encode()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            if os.path.basename(str(d.get("model") or model)) != model:
                raise RuntimeError("model changed")
            reply = THINK.sub("", str(d["choices"][0]["message"].get("content") or "")).strip()
        except Exception:
            with lock:
                inflight.discard((model, q["qid"]))
            live[port] = ""
            time.sleep(2)
            continue
        m = ANSWER.findall(reply)
        pick = "ABCD".index(m[-1].upper()) if m else None
        ok = pick == q["answer"]
        row = {"qid": q["qid"], "model": model, "reply": reply[-1500:], "pick": pick, "correct": ok, "temperature": temp, "seed": seed,
               "tokens": int((d.get("usage") or {}).get("completion_tokens") or 0), "ts": round(time.time(), 1)}
        with lock:
            inflight.discard((model, q["qid"]))
            tries[(model, q["qid"])] = tries.get((model, q["qid"]), 0) + 1
            if ok:
                solved.add(q["qid"])
            with open(f"{D}/out/{model}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def traces_models():
    """Models the queue gives to traces (queue.txt '<prio> <file> traces'); ports serving a queued JOB are driven from the PC instead."""
    try:
        with open(f"{D}/queue.txt") as f:
            return {ln.split()[1] for ln in f if len(ln.split()) >= 3 and ln.split()[2] == "traces"}
    except OSError:
        return {"Qwen3-4B-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf"}


def main():
    while not os.path.exists(QF):
        time.sleep(30)
    replay()
    load()
    for p in PORTS:
        for i in range(INFLIGHT):
            threading.Thread(target=worker, args=(p, i), daemon=True).start()
    last_load = time.time()
    while not os.path.exists(f"{D}/STOP_FEED"):
        tm = traces_models()
        for p in PORTS:
            m = served(p)
            live[p] = m if m in tm else ""
        if time.time() - last_load > 120:
            try:
                load()
            except Exception as e:
                log(f"load error {e!r}")
            last_load = time.time()
        time.sleep(3)


if __name__ == "__main__":
    main()
