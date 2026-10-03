"""Benchmark one candidate THINKING model on this machine (owner, 2 Oct 2026: Nupen must think and reason much better by morning).

    python scripts/think_model_bench.py --name qwen3-1.7b --model <gguf> --state <state/creator dir> --out <json> [--nothink] [--n 24]

Measures with Nupen running (numbers are therefore noisy and honest): load time, resident memory, generation tokens/s at 4/7/11 threads,
and QUALITY on Nupen's own tasks with the SAME prompts for every model:
  (a) creator.judgment drill cases (verdict, git_fixed): raw Brier of the model's probability (the prompt only holds the record resolved before
      the case); paired against a named baseline model's file with a 95% CI;
  (b) multiple-choice questions with exact answers from Nupen's own history: which module a test covers, which of two packages was adopted.
Read-only on the state directory. The llama server is started here with a pidfile in %TEMP% and always stopped by PID."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import device as DEV  # noqa: E402
from creator import generator as G  # noqa: E402
from creator import judgment as J  # noqa: E402

STRIP_THINK = re.compile(r"<think>.*?</think>", re.S)
SPEED_PROMPT = [{"role": "user", "content": "Explain in about eighty words why a git history can predict which commits later need a fix."}]


class Server:
    def __init__(self, model: Path, threads: int, ctx: int = 8192) -> None:
        self.port = G.free_port()
        self.pidfile = Path(tempfile.gettempdir()) / f"think_bench_{os.getpid()}_{self.port}.pid"
        t0 = time.monotonic()
        self.proc = subprocess.Popen([str(DEV.server_exe()), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(ctx),
                                      "-t", str(threads), "--log-disable"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pidfile.write_text(str(self.proc.pid), encoding="utf-8")
        while time.monotonic() - t0 < 600:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("server exited during start-up")
            time.sleep(0.3)
        self.load_s = time.monotonic() - t0

    def rss_gb(self) -> float:
        import psutil
        return psutil.Process(self.proc.pid).memory_info().rss / 2**30

    def chat(self, messages: Sequence[dict[str, str]], max_tokens: int, temperature: float = 0.2, timeout: float = 900.0) -> dict[str, Any]:
        body = json.dumps({"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature, "seed": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return dict(json.loads(r.read().decode("utf-8")))

    def stop(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            self.pidfile.unlink(missing_ok=True)


def text_of(data: dict[str, Any]) -> str:
    return STRIP_THINK.sub("", str(data["choices"][0]["message"]["content"])).strip()


def speed(model: Path, threads: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in threads:
        s = Server(model, t)
        try:
            s.chat(SPEED_PROMPT, 16)                                                   # warm-up
            runs = []
            for _ in range(2):
                tm = s.chat(SPEED_PROMPT, 80, 0.0).get("timings", {})
                runs.append(float(tm.get("predicted_per_second", 0.0)))
            out[str(t)] = {"load_s": round(s.load_s, 1), "rss_gb": round(s.rss_gb(), 2), "tok_per_s": [round(x, 2) for x in runs]}
        finally:
            s.stop()
    return out


# ------------------------------------------------------------------------------------------------ question sets (deterministic, same for every model)
def judgment_cases(state: Path, n: int) -> list[tuple[J.Case, list[J.Case]]]:
    out = []
    for topic in J.TOPICS:
        cs = J.load_cases(topic, state, ROOT)
        pick = [c for c in cs if len(J.history(cs, c)) >= 3][-n:]
        out += [(c, cs) for c in pick]
    return out


def module_questions(n: int, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    mods = {p.stem: p for p in (ROOT / "creator").glob("*.py") if p.stem != "__init__"}

    def doc1(p: Path) -> str:
        m = re.match(r'\s*"""(.*?)(?:"""|\n\n)', p.read_text(encoding="utf-8", errors="replace"), re.S)
        return " ".join((m.group(1) if m else "").split())[:140]
    qs = []
    for t in sorted((ROOT / "tests").glob("test_creator_*.py")):
        name = t.stem[len("test_creator_"):]
        if name not in mods or len(doc1(mods[name])) < 30:
            continue
        td = " ".join(re.sub(r'(?s)^\s*"""(.*?)""".*', r"\1", t.read_text(encoding="utf-8", errors="replace")[:900]).split())[:420]
        opts = [name] + rng.sample([m for m in sorted(mods) if m != name and len(doc1(mods[m])) >= 30], 3)
        rng.shuffle(opts)
        letters = "ABCD"
        body = "\n".join(f"{letters[i]}. creator/{o}.py - {doc1(mods[o])}" for i, o in enumerate(opts))
        qs.append({"kind": "test_module", "q": f"A test file begins with this description:\n\"{td}\"\n\nWhich module does the test file cover?\n{body}",
                   "answer": letters[opts.index(name)]})
    rng.shuffle(qs)
    return qs[:n]


def adopted_questions(state: Path, n: int, seed: int = 11) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cs = J.load_cases("verdict", state, ROOT)
    by: dict[str, list[J.Case]] = {}
    for c in cs:
        by.setdefault(c.group, []).append(c)
    pairs = []
    for g, xs in by.items():
        yes, no = [c for c in xs if c.y], [c for c in xs if not c.y]
        for a, b in zip(rng.sample(yes, min(len(yes), len(no))), rng.sample(no, min(len(yes), len(no)))):
            pairs.append((a, b))
    rng.shuffle(pairs)
    qs = []
    for a, b in pairs[:n]:
        first, second = (a, b) if rng.random() < 0.5 else (b, a)
        hist = [c for c in cs if c.resolved < min(a.created, b.created)]
        rate = f"Past record: {len(hist)} resolved packages, {sum(c.y for c in hist) / max(1, len(hist)):.2f} adopted. " if len(hist) >= 3 else ""
        qs.append({"kind": "adopted_pair", "q": f"{rate}Exactly one of these two packages was ADOPTED by the kernel.\nA. {first.text.rsplit(' Will', 1)[0]}\n"
                   f"B. {second.text.rsplit(' Will', 1)[0]}\nWhich one was adopted?", "answer": "A" if first is a else "B"})
    return qs


ANSWER = re.compile(r"ANSWER\s*[:=]\s*\**\s*([ABCD])\b", re.I)


# ------------------------------------------------------------------------------------------------ statistics
def mean_ci(d: Sequence[float]) -> list[float]:
    n = len(d)
    if n < 2:
        return [round(d[0], 4) if d else 0.0, 0.0, 0.0]
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return [round(m, 4), round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0, 0.0]
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--nothink", action="store_true", help="append /no_think to prompts (Qwen3)")
    ap.add_argument("--speed", default="4,7,11")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--max-tokens-mc", type=int, default=200)
    a = ap.parse_args(argv)
    model, state, outp = Path(a.model), Path(a.state), Path(a.out)
    res: dict[str, Any] = {"name": a.name, "model": str(model), "bytes": model.stat().st_size, "speed": {}, "judgment": [], "mc": []}
    if outp.is_file():
        res.update(json.loads(outp.read_text(encoding="utf-8")))

    def save() -> None:
        outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    suffix = "\n/no_think" if a.nothink else ""
    if not a.skip_speed and not res["speed"]:
        res["speed"] = speed(model, [int(x) for x in a.speed.split(",")])
        save()
    srv = Server(model, a.threads)
    try:
        res["rss_quality_gb"] = round(srv.rss_gb(), 2)
        done = {(r["topic"], r["subject"]) for r in res["judgment"]}
        strat = {"shots": 2, "hint": 1}
        for c, cs in judgment_cases(state, a.n):
            if (c.topic, c.subject) in done:
                continue
            msgs = J.build_prompt(strat, cs, c)
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + suffix}
            t0 = time.monotonic()
            reply = text_of(srv.chat(msgs, 140))
            p = J.parse(reply)
            res["judgment"].append({"topic": c.topic, "subject": c.subject, "y": c.y, "p": p, "s": round(time.monotonic() - t0, 1), "reply": reply[:200]})
            save()
        have = {r["q"] for r in res["mc"]}
        for q in module_questions(a.n) + adopted_questions(state, a.n):
            if q["q"] in have:
                continue
            msgs = [{"role": "system", "content": "Answer the multiple-choice question. Give at most two sentences of reasoning, then a last line 'ANSWER: X' with one letter."},
                    {"role": "user", "content": q["q"] + suffix}]
            t0 = time.monotonic()
            reply = text_of(srv.chat(msgs, a.max_tokens_mc))
            m = ANSWER.findall(reply)
            res["mc"].append({"kind": q["kind"], "q": q["q"], "answer": q["answer"], "got": m[-1].upper() if m else None,
                              "s": round(time.monotonic() - t0, 1), "reply": reply[:200]})
            save()
    finally:
        srv.stop()
    save()
    return 0


def compare(files: Sequence[Path], baseline: str) -> dict[str, Any]:
    """Paired comparison of benchmark result files on the same questions."""
    data = {d["name"]: d for d in (json.loads(f.read_text(encoding="utf-8")) for f in files)}
    out: dict[str, Any] = {}
    base = data[baseline]
    for name, d in data.items():
        row: dict[str, Any] = {"speed": d.get("speed"), "bytes": d.get("bytes")}
        for topic in J.TOPICS:
            mine = {r["subject"]: r for r in d["judgment"] if r["topic"] == topic}
            theirs = {r["subject"]: r for r in base["judgment"] if r["topic"] == topic}
            subj = sorted(set(mine) & set(theirs))
            br = [(mine[s]["p"] - mine[s]["y"]) ** 2 if mine[s]["p"] is not None else 0.25 for s in subj]
            bb = [(theirs[s]["p"] - theirs[s]["y"]) ** 2 if theirs[s]["p"] is not None else 0.25 for s in subj]
            row[f"brier_{topic}"] = {"n": len(subj), "mean": mean_ci(br)[0], "unparsed": sum(1 for s in subj if mine[s]["p"] is None),
                                     "gain_vs_baseline": mean_ci([x - y for x, y in zip(bb, br)])}
        for kind in ("test_module", "adopted_pair"):
            rs = [r for r in d["mc"] if r["kind"] == kind]
            k = sum(1 for r in rs if r["got"] == r["answer"])
            row[f"acc_{kind}"] = {"n": len(rs), "correct": k, "wilson95": wilson(k, len(rs))}
            bmap = {r["q"]: r for r in base["mc"] if r["kind"] == kind}
            diffs = [float(r["got"] == r["answer"]) - float(bmap[r["q"]]["got"] == bmap[r["q"]]["answer"]) for r in rs if r["q"] in bmap]
            row[f"acc_{kind}"]["gain_vs_baseline"] = mean_ci(diffs)
        out[name] = row
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        print(json.dumps(compare([Path(p) for p in sys.argv[3:]], sys.argv[2]), indent=1))
        raise SystemExit(0)
    raise SystemExit(main())
