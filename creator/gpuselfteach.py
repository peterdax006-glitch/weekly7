"""GPU ROLE IN SELF-TEACHING (h58, 3 Oct 2026): the held-out gate for a fine-tuned home model. A gpupulse EXTERNAL 'call' job.

Why this job: the GPU day (creator.gpuday) fine-tunes small home models (ft1_17b / ft1_4b on the worked-example bank the stronger model wrote)
and registers them as servable right after training; the frozen thinkbench runs after that. What was missing for the self-teaching trust gate
(creator.selfteach, check c) is the decision rule for a MODEL update: a tuned model replaces the home model only when it is significantly better
on data it was NOT trained on. This job measures exactly that, PAIRED:

  questions  creator.reasondrills questions (correct answer known by construction; frozen-benchmark collisions already removed) that the
             CURRENT home model answered with the control strategy ('plain'), that are NOT in the trace bank (the fine-tune's training data)
             and whose time `t` is AFTER every banked question (time-ordered: nothing the tuned model learned from is newer than them).
  answer     the tuned model (served by this pulse) answers the same questions with the same 'plain' prompt.
  verdict    per question d = tuned correct - home correct; mean with 95% CI. ADOPT only when the CI lower bound > 0 and n >= MIN_N;
             otherwise KEEP_HOME. The verdict is a record (thinking/gpu_heldout_gate.jsonl), never a switch: the teacher (or a later gate)
             acts on it. The home answers are read, never re-asked; nothing is banked from these questions (they stay held out).

Job spec (gpupulse.ext_job, run after `register_tuned_job` of ft1_17b):
    HELDOUT_GATE_JOB below. Dry-run: `run_gate(...)` with any object that has .chat(messages, **kw) (tests use a stub)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

MIN_N = 50
MAX_TOKENS = 16

HELDOUT_GATE_JOB: dict[str, Any] = {
    "name": "heldout_gate_ft1_17b",
    "call": "creator.gpuselfteach:heldout_gate_job",
    "model": "Qwen3-1.7B-gpuday-ft1.gguf",          # the tuned model gpuday registers (serve_as of ft1_17b)
    "minutes": 12,
    "max_minutes": 25,
    "low_util_abort_minutes": 0,                     # the home-side question generation takes a few CPU minutes first
    "args": {"n": 400, "home_model": "Qwen3-1.7B-Q4_K_M.gguf"},
}


def gate_path(state: Path) -> Path:
    return Path(state) / "thinking" / "gpu_heldout_gate.jsonl"


def mean_ci(xs: Sequence[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), -math.inf, math.inf
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return m, m - 1.96 * se, m + 1.96 * se


def heldout_set(questions: Sequence[Any], home_rows: Sequence[Mapping[str, Any]], bank: Mapping[str, Mapping[str, Any]], home_model: str,
                n: int) -> tuple[list[tuple[Any, int]], float]:
    """(question, home correct) pairs: answered by the home model with the control prompt, not banked, newer than every banked question.
    Deterministic order (hash of the id), at most n."""
    cut = max((float(r.get("t") or 0.0) for r in bank.values()), default=-math.inf)
    by = {q.qid: q for q in questions}
    home: dict[str, int] = {}
    for r in home_rows:
        if r.get("strategy") == "plain" and r.get("model") == home_model and r.get("qid") in by and r.get("correct") is not None:
            home.setdefault(str(r["qid"]), int(bool(r["correct"])))          # the first answer (no revisit)
    ids = sorted((q for q in home if q not in bank and "#" not in q and float(by[q].t) > cut),
                 key=lambda q: hashlib.sha256(q.encode("utf-8")).hexdigest())[:max(0, n)]
    return [(by[q], home[q]) for q in ids], cut


def run_gate(llm: Any, pairs: Sequence[tuple[Any, int]], workers: int = 8, deadline: float = math.inf,
             ask: Optional[Callable[[Any, list[dict[str, str]]], str]] = None) -> dict[str, Any]:
    """Ask the tuned model every held-out question (control prompt); paired difference with the home model's recorded correctness."""
    from creator import reasondrills as R
    if ask is None:
        from creator import judgment as J

        def ask(m: Any, msgs: list[dict[str, str]]) -> str:
            return J.chat_text(m, msgs, max_tokens=MAX_TOKENS, temperature=0.0, seed=0, timeout=300.0)
    st = R.BY_NAME[R.CONTROL]
    diffs: dict[str, float] = {}
    lock = threading.Lock()
    it = iter(pairs)
    errors = [0]

    def work() -> None:
        while time.monotonic() < deadline:
            with lock:
                nxt = next(it, None)
            if nxt is None:
                return
            q, home_ok = nxt
            try:
                reply = ask(llm, R.build_messages(st, q))
            except Exception:                                        # noqa: BLE001 - one failed call is one missing pair, never a guess
                with lock:
                    errors[0] += 1
                continue
            with lock:
                diffs[q.qid] = float(int(R.parse_choice(reply) == q.answer) - home_ok)
    ts = [threading.Thread(target=work, daemon=True) for _ in range(max(1, workers))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ds = list(diffs.values())
    m, lo, hi = mean_ci(ds)
    home_acc = sum(h for q, h in pairs if q.qid in diffs) / len(diffs) if diffs else None
    verdict = "ADOPT" if len(ds) >= MIN_N and lo > 0 else "KEEP_HOME"
    return {"n": len(ds), "errors": errors[0], "home_acc": None if home_acc is None else round(home_acc, 4),
            "tuned_acc": None if home_acc is None else round(home_acc + m, 4), "gain": round(m, 4),
            "gain_ci95": [None if not math.isfinite(lo) else round(lo, 4), None if not math.isfinite(hi) else round(hi, 4)],
            "verdict": verdict, "why": ("significantly better on held-out questions" if verdict == "ADOPT" else
                                        f"not significantly better (n {len(ds)}, need {MIN_N} and CI lower bound > 0)")}


def heldout_gate_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job: ctx from gpupulse.run_job_here (state, repo, workers, deadline, tunnel_file, model, args)."""
    from creator import gpupulse as GP
    from creator import reasondrills as R
    state, repo, a = Path(str(ctx["state"])), Path(str(ctx["repo"])), dict(ctx.get("args") or {})
    tuned = str(a.get("tuned") or ctx.get("model") or "")
    home = str(a.get("home_model") or R.active_tag())
    got = GP.attach(Path(str(ctx["tunnel_file"])), tuned)
    if got is None:
        return {"verdict": "NOT_RUN", "why": f"the pulse does not serve {tuned}"}
    qs = R.generate(R.repos(repo), state)
    pairs, cut = heldout_set(qs, R._jsonl(R.path(state)), R.trace_bank(state), home, int(a.get("n") or 400))
    res = run_gate(GP.PodLLM(got[0], tuned, got[1]), pairs, int(ctx.get("workers") or 8), float(ctx.get("deadline") or math.inf))
    rec = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "pulse": ctx.get("pulse"), "tuned": tuned, "home": home,
           "heldout_after_t": None if not math.isfinite(cut) else cut, "candidates": len(pairs), **res}
    p = gate_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec
