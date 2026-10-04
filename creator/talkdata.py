"""GPU data job for Nupen's voice: a big model (Qwen3-14B now; Qwen3.6-27B later) writes GROUNDED conversation training pairs for the small
voice model (creator.talk) - and the same exchanges, in the creator.lm.dialogue item format, for Nupen's own LM later (written to a separate
file; nothing here feeds or touches the LM trainer).

    python scripts/gpu_pulse.py plan --jobs-from creator.talkdata:talk_jobs      dry run: minutes and dollars
    python scripts/gpu_pulse.py run  --jobs-from creator.talkdata:talk_jobs      the 'call' job below, on the PC, while the pod serves the model
    python -m creator.talkdata --local 1.7b --per-intent 2 --docs 2              the same pipeline on this PC's model (a smoke test, slow)

Pipeline (pairs_job), every step checked:
  1 questions   the big model paraphrases each intent's example ("ways an owner might ask ...") and writes questions that a doc snippet of
                Nupen's own code answers; lines too close to a HELD-OUT eval question (creator.talkeval) are dropped
  2 verify      the big model routes each question blind with the voice's own router prompt; only questions routed to their label are kept
                -> understanding SFT rows (router prompt -> JSON)
  3 facts       the REAL layer-1 handlers (and the BM25 search for open questions) read Nupen's records on this PC; a request is answered by a
                dry handler (no proposal is written), actions are skipped
  4 answer      the big model answers from the facts only (the voice's own speak prompt); kept only when every number and id occurs in the
                facts (creator.talk.ungrounded) -> speak SFT rows + dialogue items
Privacy (as creator.gpuday's export): a prompt carrying a private marker never leaves (gpuday.private_reason before the pod's own guard),
names, e-mails and home paths are scrubbed, nothing is read from state/livesim, ~/Masterstock or ~/oldpc (the handlers read state/creator
only; private text is never indexed). Outputs live outside the repository: <runtime>/gpuday/talk/<pulse>/."""
from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from creator import conversation as CV
from creator import talk as T

M14 = "Qwen3-14B-Q4_K_M.gguf"
SKIP = ("pause", "resume", "approve", "help")                  # actions and the help list are spoken literally: nothing to learn
SMALLTALK_SEEDS = ("hello", "who are you?", "thanks, that was useful", "good morning Nupen", "how do you feel about your work?")
PARAPHRASE = ("Write {n} different short messages the owner of Nupen (a self-improving coding system on his home PC) might type to it "
              "to ask this: \"{example}\" (meaning: {about}). Vary the wording, tone and length; casual is fine. One message per line, "
              "no numbering, no quotes.")
DOC_QUESTIONS = ("Here is part of the documentation of Nupen, a self-improving coding system:\n---\n{text}\n---\nWrite {n} short questions "
                 "its owner might ask that this text answers. One per line, no numbering, no quotes.")
Chat = Callable[[list[dict[str, str]], int, float], str]


def _lines(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        ln = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip().strip("\"'").strip()
        if 4 <= len(ln) <= 160 and not ln.endswith(":"):
            out.append(ln)
    return out


def _jacc(a: str, b: str) -> float:
    x, y = set(T.tokens(a)), set(T.tokens(b))
    return len(x & y) / max(1, len(x | y))


def held_out(q: str, eval_qs: list[str], limit: float = 0.6) -> bool:
    n = q.lower().strip(" ?!.")
    return any(n == e.lower().strip(" ?!.") or _jacc(q, e) >= limit for e in eval_qs)


def dry_request(c: CV.Ctx, s: dict[str, Any]) -> CV.Facts:
    from creator import talkeval as TE
    return TE.dry_request(c, s)


def facts_for(root: Path, intent: str, question: str, cache: dict[str, CV.Facts]) -> Optional[CV.Facts]:
    """The real handler's facts (a request: the dry handler); intents without a question-specific answer are read once per run."""
    c = CV.Ctx(root)
    slots = dict(CV.RuleUnderstand.slots_from(question), question=question)
    if intent == "ask":
        f = T.h_ask(c, slots)
    elif intent == "smalltalk":
        f = T.h_smalltalk(c, slots)
    elif intent == "request":
        f = dry_request(c, dict(slots, request=re.sub(r"^(?:please|could you|can you|i want you to)\s+", "", question, flags=re.I)))
    elif intent in CV.INTENT_BY_NAME and intent not in SKIP:
        if intent == "explain" and "package" not in slots and "component" not in slots:
            return None
        key = intent if intent != "explain" else f"explain:{slots.get('package')}:{slots.get('component')}"
        if key not in cache:
            cache[key] = CV.INTENT_BY_NAME[intent].handler(c, slots)
        base = cache[key]
        f = CV.Facts(base.intent, list(base.lines), list(base.evidence))
    else:
        return None
    if f.confirm is not None:
        return None
    f.user = question
    return f


def _clean(f: CV.Facts) -> Optional[CV.Facts]:
    """None when the facts carry anything private; else a scrubbed copy (names, e-mails, home paths)."""
    from creator import gpuday as GD
    if GD.private_reason(T.facts_text(f)):
        return None
    return CV.Facts(f.intent, [GD.scrub(x) for x in f.lines], [GD.scrub(x) for x in f.evidence], user=GD.scrub(f.user))


def build(root: Path, chat: Chat, out: Path, *, per_intent: int = 16, docs: int = 60, per_doc: int = 3, workers: int = 4,
          deadline: float = 0.0, seed: int = 0, model: str = "", pulse: str = "") -> dict[str, Any]:
    """The whole pipeline with any chat function (the pod's, a local model's, or a test stub). Returns the report (also written)."""
    from creator import gpuday as GD
    from creator import talkeval as TE
    GD.outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    eval_qs = [q.text for q in TE.QUESTIONS]
    stats: dict[str, int] = {"asked": 0, "errors": 0, "private": 0, "held_out": 0, "misrouted": 0, "no_facts": 0, "ungrounded": 0}
    lock = threading.Lock()

    def bump(k: str, n: int = 1) -> None:
        with lock:
            stats[k] = stats.get(k, 0) + n

    def late() -> bool:
        return bool(deadline) and time.monotonic() > deadline

    def ask(msgs: list[dict[str, str]], max_tokens: int, temp: float) -> Optional[str]:
        if GD.private_reason(json.dumps(msgs)):
            bump("private")
            return None
        try:
            bump("asked")
            return chat(msgs, max_tokens, temp)
        except Exception:  # noqa: BLE001 - one failed request costs one item
            bump("errors")
            return None

    # 1 questions -------------------------------------------------------------------------------------------------------------
    jobs: list[tuple[str, str]] = []                    # (label, prompt)
    for it in CV.INTENTS:
        if it.name not in SKIP:
            jobs.append((it.name, PARAPHRASE.format(n=per_intent, example=it.example, about=it.about)))
    for s in SMALLTALK_SEEDS:
        jobs.append(("smalltalk", PARAPHRASE.format(n=max(2, per_intent // 4), example=s, about="small talk")))
    rng = random.Random(seed)
    corpus = [d for d in T.index_for(root).docs if d["kind"] in ("module", "doc")]
    for d in rng.sample(corpus, min(docs, len(corpus))):
        jobs.append(("ask", DOC_QUESTIONS.format(text=GD.scrub(d["text"]), n=per_doc)))
    cands: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for label, raw in zip([j[0] for j in jobs], ex.map(lambda j: None if late() else ask([{"role": "user", "content": j[1]}], 400, 0.8), jobs)):
            for q in _lines(raw or ""):
                cands.append((label, q))
    seen: set[str] = set()
    qs: list[tuple[str, str]] = []
    for label, q in cands:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        if held_out(q, eval_qs):
            bump("held_out")
            continue
        qs.append((label, q))

    # 2 verify + 3 facts + 4 answer -------------------------------------------------------------------------------------------
    cache: dict[str, CV.Facts] = {}
    cache_lock = threading.Lock()
    und: list[dict[str, Any]] = []
    spk: list[dict[str, Any]] = []
    dia: list[dict[str, Any]] = []

    def one(item: tuple[str, str]) -> None:
        label, q = item
        if late():
            return
        um = T.understand_messages(q, [])
        got = T.parse_intent_json(ask(um, 40, 0.0) or "")
        if got is None or got["intent"] != label:
            bump("misrouted")
            return
        target = {"intent": label} if label != "request" else {"intent": label, "request": str(got.get("request") or q)}
        with cache_lock:
            f = facts_for(root, label, q, cache)
        und_row = {"task": "understand", "intent": label, "messages": um + [{"role": "assistant", "content": json.dumps(target)}]}
        if f is None:
            bump("no_facts")
            with lock:
                und.append(und_row)
            return
        f = _clean(f)
        if f is None:
            bump("private")
            return
        sm = T.speak_messages(f, [])
        ans = (ask(sm, T.SPEAK_TOKENS, 0.3) or "").strip()
        if not ans or T.ungrounded(ans, T.facts_text(f)):
            bump("ungrounded")
            with lock:
                und.append(und_row)
            return
        ans = GD.scrub(ans)
        row = {"task": "speak", "intent": label, "evidence": f.evidence, "model": model, "pulse": pulse,
               "messages": sm + [{"role": "assistant", "content": ans}]}
        item_ = {"stage": 0, "split": "train", "source": "talk-gpu", "reviewed": False, "entities": f.evidence, "expect": [], "intent": label,
                 "dialogue": [{"role": "user", "text": f.user}, {"role": "nupen", "text": ans + T.evidence_line(f)}]}
        if GD.private_reason(json.dumps(row)):
            bump("private")
            return
        with lock:
            und.append(und_row)
            spk.append(row)
            dia.append(item_)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, qs))

    files = {"understand": out / "talk_understand_sft.jsonl", "speak": out / "talk_speak_sft.jsonl", "dialogues": out / "talk_dialogues.jsonl"}
    n = {k: GD._write_jsonl(files[k], rows) for k, rows in (("understand", und), ("speak", spk), ("dialogues", dia))}
    by_intent: dict[str, int] = {}
    for r in spk:
        by_intent[r["intent"]] = by_intent.get(r["intent"], 0) + 1
    rep = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "pulse": pulse, "questions": len(qs), "rows": n, "speak_by_intent": by_intent,
           "stats": stats, "files": {k: str(v) for k, v in files.items()}, "seconds": round(time.monotonic() - t0, 1),
           "note": "dialogues are NOT teacher-reviewed (reviewed=False); not fed to the own LM by this job"}
    (out / "talk_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def default_out(pulse: str = "local") -> Path:
    from creator import device as DEV
    return Path(DEV.runtime_dir()) / "gpuday" / "talk" / re.sub(r"[^\w.-]+", "_", pulse or "local")


def pairs_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job (runs on the PC at IDLE priority; the pod serves ctx['model'], requests go round robin over its forwarded ports)."""
    from creator import generator as G
    from creator import gpuday as GD
    from creator import gpupulse as GP
    GD._idle_priority()
    model = str(ctx.get("model") or M14)
    urls = (((ctx.get("endpoints") or {}).get("models") or {}).get(model) or {}).get("urls") or []
    if not urls:
        return {"error": f"model {model} is not served"}
    pods = [GP.PodLLM(int(u.rsplit(":", 1)[1].split("/", 1)[0]), model, str(ctx.get("pulse") or "")) for u in urls]
    rr = {"i": 0}
    lk = threading.Lock()

    def chat(msgs: list[dict[str, str]], max_tokens: int, temp: float) -> str:
        with lk:
            rr["i"] += 1
            pod = pods[rr["i"] % len(pods)]
        return G.THINK_BLOCK.sub("", pod.chat(G.prepare_messages(msgs, model), max_tokens=max_tokens, temperature=temp, timeout=300.0)).strip()
    a = dict(ctx.get("args") or {})
    slots = int((((ctx.get("endpoints") or {}).get("models") or {}).get(model) or {}).get("slots") or 8) * len(pods)
    return build(Path(str(ctx["repo"])), chat, default_out(str(ctx.get("pulse") or "pulse")), per_intent=int(a.get("per_intent", 16)),
                 docs=int(a.get("docs", 60)), per_doc=int(a.get("per_doc", 3)), workers=max(int(ctx.get("workers") or 4), slots),
                 deadline=float(ctx.get("deadline") or 0.0), model=model, pulse=str(ctx.get("pulse") or ""))


def estimate_minutes(per_intent: int = 16, docs: int = 60, per_doc: int = 3, agg_tok_s: float = 300.0) -> float:
    """Output tokens / aggregate decode speed (14B on a 4090 with 8 slots: ~300 tok/s assumed) + prompt processing + 1 min of slack."""
    n_int = len([i for i in CV.INTENTS if i.name not in SKIP])
    para = (n_int + len(SMALLTALK_SEEDS) + docs) * 250
    n_q = n_int * per_intent + len(SMALLTALK_SEEDS) * max(2, per_intent // 4) + docs * per_doc
    out_tok = para + n_q * (15 + 100)
    in_tok = n_q * (450 + 600)
    return round(out_tok / agg_tok_s / 60 + in_tok / (agg_tok_s * 40) / 60 + 1.0, 1)


def talk_jobs(cfg: Mapping[str, Any]) -> list[Any]:
    """`gpu_pulse.py run --jobs-from creator.talkdata:talk_jobs`: one 'call' job on the big model (device setting 'talk_model', default 14B)."""
    model = str(cfg.get("talk_model") or M14)
    args = {"per_intent": int(cfg.get("talk_per_intent") or 16), "docs": int(cfg.get("talk_docs") or 60), "per_doc": 3}
    est = estimate_minutes(args["per_intent"], args["docs"], args["per_doc"])
    return [{"name": "talk_pairs", "call": "creator.talkdata:pairs_job", "model": model, "minutes": max(5.0, round(est * 2)),
             "max_minutes": max(15.0, round(est * 4)), "low_util_abort_minutes": 0, "args": args}]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", default="", help="run on this PC's model (1.7b | 4b | path) instead of a pod: a smoke test")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--per-intent", type=int, default=2)
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--servers", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.local:
        print(json.dumps({"jobs": talk_jobs({}), "estimate_minutes": estimate_minutes(), "plan": "python scripts/gpu_pulse.py plan --jobs-from "
                          "creator.talkdata:talk_jobs", "run": "python scripts/gpu_pulse.py run --jobs-from creator.talkdata:talk_jobs"}, indent=1))
        return 0
    v = T.Voice(T.resolve_model(a.local), ctx=8192 if a.local == "1.7b" else 4096, servers=a.servers or None)
    try:
        def chat(msgs: list[dict[str, str]], max_tokens: int, temp: float) -> str:
            return v.ask(msgs, max_tokens, temp).text
        rep = build(Path(a.root), chat, default_out("local"), per_intent=a.per_intent, docs=a.docs, per_doc=2, workers=1, model=v.name)
    finally:
        v.close()
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
