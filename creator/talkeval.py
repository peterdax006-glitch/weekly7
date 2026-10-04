"""The held-out conversation eval of Nupen's terminal talk (layer 1 rules vs layer 2 voices). 50+ questions an owner would ask - status, goals,
what it learned, what it needs, explain X, open questions about its own code, small talk - each with the intent(s) a correct router picks.

Checks per answer (all against the REAL records of `root`, read-only: a request is answered by a dry handler, actions only reach the
confirmation question, nothing is logged):
    intent_ok   the routed intent is one of the expected ones
    grounded    every number and record id in the answer occurs in the facts the handler read or the question (creator.talk.ungrounded),
                and every verb of state (working on / blocked / learned / adopted) matches the record (creator.talk.state_claims)
    answered    not the 'I don't understand' list and not 'nothing found' when the question is about something Nupen has records of
    latency     seconds per answer on this PC, model seconds, tokens in/out, and who spoke (model / rule fallback)

These questions are HELD OUT: creator.talkdata never trains on them (it drops generated questions too close to any of them)."""
from __future__ import annotations

import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

from creator import conversation as CV
from creator import talk as T


@dataclasses.dataclass(frozen=True)
class Q:
    text: str
    intents: tuple[str, ...]
    kind: str


def _q(text: str, kind: str, *intents: str) -> Q:
    return Q(text, tuple(intents), kind)


QUESTIONS: list[Q] = [
    # status
    _q("how are you doing today?", "status", "status"), _q("are you up and running?", "status", "status"),
    _q("give me a quick status report", "status", "status"), _q("is the swarm working right now?", "status", "status"),
    _q("how much RAM do you have free?", "status", "status"), _q("how many patches are waiting for me?", "status", "status"),
    _q("did you adopt anything today?", "status", "status", "lessons"),
    # constraints / needs
    _q("what's holding you back?", "needs", "constraints"), _q("what is your biggest bottleneck at the moment?", "needs", "constraints"),
    _q("what do you need from me to go faster?", "needs", "constraints", "goals"), _q("anything getting in your way lately?", "needs", "constraints"),
    _q("which limit should we fix first?", "needs", "constraints"),
    # plan / work
    _q("what are you working on?", "plan", "plan"), _q("what's on your plate right now?", "plan", "plan"),
    _q("show me the critical path", "plan", "plan"), _q("why did you choose that work?", "plan", "plan"),
    _q("what is next on your schedule?", "plan", "plan", "goals"),
    _q("are you working on K28 right now?", "plan", "plan", "explain"),      # teacher 3 Oct: the overstatement case (state verbs vs record)
    # lessons
    _q("what have you learned so far?", "learned", "lessons"), _q("how is the student doing compared to the teacher?", "learned", "lessons"),
    _q("what did your last lessons teach you?", "learned", "lessons"), _q("how good is the chooser yet?", "learned", "lessons"),
    _q("are you getting less dependent on Claude?", "learned", "lessons", "constraints"),
    # goals
    _q("what would you like to build next?", "goals", "goals"), _q("any goal proposals waiting for my approval?", "goals", "goals"),
    _q("what are your goals?", "goals", "goals"),
    # requests (dry) and actions (confirmation only)
    _q("I want you to make the planner faster", "request", "request"), _q("could you add a test for the model pool?", "request", "request"),
    _q("please pause for now", "action", "pause"), _q("resume work please", "action", "resume"),
    # explain a record
    _q("why did you reject CP0207?", "explain", "explain"), _q("what happened with CP0210?", "explain", "explain"),
    _q("explain the last cycle of K28", "explain", "explain"),
    # size / checklist
    _q("how many lines of code are you?", "size", "loc"), _q("how big is the creator package?", "size", "loc"),
    _q("how far along is the master checklist?", "size", "checklist"),
    # open questions about its own code and docs
    _q("what does pulseroute do?", "ask", "ask"), _q("how does the warm model pool work?", "ask", "ask"),
    _q("explain how the GPU pulse runner keeps private files at home", "ask", "ask"),
    _q("what is focus mode?", "ask", "ask"), _q("how do you decide which student gets a task?", "ask", "ask", "lessons"),
    _q("what does the auditor protect?", "ask", "ask"), _q("how is a claim judged before adoption?", "ask", "ask"),
    _q("what is the thinkbench?", "ask", "ask"), _q("how do you keep your start-up load small?", "ask", "ask"),
    _q("what is the gpuday export for?", "ask", "ask"),
    # small talk
    _q("hello!", "smalltalk", "smalltalk"), _q("who are you?", "smalltalk", "smalltalk"), _q("thanks, that helps", "smalltalk", "smalltalk"),
    _q("good evening Nupen", "smalltalk", "smalltalk"), _q("what can you do?", "smalltalk", "help", "smalltalk"),
    _q("do you enjoy your work?", "smalltalk", "smalltalk"),
]


def dry_request(c: CV.Ctx, s: dict[str, Any]) -> CV.Facts:
    """The eval's request handler: what h_request would say, without writing a proposal into the real records."""
    text = str(s.get("request") or s.get("question") or "").strip().rstrip(".!")
    return CV.Facts("request", [f"I would draft your request as a pending goal proposal: \"{text}\". Nothing is approved yet."])


def _answered(intent: str, answer: str) -> bool:
    return not (answer.startswith("I don't understand") or "I found nothing in my records" in answer)


def run_layer(root: Path, layer: str, questions: list[Q], voice: Optional[T.Voice] = None, mode: str = "hybrid") -> dict[str, Any]:
    conv = T.conversation(root, voice, mode="rules" if layer == "1" else mode, log=False, handlers={"request": dry_request})
    sess: T.Session = conv.talk_session  # type: ignore[attr-defined]
    rows = []
    for q in questions:
        sess.history.clear()                          # each question stands alone (follow-ups are tested in the unit tests)
        conv.waiting = None
        t0 = time.monotonic()
        ans = conv.reply(q.text)
        dt_s = time.monotonic() - t0
        p, f = conv.last["parsed"], conv.last["facts"]
        body = ans.split("\n[evidence:", 1)[0]
        bad = T.ungrounded(body, T.facts_text(f)) + T.state_claims(body, T.facts_text(f))
        rows.append({"q": q.text, "kind": q.kind, "expect": list(q.intents), "intent": p.intent, "intent_ok": p.intent in q.intents,
                     "grounded": not bad, "ungrounded": bad, "answered": _answered(p.intent, ans), "seconds": round(dt_s, 2),
                     "voice_s": sess.turn.get("voice_s", 0.0), "tokens_in": sess.turn.get("tokens_in", 0), "tokens_out": sess.turn.get("tokens_out", 0),
                     "understand_by": sess.turn.get("understand_by", "rules"), "spoken_by": sess.turn.get("spoken_by", "rules"),
                     "answer": ans[:600]})
    n = len(rows)
    lat = sorted(r["seconds"] for r in rows)
    by_kind: dict[str, list[bool]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["intent_ok"])
    return {"layer": layer, "voice": voice.name if voice else "", "voice_at": voice.where() if voice else "", "n": n,
            "intent_acc": round(sum(r["intent_ok"] for r in rows) / max(1, n), 3),
            "grounded": round(sum(r["grounded"] for r in rows) / max(1, n), 3),
            "answered": round(sum(r["answered"] for r in rows) / max(1, n), 3),
            "model_spoke": round(sum(r["spoken_by"] == "model" for r in rows) / max(1, n), 3),
            "latency_mean_s": round(statistics.fmean(lat), 2) if lat else 0.0, "latency_p50_s": lat[n // 2] if lat else 0.0,
            "latency_p90_s": lat[min(n - 1, int(n * 0.9))] if lat else 0.0,
            "tokens_in_mean": round(statistics.fmean(r["tokens_in"] for r in rows), 1) if rows else 0.0,
            "tokens_out_mean": round(statistics.fmean(r["tokens_out"] for r in rows), 1) if rows else 0.0,
            "intent_acc_by_kind": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_kind.items())}, "rows": rows}


def out_dir() -> Path:
    from creator import device as DEV
    return Path(DEV.runtime_dir()) / "talk" / "eval"


def run(root: Path, layers: list[str], limit: int = 0, ctx: int = 0, servers: Optional[int] = None, out: Optional[Path] = None,
        route: str = "") -> dict[str, Any]:
    qs = QUESTIONS[:limit] if limit else QUESTIONS
    res = []
    for layer in layers:
        voice = None
        if layer != "1":
            voice = T.Voice(T.resolve_model(layer), ctx=ctx or (8192 if layer == "1.7b" else 4096), servers=servers, route=route)
        try:
            res.append(run_layer(root, layer, qs, voice))
        finally:
            if voice is not None:
                voice.close()
        if voice is not None and voice.error:
            res[-1]["voice_error"] = voice.error
    rep = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "root": str(root), "questions": len(qs), "layers": res}
    d = Path(out) if out else out_dir()
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"talkeval_{time.strftime('%Y%m%dT%H%M%S')}_{'_'.join(layers)}.json"
    f.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    rep["file"] = str(f)
    return rep


def table(rep: dict[str, Any]) -> str:
    cols = ("layer", "voice", "n", "intent_acc", "grounded", "answered", "model_spoke", "latency_mean_s", "latency_p50_s", "latency_p90_s",
            "tokens_in_mean", "tokens_out_mean")
    lines = [" | ".join(cols)]
    for r in rep["layers"]:
        lines.append(" | ".join(str(r.get(c, "")) for c in cols) + (f"   (voice error: {r['voice_error']})" if r.get("voice_error") else ""))
        lines.append("   intent by kind: " + ", ".join(f"{k} {v}" for k, v in r["intent_acc_by_kind"].items()))
    return "\n".join(lines)
