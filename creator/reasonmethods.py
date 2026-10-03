"""REASONING METHODS for the local-model judgment drills (owner, 2 Oct 2026 night: Nupen must think and reason much better).

Four methods the judgment strategy search (creator.judgment) can pick by itself, each scored on the SAME held-out subjects against the plain prompt
and the statistical predictor:

  retrieve k   RETRIEVAL MEMORY. An in-memory BM25 index over everything Nupen has on record (teacher lessons with their reasoning, kernel cycles with
               their reasons, resolved git commits and packages, plan explanations). A judgment prompt shows the k most similar PAST cases with their
               outcomes. TIME-ORDERED: a record is visible only if its outcome was known strictly before the case was created, so the case's own
               lesson/cycle (outcome known later) can never leak.
  structured   a fixed reasoning template (question, evidence for / against taken from the retrieved cases, base rate, adjustment, probability).
  samples n    SELF-CONSISTENCY: n samples at temperature > 0, aggregated by the mean probability (the majority vote is scored as well); the record
               keeps every sample so the gain per extra sample can be measured against its cost.
  traces k     TEACHER TRACES: the teacher's own reasoning (solver claude / claude-replay in lessons.jsonl) as worked examples, picked by similarity,
               only from lessons whose outcome was known before the case.
  knn k        the model-free control: the outcome rate of the k nearest past cases, shrunk toward the base rate. It costs no model call; if it beats
               the model the search will say so.

Nothing here depends on WHICH model answers: the functions take messages and a `chat` callable, so any model path (the default LocalModel or a
stronger one) works. Loaded on demand through creator.registry (name `reasonmethods`)."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

TEACHERS = ("claude", "claude-replay")
STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was will with will".split())
TOKEN = re.compile(r"[a-z][a-z0-9_]+|[0-9]{3,}")
K1, B = 1.4, 0.75


@dataclass
class Rec:
    """One thing Nupen remembers. `known` is when its outcome (or, for notes, the note itself) became known; a case may see it only if known < created."""
    known: float
    kind: str                       # lesson | cycle | case | plan
    topic: str                      # verdict | git_fixed | "" (context only)
    text: str                       # what was set up (shown, and indexed)
    outcome: str = ""               # what happened, as words
    y: Optional[int] = None         # the topic's event (1 = ADOPTED for verdict, 1 = later fixed for git_fixed)
    teacher: str = ""               # the solver name when a teacher wrote reasoning
    reasoning: str = ""             # the teacher's reasoning, as written before the outcome


def parse_time(s: Any) -> Optional[float]:
    """ISO text -> epoch seconds. A zone-less stamp is the machine's local time (how the lessons file writes it)."""
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.timestamp() if d.tzinfo else d.astimezone().timestamp()


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class Index:
    """BM25 over Rec texts, searchable 'as of' a time and restricted to kinds/topic."""

    def __init__(self, recs: Sequence[Rec]) -> None:
        self.recs = sorted(recs, key=lambda r: r.known)
        self.post: dict[str, list[tuple[int, int]]] = {}
        self.len: list[int] = []
        for i, r in enumerate(self.recs):
            toks = tokens(r.text + " " + (r.reasoning[:300] if r.reasoning else ""))
            self.len.append(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.post.setdefault(t, []).append((i, c))
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0

    def __len__(self) -> int:
        return len(self.recs)

    def search(self, query: str, before: float, k: int = 4, kinds: Optional[Sequence[str]] = None, topic: Optional[str] = None,
               teacher: bool = False) -> list[Rec]:
        """The k most similar records whose `known` is strictly before `before` (ties in score: the more recent first)."""
        n = sum(1 for r in self.recs if r.known < before)               # records are time-sorted: the visible ones are a prefix
        if not n or not k:
            return []
        lim = next((i for i, r in enumerate(self.recs) if r.known >= before), len(self.recs))
        score: dict[int, float] = {}
        for t in set(tokens(query)):
            pl = self.post.get(t)
            if not pl:
                continue
            df = sum(1 for i, _c in pl if i < lim)
            if not df:
                continue
            idf = math.log(1 + (lim - df + 0.5) / (df + 0.5))
            for i, c in pl:
                if i >= lim:
                    break
                score[i] = score.get(i, 0.0) + idf * c * (K1 + 1) / (c + K1 * (1 - B + B * self.len[i] / self.avg))
        ok = []
        for i, s in score.items():
            r = self.recs[i]
            if kinds and r.kind not in kinds:
                continue
            if topic is not None and r.topic != topic:
                continue
            if teacher and not (r.teacher and r.reasoning):
                continue
            ok.append((s, i))
        ok.sort(key=lambda x: (-x[0], -x[1]))
        return [self.recs[i] for _s, i in ok[:k]]


def _jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    except OSError:
        pass
    return out


def build_corpus(state: Path, cases: Sequence[Any] = ()) -> list[Rec]:
    """Everything on record under `state` (read-only) plus the resolved judgment `cases` (verdict packages and git commits, each known when it resolved)."""
    state = Path(state)
    recs: list[Rec] = []
    rows = _jsonl(state / "lessons.jsonl")
    out_row: dict[str, dict[str, Any]] = {r["lesson_id"]: r for r in rows if r.get("event") == "outcome" and r.get("lesson_id")}
    for r in rows:
        o = out_row.get(r.get("lesson_id", ""))
        if r.get("event") or not o:
            continue
        known = parse_time(o.get("at"))
        if known is None:
            continue
        adopted = bool(o.get("adopted"))
        recs.append(Rec(known, "lesson", "verdict",
                        f"{r.get('task_kind', '')} package for {r.get('component', '')}: {str(r.get('objective', ''))[:200]}",
                        "ADOPTED" if adopted else f"not adopted ({str(o.get('verdict', ''))[:110]})", int(adopted),
                        r.get("solver", "") if r.get("solver") in TEACHERS else "", str(r.get("reasoning", ""))[:600] if r.get("solver") in TEACHERS else ""))
    kl = {k.get("package"): k for k in _jsonl(state / "kernel_log.jsonl") if k.get("package")}
    for c in cases:
        k = kl.get(c.subject) if c.topic == "verdict" else None
        why = f" ({str(k.get('reason', ''))[:110]})" if k and k.get("reason") else ""
        word = ("ADOPTED" if c.y else "not adopted") if c.topic == "verdict" else ("later fixed" if c.y else "never fixed")
        recs.append(Rec(c.resolved, "cycle" if k else "case", c.topic, c.text, word + why, c.y))
    seen: set[tuple[str, str]] = set()
    for r in _jsonl(state / "plan_explanations.jsonl"):
        key = (str(r.get("component", "")), str(r.get("why", ""))[:80])
        t = parse_time(r.get("at"))
        if t is None or key in seen:
            continue
        seen.add(key)
        recs.append(Rec(t, "plan", "", f"plan note {r.get('component', '')} {r.get('step', '')}: {str(r.get('why', ''))[:200]}", f"chosen={r.get('chosen')}"))
    return recs


def fmt(r: Rec, width: int = 260) -> str:
    return f"[{r.kind}] {r.text[:width]} -> {r.outcome}"


# ------------------------------------------------------------------------------------------------ the methods
STRUCTURED = ("Reason with this exact template, briefly (one short line each):\n"
              "QUESTION: restate what is asked.\nFOR: evidence from the similar past cases that makes the event likely.\n"
              "AGAINST: evidence that makes it unlikely.\nBASE RATE: the event rate in the record.\nADJUST: why this case is above or below the base rate.\n"
              "Then a final line 'PROBABILITY: 0.xx'.")


def retrieved_block(index: Index, strat: dict[str, Any], c: Any) -> str:
    """The k nearest past cases with outcomes, as one prompt block ('' when none or not asked)."""
    k = int(strat.get("retrieve", 0))
    if not k or index is None:
        return ""
    kinds = None if strat.get("plans") else ("lesson", "cycle", "case")
    got = index.search(c.text, c.created, k, kinds=kinds, topic=c.topic) if kinds else index.search(c.text, c.created, k, topic=None)
    got = [r for r in got if r.topic in (c.topic, "")]
    return ("Similar past cases (all resolved before this one):\n" + "\n".join("- " + fmt(r) for r in got) + "\n") if got else ""


def trace_messages(index: Index, strat: dict[str, Any], c: Any) -> list[dict[str, str]]:
    """Teacher traces as worked examples (user = the setup, assistant = the teacher's reasoning, what happened, the probability it implies)."""
    k = int(strat.get("traces", 0))
    if not k or index is None or c.topic != "verdict":
        return []
    out: list[dict[str, str]] = []
    for r in reversed(index.search(c.text, c.created, k, kinds=("lesson",), topic="verdict", teacher=True)):
        out.append({"role": "user", "content": f"{r.text[:260]} Will the kernel ADOPT its result?"})
        out.append({"role": "assistant", "content": f"Teacher ({r.teacher}) reasoned: {' '.join(r.reasoning.split())[:300]} Resolved: {r.outcome[:60]}. "
                                                    f"PROBABILITY: {'0.85' if r.y else '0.15'}"})
    return out


def knn_probability(index: Index, c: Any, k: int, base: float) -> Optional[float]:
    """Model-free control: outcome rate of the k nearest same-topic past cases, shrunk toward `base` with weight 2."""
    got = [r for r in index.search(c.text, c.created, k * 3, kinds=("lesson", "cycle", "case"), topic=c.topic) if r.y is not None][:k]
    if not got:
        return None
    return min(0.97, max(0.03, (sum(r.y or 0 for r in got) + 2 * base) / (len(got) + 2)))


def aggregate(ps: Sequence[float]) -> Optional[float]:
    """Self-consistency: the mean of the samples' probabilities."""
    return (sum(ps) / len(ps)) if ps else None


def majority(ps: Sequence[float]) -> Optional[float]:
    """Self-consistency by vote: the share of samples that say 'more likely than not', pulled off the extremes."""
    return (0.1 + 0.8 * sum(1 for p in ps if p >= 0.5) / len(ps)) if ps else None


def sample(chat: Callable[..., str], msgs: list[dict[str, str]], parse: Callable[[str], Optional[float]], n: int, max_tokens: int,
           ) -> tuple[list[float], str, int]:
    """n samples (temperature 0.2 for one, 0.7 for several; distinct seeds) -> (parsed probabilities, first reply, approximate tokens used)."""
    ps: list[float] = []
    first, toks = "", sum(len(m["content"]) for m in msgs) // 4
    for i in range(max(1, n)):
        reply = chat(msgs, max_tokens=max_tokens, temperature=0.2 if n <= 1 else 0.7, seed=i, timeout=300.0)
        first = first or reply
        toks += (sum(len(m["content"]) for m in msgs) + len(reply)) // 4 if i else len(reply) // 4
        p = parse(reply)
        if p is not None:
            ps.append(p)
    return ps, first, toks


def paired_gain(base: Sequence[float], other: Sequence[float], ys: Sequence[int]) -> list[float]:
    """Mean Brier gain of `other` over `base` on the same items (positive = other is better) and its 95% CI [lo, hi]."""
    d = [(b - y) ** 2 - (o - y) ** 2 for b, o, y in zip(base, other, ys)]
    n = len(d)
    if n < 2:
        return [round(sum(d) / n, 4) if n else 0.0, 0.0, 0.0]
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return [round(m, 4), round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]
