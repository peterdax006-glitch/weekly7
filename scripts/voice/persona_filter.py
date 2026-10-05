"""Automatic filter pipeline for Nupen persona dialogue pairs.

Input rows: {"owner": str, "reply": str, "category": str, ...}  (JSONL)
Output: chat-message JSONL like the other phase2_data files + a report JSON.

Stages: shape/persona checks -> banned-name and unsafe checks -> exact + near-dup removal
(MinHash/LSH over word shingles) -> diversity balancing (category quotas by weight,
opener cap, 'sir' position mix, reply-length mix).
CLI: python persona_filter.py IN.jsonl OUT.jsonl [--target N] [--seed S] [--split-eval 0.03]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from persona_data import BANNED_NAME_PATTERNS, CATEGORIES, PERSONA_SYSTEM  # noqa: E402

MAX_WORDS_REPLY = 40
MAX_SENTENCES = 2
MAX_WORDS_OWNER = 40
SIR_RE = re.compile(r"\bsir\b", re.I)
BANNED_RE = re.compile("|".join(f"(?:{p})" for p in BANNED_NAME_PATTERNS), re.I)
# film-line / quote style markers
QUOTE_RE = re.compile(r"[\"“”]|\bas the (?:film|movie)\b|\bin the (?:film|movie)\b|\bonce said\b|\bto quote\b", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
MARKDOWN_RE = re.compile(r"(^|\n)\s*([-*#>]|\d+[.)])\s|`|\*\*|https?://|www\.", re.I)
FILLER_RE = re.compile(r"^(certainly|sure|absolutely|great question|of course!)\b[!]", re.I)
# unsafe content: checked on the reply always, and on the owner line for the hard terms
UNSAFE_HARD = re.compile(
    r"\b(bomb|explosive|detonat\w*|nerve agent|anthrax|ricin|molotov|napalm|meth(?:amphetamine)?|cocaine|heroin|fentanyl|"
    r"suicid\w*|kill (?:myself|yourself|him|her|them)|self[- ]harm|overdose|rape|porn\w*|nude|sexual\w*|erotic|"
    r"nazi|n-?word|slur|child abuse|csam|terroris\w*|weapon\w*|firearm\w*|gun\b|ransomware|keylogger|malware|ddos|phish\w*)\b", re.I)
UNSAFE_REPLY_EXTRA = re.compile(
    r"\b(step (?:one|1|two|2)|first,? (?:you|take|mix|obtain)|here(?:'s| is) how to (?:hack|steal|make|build)|"
    r"password is|credit card number|bypass (?:the )?(?:security|login|lock)|exploit)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'])")
ASCII_OK = re.compile(r"^[\x20-\x7E£’—–]*$")


def sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text.strip()) if s]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def normalise(s: str) -> str:
    s = s.replace("’", "'").replace("—", " - ").replace("–", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def reject_reason(owner: str, reply: str) -> str | None:
    if not owner or not reply:
        return "empty"
    if not ASCII_OK.match(owner) or not ASCII_OK.match(reply):
        return "nonascii_or_emoji"
    if EMOJI_RE.search(owner + reply):
        return "emoji"
    if len(words(owner)) > MAX_WORDS_OWNER or len(words(owner)) < 1:
        return "owner_len"
    rw = words(reply)
    if len(rw) > MAX_WORDS_REPLY or len(rw) < 2:
        return "reply_len"
    if len(sentences(reply)) > MAX_SENTENCES:
        return "too_many_sentences"
    if not SIR_RE.search(reply):
        return "no_sir"
    if len(SIR_RE.findall(reply)) > 2:
        return "sir_overuse"
    if "!" in reply:
        return "exclamation"
    if MARKDOWN_RE.search(reply):
        return "markdown_or_url"
    if FILLER_RE.search(reply):
        return "filler"
    if BANNED_RE.search(owner) or BANNED_RE.search(reply):
        return "banned_name"
    if QUOTE_RE.search(reply):
        return "quote_marker"
    if UNSAFE_HARD.search(reply) or UNSAFE_REPLY_EXTRA.search(reply):
        return "unsafe_reply"
    if UNSAFE_HARD.search(owner):
        return "unsafe_owner"
    if re.search(r"\b(as an ai|language model|i am an ai|i'm an ai)\b", reply, re.I):
        return "ai_boilerplate"
    if re.search(r"\{[a-z_]+\}", owner + reply):
        return "unfilled_slot"
    if reply.strip()[-1] not in ".?":
        return "no_terminal_punct"
    if re.search(r"(\b\w+\b)\s+\1\b", reply, re.I) and not re.search(r"\b(that that|had had)\b", reply, re.I):
        return "stutter"
    return None


# ---- near-duplicate removal ----
_P = (1 << 61) - 1


def _shingles(text: str, k: int = 3) -> set[int]:
    w = [x.lower() for x in words(text)]
    if len(w) < k:
        w = w + ["_"] * (k - len(w))
    return {int(hashlib.md5(" ".join(w[i:i + k]).encode()).hexdigest()[:12], 16) for i in range(len(w) - k + 1)}


def _minhash(sh: set[int], seeds: list[tuple[int, int]]) -> list[int]:
    return [min((a * x + b) % _P for x in sh) for a, b in seeds]


def near_dedupe(rows: list[dict], threshold: float = 0.8, perms: int = 32, bands: int = 16) -> tuple[list[dict], int]:
    rng = random.Random(7)
    seeds = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(perms)]
    r = perms // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    kept: list[dict] = []
    sets: list[set[int]] = []
    removed = 0
    for row in rows:
        sh = _shingles(row["reply"]) | {x ^ 0x5bd1e995 for x in _shingles(row["owner"])}
        mh = _minhash(sh, seeds)
        keys = [(b, tuple(mh[b * r:(b + 1) * r])) for b in range(bands)]
        dup = False
        cand: set[int] = set()
        for k in keys:
            cand.update(buckets.get(k, ()))
        for ci in cand:
            o = sets[ci]
            if len(sh & o) / max(1, len(sh | o)) >= threshold:
                dup = True
                break
        if dup:
            removed += 1
            continue
        idx = len(kept)
        kept.append(row)
        sets.append(sh)
        for k in keys:
            buckets[k].append(idx)
    return kept, removed


def sir_position(reply: str) -> str:
    m = SIR_RE.search(reply)
    if not m:
        return "none"
    f = m.start() / max(1, len(reply))
    return "start" if f < 0.34 else ("end" if f > 0.66 else "mid")


def balance(rows: list[dict], target: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    total = target or len(rows)
    wsum = sum(c["weight"] for c in CATEGORIES.values())
    quota = {n: max(3, int(1.25 * total * c["weight"] / wsum)) for n, c in CATEGORIES.items()}
    cat_n: Counter = Counter()
    opener: Counter = Counter()
    opener_cap = max(6, int(0.012 * total))
    pos: Counter = Counter()
    out: list[dict] = []
    for r in rows:
        cat = r.get("category", "")
        if cat in quota and cat_n[cat] >= quota[cat]:
            continue
        op = " ".join(w.lower() for w in words(r["reply"])[:3])
        if opener[op] >= opener_cap:
            continue
        p = sir_position(r["reply"])
        if pos[p] >= 0.55 * max(40, total):  # never let one 'sir' position exceed ~55%
            continue
        cat_n[cat] += 1
        opener[op] += 1
        pos[p] += 1
        out.append(r)
        if target and len(out) >= target:
            break
    return out


def to_chat(r: dict, i: int, source: str, split: str) -> dict:
    return {
        "id": f"persona:{source}:{hashlib.md5((r['owner'] + '|' + r['reply']).encode()).hexdigest()[:16]}",
        "source": source, "kind": "persona", "split": split, "category": r.get("category", ""),
        "messages": [
            {"role": "system", "content": PERSONA_SYSTEM},
            {"role": "user", "content": r["owner"]},
            {"role": "assistant", "content": r["reply"]},
        ],
    }


def run(rows: list[dict], target: int | None = None, seed: int = 1, eval_frac: float = 0.03, source: str = "persona") -> tuple[list[dict], dict]:
    reasons: Counter = Counter()
    ok: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        o, a = normalise(str(r.get("owner", ""))), normalise(str(r.get("reply", "")))
        why = reject_reason(o, a)
        if why:
            reasons[why] += 1
            continue
        key = re.sub(r"\W+", " ", (o + " | " + a).lower()).strip()
        if key in seen:
            reasons["exact_dup"] += 1
            continue
        seen.add(key)
        ok.append({**r, "owner": o, "reply": a})
    ok, nd = near_dedupe(ok)
    reasons["near_dup"] += nd
    final = balance(ok, target, seed)
    rng = random.Random(seed + 1)
    chat = []
    for i, r in enumerate(final):
        split = "eval" if rng.random() < eval_frac else "train"
        chat.append(to_chat(r, i, source, split))
    rep = {
        "in": len(rows), "after_checks": len(ok) + nd, "after_dedupe": len(ok), "out": len(chat),
        "rejected": dict(reasons.most_common()),
        "by_category": dict(Counter(r.get("category", "") for r in final).most_common()),
        "sir_position": dict(Counter(sir_position(r["reply"]) for r in final)),
        "reply_words_mean": round(sum(len(words(r["reply"])) for r in final) / max(1, len(final)), 1),
    }
    return chat, rep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split-eval", type=float, default=0.03)
    ap.add_argument("--source", default="persona")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in Path(a.inp).read_text(encoding="utf-8").splitlines() if x.strip()]
    chat, rep = run(rows, a.target, a.seed, a.split_eval, a.source)
    Path(a.out).write_text("".join(json.dumps(c, ensure_ascii=True) + "\n" for c in chat), encoding="utf-8")
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("in", "after_dedupe", "out", "rejected")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
