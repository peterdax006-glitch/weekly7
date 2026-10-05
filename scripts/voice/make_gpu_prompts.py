"""Build the prompt/seed set for the Phase 2 GPU job p2.voice_persona_data.

A large model (27B dense or 30B-A3B MoE served with vLLM on the pod) is asked, per prompt, for
10 ORIGINAL owner->Nupen exchanges in a given category/situation. Prompts are derived from the
~84 categories in persona_data.py (weights), crossed with situation topics, owner moods, owner
line styles, 'sir' placement hints and reply length hints, so the output set is diverse.

Usage: python make_gpu_prompts.py OUT.jsonl [--prompts 5000] [--seed 5]
Each output line: {"id", "category", "messages": [system, user], "params": {...}}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import persona_data as pd  # noqa: E402

MOODS = ["calm", "cheerful", "tired", "hurried", "curious", "grumpy", "playful", "anxious", "focused", "distracted", "sarcastic", "sincere"]
OWNER_STYLES = [
    "a terse command of three words or fewer", "a casual spoken sentence", "a rambling half-sentence with a correction",
    "a polite full request", "a question with a typo-free but informal tone", "a statement that implies a request",
    "a one-word or two-word utterance", "a complaint", "a speech-to-text style line without punctuation",
]
SIR_HINTS = ["open the reply with 'sir' or a phrase containing it", "place 'sir' in the middle of the reply",
             "end the reply with ', sir.'", "use 'sir' once, anywhere natural"]
LEN_HINTS = ["one short sentence (under 12 words)", "one sentence of 12-22 words", "two short sentences", "one sentence plus a brief dry aside"]

SYSTEM = (
    "You are a dialogue writer producing ORIGINAL training data for a British butler-style AI assistant called Nupen. "
    "You never quote, paraphrase or allude to any film, television, book or game, and never use character, actor or "
    "franchise names. You output only the requested lines."
)


def render_shots(cat: str, rng: random.Random, k: int = 3) -> str:
    c = pd.CATEGORIES[cat]
    pairs = rng.sample(c["pairs"], min(k, len(c["pairs"])))
    out = []
    for o, r in pairs:
        names = set(re.findall(r"\{(\w+)\}", o + r))
        s = {n: rng.choice(pd.SLOTS[n]) for n in names}
        f = lambda t: re.sub(r"\{(\w+)\}", lambda m: s[m.group(1)], t)
        out.append(f"OWNER: {f(o)[:1].upper() + f(o)[1:]}\nNUPEN: {f(r)[:1].upper() + f(r)[1:]}")
    return "\n\n".join(out)


def build(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    names = list(pd.CATEGORIES)
    wsum = sum(c["weight"] for c in pd.CATEGORIES.values())
    weights = [pd.CATEGORIES[x]["weight"] / wsum for x in names]
    rules = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(pd.PERSONA_RULES))
    out = []
    for i in range(n):
        cat = rng.choices(names, weights)[0]
        c = pd.CATEGORIES[cat]
        topic, mood = rng.choice(pd.TOPICS), rng.choice(MOODS)
        style, sirh, lenh = rng.choice(OWNER_STYLES), rng.choice(SIR_HINTS), rng.choice(LEN_HINTS)
        user = (
            f"Write 10 NEW exchanges. Situation type: {c['intent']} (group: {c['group']}).\n"
            f"Owner mood: {mood}. Loose scene inspiration, use or ignore: {topic}. Owner line style: {style}.\n"
            f"Nupen reply length: {lenh}. For this batch: {sirh}.\n\n"
            f"Persona rules:\n{rules}\n\n"
            f"Style examples (never copy them; vary the wording, the details and the jokes):\n{render_shots(cat, rng)}\n\n"
            "Every exchange must differ in content from the others. Output exactly 10 blocks in this format and nothing else:\n"
            "OWNER: <owner line>\nNUPEN: <reply>\n"
        )
        out.append({
            "id": f"pp{i:06d}", "category": cat,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "params": {"temperature": round(rng.uniform(0.8, 1.05), 2), "top_p": 0.95, "max_tokens": 700, "seed": rng.randrange(1 << 30)},
        })
    return out


def spec(n_prompts: int) -> dict:
    import persona_filter as pf
    return {
        "name": "voice_persona_data",
        "kind": "data_generation (pod GPU, no training)",
        "purpose": "20,000-50,000 ORIGINAL owner->Nupen persona dialogue pairs for p2.voice_persona",
        "generator": {
            "primary": "Qwen3-30B-A3B-Instruct (vLLM, fp8/AWQ-int4; MoE, ~3B active so fast)",
            "fallback": "a 27B instruct model (bf16, 80 GB class) if the MoE is unavailable",
            "per_prompt": "10 exchanges, max_tokens 700, temperature 0.8-1.05, top_p 0.95, seeded",
            "prompts": n_prompts, "raw_pairs_expected": n_prompts * 10,
            "kept_expected": "55-70% after checks; 25-35k after near-dup + balancing (target 30000, hard max 50000)",
        },
        "persona_system_prompt": pd.PERSONA_SYSTEM,
        "persona_rules": pd.PERSONA_RULES,
        "length_limits": {"reply_max_sentences": pf.MAX_SENTENCES, "reply_max_words": pf.MAX_WORDS_REPLY, "owner_max_words": pf.MAX_WORDS_OWNER},
        "hard_rules": ["no film/TV/book lines or quotes, no character or actor or franchise names (regex list in persona_data.BANNED_NAME_PATTERNS)",
                       "no real actor audio or voice reference anywhere in the voice pipeline", "'sir' present in every reply (max twice)",
                       "no exclamation marks, no markdown, no emoji, no URLs", "no unsafe content (persona_filter.UNSAFE_HARD / UNSAFE_REPLY_EXTRA)"],
        "categories": {k: {"group": v["group"], "weight": v["weight"], "intent": v["intent"], "curated_pairs": len(v["pairs"])}
                       for k, v in pd.CATEGORIES.items()},
        "group_weights": {g: round(sum(v["weight"] for v in pd.CATEGORIES.values() if v["group"] == g), 2)
                          for g in sorted({v["group"] for v in pd.CATEGORIES.values()})},
        "situation_topics": len(pd.TOPICS), "owner_moods": MOODS, "owner_styles": OWNER_STYLES, "sir_hints": SIR_HINTS, "length_hints": LEN_HINTS,
        "filter_pipeline": ["parse OWNER/NUPEN blocks", "shape + persona checks", "banned-name / quote / unsafe checks",
                            "exact dedupe", "near-dup MinHash/LSH (3-gram, Jaccard>=0.7) over owner+reply",
                            "diversity balancing: category quota by weight x1.25, opener cap 1.2%, 'sir' position cap 55%",
                            "3% eval split", "report json (rejections by reason, per-category counts)"],
        "pilot": "first run 50 prompts; require >=50% pass rate and zero banned-name hits before the full run",
        "output": "persona_gpu.jsonl (chat messages: system/user/assistant, kind=persona), merged with the 1.5k PC seed by the trainer",
        "code": ["scripts/voice/persona_data.py", "scripts/voice/make_gpu_prompts.py", "scripts/voice/persona_filter.py"],
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prompts", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--spec", default="", help="also write the generation spec JSON here")
    a = ap.parse_args(argv)
    if a.spec:
        Path(a.spec).write_text(json.dumps(spec(a.prompts), indent=1), encoding="utf-8")
    rows = build(a.prompts, a.seed)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "prompts,", len(pd.CATEGORIES), "categories")


if __name__ == "__main__":
    main(sys.argv[1:])
