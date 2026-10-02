"""Mixed training stream: TinyStories tokens + dialogue text (User:/Nupen: turns with the end-of-turn marker). Torch-free.

Dialogue pool = teacher train-split dialogues + train-template practice items (gen_stage3/4) + terminal conversation exports.
Held-out dialogues and templates never enter; `build_pool` asserts disjointness exactly as dialogue.split_overlap does.
The tokenizer is byte-level BPE, so the `<|eot|>` turn marker is encoded as ordinary bytes: no tokenizer change, old checkpoints stay comparable.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from creator.lm import dialogue as D

DEFAULT_SHARE = 0.20
GEN_PER_STAGE = 400
CONVERSATIONS = Path(__file__).resolve().parent.parent.parent / "state" / "creator" / "conversations.jsonl"


def load_conversations(path: Path | None = None) -> list[dict[str, Any]]:
    """Terminal conversation exports as train-split items. Accepts {"dialogue": [{role,text}..]} or {"user":..,"nupen":..} lines."""
    path = path or CONVERSATIONS
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        turns = d.get("dialogue")
        if not turns and d.get("user") and d.get("nupen"):
            turns = [{"role": "user", "text": str(d["user"])}, {"role": "nupen", "text": str(d["nupen"])}]
        if isinstance(turns, list) and turns and all(isinstance(t, dict) and t.get("role") in D.ROLE_NAMES and str(t.get("text", "")).strip() for t in turns):
            out.append({"stage": 5, "dialogue": turns, "split": "train", "source": "terminal", "entities": []})
    return out


def build_pool(teacher_path: Path | None = None, conv_path: Path | None = None, gen: int = GEN_PER_STAGE, seed: int = 0) -> list[dict[str, Any]]:
    """Train-split items only. Raises if any held-out entity or dialogue text leaks into it."""
    items = D.load_items(teacher_path or D.TEACHER_FILE, generated=0)
    held = [i for i in items if i["split"] == "heldout"] + D.gen_stage3(40, 0, "heldout") + D.gen_stage4(40, 0, "heldout")
    train = [i for i in items if i["split"] == "train"] + D.gen_stage3(gen, seed, "train") + D.gen_stage4(gen, seed, "train") + load_conversations(conv_path)
    held_texts = {D.to_training_text(i) for i in held}
    train = [i for i in train if D.to_training_text(i) not in held_texts]
    bad = D.split_overlap([i for i in train if i["source"] != "terminal"] + held)
    if bad:
        raise ValueError(f"held-out entities leak into the dialogue training pool: {bad}")
    return train


def pool_texts(pool: list[dict[str, Any]]) -> list[str]:
    return [D.to_training_text(i) for i in pool]


def encode_stream(tok: Any, texts: list[str], seed: int, min_tokens: int) -> tuple[list[int], list[int]]:
    """Shuffled, repeated pool encoded to ids (>= min_tokens, with room for a context window past every start). Returns (ids, dialogue start offsets)."""
    rng = random.Random(seed)
    ids: list[int] = []
    starts: list[int] = []
    while len(ids) < min_tokens and texts:
        order = list(texts)
        rng.shuffle(order)
        for t in order:
            starts.append(len(ids))
            ids += tok.encode(t)
    return ids, starts[: max(1, len(starts) - 20)]


def sample_sources(n: int, share: float, rng: Any) -> list[bool]:
    """For n sequences: True = draw from the dialogue stream. Exactly round(n*share) of them, in random order."""
    k = round(n * share)
    flags = [True] * k + [False] * (n - k)
    rng.shuffle(flags)
    return flags


def mix_offsets(n: int, share: float, rng: Any, story_len: int, dlg_starts: list[int], ctx: int) -> list[tuple[bool, int]]:
    """(is_dialogue, start offset) per sequence. Dialogue offsets are aligned to the start of a dialogue."""
    out: list[tuple[bool, int]] = []
    for is_d in sample_sources(n, share, rng):
        out.append((True, int(dlg_starts[int(rng.integers(0, len(dlg_starts)))])) if is_d else (False, int(rng.integers(0, story_len - ctx - 1))))
    return out
