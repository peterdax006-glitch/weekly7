"""Nupen's language curriculum: the teacher (Claude) teaches, Nupen (a model trained from scratch in creator/lm/) is the student.

Format: `User: ...<|eot|>` / `Nupen: ...<|eot|>` turns. Five stages of two-way English, each with a measurable test on HELD-OUT items;
a stage's training data unlocks only when every earlier stage has passed. Stage 5 (open conversation) is graded by the teacher.
The model is addressed through a tiny protocol (`generate`, `bits_per_byte`) so tests use a fake.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from creator import device
from creator.lm.dialogue_seed import seed_items

EOT = "<|eot|>"
ROLE_NAMES = {"user": "User", "nupen": "Nupen"}

RUNTIME = device.runtime_dir()
TEACHER_FILE = RUNTIME / "lmdata" / "teacher_dialogues.jsonl"
TRANSCRIPTS_FILE = RUNTIME / "lmdata" / "chat_transcripts.jsonl"
STAGES_FILE = RUNTIME / "lmckpt" / "stages.json"


class LMLike(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8) -> str: ...
    def bits_per_byte(self, texts: list[str]) -> float: ...


# ----------------------------------------------------------------------------- format

def format_turn(role: str, text: str) -> str:
    return f"{ROLE_NAMES[role]}: {text.strip()}{EOT}\n"


def format_dialogue(turns: list[dict[str, str]]) -> str:
    return "".join(format_turn(t["role"], t["text"]) for t in turns)


def to_training_text(dialogue: list[dict[str, str]] | dict[str, Any]) -> str:
    """The exact string the trainer sees for one dialogue (an item dict or its turn list)."""
    turns = dialogue["dialogue"] if isinstance(dialogue, dict) else dialogue
    return format_dialogue(turns)


_TURN_RE = re.compile(r"(User|Nupen): (.*?)" + re.escape(EOT), re.S)


def parse_dialogue(text: str) -> list[dict[str, str]]:
    inv = {v: k for k, v in ROLE_NAMES.items()}
    return [{"role": inv[m.group(1)], "text": m.group(2)} for m in _TURN_RE.finditer(text)]


def prompt_for(turns: list[dict[str, str]]) -> str:
    """Conversation so far, ending with the cue for Nupen's reply."""
    return format_dialogue(turns) + "Nupen:"


def extract_reply(raw: str, prompt: str = "") -> tuple[str, bool]:
    """(reply text, turn_ok). turn_ok = it ended at the end-of-turn marker and never wrote a speaker label."""
    if prompt and raw.startswith(prompt):
        raw = raw[len(prompt):]
    stopped = EOT in raw
    reply = raw.split(EOT, 1)[0]
    leaked = re.search(r"(^|\n|\s)(User|Nupen):", reply) is not None
    for label in ("\nUser:", "\nNupen:", " User:", " Nupen:"):
        reply = reply.split(label, 1)[0]
    reply = reply.strip()
    return reply, bool(stopped and reply and not leaked)


def check_turn_taking(raw: str, prompt: str = "") -> bool:
    return extract_reply(raw, prompt)[1]


def answer_matches(reply: str, expect: list[str]) -> bool:
    """Whole-word, case-insensitive containment of any acceptable answer."""
    low = reply.lower()
    return any(re.search(r"(?<!\w)" + re.escape(e.lower()) + r"(?!\w)", low) for e in expect)


# ----------------------------------------------------------------------------- generated practice items

COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown"]
NUMWORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
POOLS: dict[str, dict[str, list[str]]] = {
    "train": {
        "name": ["Ada", "Bert", "Carl", "Dev", "Elsa", "Finn", "Gus", "Hilda", "Iris", "Jon", "Kira", "Leo", "Mona", "Ned", "Olga", "Pablo", "Rosa", "Stan", "Tina", "Walt"],
        "city": ["Paris", "Cairo", "Oslo", "Delhi", "Seoul", "Tokyo", "Dublin", "Vienna"],
        "animal": ["cat", "rabbit", "duck", "frog", "horse", "mouse"],
        "obj": ["hat", "ball", "book", "cup", "bag", "box"],
        "food": ["apple", "bread", "rice", "soup", "cake"],
    },
    "heldout": {
        "name": ["Zoe", "Yusuf", "Quentin", "Wilma", "Vera", "Ximena", "Bruno", "Cyrus", "Dora", "Felix"],
        "city": ["Madrid", "Nairobi", "Prague", "Hanoi", "Quito", "Lisbon"],
        "animal": ["parrot", "pony", "lizard", "goose"],
        "obj": ["kite", "lamp", "scarf", "drum"],
        "food": ["pear", "cheese", "noodles"],
    },
}
HELD_ADD_PAIRS = {(2, 7), (3, 5), (4, 4), (1, 8), (6, 2), (5, 3), (7, 1), (2, 6), (3, 3), (6, 3), (4, 2), (5, 4)}
TRAIN_PHRASES = ["the sky is high", "I like to read", "the road is long", "we eat at noon", "the door is open"]
HELD_PHRASES = ["birds can sing", "the tea is hot", "snow falls in winter", "my shoes are wet"]


def _d(*turns: str) -> list[dict[str, str]]:
    return [{"role": "user" if i % 2 == 0 else "nupen", "text": t} for i, t in enumerate(turns)]


def _item(stage: int, split: str, turns: list[dict[str, str]], expect: list[str], ents: list[str]) -> dict[str, Any]:
    return {"stage": stage, "dialogue": turns, "split": split, "source": "template", "entities": ents, "expect": expect}


def gen_stage3(n: int, seed: int, split: str) -> list[dict[str, Any]]:
    """Context-QA practice: names, colours, numbers, places stated earlier. Templates are disjoint between splits as well as entities."""
    rng = random.Random(f"s3-{seed}-{split}")
    p = POOLS[split]
    out: list[dict[str, Any]] = []
    for _ in range(n):
        nm, city, an, ob, fd = (rng.choice(p[k]) for k in ("name", "city", "animal", "obj", "food"))
        col, num = rng.choice(COLOURS), rng.randint(2, 9)
        kind = rng.randrange(3) if split == "train" else 3 + rng.randrange(2)
        if kind == 0:
            t = _d(f"My name is {nm}. I like {fd}.", f"Nice to meet you, {nm}.", "What do I like?", f"You like {fd}.")
            out.append(_item(3, split, t, [fd], [nm, fd]))
        elif kind == 1:
            t = _d(f"I live in {city}. My {an} is {col}.", f"That sounds nice.", "Where do I live?", f"You live in {city}.")
            out.append(_item(3, split, t, [city], [city, an]))
        elif kind == 2:
            t = _d(f"{nm} has {NUMWORDS[num]} {ob}s.", "I see.", f"How many {ob}s does {nm} have?", f"{nm} has {NUMWORDS[num]}.")
            out.append(_item(3, split, t, [NUMWORDS[num], str(num)], [nm, ob]))
        elif kind == 3:
            t = _d(f"Hello, I am {nm}. I have a {col} {ob}.", f"Hello, {nm}.", f"What colour is my {ob}?", f"Your {ob} is {col}.")
            out.append(_item(3, split, t, [col], [nm, ob]))
        else:
            t = _d(f"Today {nm} went to {city} with a {an}.", "What a trip.", f"Where did {nm} go?", f"{nm} went to {city}.")
            out.append(_item(3, split, t, [city], [nm, city, an]))
    return out


def gen_stage4(n: int, seed: int, split: str) -> list[dict[str, Any]]:
    """Instruction practice. Held-out uses other phrases, names, a different operation and a disjoint set of addition pairs."""
    rng = random.Random(f"s4-{seed}-{split}")
    p = POOLS[split]
    out: list[dict[str, Any]] = []
    for _ in range(n):
        kind = rng.randrange(3) if split == "train" else 3 + rng.randrange(3)
        if kind == 0:
            ph = rng.choice(TRAIN_PHRASES)
            out.append(_item(4, split, _d(f"Repeat after me: {ph}.", ph[0].upper() + ph[1:] + "."), [ph.lower()], [ph]))
        elif kind == 1:
            a, b = rng.randint(1, 8), rng.randint(1, 8)
            while (a, b) in HELD_ADD_PAIRS or a + b > 9:
                a, b = rng.randint(1, 8), rng.randint(1, 8)
            out.append(_item(4, split, _d(f"What is {a}+{b}?", f"{a}+{b} is {a + b}."), [str(a + b)], [f"{a}+{b}"]))
        elif kind == 2:
            nm = rng.choice(p["name"])
            out.append(_item(4, split, _d(f"Say hello to {nm}.", f"Hello, {nm}."), [nm.lower()], [nm]))
        elif kind == 3:
            ph = rng.choice(HELD_PHRASES)
            out.append(_item(4, split, _d(f"Please say: {ph}.", ph[0].upper() + ph[1:] + "."), [ph.lower()], [ph]))
        elif kind == 4:
            a, b = rng.choice(sorted(HELD_ADD_PAIRS))
            out.append(_item(4, split, _d(f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."), [str(a + b)], [f"{a}+{b}"]))
        else:
            nm = rng.choice(p["name"])
            out.append(_item(4, split, _d(f"Say goodbye to {nm}.", f"Goodbye, {nm}."), [nm.lower()], [nm]))
    return out


# ----------------------------------------------------------------------------- teacher file

def write_seed_file(path: Path = TEACHER_FILE, overwrite: bool = False) -> int:
    """Create the teacher file with the 60 seed lessons if it is absent (the teacher appends more later). Returns lines written."""
    if path.exists() and not overwrite:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    items = seed_items()
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return len(items)


def load_items(path: Path = TEACHER_FILE, generated: int = 0, seed: int = 0) -> list[dict[str, Any]]:
    """Teacher lessons from the JSONL (the built-in seeds if the file is missing) plus `generated` templated items per stage and split."""
    items: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("dialogue") and d.get("split") in ("train", "heldout") and isinstance(d.get("stage"), int):
                items.append(d)
    else:
        items = seed_items()
    if generated:
        for split in ("train", "heldout"):
            items += gen_stage3(generated, seed, split) + gen_stage4(generated, seed, split)
    return items


def split_overlap(items: list[dict[str, Any]]) -> list[str]:
    """Held-out entities that leak into train: single-word entities appearing as a word in any train text, or multi-part entities equal to a train one."""
    train = [i for i in items if i["split"] == "train"]
    words: set[str] = set()
    ents: set[str] = set()
    for i in train:
        words |= set(re.findall(r"[a-z0-9']+", to_training_text(i).lower()))
        ents |= {e.lower() for e in i.get("entities", [])}
    bad: list[str] = []
    for i in items:
        if i["split"] != "heldout":
            continue
        for e in i.get("entities", []):
            el = e.lower()
            if re.fullmatch(r"[a-z0-9']+", el):
                if el in words:
                    bad.append(e)
            elif el in ents:
                bad.append(e)
    return sorted(set(bad))


# ----------------------------------------------------------------------------- stages

@dataclass(frozen=True)
class Stage:
    n: int
    name: str
    metric: str
    threshold: float
    lower_better: bool
    min_items: int


STAGES: list[Stage] = [
    Stage(1, "fluent English continuation", "bits_per_byte on held-out English text", 2.2, True, 5),
    Stage(2, "turn-taking", "fraction of replies that stop at the end-of-turn marker without writing the user's turn", 0.9, False, 15),
    Stage(3, "answer from context", "fraction of held-out questions answered with the stated name/colour/number/place", 0.7, False, 12),
    Stage(4, "follow simple instructions", "fraction of held-out requests (repeat, yes/no, arithmetic, greet) done correctly", 0.7, False, 12),
    Stage(5, "open conversation", "teacher grade (0..1) of recorded transcripts", 0.7, False, 1),
]

HELDOUT_TEXTS = [
    "The old man walked slowly down the road to the market. He bought bread, two apples and a small jar of honey.",
    "When it rains, the river grows wide and brown. The children stay inside and play games until the sun comes back.",
    "She opened the window and listened. Somewhere far away a bell rang, and the street was quiet again.",
    "A good friend will tell you the truth, even when it is hard to hear. That is why we keep them close.",
    "The train left the station at noon. By evening it had crossed the mountains and reached the sea.",
    "We planted seeds in the spring. In summer the garden was full of beans, tomatoes and tall yellow flowers.",
    "He read the letter twice, folded it, and put it in his pocket. Then he went out to find his sister.",
    "Bread is made from flour, water and salt. It takes time to rise, so you have to be patient.",
]


def stage_items(items: list[dict[str, Any]], stage: int, split: str) -> list[dict[str, Any]]:
    return [i for i in items if i["stage"] == stage and i["split"] == split]


def unlocked_stage(passed: dict[int, bool]) -> int:
    """Highest stage whose training data is open: stage k opens when stages 1..k-1 have all passed (stage 1 is always open)."""
    k = 1
    while k < len(STAGES) and passed.get(k, False):
        k += 1
    return k


def training_items(items: list[dict[str, Any]], passed: dict[int, bool]) -> list[dict[str, Any]]:
    """Train-split lessons the trainer may use now. Locked stages contribute nothing."""
    top = unlocked_stage(passed)
    return [i for i in items if i["split"] == "train" and i["stage"] <= top]


def training_texts(items: list[dict[str, Any]], passed: dict[int, bool]) -> list[str]:
    return [to_training_text(i) for i in training_items(items, passed)]


def _reply(lm: LMLike, turns: list[dict[str, str]]) -> tuple[str, bool]:
    prompt = prompt_for(turns)
    return extract_reply(lm.generate(prompt, max_new_tokens=48, temperature=0.2), prompt)


def eval_stage1(lm: LMLike) -> dict[str, Any]:
    bpb = float(lm.bits_per_byte(HELDOUT_TEXTS))
    return {"score": bpb, "n": len(HELDOUT_TEXTS)}


def eval_stage2(lm: LMLike, items: list[dict[str, Any]]) -> dict[str, Any]:
    pool = [i for i in items if i["split"] == "heldout" and i["stage"] in (2, 3, 4)]
    ok = 0
    for it in pool:
        ok += _reply(lm, it["dialogue"][:-1])[1]
    return {"score": ok / len(pool) if pool else 0.0, "n": len(pool)}


def eval_qa(lm: LMLike, items: list[dict[str, Any]], stage: int) -> dict[str, Any]:
    pool = [i for i in stage_items(items, stage, "heldout") if i.get("expect")]
    hits = 0
    fails: list[dict[str, str]] = []
    for it in pool:
        reply, ok = _reply(lm, it["dialogue"][:-1])
        if ok and answer_matches(reply, it["expect"]):
            hits += 1
        elif len(fails) < 5:
            fails.append({"q": it["dialogue"][-2]["text"], "reply": reply})
    return {"score": hits / len(pool) if pool else 0.0, "n": len(pool), "sample_fails": fails}


STAGE5_PROMPTS = ["Hello! Tell me about your day.", "What do you like to do?", "Can you tell me a short story?", "What should I make for dinner?"]


def record_stage5(lm: LMLike, path: Path | None = None) -> int:
    """Stub: let Nupen answer open prompts and store the transcripts for the teacher to grade (the teacher writes stage5_grade into stages.json)."""
    path = path or TRANSCRIPTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for p in STAGE5_PROMPTS:
            reply, ok = _reply(lm, [{"role": "user", "text": p}])
            f.write(json.dumps({"ts": time.time(), "source": "stage5", "user": p, "nupen": reply, "turn_ok": ok, "grade": None}, ensure_ascii=False) + "\n")
    return len(STAGE5_PROMPTS)


def stage_passed(st: Stage, score: float, n: int) -> bool:
    if n < st.min_items:
        return False
    return score <= st.threshold if st.lower_better else score >= st.threshold


def run_stages(lm: LMLike | None, items: list[dict[str, Any]], prior: dict[str, Any] | None = None, transcripts: Path | None = None) -> dict[str, Any]:
    """Scorecard. Stages run in order; a stage is only tested once every earlier stage has passed (locked otherwise)."""
    prior = prior or {}
    card: dict[str, Any] = {"ts": time.time(), "model": lm is not None, "stages": {}}
    passed: dict[int, bool] = {}
    blocked = lm is None
    for st in STAGES:
        info: dict[str, Any] = {"name": st.name, "metric": st.metric, "threshold": st.threshold, "lower_better": st.lower_better}
        if blocked:
            info.update(status="locked" if lm is not None else "no_model", passed=False)
        elif st.n == 1:
            info.update(eval_stage1(lm))  # type: ignore[arg-type]
        elif st.n == 2:
            info.update(eval_stage2(lm, items))  # type: ignore[arg-type]
        elif st.n in (3, 4):
            info.update(eval_qa(lm, items, st.n))  # type: ignore[arg-type]
        else:
            n = record_stage5(lm, transcripts)  # type: ignore[arg-type]
            grade = float(prior.get("stage5_grade", 0.0) or 0.0)
            info.update(score=grade, n=1 if prior.get("stage5_grade") is not None else 0, recorded=n, status="awaiting_teacher" if prior.get("stage5_grade") is None else "graded")
        if not blocked:
            info["passed"] = stage_passed(st, float(info["score"]), int(info["n"]))
            info.setdefault("status", "tested")
            if not info["passed"]:
                blocked = True
        passed[st.n] = bool(info["passed"])
        card["stages"][str(st.n)] = info
    card["unlocked_stage"] = unlocked_stage(passed)
    card["passed"] = [k for k, v in passed.items() if v]
    card["stage5_grade"] = prior.get("stage5_grade")
    return card


def passed_from_card(card: dict[str, Any]) -> dict[int, bool]:
    return {int(k): bool(v.get("passed")) for k, v in card.get("stages", {}).items()}


def load_state(path: Path = STAGES_FILE) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(card: dict[str, Any], path: Path = STAGES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(card, indent=1), encoding="utf-8")
    tmp.replace(path)


def append_transcript(rec: dict[str, Any], path: Path | None = None) -> None:
    path = path or TRANSCRIPTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
