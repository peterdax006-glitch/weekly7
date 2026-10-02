"""Nupen's dialogue curriculum: format, turn-taking check, QA scoring, stage gating, split disjointness (FakeLM, no model)."""
from __future__ import annotations

import json
from pathlib import Path

from creator.lm import dialogue as D
from creator.lm.dialogue_seed import seed_items


class FakeLM:
    """Answers perfectly by looking up the reference reply for the prompt (an oracle), or misbehaves on demand."""

    def __init__(self, items: list[dict], mode: str = "oracle", bpb: float = 1.5) -> None:
        self.ref = {D.prompt_for(i["dialogue"][:-1]): i["dialogue"][-1]["text"] for i in items}
        self.mode, self.bpb = mode, bpb

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8) -> str:
        ans = self.ref.get(prompt, "I do not know")
        if self.mode == "leak":
            return f" {ans}\nUser: and then?"
        if self.mode == "wrong":
            return f" banana{D.EOT}"
        return f" {ans}{D.EOT}"

    def bits_per_byte(self, texts: list[str]) -> float:
        return self.bpb


def all_items() -> list[dict]:
    return D.load_items(Path("does-not-exist.jsonl"), generated=15, seed=1)


def test_seed_counts() -> None:
    s = seed_items()
    assert len(s) == 60
    assert sum(i["split"] == "heldout" for i in s) == 20
    assert {i["stage"] for i in s} == {2, 3, 4}
    assert all(i["source"] == "teacher" for i in s)


def test_format_round_trip() -> None:
    for it in seed_items():
        txt = D.to_training_text(it)
        assert D.parse_dialogue(txt) == it["dialogue"]
        assert txt.count(D.EOT) == len(it["dialogue"])
    assert D.prompt_for([{"role": "user", "text": "Hi"}]) == f"User: Hi{D.EOT}\nNupen:"


def test_turn_taking_catches_user_leak() -> None:
    assert D.check_turn_taking(f" Hello there.{D.EOT}")
    assert not D.check_turn_taking(" Hello there.\nUser: what next?")
    assert not D.check_turn_taking(f" Hello. User: hi{D.EOT}")
    assert not D.check_turn_taking(" Hello with no end marker")
    assert not D.check_turn_taking(D.EOT)


def test_qa_scoring() -> None:
    assert D.answer_matches("Your name is Anna.", ["anna"])
    assert D.answer_matches("There are four cups.", ["four", "4"])
    assert not D.answer_matches("I know.", ["no"])  # whole word only
    assert not D.answer_matches("Your name is Bo.", ["anna"])
    items = all_items()
    good = D.eval_qa(FakeLM(items), items, 3)
    assert good["score"] == 1.0 and good["n"] >= 12
    bad = D.eval_qa(FakeLM(items, "wrong"), items, 3)
    assert bad["score"] == 0.0 and bad["sample_fails"]
    assert D.eval_qa(FakeLM(items, "leak"), items, 4)["score"] == 0.0  # right answer but writes the user's turn


def test_stage_gating(tmp_path: Path) -> None:
    items = all_items()
    n_train = lambda st: sum(1 for i in D.training_items(items, st) if i["stage"] == 3)  # noqa: E731
    assert n_train({}) == 0 and n_train({1: True}) == 0
    assert n_train({1: True, 2: True}) > 0
    assert D.unlocked_stage({1: True, 3: True}) == 2  # a later pass does not skip a failed stage
    assert D.unlocked_stage({1: True, 2: True, 3: True, 4: True}) == 5
    tr = tmp_path / "t.jsonl"
    card = D.run_stages(FakeLM(items, bpb=3.0), items, transcripts=tr)  # fails stage 1: everything else locked
    assert card["unlocked_stage"] == 1 and card["stages"]["2"]["status"] == "locked"
    card = D.run_stages(FakeLM(items), items, transcripts=tr)
    assert card["passed"] == [1, 2, 3, 4] and card["unlocked_stage"] == 5
    assert card["stages"]["5"]["status"] == "awaiting_teacher" and not card["stages"]["5"]["passed"]
    assert len(tr.read_text().splitlines()) == len(D.STAGE5_PROMPTS)
    card = D.run_stages(FakeLM(items), items, prior={"stage5_grade": 0.8}, transcripts=tr)
    assert card["stages"]["5"]["passed"]
    assert D.run_stages(None, items)["stages"]["1"]["status"] == "no_model"
    st = tmp_path / "stages.json"
    D.save_state(card, st)
    assert D.load_state(st)["passed"] == [1, 2, 3, 4, 5]


def test_too_few_items_never_passes() -> None:
    assert not D.stage_passed(D.STAGES[2], 1.0, 3)


def test_heldout_disjoint_from_train() -> None:
    items = all_items() + D.load_items(Path("x"), generated=40, seed=7)
    assert D.split_overlap(items) == []
    leaky = items + [{"stage": 3, "split": "train", "dialogue": D.parse_dialogue(f"User: Zed is here.{D.EOT}\nNupen: ok{D.EOT}\n"), "entities": []}]
    assert "Zed" in D.split_overlap(leaky)
    tr3 = {json.dumps(i["dialogue"][0]) for i in D.gen_stage3(60, 1, "train")}
    he3 = {json.dumps(i["dialogue"][0]) for i in D.gen_stage3(60, 1, "heldout")}
    assert not tr3 & he3
    pairs = lambda sp: {e for i in D.gen_stage4(80, 3, sp) for e in i["entities"] if "+" in e}  # noqa: E731
    assert not pairs("train") & pairs("heldout")


def test_seed_file_written(tmp_path: Path) -> None:
    p = tmp_path / "lmdata" / "t.jsonl"
    assert D.write_seed_file(p) == 60
    assert D.write_seed_file(p) == 0  # never clobbers the teacher's file
    assert len(D.load_items(p)) == 60
