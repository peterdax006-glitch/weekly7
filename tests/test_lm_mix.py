"""Mixed story+dialogue training stream: share, held-out disjointness, turn format, supervisor flag (torch-free)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from creator.lm import dialogue as D
from creator.lm import mix

NOFILE = Path("does-not-exist.jsonl")


class FakeTok:
    def encode(self, t: str) -> list[int]:
        return list(t.encode())[:200]


def test_pool_is_train_only_and_disjoint() -> None:
    pool = mix.build_pool(NOFILE, NOFILE, gen=50, seed=3)
    assert pool and all(i["split"] == "train" for i in pool)
    held = [i for i in D.load_items(NOFILE) if i["split"] == "heldout"] + D.gen_stage3(30, 0, "heldout") + D.gen_stage4(30, 0, "heldout")
    held_texts = {D.to_training_text(i) for i in held}
    assert not held_texts & set(mix.pool_texts(pool))
    assert D.split_overlap(pool + held) == []
    assert {i["stage"] for i in pool} >= {2, 3, 4}


def test_pool_uses_turn_format() -> None:
    for t in mix.pool_texts(mix.build_pool(NOFILE, NOFILE, gen=5)):
        assert t.startswith("User: ") and "\nNupen: " in t and t.endswith(D.EOT + "\n")
        assert t.count(D.EOT) == len(D.parse_dialogue(t))


def test_terminal_conversations_included(tmp_path: Path) -> None:
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"user": "Hi Nupen", "nupen": "Hello."}) + "\nnot json\n", encoding="utf-8")
    pool = mix.build_pool(NOFILE, f, gen=2)
    assert any(i["source"] == "terminal" for i in pool)


def test_mix_share() -> None:
    rng = np.random.default_rng(0)
    offs = mix.mix_offsets(100, 0.2, rng, 10_000, [0, 50, 90], 64)
    assert sum(d for d, _ in offs) == 20
    assert all(o in (0, 50, 90) for d, o in offs if d)
    ids, starts = mix.encode_stream(FakeTok(), mix.pool_texts(mix.build_pool(NOFILE, NOFILE, gen=5)), 1, 5000)
    assert len(ids) >= 5000 and starts[0] == 0


def test_supervisor_command_carries_mix() -> None:
    from scripts import nupen_service as S
    c = S.LMTrainer(idle=lambda: 0.0, free_gb=lambda: 9.0, spawn=lambda c: None, stop=lambda p: None, log=lambda *a: None).cmd()
    assert "--mix" in c and c[c.index("--mix") + 1] == "dialogue" and "--dialogue-share" in c
