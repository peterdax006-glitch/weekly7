"""The interface other engineers code against: load_current() -> LM | None; LM.generate; LM.bits_per_byte."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creator.lm.model import LMConfig
from creator.lm.paths import ckpt_dir, current_json
from creator.lm.tokenizer import BPETokenizer


class LM:
    def __init__(self, model: Any, tok: BPETokenizer, cfg: LMConfig) -> None:
        self.model = model
        self.tok = tok
        self.cfg = cfg

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8) -> str:
        ids = [self.tok.eot_id] + self.tok.encode(prompt)
        out = self.model.generate(ids, max_new_tokens, temperature=temperature, stop_id=self.tok.eot_id)
        new = [i for i in out[len(ids):] if i != self.tok.eot_id]
        return prompt + self.tok.decode(new)

    def bits_per_byte(self, texts: list[str]) -> float:
        from creator.lm.evaluate import per_text_nats, summarize
        return summarize(per_text_nats(self.model, self.tok, texts, self.cfg.ctx), n_boot=1)["bpb"]


def load_weights(weights: Path | str, tok_path: Path | str | None = None) -> LM:
    import torch
    from creator.lm.model import build_model
    blob = torch.load(str(weights), map_location="cpu", weights_only=True)
    cfg = LMConfig(**blob["cfg"])
    model = build_model(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    tp = Path(tok_path) if tok_path else ckpt_dir() / "tokenizer.json"
    return LM(model, BPETokenizer.load(tp), cfg)


def load_current() -> LM | None:
    """The promoted checkpoint, or None when there is none (or torch is unavailable)."""
    p = current_json()
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return load_weights(d["weights"], d.get("tokenizer"))
    except (OSError, ValueError, KeyError, ImportError):
        return None
