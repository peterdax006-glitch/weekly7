"""Data preparation: tokenizer training, train.bin (uint16 ids), fixed held-out eval set. No torch needed."""
from __future__ import annotations

import hashlib
from pathlib import Path

from creator.lm.paths import data_dir
from creator.lm.tokenizer import EOT, BPETokenizer

EVAL_STORIES = 300           # fixed: the first 300 stories of the official validation file; never used for training


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def eval_texts(valid_path: Path | None = None) -> list[str]:
    p = valid_path or data_dir() / "valid.txt"
    text = p.read_text(encoding="utf-8", errors="replace")
    stories = [s.strip() for s in text.split(EOT) if s.strip()]
    return stories[:EVAL_STORIES]


def eval_set_sha256(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def prepare(train_mb: int = 40, tok_mb: int = 6, vocab_size: int = 4096) -> dict[str, object]:
    """Build train.txt boundary-clean, train the tokenizer, encode the first train_mb MB into train.bin."""
    import numpy as np
    d = data_dir()
    raw = b"".join((d / n).read_bytes() for n in ("train_slice.txt", "part2.txt"))
    raw = raw[: raw.rfind(EOT.encode()) + len(EOT)]            # the range cut ended mid-story: drop the stump
    text = raw.decode("utf-8", errors="replace")
    (d / "train.txt").write_text(text, encoding="utf-8", newline="")
    sample = text[: tok_mb * 1_000_000]
    tok = BPETokenizer.train(sample, vocab_size)
    tok.save(d / "tokenizer.json")
    part = text[: train_mb * 1_000_000]
    part = part[: part.rfind(EOT) + len(EOT)]
    ids = np.asarray(tok.encode(part), dtype=np.uint16)
    ids.tofile(d / "train.bin")
    texts = eval_texts()
    return {
        "train_txt_bytes": len(raw), "train_bin_tokens": int(ids.size), "train_bytes_encoded": len(part.encode()),
        "vocab_size": tok.vocab_size, "valid_sha256": sha256_file(d / "valid.txt"),
        "eval_set_sha256": eval_set_sha256(texts), "eval_stories": len(texts),
        "eval_bytes": sum(len(t.encode()) for t in texts),
        "bytes_per_token": len(part.encode()) / int(ids.size),
    }
