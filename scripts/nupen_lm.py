"""Nupen's own language model. Run with the LM env at idle priority:
    python scripts/lowprio.py --idle C:/Users/Peter/creator_runtime/lmenv/Scripts/python.exe scripts/nupen_lm.py train --minutes 15
Commands: prep | train --minutes N | eval | sample "prompt"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_eval() -> int:
    from creator.lm import data
    from creator.lm.api import load_current
    from creator.lm.evaluate import per_text_nats, read_current, summarize
    lm = load_current()
    if lm is None:
        print("no current checkpoint")
        return 1
    res = summarize(per_text_nats(lm.model, lm.tok, data.eval_texts(), lm.cfg.ctx))
    print(json.dumps({"fresh": res, "recorded": read_current()}, indent=1))
    return 0


def cmd_train(minutes: float, threads: int, mix: str = "none", share: float = 0.2) -> int:
    from creator.lm import data, evaluate, train
    from creator.lm.model import LMConfig, build_model
    from creator.lm.tokenizer import BPETokenizer
    tok = BPETokenizer.load(data.data_dir() / "tokenizer.json")
    texts = data.eval_texts()
    if evaluate.read_current() is None:
        import torch
        torch.manual_seed(0)
        m = build_model(LMConfig(vocab_size=tok.vocab_size))
        base = evaluate.summarize(evaluate.per_text_nats(m, tok, texts, 256))
        print("BEFORE (untrained init) bits/byte:", json.dumps(base))
    out = train.train(minutes, threads=threads, dialogue_share=share if mix == "dialogue" else 0.0)
    print("TRAINED:", json.dumps(out))
    ok, why, res = evaluate.consider(Path(out["weights"]), {"step": out["step"], "tokens": out["tokens"], "params": out["params"]})
    print("PROMOTED" if ok else "NOT PROMOTED", why)
    print("AFTER:", json.dumps({k: res[k] for k in ("bpb", "lo", "hi", "n")}))
    return 0


def cmd_sample(prompt: str) -> int:
    from creator.lm.api import load_current
    lm = load_current()
    if lm is None:
        print("no current checkpoint")
        return 1
    print(lm.generate(prompt, 120, 0.8))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=15)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--mix", choices=["none", "dialogue"], default="none", help="dialogue: mix User:/Nupen: dialogue text into the story stream")
    t.add_argument("--dialogue-share", type=float, default=0.2, help="fraction of training sequences drawn from dialogue when --mix dialogue")
    sub.add_parser("eval")
    s = sub.add_parser("sample")
    s.add_argument("prompt")
    a = ap.parse_args()
    if a.cmd == "prep":
        from creator.lm import data
        print(json.dumps(data.prepare(), indent=1))
        return 0
    if a.cmd == "train":
        return cmd_train(a.minutes, a.threads, a.mix, a.dialogue_share)
    if a.cmd == "eval":
        return cmd_eval()
    return cmd_sample(a.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
