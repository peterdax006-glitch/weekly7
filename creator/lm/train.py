"""Resumable CPU training loop. Checkpoints to lmckpt/, stops cleanly on lmckpt/STOP, promotes only via evaluate.consider."""
from __future__ import annotations

import math
import time
from typing import Any

from creator.lm.model import LMConfig, build_model, count_params
from creator.lm.paths import ckpt_dir, data_dir

STOP_FILE = "STOP"


def _batch(data: Any, rng: Any, bs: int, ctx: int, device: str = "cpu") -> tuple[Any, Any]:
    import numpy as np
    import torch
    ix = rng.integers(0, len(data) - ctx - 1, size=bs)
    x = np.stack([data[i: i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1: i + 1 + ctx] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def train(minutes: float, threads: int = 4, batch: int = 8, accum: int = 2, lr: float = 2e-3,
          cfg: LMConfig | None = None, save_every_min: float = 5.0, log: Any = print,
          device: str = "cpu") -> dict[str, Any]:
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    ck = ckpt_dir()
    ck.mkdir(parents=True, exist_ok=True)
    stop = ck / STOP_FILE
    if stop.exists():
        stop.unlink()
    from creator.lm.tokenizer import BPETokenizer
    tok = BPETokenizer.load(data_dir() / "tokenizer.json")
    resume = ck / "resume.pt"
    step, tokens_seen, elapsed_total = 0, 0, 0.0
    blob = torch.load(str(resume), map_location="cpu", weights_only=True) if resume.exists() else None
    cfg = LMConfig(**blob["cfg"]) if blob else (cfg or LMConfig(vocab_size=tok.vocab_size))
    model = build_model(cfg).to(device)                              # device: 'cuda' on a machine whose LM env has a CUDA torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if blob:
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        step, tokens_seen, elapsed_total = blob["step"], blob["tokens"], blob["elapsed"]
    data = np.memmap(data_dir() / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1234 + step)
    n_params = count_params(model)

    def save(final_name: str | None = None) -> None:
        state = {"cfg": cfg.to_dict(), "model": {k: v.cpu() for k, v in model.state_dict().items()}}   # checkpoints are device-free
        torch.save({**state, "opt": opt.state_dict(), "step": step, "tokens": tokens_seen, "elapsed": elapsed_total + time.time() - t0},
                   str(resume) + ".tmp")
        (ck / "resume.pt.tmp").replace(resume)
        if final_name:
            torch.save(state, str(ck / final_name))

    budget = minutes * 60.0
    t0 = time.time()
    last_save = t0
    model.train()
    losses: list[float] = []
    warm = 30
    s0 = step
    while True:
        spent = time.time() - t0
        if spent >= budget or stop.exists():
            break
        frac = min(1.0, spent / budget)                              # time-based cosine schedule for this session
        cur_lr = lr * min(1.0, (step - s0 + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for _ in range(accum):
            x, y = _batch(data, rng, batch, cfg.ctx, device)
            _, loss = model(x, y)
            (loss / accum).backward()
            losses.append(float(loss))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += batch * accum * cfg.ctx
        if step % 10 == 0:
            dt = time.time() - t0
            log(f"step {step} loss {sum(losses[-20:]) / len(losses[-20:]):.3f} lr {cur_lr:.2e} "
                f"tok/s {(step - s0) * batch * accum * cfg.ctx / dt:.0f} elapsed {dt / 60:.1f}m")
        if time.time() - last_save > save_every_min * 60:
            save()
            last_save = time.time()
    session = time.time() - t0
    elapsed_total += session
    name = f"weights_step{step}.pt"
    save(name)
    return {"step": step, "tokens": tokens_seen, "params": n_params, "weights": str(ck / name),
            "session_seconds": session, "session_tok_per_s": (step - s0) * batch * accum * cfg.ctx / max(session, 1e-9),
            "last_loss": sum(losses[-20:]) / max(1, len(losses[-20:]))}
