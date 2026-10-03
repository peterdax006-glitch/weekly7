"""Resumable CPU training loop. Checkpoints to lmckpt/, stops cleanly on lmckpt/STOP, promotes only via evaluate.consider."""
from __future__ import annotations

import math
import time
from typing import Any

from creator.lm.model import LMConfig, build_model, count_params
from creator.lm.paths import ckpt_dir, data_dir

STOP_FILE = "STOP"


def _batch(data: Any, rng: Any, bs: int, ctx: int, device: str = "cpu", dlg: Any = None, starts: list[int] | None = None,
           share: float = 0.0) -> tuple[Any, Any]:
    import numpy as np
    import torch
    if dlg is not None and starts and share > 0:
        from creator.lm import mix
        rows = [((dlg if d else data), i) for d, i in mix.mix_offsets(bs, share, rng, len(data), starts, ctx)]
    else:
        rows = [(data, int(i)) for i in rng.integers(0, len(data) - ctx - 1, size=bs)]
    x = np.stack([a[i: i + ctx] for a, i in rows]).astype(np.int64)
    y = np.stack([a[i + 1: i + 1 + ctx] for a, i in rows]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(lr: float, cycle_spent: float, cycle_budget: float, cycle_steps: int, warm: int = 30) -> float:
    """Learning rate inside an annealing CYCLE (a cosine over cycle_budget seconds of training, spanning as many sessions as it takes).
    Warm-up only at the very start of a cycle (cycle_steps counts steps taken in it), so a resumed session continues where the last
    one stopped instead of re-warming and re-annealing from the top."""
    frac = min(1.0, max(0.0, cycle_spent / max(cycle_budget, 1e-9)))
    return lr * min(1.0, (cycle_steps + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def pending_path() -> Any:
    return ckpt_dir() / "cycle_complete_pending.json"


def read_pending() -> dict[str, Any] | None:
    """The completed-cycle weights whose promotion check has not run yet (the process died between the final save and evaluate.consider)."""
    import json
    try:
        d = json.loads(pending_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("weights") else None


def clear_pending() -> None:
    try:
        pending_path().unlink()
    except OSError:
        pass


def session_limits(minutes: float, cycle_minutes: float | None, cycle_spent: float) -> tuple[float, float]:
    """(cycle_budget_s, session_budget_s): the session ends at its own budget or when the cycle completes, whichever is first."""
    cycle_budget = (cycle_minutes if cycle_minutes else minutes) * 60.0
    return cycle_budget, min(minutes * 60.0, max(0.0, cycle_budget - cycle_spent))


def train(minutes: float, threads: int = 4, batch: int = 8, accum: int = 2, lr: float = 2e-3,
          cfg: LMConfig | None = None, save_every_min: float = 5.0, log: Any = print,
          device: str = "cpu", dialogue_share: float = 0.0, cycle_minutes: float | None = None) -> dict[str, Any]:
    """Train for up to `minutes`. The cosine schedule runs over a CYCLE of cycle_minutes (default: this session's minutes) that is
    saved in the checkpoint: a session cut short (owner back, NUPEN_STOP, a yield to practice) resumes the same cycle next time, and
    only a COMPLETED cycle ends annealed - the result says so ('cycle_complete'), and only then is a promotion attempt meaningful."""
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
    cycle_spent0, cycle_steps0 = 0.0, 0                              # progress inside the current annealing cycle (old checkpoints: a new cycle)
    blob = torch.load(str(resume), map_location="cpu", weights_only=True) if resume.exists() else None
    cfg = LMConfig(**blob["cfg"]) if blob else (cfg or LMConfig(vocab_size=tok.vocab_size))
    model = build_model(cfg).to(device)                              # device: 'cuda' on a machine whose LM env has a CUDA torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if blob:
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        step, tokens_seen, elapsed_total = blob["step"], blob["tokens"], blob["elapsed"]
        cycle_spent0, cycle_steps0 = float(blob.get("cycle_spent", 0.0)), int(blob.get("cycle_steps", 0))
    data = np.memmap(data_dir() / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1234 + step)
    dlg: Any = None
    starts: list[int] = []
    if dialogue_share > 0:
        from creator.lm import mix
        texts = mix.pool_texts(mix.build_pool())
        ids, starts = mix.encode_stream(tok, texts, 7, 200_000)
        dlg = np.asarray(ids, dtype=np.uint16)
        log(f"dialogue mix: share {dialogue_share:.2f}, {len(texts)} dialogues, {len(ids)} tokens in the stream")
    n_params = count_params(model)

    done = False

    def save(final_name: str | None = None) -> None:
        cyc = {"cycle_spent": 0.0, "cycle_steps": 0} if done else {"cycle_spent": cycle_spent0 + time.time() - t0,
                                                                    "cycle_steps": cycle_steps0 + step - s0}
        state = {"cfg": cfg.to_dict(), "model": {k: v.cpu() for k, v in model.state_dict().items()}}   # checkpoints are device-free
        torch.save({**state, "opt": opt.state_dict(), "step": step, "tokens": tokens_seen, "elapsed": elapsed_total + time.time() - t0, **cyc},
                   str(resume) + ".tmp")
        (ck / "resume.pt.tmp").replace(resume)
        if final_name:
            torch.save(state, str(ck / final_name))

    cycle_budget, budget = session_limits(minutes, cycle_minutes, cycle_spent0)
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
        cur_lr = lr_at(lr, cycle_spent0 + spent, cycle_budget, cycle_steps0 + step - s0, warm)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for _ in range(accum):
            x, y = _batch(data, rng, batch, cfg.ctx, device, dlg, starts, dialogue_share)
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
    done = cycle_spent0 + session >= cycle_budget - 1e-6
    elapsed_total += session
    name = f"weights_step{step}.pt"
    save(name)
    if done:                                                         # written AFTER the weights exist, cleared by the caller once consider() ran
        import json
        pending_path().write_text(json.dumps({"weights": str(ck / name), "step": step, "tokens": tokens_seen, "params": n_params}),
                                  encoding="utf-8")
    return {"step": step, "tokens": tokens_seen, "params": n_params, "weights": str(ck / name),
            "session_seconds": session, "cycle_complete": done, "cycle_progress": min(1.0, (cycle_spent0 + session) / cycle_budget), "session_tok_per_s": (step - s0) * batch * accum * cfg.ctx / max(session, 1e-9),
            "last_loss": sum(losses[-20:]) / max(1, len(losses[-20:]))}
