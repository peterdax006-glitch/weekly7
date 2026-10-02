"""Held-out evaluation in bits per byte (tokenizer-independent) with a bootstrap CI, and the computed promotion rule."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from creator.lm.paths import ckpt_dir, current_json


def summarize(per_text: list[tuple[float, int]], seed: int = 0, n_boot: int = 1000) -> dict[str, float]:
    """per_text: (total nats, byte count) for each held-out text. Pooled bits/byte; 95% bootstrap CI over texts."""
    import numpy as np
    nats = np.asarray([p[0] for p in per_text], dtype=np.float64)
    nbytes = np.asarray([p[1] for p in per_text], dtype=np.float64)
    bpb = float(nats.sum() / nbytes.sum() / math.log(2))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_text), size=(n_boot, len(per_text)))
    boots = nats[idx].sum(axis=1) / nbytes[idx].sum(axis=1) / math.log(2)
    return {"bpb": bpb, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "n": float(len(per_text))}


def should_promote(new: dict[str, Any], cur: dict[str, Any] | None) -> tuple[bool, str]:
    """Computed, never self-graded: promote only if the new CI lies entirely below the current CI (same eval set)."""
    if cur is None:
        return True, "no current checkpoint"
    if new.get("eval_sha256") != cur.get("eval_sha256"):
        return False, "different eval set: not comparable"
    if new["hi"] < cur["lo"]:
        return True, f"CI separated: {new['bpb']:.4f} [{new['lo']:.4f},{new['hi']:.4f}] < [{cur['lo']:.4f},{cur['hi']:.4f}]"
    return False, f"within noise or worse: {new['bpb']:.4f} [{new['lo']:.4f},{new['hi']:.4f}] vs current [{cur['lo']:.4f},{cur['hi']:.4f}]"


def window_nats(model: Any, ids: list[int], eot_id: int, ctx: int) -> float:
    """Total NLL (nats) of ids, each window opened with a start token (the end-of-text id)."""
    import torch
    from torch.nn import functional as F
    total = 0.0
    step = ctx
    with torch.no_grad():
        for s in range(0, len(ids), step):
            chunk = [eot_id] + ids[s: s + step]
            x = torch.tensor([chunk[:-1]], dtype=torch.long)
            y = torch.tensor([chunk[1:]], dtype=torch.long)
            total += float(F.cross_entropy(model(x)[0], y[0], reduction="sum"))
    return total


def per_text_nats(model: Any, tok: Any, texts: list[str], ctx: int) -> list[tuple[float, int]]:
    was = model.training
    model.eval()
    out = [(window_nats(model, tok.encode(t), tok.eot_id, ctx), len(t.encode("utf-8"))) for t in texts]
    model.train(was)
    return out


def read_current() -> dict[str, Any] | None:
    p = current_json()
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def evaluate_weights(weights: Path, texts: list[str] | None = None) -> dict[str, Any]:
    """Score a weights file on the fixed held-out set."""
    from creator.lm import data
    from creator.lm.api import load_weights
    lm = load_weights(weights)
    texts = texts if texts is not None else data.eval_texts()
    res: dict[str, Any] = summarize(per_text_nats(lm.model, lm.tok, texts, lm.cfg.ctx))
    res["eval_sha256"] = data.eval_set_sha256(texts)
    res["weights"] = str(weights)
    return res


def consider(weights: Path, meta: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate a checkpoint and, if the rule says so, make it current. Returns (promoted, reason, result)."""
    res = evaluate_weights(weights)
    res.update(meta or {})
    cur = read_current()
    ok, why = should_promote(res, cur)
    if ok:
        tokp = ckpt_dir() / "tokenizer.json"
        if not tokp.exists():
            from creator.lm.paths import data_dir
            shutil.copyfile(data_dir() / "tokenizer.json", tokp)
        res["tokenizer"] = str(tokp)
        tmp = current_json().with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        tmp.replace(current_json())
    return ok, why, res
