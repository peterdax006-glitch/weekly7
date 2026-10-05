"""Merge a LoRA adapter GGUF into a base GGUF on the CPU (no torch needed).

Why: the official llama.cpp release zips carry no llama-export-lora, and runtime
--lora is ~2.5x slower on CPU, so every slot must be a merged GGUF.

Steps: dequantize the base (Q4_K/Q6_K/..) with the gguf package, add
scale * (B @ A) to every adapted tensor, write an F16 GGUF, then run
llama-quantize to Q4_K_M. Needs `pip install gguf numpy` (any env or --target dir).

usage: merge_lora_gguf.py BASE.gguf ADAPTER.gguf OUT_F16.gguf [--scale S]
       then: llama-quantize OUT_F16.gguf OUT_Q4_K_M.gguf Q4_K_M
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def merge(base_path: str, adapter_path: str, out_path: str, scale_mult: float = 1.0) -> dict:
    import gguf
    from gguf import quants

    ad = gguf.GGUFReader(adapter_path)
    alpha = float(ad.fields["adapter.lora.alpha"].contents())
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for t in ad.tensors:
        stem, _, kind = t.name.rpartition(".")
        pairs.setdefault(stem, {})[kind] = np.asarray(t.data, dtype=np.float32)
    rank = next(iter(pairs.values()))["lora_a"].shape[0]
    scale = alpha / rank * scale_mult

    base = gguf.GGUFReader(base_path)
    arch = base.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(out_path, arch)
    for key, f in base.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key == "general.file_type":
            continue
        val = f.contents()
        vtype = f.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            w.add_array(key, val)
        else:
            w.add_key_value(key, val, vtype)
    merged = 0
    for t in base.tensors:
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            arr = np.asarray(t.data)
        else:
            arr = quants.dequantize(t.data, t.tensor_type)
        arr = arr.astype(np.float32)
        if t.name in pairs:
            p = pairs[t.name]
            arr = arr + scale * (p["lora_b"] @ p["lora_a"]).reshape(arr.shape)
            merged += 1
        one_d = arr.ndim == 1
        w.add_tensor(t.name, arr if one_d else arr.astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    if merged != len(pairs):
        raise SystemExit(f"adapter had {len(pairs)} targets, merged {merged}")
    return {"targets": merged, "rank": rank, "alpha": alpha, "scale": scale}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("adapter")
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    print(merge(a.base, a.adapter, a.out, a.scale))
    sys.exit(0)
