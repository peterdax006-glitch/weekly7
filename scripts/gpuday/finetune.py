"""GPU-day fine-tune pipeline: LoRA/QLoRA SFT, preference (DPO/ORPO), merge, GGUF, sha256 - ONE script for the pod and the CPU test.

Runs on the pod (Vast.ai image vastai/unsloth-studio, CUDA, Unsloth present -> Unsloth's FastLanguageModel loader, bf16) and on a CPU
(no Unsloth -> plain transformers + peft, fp32) - the training loop itself (TRL SFTTrainer / DPOTrainer / ORPOTrainer) is shared, so the
CPU test with a tiny model proves the data path, the trainer configuration, the merge and the GGUF conversion before any paid minute.
Nupen's own code is NOT imported here (the pod has no repo checkout; data arrives as the JSONL files of creator/gpuday.py export).

  python finetune.py sft   --base Qwen/Qwen3-1.7B --data handoff_train.jsonl --out runs/sft17 [--qlora] [--max-steps N] [--epochs E]
  python finetune.py pref  --method dpo|orpo --base Qwen/Qwen3-1.7B --data pref_train.jsonl --out runs/dpo17 [--adapter runs/sft17/adapter]
  python finetune.py merge --base Qwen/Qwen3-1.7B --adapter runs/sft17/adapter --out runs/sft17/merged
  python finetune.py gguf  --merged runs/sft17/merged --llama-cpp /workspace/llama.cpp --quant Q4_K_M --out runs/sft17/model-Q4_K_M.gguf
  python finetune.py lora-gguf --base Qwen/Qwen3-1.7B --adapter runs/sft17/adapter --llama-cpp ... --out runs/sft17/adapter.gguf
  python finetune.py pipeline --base ... --data ... --out runs/x [--pref pref.jsonl --method dpo] --llama-cpp ... [--quant Q4_K_M]
Every step writes <out>/result.json (seconds, losses, files with bytes + sha256): the small file that comes home."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files_info(paths: list[Path]) -> dict[str, Any]:
    return {str(p.name): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths if p.is_file()}


def write_result(out: Path, step: str, rec: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "result.json"
    try:
        allr = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        allr = {}
    allr[step] = rec
    p.write_text(json.dumps(allr, indent=1), encoding="utf-8")
    print("@@result=" + json.dumps({step: rec}), flush=True)
    return rec


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def use_unsloth(force_hf: bool) -> bool:
    if force_hf:
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import unsloth  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001 - no GPU / no unsloth: the plain HF path
        return False


def load(base: str, max_seq: int, qlora: bool, r: int, alpha: int, force_hf: bool, adapter: str = "", train: bool = True) -> tuple[Any, Any, str]:
    """(model with a trainable LoRA, tokenizer, backend). `adapter`: continue from an existing LoRA (e.g. SFT before DPO)."""
    if use_unsloth(force_hf):
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(model_name=adapter or base, max_seq_length=max_seq, load_in_4bit=qlora, dtype=None)
        if not adapter:
            model = FastLanguageModel.get_peft_model(model, r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none",
                                                     use_gradient_checkpointing="unsloth", random_state=3407)
        return model, tok, "unsloth"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    kw: dict[str, Any] = {"torch_dtype": (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32}
    if qlora and cuda:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    else:
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none",
                                                 task_type="CAUSAL_LM"))
    return model, tok, "hf"


def to_prompt_completion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """{'messages': [..., assistant]} -> TRL conversational prompt/completion (loss on the answer only)."""
    out = []
    for r in rows:
        m = r["messages"]
        if m and m[-1]["role"] == "assistant":
            out.append({"prompt": m[:-1], "completion": [m[-1]]})
    return out


def fit_rows(rows: list[dict[str, Any]], tok: Any, max_seq: int) -> tuple[list[dict[str, Any]], int]:
    """Rows whose whole conversation fits `max_seq` tokens. A longer row would be truncated from the END - its answer cut off or masked
    away entirely (TRL then drops it silently) - so it is left out and counted instead."""
    keep, n = [], 0
    for r in rows:
        msgs = list(r.get("prompt") or []) + list(r.get("completion") or r.get("chosen") or [])
        try:
            ids = tok.apply_chat_template(msgs, tokenize=True)
            length = len(ids["input_ids"] if isinstance(ids, dict) else ids)
        except Exception:                               # noqa: BLE001 - no template: count the plain text
            length = len(tok(" ".join(str(m.get("content", "")) for m in msgs))["input_ids"])
        if length <= max_seq:
            keep.append(r)
        else:
            n += 1
    return keep, n


def common_args(a: argparse.Namespace, cls: Any, **extra: Any) -> Any:
    import torch
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()       # Ampere and newer: bf16; older cards fp16; no FP8 path is used anywhere
    kw: dict[str, Any] = dict(output_dir=str(Path(a.out) / "ckpt"), per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
                              learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps else -1,
                              logging_steps=1, save_strategy="no", report_to=[], bf16=bf16, fp16=cuda and not bf16, seed=3407,
                              lr_scheduler_type="cosine", warmup_ratio=0.05, gradient_checkpointing=cuda and a.backend == "hf",
                              use_cpu=not cuda)
    kw.update(extra)
    import inspect
    ok = set(inspect.signature(cls.__init__).parameters)
    return cls(**{k: v for k, v in kw.items() if k in ok})


def cmd_sft(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    model, tok, a.backend = load(a.base, a.max_seq, a.qlora, a.r, a.alpha, a.hf, a.adapter)
    rows, too_long = fit_rows(to_prompt_completion(read_jsonl(Path(a.data))), tok, a.max_seq)
    if not rows:
        raise SystemExit(f"no training rows in {a.data} fit {a.max_seq} tokens ({too_long} too long)")
    ds = Dataset.from_list(rows)
    ev_rows: list[dict[str, Any]] = []
    if getattr(a, "eval_data", "") and Path(a.eval_data).is_file():
        ev_rows, _ = fit_rows(to_prompt_completion(read_jsonl(Path(a.eval_data))), tok, a.max_seq)
    extra: dict[str, Any] = {}
    callbacks: list[Any] = []
    if ev_rows:                         # early stopping on the held-out dev split: evaluate ~4x per epoch, keep the best adapter
        es = eval_every(len(rows), a.batch, a.accum, a.eval_steps)
        extra = dict(eval_strategy="steps", evaluation_strategy="steps", eval_steps=es, save_strategy="steps", save_steps=es,
                     save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                     per_device_eval_batch_size=a.batch)
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=max(1, a.patience)))
    cfg = common_args(a, SFTConfig, max_length=a.max_seq, completion_only_loss=True, packing=False, **extra)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, eval_dataset=Dataset.from_list(ev_rows) if ev_rows else None,
                    processing_class=tok, callbacks=callbacks or None)
    st = tr.train()
    ad = Path(a.out) / "adapter"
    model.save_pretrained(str(ad))
    tok.save_pretrained(str(ad))
    losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    evl = [(h.get("step"), h["eval_loss"]) for h in tr.state.log_history if "eval_loss" in h]
    if ev_rows:
        import shutil
        shutil.rmtree(Path(a.out) / "ckpt", ignore_errors=True)       # the best weights are loaded and saved as adapter/; checkpoints go
    return write_result(Path(a.out), "sft", {"backend": a.backend, "base": a.base, "rows": len(rows), "too_long": too_long, "steps": st.global_step,
                                             "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
                                             "dev_rows": len(ev_rows), "eval_loss": evl,
                                             "eval_loss_best": min((v for _s, v in evl), default=None),
                                             "best_step": getattr(tr.state, "best_global_step", None) or (min(evl, key=lambda x: x[1])[0] if evl else None),
                                             "max_steps_planned": tr.state.max_steps,
                                             "stopped_early": bool(evl) and st.global_step < tr.state.max_steps,
                                             "seconds": round(time.time() - t0, 1), "adapter": str(ad),
                                             "files": files_info(sorted(ad.glob("adapter_*")))})


def eval_every(rows: int, batch: int, accum: int, eval_steps: int = 0) -> int:
    """Optimizer steps between dev evaluations: `eval_steps` when given, else ~4 per epoch (at least 1)."""
    if eval_steps > 0:
        return eval_steps
    per_epoch = max(1, -(-rows // max(1, batch * accum)))
    return max(1, per_epoch // 4)


def cmd_pref(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    from datasets import Dataset
    model, tok, a.backend = load(a.base, a.max_seq, a.qlora, a.r, a.alpha, a.hf, a.adapter)
    rows, too_long = fit_rows(read_jsonl(Path(a.data)), tok, a.max_seq)
    if not rows:
        raise SystemExit(f"no preference rows in {a.data} fit {a.max_seq} tokens ({too_long} too long)")
    ds = Dataset.from_list(rows)
    if a.method == "dpo":
        from trl import DPOConfig, DPOTrainer
        cfg = common_args(a, DPOConfig, beta=a.beta, max_length=a.max_seq)
        tr = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok)
    else:
        try:
            from trl import ORPOConfig, ORPOTrainer
        except ImportError:                              # newer TRL keeps ORPO under trl.experimental
            from trl.experimental.orpo import ORPOConfig, ORPOTrainer  # type: ignore[no-redef]
        cfg = common_args(a, ORPOConfig, beta=a.beta, max_length=a.max_seq)
        tr = ORPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    st = tr.train()
    ad = Path(a.out) / "adapter"
    model.save_pretrained(str(ad))
    tok.save_pretrained(str(ad))
    losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    return write_result(Path(a.out), a.method, {"backend": a.backend, "base": a.base, "rows": len(rows), "too_long": too_long,
                                                "steps": st.global_step,
                                                "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
                                                "seconds": round(time.time() - t0, 1), "adapter": str(ad),
                                                "files": files_info(sorted(ad.glob("adapter_*")))})


def cmd_merge(a: argparse.Namespace) -> dict[str, Any]:
    """Base + adapter -> one 16-bit HF model directory (the input of convert_hf_to_gguf.py). Always merged into a 16-bit base (also
    after QLoRA: merging into 4-bit weights loses precision)."""
    t0 = time.time()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) or a.bf16 else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    out = Path(a.out)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter if (Path(a.adapter) / "tokenizer_config.json").is_file() else a.base).save_pretrained(str(out))
    return write_result(out.parent, "merge", {"merged": str(out), "dtype": str(dtype), "seconds": round(time.time() - t0, 1),
                                              "bytes": sum(p.stat().st_size for p in out.glob("*.safetensors"))})


def _run(argv: list[str]) -> str:
    print("$ " + " ".join(argv), flush=True)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed (rc {r.returncode}): {' '.join(argv)}\n{(r.stdout + r.stderr)[-3000:]}")
    return r.stdout + r.stderr


def quantize_exe(llama_cpp: Path) -> str:
    for c in ("llama-quantize", "llama-quantize.exe", "build/bin/llama-quantize", "bin/llama-quantize"):
        if (llama_cpp / c).is_file():
            return str(llama_cpp / c)
    for p in llama_cpp.rglob("llama-quantize*"):
        if p.is_file() and p.suffix in ("", ".exe"):
            return str(p)
    return "llama-quantize"


def cmd_gguf(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    lc, out = Path(a.llama_cpp), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f16 = out.with_name(out.stem + ".f16.gguf")
    conv = Path(a.convert_dir or lc) / "convert_hf_to_gguf.py"
    _run([sys.executable, str(conv), str(a.merged), "--outfile", str(f16), "--outtype", "f16"])
    if a.quant.lower() in ("f16", "none"):
        f16.replace(out)
    else:
        _run([a.quantize or quantize_exe(lc), str(f16), str(out), a.quant])
        if not a.keep_f16:
            f16.unlink(missing_ok=True)
    return write_result(out.parent, "gguf", {"quant": a.quant, "seconds": round(time.time() - t0, 1), "files": files_info([out])})


def cmd_lora_gguf(a: argparse.Namespace) -> dict[str, Any]:
    """The adapter alone as GGUF (tens of MB): llama-server --model <base gguf> --lora <this>. The cheapest thing to bring home."""
    t0 = time.time()
    out = Path(a.out)
    conv = Path(a.convert_dir or a.llama_cpp) / "convert_lora_to_gguf.py"
    base = str(a.base)
    if not Path(base).is_dir():                         # an HF repo id: the converter wants a local folder with the base's config files
        from huggingface_hub import snapshot_download
        base = snapshot_download(base, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
    _run([sys.executable, str(conv), str(a.adapter), "--base", base, "--outfile", str(out), "--outtype", "f16"])
    return write_result(out.parent, "lora_gguf", {"seconds": round(time.time() - t0, 1), "files": files_info([out])})


def cmd_pipeline(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    res: dict[str, Any] = {"sft": cmd_sft(a)}
    adapter = str(out / "adapter")
    if a.pref:
        pa = argparse.Namespace(**vars(a))
        pa.data, pa.adapter, pa.out, pa.eval_data = a.pref, adapter, str(out / a.method), ""
        res[a.method] = cmd_pref(pa)
        adapter = str(Path(pa.out) / "adapter")
    if a.llama_cpp:
        ma = argparse.Namespace(**vars(a))
        ma.adapter, ma.out = adapter, str(out / "merged")
        ma.base = a.merge_base or a.base                  # QLoRA trains on a 4-bit repo; the merge always goes into the 16-bit weights
        res["merge"] = cmd_merge(ma)
        ga = argparse.Namespace(**vars(a))
        ga.merged, ga.out = str(out / "merged"), str(out / f"model-{a.quant}.gguf")
        res["gguf"] = cmd_gguf(ga)
        if not a.keep_merged:
            import shutil
            shutil.rmtree(out / "merged", ignore_errors=True)
    if a.llama_cpp and a.adapter_gguf:                    # the adapter alone (tens of MB): what comes home; never fails the pipeline
        la = argparse.Namespace(**vars(a))
        la.adapter, la.out = adapter, str(out / "adapter.gguf")
        try:
            res["lora_gguf"] = cmd_lora_gguf(la)
        except (SystemExit, Exception) as e:            # noqa: BLE001 - the merged GGUF and the adapter are the results that matter
            res["lora_gguf"] = write_result(out, "lora_gguf", {"error": str(e)[-800:]})
    return write_result(out, "pipeline", {"steps": list(res)})


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def train_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--adapter", default="")
        p.add_argument("--qlora", action="store_true")
        p.add_argument("--hf", action="store_true", help="force the plain transformers+peft loader (no Unsloth)")
        p.add_argument("--r", type=int, default=16)
        p.add_argument("--alpha", type=int, default=32)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--epochs", type=float, default=2.0)
        p.add_argument("--max-steps", type=int, default=0)
        p.add_argument("--max-seq", type=int, default=8192)
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--accum", type=int, default=8)
        p.add_argument("--beta", type=float, default=0.1)
        p.add_argument("--method", choices=["dpo", "orpo"], default="dpo")
        p.add_argument("--eval-data", default="", help="held-out dev split (same format): early stopping on its loss, best adapter kept")
        p.add_argument("--eval-steps", type=int, default=0, help="optimizer steps between dev evaluations (0 = ~4 per epoch)")
        p.add_argument("--patience", type=int, default=2, help="dev evaluations without improvement before training stops")
    p = sub.add_parser("sft")
    train_opts(p)
    p = sub.add_parser("pref")
    train_opts(p)
    p = sub.add_parser("merge")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", action="store_true")
    for name in ("gguf", "lora-gguf"):
        p = sub.add_parser(name)
        p.add_argument("--llama-cpp", required=True, help="llama.cpp directory (convert scripts; llama-quantize binary)")
        p.add_argument("--convert-dir", default="", help="where convert_*_to_gguf.py live when not in --llama-cpp")
        p.add_argument("--quantize", default="", help="path of llama-quantize when not under --llama-cpp")
        p.add_argument("--out", required=True)
        p.add_argument("--quant", default="Q4_K_M")
        p.add_argument("--keep-f16", action="store_true")
        if name == "gguf":
            p.add_argument("--merged", required=True)
        else:
            p.add_argument("--base", required=True)
            p.add_argument("--adapter", required=True)
    p = sub.add_parser("pipeline")
    train_opts(p)
    p.add_argument("--pref", default="")
    p.add_argument("--llama-cpp", default="")
    p.add_argument("--convert-dir", default="")
    p.add_argument("--quantize", default="")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--keep-f16", action="store_true")
    p.add_argument("--keep-merged", action="store_true")
    p.add_argument("--merge-base", default="", help="16-bit base for the merge (QLoRA runs train on a pre-quantised repo)")
    p.add_argument("--adapter-gguf", action="store_true", help="also write the adapter alone as GGUF (llama-server --lora)")
    p.add_argument("--bf16", action="store_true")
    a = ap.parse_args(argv)
    a.backend = "hf"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"sft": cmd_sft, "pref": cmd_pref, "merge": cmd_merge, "gguf": cmd_gguf, "lora-gguf": cmd_lora_gguf, "pipeline": cmd_pipeline}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
