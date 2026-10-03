"""Prove the GPU-day training scripts end to end on this PC's CPU with a TINY Qwen3 (random weights), before any paid minute.

Run with the separate training venv (torch CPU, transformers, peft, trl, datasets, gguf - NOT Nupen's venv), at IDLE priority:
  ~/weekly7/.venv/Scripts/python.exe ~/weekly7/scripts/lowprio.py --idle ~/gpuday_cpu/env/Scripts/python.exe scripts/gpuday/selftest_cpu.py
      [--model ~/creator_runtime/gpuday/hf/tiny-qwen3] [--export ~/creator_runtime/gpuday/export] [--llama-cpp-src DIR] [--llama-bin DIR]
Steps (each one's result.json lands in --out): SFT on 4 exported coder rows -> DPO on the exported preference pairs -> merge -> GGUF f16 ->
llama-quantize Q4_K_M -> adapter GGUF -> GRPO 2 steps on 4 RL tasks. Prints one summary line per step and SELFTEST OK / FAILED."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(argv: list[str], log: Path) -> bool:
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT).returncode
    what = " ".join(Path(x).name for x in argv[1:4])
    status = "ok  " if rc == 0 else "FAIL"
    print(f"{status} {time.time() - t0:7.1f}s  {what}  (log {log.name})", flush=True)
    if rc != 0:
        print(log.read_text(encoding="utf-8")[-2500:])
    return rc == 0


def main(argv: list[str]) -> int:
    home = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(home / "creator_runtime" / "gpuday" / "hf" / "tiny-qwen3"))
    ap.add_argument("--export", default=str(home / "creator_runtime" / "gpuday" / "export"))
    ap.add_argument("--out", default=str(home / "creator_runtime" / "gpuday" / "selftest"))
    ap.add_argument("--llama-cpp-src", default=str(home / "gpuday_cpu" / "llama.cpp-b11351"))
    ap.add_argument("--llama-bin", default=str(home / "creator_runtime" / "llama"))
    a = ap.parse_args(argv)
    out, ex = Path(a.out), Path(a.export)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "sft4.jsonl"
    rows = sorted((ln for ln in (ex / "coder_sft_mix.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()), key=len)[:4]
    data.write_text("\n".join(rows) + "\n", encoding="utf-8")
    pref = ex / "pref_train.jsonl"
    rl = out / "rl4.jsonl"
    rlrows = [json.loads(ln) for ln in (ex / "rl_tasks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    pick = [r for r in rlrows if r["split"] == "train"][:4] + [r for r in rlrows if r["split"] == "eval"][:2]
    rl.write_text("".join(json.dumps(r) + "\n" for r in pick), encoding="utf-8")
    py, ft = sys.executable, str(HERE / "finetune.py")
    tiny = ["--max-seq", "16384", "--max-steps", "2", "--batch", "1", "--accum", "1", "--r", "4", "--alpha", "8", "--hf"]
    q = Path(a.llama_bin) / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")
    ok = run([py, ft, "sft", "--base", a.model, "--data", str(data), "--out", str(out / "sft"), *tiny], out / "sft.log")
    ok = ok and run([py, ft, "pref", "--method", "dpo", "--base", a.model, "--adapter", str(out / "sft" / "adapter"), "--data", str(pref),
                     "--out", str(out / "dpo"), *tiny], out / "dpo.log")
    ok = ok and run([py, ft, "merge", "--base", a.model, "--adapter", str(out / "dpo" / "adapter"), "--out", str(out / "merged")], out / "merge.log")
    ok = ok and run([py, ft, "gguf", "--merged", str(out / "merged"), "--llama-cpp", a.llama_cpp_src, "--quantize", str(q), "--quant", "Q4_K_M",
                     "--out", str(out / "gguf" / "tiny-Q4_K_M.gguf")], out / "gguf.log")
    ok = ok and run([py, ft, "lora-gguf", "--base", a.model, "--adapter", str(out / "dpo" / "adapter"), "--llama-cpp", a.llama_cpp_src,
                     "--out", str(out / "gguf" / "tiny-adapter.gguf")], out / "lora_gguf.log")
    ok = ok and run([py, str(HERE / "rl_grpo.py"), "--base", a.model, "--tasks", str(rl), "--out", str(out / "rl"), "--steps", "2", "--gens", "2",
                     "--batch", "1", "--max-seq", "640", "--max-new", "48", "--eval-n", "2", "--r", "4", "--alpha", "8", "--hf"], out / "rl.log")
    print("SELFTEST OK" if ok else "SELFTEST FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
