"""GPU-day RL proof of concept: GRPO (TRL; Unsloth's loader on the GPU) with a reward computed by RUNNING TESTS.

Smallest honest design:
  tasks   `make-tasks` (on the PC, in the repo): pure top-level functions of Nupen's own creator/*.py with simple annotated parameters
          (str/int/float/bool and lists of them). For each, inputs are sampled and the ORIGINAL function is the oracle: expected outputs
          become asserts (deterministic functions only: called twice, same answer). The prompt shows the signature + docstring + 2 example
          calls; the model writes the body. Held-out split by function (time order of the file's last commit is not needed: tasks are
          split by module so no function's neighbours leak).
  reward  fraction of asserts that pass when the completion is executed in a separate `python -I` process with a timeout (bad code,
          loops, crashes = 0). Format bonus 0.1 for a parsable `def`.
  run     `train` (the pod, or the CPU with a tiny model): GRPOTrainer, num_generations G, a few hundred steps on the 4090; before/after
          mean reward on the held-out tasks is the result (result.json). 'Worked' means held-out reward rose - not train reward.
Usage:
  python scripts/gpuday/rl_grpo.py make-tasks --repo . --out ~/creator_runtime/gpuday/export/rl_tasks.jsonl
  python rl_grpo.py --base Qwen/Qwen3-1.7B --tasks rl_tasks.jsonl --out runs/rl_poc --steps 150      (train is the default)"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

SIMPLE = {"str": lambda r: r.choice(["", "a", "abc", "Hello World", "x_y-z", "  pad  ", "K07.exists", "line1\nline2"]),
          "int": lambda r: r.choice([0, 1, 2, 3, 7, 10, -1, 42, 100]),
          "float": lambda r: r.choice([0.0, 0.5, 1.0, 2.5, -1.0, 3.14]),
          "bool": lambda r: r.choice([True, False])}
LISTS = {f"list[{k}]": k for k in SIMPLE} | {f"Sequence[{k}]": k for k in SIMPLE}
BANNED = re.compile(r"write|save|run|delete|remove|open|kill|start|stop|lock|git|http|request|sleep|print|main|rand|now|time", re.I)


def gen_value(ann: str, r: random.Random) -> Any:
    if ann in SIMPLE:
        return SIMPLE[ann](r)
    k = LISTS[ann]
    return [SIMPLE[k](r) for _ in range(r.randint(0, 4))]


def candidates(repo: Path) -> list[tuple[str, ast.FunctionDef, str]]:
    out = []
    for p in sorted((repo / "creator").glob("*.py")):
        src = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_") or BANNED.search(node.name):
                continue
            lines = (node.end_lineno or node.lineno) - node.lineno + 1
            args = node.args.args
            if not (1 <= len(args) <= 3 and 3 <= lines <= 30) or node.args.vararg or node.args.kwarg or node.decorator_list:
                continue
            anns = [ast.unparse(a.annotation) if a.annotation else "" for a in args]
            if not all(a in SIMPLE or a in LISTS for a in anns):
                continue
            out.append((f"creator.{p.stem}", node, ast.get_source_segment(src, node) or ""))
    return out


def make_tasks(repo: Path, out: Path, per_fn: int = 8, seed: int = 7, eval_frac: float = 0.25) -> dict[str, Any]:
    sys.path.insert(0, str(repo))
    r = random.Random(seed)
    tasks, skipped = [], 0
    for mod, node, src in candidates(repo):
        try:
            fn: Callable[..., Any] = getattr(importlib.import_module(mod), node.name)
        except Exception:                                        # noqa: BLE001 - a module that does not import is skipped
            skipped += 1
            continue
        anns = [ast.unparse(a.annotation) for a in node.args.args if a.annotation]
        cases = []
        for _ in range(per_fn * 3):
            args = [gen_value(a, r) for a in anns]
            try:
                y1, y2 = fn(*args), fn(*args)
                json.dumps(y1)
            except Exception:                                    # noqa: BLE001 - inputs the function rejects are not cases
                continue
            if y1 != y2:
                cases = []
                break
            if (args, y1) not in cases:
                cases.append((args, y1))
            if len(cases) >= per_fn:
                break
        if len(cases) < 3 or len({json.dumps(y) for _, y in cases}) < 2:   # constant functions teach nothing
            skipped += 1
            continue
        sig = src.split(":\n", 1)[0] if ":\n" in src else f"def {node.name}(...)"
        doc = ast.get_docstring(node) or ""
        shown = cases[:2]
        prompt = (f"Write the Python function below (standard library only; return the function definition in one ```python block).\n\n"
                  f"{sig}:\n    \"\"\"{doc}\"\"\"\n\nExamples:\n" + "\n".join(f"{node.name}(*{json.dumps(a)}) == {json.dumps(y)}" for a, y in shown))
        tasks.append({"id": f"{mod}:{node.name}", "module": mod, "name": node.name, "prompt": prompt,
                      "tests": [{"args": a, "expect": y} for a, y in cases]})
    mods = sorted({t["module"] for t in tasks})
    r.shuffle(mods)
    held = set(mods[: max(1, int(len(mods) * eval_frac))]) if mods else set()
    for t in tasks:
        t["split"] = "eval" if t["module"] in held else "train"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    return {"tasks": len(tasks), "train": sum(t["split"] == "train" for t in tasks), "eval": sum(t["split"] == "eval" for t in tasks),
            "skipped": skipped}


CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
RUNNER = """
import json, sys
src, name, tests = json.loads(sys.stdin.read())
ns = {}
try:
    exec(compile(src, "<completion>", "exec"), ns)
    f = ns[name]
except Exception:
    print(0); raise SystemExit
ok = 0
for t in tests:
    try:
        ok += json.loads(json.dumps(f(*t["args"]))) == t["expect"]
    except Exception:
        pass
print(ok)
"""


def reward_one(completion: str, name: str, tests: list[dict[str, Any]], timeout: float = 5.0) -> float:
    m = CODE_RE.search(completion)
    src = m.group(1) if m else completion
    if f"def {name}" not in src:
        return 0.0
    try:
        ast.parse(src)
    except SyntaxError:
        return 0.0
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps([src, name, tests]), capture_output=True, text=True,
                               timeout=timeout, cwd=td)
            ok = int((p.stdout.strip() or "0").splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError):
            ok = 0
    return 0.1 + 0.9 * ok / max(1, len(tests))


def completion_text(c: Any) -> str:
    if isinstance(c, list):                                     # conversational completions: [{'role': 'assistant', 'content': ...}]
        return "".join(str(m.get("content", "")) for m in c)
    return str(c)


def reward_fn(completions: list[Any], name: list[str], tests: list[str], **_: Any) -> list[float]:
    return [reward_one(completion_text(c), n, json.loads(t)) for c, n, t in zip(completions, name, tests)]


def mean_reward(model: Any, tok: Any, rows: list[dict[str, Any]], max_new: int) -> float:
    import torch
    scores = []
    for r in rows:
        ids = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        scores.append(reward_one(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), r["name"], json.loads(r["tests"])))
    return round(sum(scores) / max(1, len(scores)), 4)


def train(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import finetune as FT                                       # the same loader (Unsloth on the GPU, transformers+peft on a CPU)
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    rows = [json.loads(ln) for ln in Path(a.tasks).read_text(encoding="utf-8").splitlines() if ln.strip()]
    conv = lambda t: {"prompt": [{"role": "user", "content": t["prompt"]}], "name": t["name"], "tests": json.dumps(t["tests"])}  # noqa: E731
    tr = [conv(t) for t in rows if t.get("split") != "eval"]
    ev = [conv(t) for t in rows if t.get("split") == "eval"][: a.eval_n]
    if not tr:
        raise SystemExit("no training tasks")
    model, tok, backend = FT.load(a.base, a.max_seq, a.qlora, a.r, a.alpha, a.hf)
    before = mean_reward(model, tok, ev, a.max_new) if ev else None
    cfg = FT.common_args(argparse.Namespace(out=a.out, batch=a.batch, accum=1, lr=a.lr, epochs=1, max_steps=a.steps, backend=backend),
                         GRPOConfig, num_generations=a.gens, max_completion_length=a.max_new, max_prompt_length=a.max_seq - a.max_new,
                         temperature=0.9, beta=0.0, generation_batch_size=a.batch * a.gens if a.batch * a.gens >= a.gens else a.gens)
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=cfg, train_dataset=Dataset.from_list(tr), processing_class=tok)
    st = trainer.train()
    after = mean_reward(model, tok, ev, a.max_new) if ev else None
    rew = [h.get("reward") for h in trainer.state.log_history if "reward" in h]
    ad = Path(a.out) / "adapter"
    model.save_pretrained(str(ad))
    res = {"backend": backend, "base": a.base, "train_tasks": len(tr), "eval_tasks": len(ev), "steps": st.global_step,
           "train_reward_first": rew[0] if rew else None, "train_reward_last": rew[-1] if rew else None,
           "heldout_reward_before": before, "heldout_reward_after": after,
           "worked": bool(before is not None and after is not None and after > before), "seconds": round(time.time() - t0, 1)}
    return FT.write_result(Path(a.out), "grpo", res)


def main(argv: list[str]) -> int:
    if argv and argv[0] == "make-tasks":
        ap = argparse.ArgumentParser()
        ap.add_argument("--repo", default=".")
        ap.add_argument("--out", required=True)
        ap.add_argument("--per-fn", type=int, default=8)
        a = ap.parse_args(argv[1:])
        print(json.dumps(make_tasks(Path(a.repo).resolve(), Path(a.out).expanduser(), a.per_fn)))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-seq", type=int, default=1536)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--hf", action="store_true")
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
