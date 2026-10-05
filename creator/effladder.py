"""EFFICIENCY LADDER (owner, 3 Oct 2026: "we should probably spend today on the GPU collecting data to make Nupen as efficent as possible getting
the most done with the least prossessing power, and memory, and disk and any other limitation").

Loaded ON DEMAND only (python -m creator.effladder ..., or a GPU-runner 'call' job); never part of the swarm's eager start load.

The data lets Nupen pick, PER TASK TYPE, the CHEAPEST setup that is good enough (router / cascade data) and what each setup costs AT HOME:

  freeze     the item sets, chosen ONCE and stored with a content hash (<runtime>/gpu/ladder/items.json) so every model answers the SAME items:
               reasoning   held-out creator.reasondrills questions (frozen-benchmark collisions already removed by reasondrills.generate; a
                           deterministic hash split; never a question of the worked-example bank)
               judgment    held-out public-commit cases of creator.judgment (pub_git_fixed / pub_git_churn, >= PUB_MIN_HISTORY resolved before)
               thinkbench  part c of the FROZEN thinking benchmark (hash FROZEN_HASH, read through thinkbench.load_frozen): READ-ONLY, measured,
                           never a distillation row
               coding      creator.gpuday's rl_tasks eval split + public MBPP (sanitized test) + HumanEval, scored by EXECUTING their tests
             plus a separate DISTILL pool (reasoning questions and MBPP train/validation + rl_tasks train) that is never an eval item.
  run        one model (one served endpoint) answers every eval item under several STRATEGIES x max_tokens budgets (CONFIGS); per answer one
             row in <state>/thinking/effladder.jsonl: correct, tokens in/out, latency, model, quant, strategy, budget. Resumable (an answered
             (model, item, config) is skipped). As a GPU-runner job: call 'creator.effladder:ladder_job' (ctx['args'] = run options).
  distill    when a big model's answer to a DISTILL-pool item is verified correct (known answer / executed tests) the (prompt, answer) pair is
             appended to <runtime>/gpuday/distill/ladder_distill.jsonl as a chat row for the small home models. Same rules as the gpuday
             export: gpuday.private_reason drops the row, gpuday.scrub cleans it, no eval item and no frozen-benchmark item is ever written.
  summary    accuracy vs cost per task type and config; per task type the CHEAPEST setup within `points` of the best (cost = estimated
             home CPU seconds from the home-cost table, else weight-bytes x tokens).
  homecost   on THIS PC, at IDLE priority: llama-bench tokens/s (prompt + generation), peak RSS and disk per local GGUF; quants that are not
             on disk are ESTIMATED from the measured file of the same size (decode is memory-bound: tok/s ~ 1 / weight bytes).
  jobs       the GPU-runner job list (fetch -> ladder -> delete per model, disk-safe on a pod with ~10 GB free).

Nothing here starts or stops a server: a job talks to whatever the runner (creator.gpupulse) serves, or to an explicit port."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1

# ------------------------------------------------------------------------------------------------ the model ladder (HF API, read 3 Oct 2026)
# sha256 = the Hugging Face LFS oid (api/models/<repo>/tree/main). 4B Q4/Q5/Q8 come from the official Qwen repo (its Q4_K_M is the file the
# runner's CATALOG already pins); everything else from unsloth (the official repos ship only Q8_0 for 0.6B/1.7B).
LADDER: dict[str, dict[str, Any]] = {
    "Qwen3-0.6B-Q3_K_M.gguf": {"repo": "unsloth/Qwen3-0.6B-GGUF", "bytes": 347127488, "sha256": "30bb9f9222cf58c076277879881a24e7127a10be32076d76d8c70f8defc01f5d"},
    "Qwen3-0.6B-Q4_K_M.gguf": {"repo": "unsloth/Qwen3-0.6B-GGUF", "bytes": 396705472, "sha256": "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"},
    "Qwen3-0.6B-Q5_K_M.gguf": {"repo": "unsloth/Qwen3-0.6B-GGUF", "bytes": 444415680, "sha256": "03c6e2127d155b89c21a512954010486b1e00e1a9eebdfad650d03b53ab4c74a"},
    "Qwen3-0.6B-Q8_0.gguf": {"repo": "unsloth/Qwen3-0.6B-GGUF", "bytes": 639447744, "sha256": "e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c"},
    "Qwen3-1.7B-Q3_K_M.gguf": {"repo": "unsloth/Qwen3-1.7B-GGUF", "bytes": 939539008, "sha256": "8070f21b9763ef653289e20252592d0aab6680948f36eb5b4362d403fef09170"},
    "Qwen3-1.7B-Q4_K_M.gguf": {"repo": "unsloth/Qwen3-1.7B-GGUF", "bytes": 1107409472, "sha256": "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897"},
    "Qwen3-1.7B-Q5_K_M.gguf": {"repo": "unsloth/Qwen3-1.7B-GGUF", "bytes": 1257880128, "sha256": "b0949de5b2e06cbed6aa96517f9bd8afb334584b6f95ee83479292ff4bdd8ed3"},
    "Qwen3-1.7B-Q8_0.gguf": {"repo": "unsloth/Qwen3-1.7B-GGUF", "bytes": 1834426944, "sha256": "0becaa825564295d82e9af4d008bca5f8b7f5f73bf1c6a0b58f7c53ef26b47fd"},
    "Qwen3-4B-Q3_K_M.gguf": {"repo": "unsloth/Qwen3-4B-GGUF", "bytes": 2075618592, "sha256": "e78ff54a67f3b3f57e637fbf0770d5b73c1e215ea7c0153980f1e8140c2b2486"},
    "Qwen3-4B-Q4_K_M.gguf": {"repo": "Qwen/Qwen3-4B-GGUF", "bytes": 2497280256, "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"},
    "Qwen3-4B-Q5_K_M.gguf": {"repo": "Qwen/Qwen3-4B-GGUF", "bytes": 2889513184, "sha256": "aca596860e8cb40af6539e3f2ea40df305f42515deac56d49c08d39a02e6533f"},
    "Qwen3-4B-Q8_0.gguf": {"repo": "Qwen/Qwen3-4B-GGUF", "bytes": 4280404704, "sha256": "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300"},
    "Qwen3-14B-Q4_K_M.gguf": {"repo": "Qwen/Qwen3-14B-GGUF", "bytes": 9001752960, "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"},
}
CODER_27B = "Qwen3.6-27B-Q4_K_M.gguf"     # on the pod's RAM disk (/dev/shm/coder), served by the teacher's coder stage on pod/local port 18400
CODER_27B_PORT = 18400
# models the runner keeps on the pod (pulse.json 'models' / its CATALOG): the ladder never deletes these
KEEP_ON_POD = ("Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf")
NAME_RE = re.compile(r"Qwen3(?:\.\d+)?-([\d.]+)B-(Q\d_K_M|Q\d_0|[A-Z0-9_]+)\.gguf$", re.I)


def model_facts(name: str) -> dict[str, Any]:
    """{'params_b', 'quant', 'bytes'} from the file name (and LADDER)."""
    m = NAME_RE.search(Path(name).name)
    e = LADDER.get(Path(name).name, {})
    return {"params_b": float(m.group(1)) if m else None, "quant": m.group(2).upper() if m else "?", "bytes": e.get("bytes")}


# ------------------------------------------------------------------------------------------------ configs: strategy x max_tokens per suite
@dataclass(frozen=True)
class Config:
    strategy: str               # plain | cot | retrieval | think   (the prompt variant; 'think' = the model's own thinking mode on)
    max_tokens: int

    @property
    def think(self) -> bool:
        return self.strategy == "think"

    @property
    def key(self) -> str:
        return f"{self.strategy}@{self.max_tokens}"


def cfgs(*pairs: tuple[str, int]) -> list[Config]:
    return [Config(s, n) for s, n in pairs]


# per-slot context on the pod is 4096 tokens (pulse.json ctx_per_slot): prompt + budget stays under it
CONFIGS: dict[str, list[Config]] = {
    "reasoning": cfgs(("plain", 32), ("cot", 256), ("retrieval", 256), ("think", 1024), ("think", 2048)),
    "thinkbench": cfgs(("plain", 32), ("cot", 256), ("think", 1024), ("think", 2048)),
    "judgment": cfgs(("plain", 64), ("cot", 256), ("retrieval", 64), ("think", 1024)),
    "coding": cfgs(("plain", 1024), ("retrieval", 1024), ("think", 1536), ("think", 3000)),
}
SUITES = tuple(CONFIGS)
DISTILL_CONFIG = {"reasoning": Config("cot", 256), "coding": Config("plain", 1024)}   # the answer format a small home model is trained on


def parse_configs(spec: str, suite: str) -> list[Config]:
    """'plain@32,think@1024' -> those configs (restricted to the suite's strategies); '' -> the suite's defaults."""
    if not spec:
        return list(CONFIGS[suite])
    known = {c.strategy for c in CONFIGS[suite]}
    out = []
    for part in spec.split(","):
        s, _, n = part.strip().partition("@")
        if s in known and n.isdigit():
            out.append(Config(s, int(n)))
    return out


# ------------------------------------------------------------------------------------------------ paths
def ladder_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    from creator import device as DEV
    return DEV.runtime_dir(env) / "gpu" / "ladder"


def items_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return ladder_dir(env) / "items.json"


def results_path(state: Path) -> Path:
    return Path(state) / "thinking" / "effladder.jsonl"


def distill_path(env: Optional[Mapping[str, str]] = None) -> Path:
    from creator import device as DEV
    return DEV.runtime_dir(env) / "gpuday" / "distill" / "ladder_distill.jsonl"


def homecost_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return ladder_dir(env) / "homecost.json"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


# ------------------------------------------------------------------------------------------------ item builders (freeze)
CODE_SYSTEM = "You are a careful Python programmer. Reply with the complete function(s) in one ```python code block and nothing else."


def _user(msgs: list[dict[str, str]]) -> str:
    return next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")


def reasoning_items(state: Path, repo: Path, n_eval: int, n_distill: int, eval_mod: int = 5) -> list[dict[str, Any]]:
    """Held-out reasoning questions (qid hash % eval_mod == 0 and not in the worked-example bank) and a distill pool (the rest, bank last)."""
    from creator import reasondrills as R
    qs = R.generate(R.repos(repo), state)
    bank = R.trace_bank(state)
    ordered = R.order(qs)
    ev = [q for q in ordered if _h(q.qid) % eval_mod == 0 and q.qid not in bank][:n_eval]
    rest = [q for q in ordered if _h(q.qid) % eval_mod != 0]
    pool = [q for q in rest if q.qid not in bank][: max(n_distill, 3000)]
    traces = {k: str(v.get("trace", "")) for k, v in bank.items()}
    solved = [(s, None, 0.0) for s in pool]
    st = {s.name: s for s in R.STRATEGIES}
    styles = {"plain": st["plain"], "cot": st["cot"], "think": R.Strategy("think", "think")}
    out: list[dict[str, Any]] = []
    for split, group in (("eval", ev), ("distill", pool[:n_distill])):
        for q in group:
            msgs = {k: R.build_messages(s, q) for k, s in styles.items()}
            if split == "eval":
                ex = R.solved_examples(q, solved, 4, math.inf)
                msgs["retrieval"] = R.build_messages(st["retrieve4"], q, ex, traces)
            out.append({"id": f"reasoning:{q.qid}", "suite": "reasoning", "kind": q.kind, "split": split, "answer": q.answer,
                        "check": "mc", "messages": msgs})
    return out


def judgment_items(state: Path, repo: Path, n_eval: int) -> list[dict[str, Any]]:
    """Held-out public-commit cases (n_eval/2 per topic, newest-first by a hash among cases with a long resolved history before them)."""
    from creator import judgment as J
    strategies = {"plain": dict(J.REF_STRATEGY), "cot": dict(J.REF_STRATEGY, structured=1), "retrieval": {"shots": 5, "hint": 1},
                  "think": dict(J.REF_STRATEGY)}
    out: list[dict[str, Any]] = []
    for topic in J.PUB_TOPICS:
        cases = J.load_cases(topic, state, repo)
        if not cases:
            continue
        created = sorted(c.resolved for c in cases)
        import bisect
        ok = [c for c in cases if bisect.bisect_left(created, c.created) >= J.PUB_MIN_HISTORY]
        ok.sort(key=lambda c: _h(f"ladder:{c.subject}"))
        for c in ok[: n_eval // 2]:
            msgs = {k: J.build_prompt(s, cases, c) for k, s in strategies.items()}
            out.append({"id": f"judgment:{topic}:{c.subject}", "suite": "judgment", "kind": topic, "split": "eval", "answer": int(c.y),
                        "check": "prob", "messages": msgs})
    return out


def thinkbench_items(state: Path) -> list[dict[str, Any]]:
    """Part c of the FROZEN benchmark (read-only; the content hash is checked by thinkbench.load_frozen and must match gpuday.FROZEN_HASH)."""
    from creator import gpuday as GD
    from creator import thinkbench as TB
    fz = TB.load_frozen(state)
    if not str(fz.get("hash", "")).startswith(GD.FROZEN_HASH):
        raise ValueError(f"thinkbench items hash {str(fz.get('hash'))[:12]} is not the frozen {GD.FROZEN_HASH}")
    out = []
    for q in fz.get("c") or []:
        cot = TB._c_messages(q)
        plain = [{"role": "system", "content": cot[0]["content"].replace(
            "Use the procedure: think briefly (at most three short lines), then finish with exactly 'ANSWER: <letter>'.",
            "Reply with exactly one line: 'ANSWER: <letter>'.")}, cot[1]]
        out.append({"id": f"thinkbench:{q['id']}", "suite": "thinkbench", "kind": str(q.get("kind")), "split": "eval", "frozen": True,
                    "answer": int(q["answer"]), "check": "mc", "messages": {"plain": plain, "cot": cot, "think": cot}})
    return out


HF_ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"


def hf_rows(ds: str, cfg: str, split: str, fetch: Optional[Callable[[str], bytes]] = None) -> list[dict[str, Any]]:
    """Every row of a public Hugging Face dataset split through the datasets-server JSON API (no pyarrow needed)."""
    get = fetch or (lambda u: urllib.request.urlopen(u, timeout=120).read())
    out: list[dict[str, Any]] = []
    off = 0
    while True:
        d = json.loads(get(HF_ROWS.format(ds=ds, cfg=cfg, split=split, off=off)))
        rows = [r["row"] for r in d.get("rows") or []]
        out += rows
        off += len(rows)
        if not rows or off >= int(d.get("num_rows_total") or 0):
            return out


def _mbpp_prompt(r: Mapping[str, Any]) -> str:
    return f"{r['prompt']}\nYour code should pass this test:\n{r['test_list'][0]}"


def _code_msgs(prompt: str, examples: Sequence[tuple[str, str]] = ()) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": CODE_SYSTEM}]
    for p, code in examples:
        msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": f"```python\n{code.strip()}\n```"}]
    return msgs + [{"role": "user", "content": prompt}]


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def coding_items(export_dir: Path, fetch: Optional[Callable[[str], bytes]] = None, mbpp: bool = True, humaneval: bool = True) -> list[dict[str, Any]]:
    """Eval: rl_tasks eval split + MBPP sanitized test + HumanEval. Distill pool: MBPP sanitized train/validation/prompt (with reference code:
    also the retrieval examples) + rl_tasks train. Tests travel with the item; they are executed on THIS PC (run_code)."""
    from creator import gpuday as GD
    rl = GD.jsonl_rows(export_dir / "rl_tasks.jsonl") if (export_dir / "rl_tasks.jsonl").is_file() else []
    out: list[dict[str, Any]] = []
    pool: list[tuple[str, str]] = []
    if mbpp:
        for split in ("train", "validation", "prompt"):
            for r in hf_rows("google-research-datasets/mbpp", "sanitized", split, fetch):
                p = _mbpp_prompt(r)
                pool.append((p, str(r["code"])))
                out.append({"id": f"coding:mbpp:{r['task_id']}", "suite": "coding", "kind": "mbpp", "split": "distill", "check": "py",
                            "test": {"asserts": list(r["test_list"]), "imports": list(r.get("test_imports") or [])},
                            "messages": {"plain": _code_msgs(p)}})
    for r in rl:
        if r.get("split") not in ("train", "eval"):
            continue
        out.append({"id": f"coding:rl:{r['id']}", "suite": "coding", "kind": "nupen_fn", "split": "eval" if r["split"] == "eval" else "distill",
                    "check": "py", "test": {"name": r["name"], "cases": r["tests"]}, "messages": {"plain": _code_msgs(str(r["prompt"]))}})
    if mbpp:
        for r in hf_rows("google-research-datasets/mbpp", "sanitized", "test", fetch):
            out.append({"id": f"coding:mbpp:{r['task_id']}", "suite": "coding", "kind": "mbpp", "split": "eval", "check": "py",
                        "test": {"asserts": list(r["test_list"]), "imports": list(r.get("test_imports") or [])},
                        "messages": {"plain": _code_msgs(_mbpp_prompt(r))}})
    if humaneval:
        for r in hf_rows("openai/openai_humaneval", "openai_humaneval", "test", fetch):
            p = f"Complete this Python function:\n\n```python\n{r['prompt']}```"
            out.append({"id": f"coding:{r['task_id']}", "suite": "coding", "kind": "humaneval", "split": "eval", "check": "py",
                        "test": {"prefix": r["prompt"], "test": r["test"], "entry": r["entry_point"]}, "messages": {"plain": _code_msgs(p)}})
    for it in out:                                            # retrieval: the 2 most similar distill-pool tasks with their reference code
        if it["split"] == "eval":
            p = _user(it["messages"]["plain"])
            w = _words(p)
            ex = sorted((x for x in pool if x[0] != p), key=lambda x: (-len(w & _words(x[0])), x[0]))[:2]
            it["messages"]["retrieval"] = _code_msgs(p, ex)
        it["messages"]["think"] = it["messages"]["plain"]
    return out


def freeze(state: Path, repo: Path, out: Optional[Path] = None, n_reason: int = 160, n_distill: int = 600, n_judge: int = 160,
           export_dir: Optional[Path] = None, fetch: Optional[Callable[[str], bytes]] = None, suites: Sequence[str] = SUITES) -> dict[str, Any]:
    """Choose every item ONCE; store with a content hash. Refuses a private-marker prompt (the items go to the pod)."""
    from creator import device as DEV
    from creator import gpupulse as GP
    items: list[dict[str, Any]] = []
    if "reasoning" in suites:
        items += reasoning_items(state, repo, n_reason, n_distill)
    if "judgment" in suites:
        items += judgment_items(state, repo, n_judge)
    if "thinkbench" in suites:
        items += thinkbench_items(state)
    if "coding" in suites:
        items += coding_items(export_dir or DEV.runtime_dir() / "gpuday" / "export", fetch)
    kept = []
    for it in items:
        try:
            for msgs in it["messages"].values():
                GP.outbound_ok(msgs)
            kept.append(it)
        except GP.PulseError:
            continue
    h = _sha({"version": VERSION, "items": kept})
    body: dict[str, Any] = {"version": VERSION, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "items": kept, "hash": h}
    p = GP.outside_repo(out or items_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body), encoding="utf-8")
    return {"file": str(p), "hash": h[:12], "counts": counts(kept), "dropped_private": len(items) - len(kept)}


def counts(items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    c: dict[str, dict[str, int]] = {}
    for it in items:
        d = c.setdefault(str(it["suite"]), {})
        d[str(it["split"])] = d.get(str(it["split"]), 0) + 1
    return c


def load_items(path: Optional[Path] = None) -> dict[str, Any]:
    body = json.loads((path or items_path()).read_text(encoding="utf-8"))
    if _sha({"version": body.get("version"), "items": body.get("items")}) != body.get("hash"):
        raise ValueError("ladder items.json was edited after it was frozen (content hash mismatch)")
    return dict(body)


# ------------------------------------------------------------------------------------------------ scoring
THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*\(?\**\s*([A-D])\b", re.I)
PROB_RE = re.compile(r"PROBABILITY\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)", re.I)
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
RUNNER = r"""
import json, sys
prog = sys.stdin.buffer.read().decode("utf-8")
ns = {"__name__": "__ladder__"}
try:
    exec(compile(prog, "<candidate>", "exec"), ns)
except BaseException as e:
    print("FAIL", type(e).__name__); raise SystemExit(0)
print("PASS")
"""


def answer_text(content: str) -> str:
    return THINK_RE.sub("", content or "").strip()


def mc_pick(text: str) -> Optional[int]:
    m = ANSWER_RE.findall(text)
    return "ABCD".index(m[-1].upper()) if m else None


def prob_of(text: str) -> Optional[float]:
    m = PROB_RE.findall(text)
    return min(0.97, max(0.03, float(m[-1]))) if m else None


def extract_code(text: str, name: str = "") -> str:
    blocks = CODE_RE.findall(text) or ([text] if "def " in text else [])
    if name:
        named = [b for b in blocks if f"def {name}" in b]
        blocks = named or blocks
    return max(blocks, key=len) if blocks else ""


def program(test: Mapping[str, Any], code: str) -> str:
    """The test program for one candidate (stdlib only; the candidate's code first)."""
    if "asserts" in test:
        return "\n".join(list(test.get("imports") or []) + [code, ""] + list(test["asserts"])) + "\n"
    if "cases" in test:
        cases = json.dumps(test["cases"])
        return (f"{code}\n\nimport json as _j\n_f = {test['name']}\nfor _t in _j.loads({cases!r}):\n"
                f"    assert _j.loads(_j.dumps(_f(*_t['args']))) == _t['expect']\n")
    entry = str(test["entry"])
    src = code if f"def {entry}" in code else str(test["prefix"]) + code
    head = "\n".join(ln for ln in str(test["prefix"]).splitlines() if ln.startswith(("import ", "from ")))
    return f"{head}\n{src}\n\n{test['test']}\n\ncheck({entry})\n"


IDLE_PRIORITY_CLASS = 0x00000040


def run_code(prog: str, timeout: float = 10.0) -> bool:
    """Execute a candidate in a separate isolated interpreter (python -I, a temporary cwd, a timeout) at IDLE priority: True iff it passes."""
    flags = IDLE_PRIORITY_CLASS if sys.platform == "win32" else 0
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=prog.encode("utf-8"), capture_output=True, timeout=timeout, cwd=td,
                               creationflags=flags)
        except subprocess.TimeoutExpired:
            return False
    return p.stdout.decode("utf-8", "replace").strip().startswith("PASS")


def score(item: Mapping[str, Any], content: str) -> dict[str, Any]:
    """{'correct': bool, ...} for one reply (a truncated think block leaves no answer: wrong)."""
    text = answer_text(content)
    if item["check"] == "mc":
        pick = mc_pick(text)
        return {"correct": pick == item["answer"], "pick": pick}
    if item["check"] == "prob":
        p = prob_of(text)
        y = int(item["answer"])
        return {"correct": p is not None and (p >= 0.5) == bool(y), "p": p,
                "brier": round((p - y) ** 2, 4) if p is not None else None}
    t = item["test"]
    code = extract_code(text, str(t.get("name") or t.get("entry") or ""))
    return {"correct": bool(code) and run_code(program(t, code)), "code_chars": len(code)}


# ------------------------------------------------------------------------------------------------ the client and one run
class Endpoint:
    """An OpenAI-compatible llama-server (through the pulse tunnel or local). Records tokens in/out and latency per call."""

    def __init__(self, port: int, model: str, host: str = "127.0.0.1", timeout: float = 900.0) -> None:
        self.url, self.model, self.timeout = f"http://{host}:{int(port)}/v1/chat/completions", model, timeout

    def messages_for(self, msgs: Sequence[Mapping[str, str]], think: bool) -> list[dict[str, str]]:
        out = [dict(m) for m in msgs]
        if Path(self.model).name.lower().startswith("qwen3") and out and out[-1]["role"] == "user":
            out[-1]["content"] += "\n/think" if think else "\n/no_think"      # Qwen3 soft switch (works with or without --jinja)
        return out

    def call(self, msgs: Sequence[Mapping[str, str]], max_tokens: int, think: bool, seed: int = 0) -> dict[str, Any]:
        from creator import gpupulse as GP
        m = self.messages_for(msgs, think)
        GP.outbound_ok(m)
        body = {"messages": m, "max_tokens": int(max_tokens), "temperature": 0.6 if think else 0.2, "top_p": 0.95 if think else 0.8,
                "seed": seed, "chat_template_kwargs": {"enable_thinking": bool(think)}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        from creator import slowpath as SP                    # P0.2: the one accounting path of a model call
        with SP.model_call(self.model, backend_kind="effladder") as mc:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
            u = d.get("usage") or {}
            mc.tokens(u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0)
        ch = d["choices"][0]
        tm = d.get("timings") or {}
        return {"content": str(ch["message"].get("content") or ""), "reasoning_chars": len(str(ch["message"].get("reasoning_content") or "")),
                "finish": ch.get("finish_reason"), "tok_in": int(u.get("prompt_tokens") or 0), "tok_out": int(u.get("completion_tokens") or 0),
                "latency_s": round(time.monotonic() - t0, 3), "gen_tps": round(float(tm["predicted_per_second"]), 1) if tm.get("predicted_per_second") else None}


_LOCK = threading.Lock()


def _append(p: Path, row: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_rows(p: Path) -> list[dict[str, Any]]:
    out = []
    try:
        with p.open(encoding="utf-8") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def distill_row(item: Mapping[str, Any], cfg: Config, content: str, model: str, eval_prompts: set[str]) -> Optional[dict[str, Any]]:
    """The (prompt, verified answer) chat row for small models, or None: only DISTILL-pool items, never a frozen item or an eval prompt,
    dropped when gpuday.private_reason objects; scrubbed like the gpuday export."""
    from creator import gpuday as GD
    if item.get("split") != "distill" or item.get("frozen"):
        return None
    msgs = [dict(m) for m in item["messages"][cfg.strategy]]
    if _user(msgs) in eval_prompts:
        return None
    ans = answer_text(content)
    if item["check"] == "py":
        code = extract_code(ans, str(item["test"].get("name") or ""))
        ans = f"```python\n{code.strip()}\n```"
    text = json.dumps(msgs) + ans
    if GD.private_reason(text):
        return None
    return {"messages": GD.scrub_obj(msgs) + [{"role": "assistant", "content": GD.scrub(ans)}],
            "meta": {"id": item["id"], "suite": item["suite"], "kind": item["kind"], "teacher": model, "config": cfg.key, "verified": True}}


def run(items_body: Mapping[str, Any], ep: Endpoint, state: Path, *, suites: Sequence[str] = SUITES, configs: Optional[Mapping[str, str]] = None,
        n: Optional[Mapping[str, int]] = None, workers: int = 8, deadline: float = math.inf, distill: bool = False, n_distill: int = 0,
        run_id: str = "", pulse: str = "", say: Callable[[str], None] = print, out: Optional[Path] = None,
        distill_out: Optional[Path] = None, score_workers: int = 4) -> dict[str, Any]:
    """One model answers every chosen eval item under every config (plus, with distill, the DISTILL pool under DISTILL_CONFIG). Rows are
    appended as they come; an (items hash, model, item, config) already recorded is skipped (restart-safe)."""
    out = out or results_path(state)
    distill_out = distill_out or distill_path()
    ih = str(items_body["hash"])[:12]
    facts = model_facts(ep.model)
    done = {(r["item"], r["config"]) for r in read_rows(out) if r.get("items_hash") == ih and r.get("model") == ep.model}
    items = list(items_body["items"])
    eval_prompts = {_user(m) for it in items if it["split"] == "eval" for m in it["messages"].values()}
    work: list[tuple[dict[str, Any], Config]] = []
    for s in suites:
        cs = parse_configs((configs or {}).get(s, ""), s)
        ev = sorted((it for it in items if it["suite"] == s and it["split"] == "eval"), key=lambda it: _h(str(it["id"])))
        ev = ev[: (n or {}).get(s) or None]                   # a fixed hash-ordered sample: every model gets the SAME subset, kinds mixed
        for c in cs:                                          # config-major: a deadline leaves whole configs, not a ragged mix
            work += [(it, c) for it in ev if c.strategy in it["messages"]]
        if distill and s in DISTILL_CONFIG:
            dc = DISTILL_CONFIG[s]
            pool = sorted((x for x in items if x["suite"] == s and x["split"] == "distill"), key=lambda x: _h(str(x["id"])))
            work += [(it, dc) for it in pool[:n_distill or None]]
    work = [(it, c) for it, c in work if (it["id"], c.key) not in done]
    say(f"ladder {ep.model}: {len(work)} calls to make ({len(done)} already recorded), {workers} in flight")
    stats: dict[str, Any] = {"calls": 0, "correct": 0, "errors": 0, "distilled": 0, "tok_out": 0}

    def finish(it: dict[str, Any], c: Config, r: dict[str, Any]) -> None:
        sc = score(it, r["content"])
        row = {"ts": round(time.time(), 1), "run": run_id, "gpu_pulse": pulse, "items_hash": ih, "model": ep.model, "quant": facts["quant"],
               "params_b": facts["params_b"], "suite": it["suite"], "kind": it["kind"], "split": it["split"], "item": it["id"],
               "strategy": c.strategy, "max_tokens": c.max_tokens, "config": c.key, "think": c.think,
               "truncated": r["finish"] == "length", **{k: v for k, v in r.items() if k != "content"}, **sc}
        _append(out, row)
        with _LOCK:
            stats["calls"] += 1
            stats["correct"] += int(bool(sc["correct"]))
            stats["tok_out"] += int(r["tok_out"])
        if distill and sc["correct"] and it["split"] == "distill":
            d = distill_row(it, c, r["content"], ep.model, eval_prompts)
            if d is not None:
                _append(distill_out, d)
                with _LOCK:
                    stats["distilled"] += 1

    pending: list[cf.Future[None]] = []
    scorer = cf.ThreadPoolExecutor(max_workers=max(1, score_workers))   # executed tests never hold a request slot (the GPU stays fed)

    def one(it: dict[str, Any], c: Config) -> None:
        if time.monotonic() > deadline:
            return
        try:
            r = ep.call(it["messages"][c.strategy], c.max_tokens, c.think)
        except Exception as e:                                # noqa: BLE001 - a failed call is recorded, the run goes on
            with _LOCK:
                stats["errors"] += 1
            say(f"  {it['id']} {c.key}: {type(e).__name__}: {str(e)[:120]}")
            return
        if it["check"] == "py":
            with _LOCK:
                pending.append(scorer.submit(finish, it, c, r))
        else:
            finish(it, c, r)

    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(lambda w: one(*w), work))
    for f in list(pending):
        f.result()
    scorer.shutdown()
    stats["seconds"] = round(time.monotonic() - t0, 1)
    stats["agg_tok_s"] = round(stats["tok_out"] / max(1e-6, stats["seconds"]), 1)
    stats["stopped_at_deadline"] = time.monotonic() > deadline
    say(f"ladder {ep.model}: {json.dumps(stats)}")
    return stats


def ladder_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """GPU-runner 'call' job (creator.gpupulse.ext_job). ctx['args']: model (label; default ctx['model']), port (explicit endpoint, e.g. 18400
    for the 27B coder stage; default: the runner's tunnel for the model), suites, configs {suite: 'strategy@n,...'}, n {suite: items},
    distill (bool), n_distill, workers, items (path)."""
    from creator import gpupulse as GP
    a = dict(ctx.get("args") or {})
    model = str(a.get("model") or ctx.get("model") or "")
    if a.get("port"):
        port = int(a["port"])
    else:
        got = GP.attach(Path(str(ctx["tunnel_file"])), model)
        if got is None:
            raise GP.PulseError(f"the pulse does not serve {model}")
        port = got[0]
    body = load_items(Path(a["items"]) if a.get("items") else None)
    workers = int(a.get("workers") or ctx.get("workers") or 8)
    ep = Endpoint(port, model)
    res = run(body, ep, Path(ctx["state"]), suites=list(a.get("suites") or SUITES), configs=a.get("configs") or None, n=a.get("n") or None,
              workers=workers, deadline=float(ctx.get("deadline") or math.inf), distill=bool(a.get("distill")),
              n_distill=int(a.get("n_distill") or 0), run_id=str(a.get("run") or f"ladder-{model}"), pulse=str(ctx.get("pulse") or ""))
    return dict(res, model=model, items_hash=str(body["hash"])[:12])


# ------------------------------------------------------------------------------------------------ the GPU-runner job list
def fetch_script(rdir: str, name: str) -> str:
    """Pod-side: enough disk? then fetch + sha256-verify (gpupulse.model_script: a verified copy is kept, a bad file deleted)."""
    from creator import gpupulse as GP
    e = LADDER[name]
    need_gb = e["bytes"] / 1e9 + 1.0
    head = (f"mkdir -p {shlex.quote(rdir)}/models; free=$(df -P {shlex.quote(rdir)}/models | awk 'NR==2 {{print int($4/1048576)}}'); "
            f"if [ ! -f {shlex.quote(rdir)}/models/{shlex.quote(name)} ] && [ \"$free\" -lt {math.ceil(need_gb)} ]; then "
            f"echo '@@result={{\"fetched\": false, \"error\": \"disk: '$free' GB free\"}}'; exit 3; fi\n")
    body = GP.model_script(rdir, name, f"{GP.HF_BASE}/{e['repo']}/resolve/main/{name}", int(e["bytes"]), str(e["sha256"]))
    return head + "( " + body.replace("set -u\n", "", 1) + ") | tee /tmp/ladder_fetch.out\n" + \
        "grep -q '@@error' /tmp/ladder_fetch.out && { echo '@@result={\"fetched\": false}'; exit 4; }\n" + \
        f"echo '@@result={{\"fetched\": true, \"model\": \"{name}\"}}'\n"


def delete_script(rdir: str, name: str) -> str:
    if name in KEEP_ON_POD:
        return "echo '@@result={\"deleted\": false, \"kept\": true}'\n"
    q = shlex.quote(f"{rdir}/models/{name}")
    return f"rm -f {q} {q}.ok {q}.part; echo '@@result={{\"deleted\": true, \"model\": \"{name}\"}}'\n"


# The plan, in tiers (the teacher cuts from the end). Expected minutes = the job's estimate on the RTX 5090 at 32 slots from assumed aggregate
# decode rates (0.6B ~2500, 1.7B ~1600, 4B ~900, 14B ~300 tok/s) and ~950k output tokens for a full ladder, ~580k for a quant variant;
# max_minutes = 1.6x caps a job (configs run cheapest first, so a cap drops the expensive thinking budgets, never a ragged mix).
FULL: dict[str, Any] = {"configs": {"coding": "plain@1024,retrieval@1024,think@2048"}, "n": {"coding": 200}}
VARIANT: dict[str, Any] = {"configs": {"reasoning": "plain@32,cot@256,retrieval@256,think@1024", "thinkbench": "plain@32,cot@256,think@1024",
                                       "judgment": "plain@64,cot@256,retrieval@64,think@1024", "coding": "plain@1024,retrieval@1024,think@2048"},
                           "n": {"coding": 120}}
LADDER_PLAN: list[dict[str, Any]] = [
    {"tier": 1, "model": "Qwen3-0.6B-Q4_K_M.gguf", "minutes": 8, **FULL},
    {"tier": 1, "model": "Qwen3-1.7B-Q4_K_M.gguf", "minutes": 11, **FULL},
    {"tier": 1, "model": "Qwen3-4B-Q4_K_M.gguf", "minutes": 19, **FULL},
    {"tier": 1, "model": "Qwen3-14B-Q4_K_M.gguf", "minutes": 50, "configs": FULL["configs"], "n": {"coding": 160}, "distill": True,
     "n_distill": 800},
    {"tier": 2, "model": "Qwen3-0.6B-Q8_0.gguf", "minutes": 5, **VARIANT}, {"tier": 2, "model": "Qwen3-0.6B-Q3_K_M.gguf", "minutes": 5, **VARIANT},
    {"tier": 2, "model": "Qwen3-1.7B-Q8_0.gguf", "minutes": 7, **VARIANT}, {"tier": 2, "model": "Qwen3-1.7B-Q3_K_M.gguf", "minutes": 7, **VARIANT},
    {"tier": 2, "model": "Qwen3-4B-Q8_0.gguf", "minutes": 12, **VARIANT}, {"tier": 2, "model": "Qwen3-4B-Q3_K_M.gguf", "minutes": 12, **VARIANT},
    {"tier": 3, "model": "Qwen3-0.6B-Q5_K_M.gguf", "minutes": 5, **VARIANT}, {"tier": 3, "model": "Qwen3-1.7B-Q5_K_M.gguf", "minutes": 7, **VARIANT},
    {"tier": 3, "model": "Qwen3-4B-Q5_K_M.gguf", "minutes": 12, **VARIANT},
]


def jobs(plan: Sequence[Mapping[str, Any]] = LADDER_PLAN, rdir: str = "/workspace/nupen", items: str = "", workers: int = 0,
         tiers: Sequence[int] = (1, 2, 3)) -> list[dict[str, Any]]:
    """fetch (remote) -> ladder (call, the model served by the runner) -> delete (remote) per model; at most one ladder-only file on the pod."""
    out: list[dict[str, Any]] = []
    for p in [x for x in plan if int(x.get("tier", 1)) in tiers]:
        m = str(p["model"])
        tag = m.replace(".gguf", "").replace(".", "_")
        if m not in KEEP_ON_POD:
            out.append({"name": f"ladder_fetch_{tag}", "remote": fetch_script(rdir, m), "minutes": round(LADDER[m]["bytes"] / 1e9 / 6.0 + 0.3, 1),
                        "low_util_abort_minutes": 0})
        args: dict[str, Any] = {k: v for k, v in p.items() if k not in ("model", "minutes", "tier")}
        args.update(model=m, run=f"ladder-{tag}")
        if items:
            args["items"] = items
        if workers:
            args["workers"] = workers
        out.append({"name": f"ladder_{tag}", "call": "creator.effladder:ladder_job", "model": m, "minutes": p["minutes"],
                    "max_minutes": round(float(p["minutes"]) * 1.6, 1), "args": args, "low_util_abort_minutes": 0})
        if m not in KEEP_ON_POD:
            out.append({"name": f"ladder_delete_{tag}", "remote": delete_script(rdir, m), "minutes": 0.1, "low_util_abort_minutes": 0})
    return out


def coder_job(minutes: float = 35.0, n_coding: int = 160, items: str = "", configs: str = "plain@1024") -> dict[str, Any]:
    """The 27B coder (teacher's coder stage on port 18400, 4 slots; the runner serves nothing for it): coding eval items + the coding distill pool
    (verified rows for a small home coder). ~110k output tokens at ~60 tok/s aggregate when the 27B is not shared."""
    args: dict[str, Any] = {"model": CODER_27B, "port": CODER_27B_PORT, "suites": ["coding"], "configs": {"coding": configs},
                            "distill": True, "workers": 4, "run": "ladder-coder27"}
    if n_coding:
        args["n"] = {"coding": n_coding}
    if items:
        args["items"] = items
    return {"name": "ladder_coder27", "call": "creator.effladder:ladder_job", "model": "", "minutes": minutes, "max_minutes": minutes * 1.5,
            "args": args, "low_util_abort_minutes": 0}


# ------------------------------------------------------------------------------------------------ home cost (this PC, CPU, IDLE priority)
def llama_bench_exe() -> Path:
    from creator import device as DEV
    return DEV.runtime_dir() / "llama" / DEV.exe_name("llama-bench")


def bench_one(gguf: Path, threads: int, n_prompt: int = 256, n_gen: int = 64, reps: int = 2, exe: Optional[Path] = None,
              timeout: float = 1800.0) -> dict[str, Any]:
    """llama-bench at IDLE priority: prompt / generation tok/s; peak RSS sampled every 0.25 s; machine CPU% during the run (contention)."""
    import psutil
    argv = [str(exe or llama_bench_exe()), "-m", str(gguf), "-p", str(n_prompt), "-n", str(n_gen), "-t", str(threads), "-r", str(reps), "-o", "json"]
    flags = IDLE_PRIORITY_CLASS if sys.platform == "win32" else 0
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    peak, cpu = 0, []
    ps = psutil.Process(proc.pid)
    psutil.cpu_percent(None)
    while proc.poll() is None and time.monotonic() - t0 < timeout:
        try:
            peak = max(peak, int(ps.memory_info().rss))
        except psutil.Error:
            pass
        time.sleep(0.25)
        cpu.append(psutil.cpu_percent(None))
    if proc.poll() is None:
        proc.kill()
    so, se = proc.communicate()
    res: dict[str, Any] = {"model": gguf.name, "disk_bytes": gguf.stat().st_size, "threads": threads, "peak_rss_mb": round(peak / 2**20),
                           "machine_cpu_pct_mean": round(sum(cpu) / len(cpu), 1) if cpu else None, "seconds": round(time.monotonic() - t0, 1)}
    try:
        for r in json.loads(so.decode("utf-8", "replace")):
            k = "pp_tok_s" if int(r.get("n_prompt") or 0) else "tg_tok_s"
            res[k] = round(float(r["avg_ts"]), 2)
    except (ValueError, KeyError, TypeError):
        res["error"] = (se.decode("utf-8", "replace") or so.decode("utf-8", "replace"))[-400:]
    return res


def local_ggufs() -> dict[str, Path]:
    """Ladder model files already on this PC (Nupen's models dir, its download dirs, the ladder download dir)."""
    from creator import device as DEV
    rt = DEV.runtime_dir()
    dirs = [rt / "models", rt / "models" / "dl", rt / "models" / "ladder", Path.home() / "h33_models"]
    out: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(d.glob("Qwen3-*.gguf")) if d.is_dir() else []:
            if f.name in LADDER and f.stat().st_size == LADDER[f.name]["bytes"]:
                out.setdefault(f.name, f)
    return out


def homecost(threads: int = 4, models: Sequence[str] = (), say: Callable[[str], None] = print, bench: Callable[..., dict[str, Any]] = bench_one,
             out: Optional[Path] = None) -> dict[str, Any]:
    """Measure every local ladder GGUF (smallest first), then estimate the rest from the measured file of the same parameter size."""
    have = local_ggufs()
    names = [m for m in (models or sorted(LADDER, key=lambda k: LADDER[k]["bytes"])) if m in LADDER]
    rows: dict[str, dict[str, Any]] = {}
    for m in names:
        if m in have and LADDER[m]["bytes"] < 6e9:
            say(f"homecost: measuring {m} ({threads} threads, IDLE priority) ...")
            rows[m] = dict(bench(have[m], threads), measured=True)
            say(f"  {json.dumps(rows[m])}")
    for m in names:
        if m in rows:
            continue
        pb = model_facts(m)["params_b"]
        # the least-contended measurement of this size is the reference (a starved run on a busy PC must not drag every estimate down)
        refs = [r for k, r in rows.items() if r.get("measured") and model_facts(k)["params_b"] == pb and r.get("tg_tok_s")]
        ref = max(refs, key=lambda r: float(r["tg_tok_s"]) * float(r["disk_bytes"])) if refs else None
        b = LADDER[m]["bytes"]
        if ref is None:
            rows[m] = {"model": m, "disk_bytes": b, "measured": False, "note": "no measured file of this size on the PC"}
            continue
        ratio = ref["disk_bytes"] / b
        rows[m] = {"model": m, "disk_bytes": b, "measured": False, "estimated_from": ref["model"], "tg_tok_s": round(ref["tg_tok_s"] * ratio, 2),
                   "pp_tok_s": ref.get("pp_tok_s"), "peak_rss_mb": round(ref["peak_rss_mb"] + (b - ref["disk_bytes"]) / 2**20),
                   "note": "estimate: decode tok/s ~ 1/weight bytes, prompt tok/s ~ unchanged, RSS shifted by the size difference"}
    import psutil
    body = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "threads": threads,
            "machine": {"logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
            "contended": "measured while Nupen runs (llama-bench at IDLE priority on a machine ~85% CPU busy): tok/s are LOWER BOUNDS of an idle "
                         "machine and vary run to run; machine_cpu_pct_mean says how busy the PC was during each measurement",
            "rows": rows}
    p = out or homecost_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return body


# ------------------------------------------------------------------------------------------------ the summary (router / cascade table)
def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(c - h, 3), round(c + h, 3))


def home_seconds(model: str, tok_in: float, tok_out: float, hc: Mapping[str, Any]) -> Optional[float]:
    r = (hc.get("rows") or {}).get(model) or {}
    if not r.get("tg_tok_s") or not r.get("pp_tok_s"):
        return None
    return tok_in / float(r["pp_tok_s"]) + tok_out / float(r["tg_tok_s"])


def summary(rows: Sequence[Mapping[str, Any]], hc: Optional[Mapping[str, Any]] = None, points: float = 5.0) -> dict[str, Any]:
    """Per suite: every (model, config) on the eval items ALL of them answered (a fair comparison), accuracy + Wilson CI, mean tokens,
    latency, home cost; and the cheapest setup whose accuracy is within `points` of the best."""
    hc = hc or {}
    ev = [r for r in rows if r.get("split") == "eval"]
    out: dict[str, Any] = {}
    for s in sorted({str(r["suite"]) for r in ev}):
        rs = [r for r in ev if r["suite"] == s]
        by: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
        for r in rs:
            by.setdefault((str(r["model"]), str(r["config"])), {})[str(r["item"])] = r
        common = set.intersection(*(set(v) for v in by.values())) if by else set()
        table: list[dict[str, Any]] = []
        for (m, c), got in by.items():
            xs = [got[i] for i in common]
            if not xs:
                continue
            k = sum(1 for x in xs if x.get("correct"))
            ti = sum(float(x.get("tok_in") or 0) for x in xs) / len(xs)
            to = sum(float(x.get("tok_out") or 0) for x in xs) / len(xs)
            b = LADDER.get(m, {}).get("bytes") or 0
            hs = home_seconds(m, ti, to, hc)
            row: dict[str, Any] = {"model": m, "config": c, "n": len(xs), "acc": round(k / len(xs), 3), "ci": wilson(k, len(xs)),
                   "tok_in": round(ti), "tok_out": round(to), "truncated": round(sum(1 for x in xs if x.get("truncated")) / len(xs), 3),
                   "gpu_latency_s": round(sum(float(x.get("latency_s") or 0) for x in xs) / len(xs), 2),
                   "home_s_per_item": round(hs, 1) if hs is not None else None, "gb_tokens": round(b / 1e9 * (ti / 40 + to), 1),
                   "disk_gb": round(b / 1e9, 2) if b else None}
            if s == "judgment":
                bs = [float(x["brier"]) for x in xs if x.get("brier") is not None]
                row["brier"] = round(sum(bs) / len(bs), 4) if bs else None
            table.append(row)
        if not table:
            continue
        best = max(float(r["acc"]) for r in table)
        ok = [r for r in table if r["acc"] >= best - points / 100.0]
        def cost(r: Mapping[str, Any]) -> tuple[float, float]:
            return (float(r["home_s_per_item"]) if r["home_s_per_item"] is not None else math.inf, float(r["gb_tokens"]))

        table.sort(key=lambda r: (-r["acc"], cost(r)))
        out[s] = {"common_items": len(common), "best_acc": best, "cheapest_within": min(ok, key=cost), "points": points, "table": table}
    return out


def format_summary(sm: Mapping[str, Any]) -> str:
    lines = []
    for s, d in sm.items():
        ch = d["cheapest_within"]
        lines.append(f"\n== {s}: {d['common_items']} common eval items; best acc {d['best_acc']:.3f}; CHEAPEST within {d['points']:.0f} pts: "
                     f"{ch['model']} {ch['config']} (acc {ch['acc']:.3f}, home {ch['home_s_per_item']} s/item, {ch['gb_tokens']} GB-tok)")
        lines.append(f"{'model':34} {'config':14} {'acc':>6} {'ci':>13} {'in':>6} {'out':>6} {'trunc':>6} {'home s':>8} {'GB-tok':>8}")
        for r in d["table"]:
            lines.append(f"{r['model']:34} {r['config']:14} {r['acc']:6.3f} {str(r['ci']):>13} {r['tok_in']:6} {r['tok_out']:6} "
                         f"{r['truncated']:6.2f} {str(r['home_s_per_item']):>8} {r['gb_tokens']:8}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m creator.effladder", description=__doc__.split("\n\n")[0] if __doc__ else "")
    ap.add_argument("cmd", choices=("freeze", "run", "summary", "homecost", "jobs", "counts"))
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--items", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--n", default="", help="items per suite, e.g. reasoning=10,coding=5")
    ap.add_argument("--configs", default="", help="suite:strategy@n,...;suite2:...")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--n-distill", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--points", type=float, default=5.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models", default="")
    ap.add_argument("--no-coder", action="store_true")
    ap.add_argument("--tiers", default="1,2,3")
    a = ap.parse_args(argv)
    suites = [s for s in a.suites.split(",") if s in SUITES]
    if a.cmd == "freeze":
        print(json.dumps(freeze(Path(a.state), Path(a.repo), Path(a.items) if a.items else None, suites=suites), indent=1))
    elif a.cmd == "counts":
        b = load_items(Path(a.items) if a.items else None)
        print(json.dumps({"hash": b["hash"][:12], "counts": counts(b["items"])}, indent=1))
    elif a.cmd == "run":
        b = load_items(Path(a.items) if a.items else None)
        n = {k: int(v) for k, v in (p.split("=") for p in a.n.split(",") if "=" in p)}
        configs = {k: v for k, v in (p.split(":", 1) for p in a.configs.split(";") if ":" in p)}
        run(b, Endpoint(a.port, a.model), Path(a.state), suites=suites, configs=configs, n=n, workers=a.workers, distill=a.distill,
            n_distill=a.n_distill, run_id=f"cli-{a.model}", out=Path(a.out) if a.out else None)
    elif a.cmd == "summary":
        hc = json.loads(homecost_path().read_text(encoding="utf-8")) if homecost_path().is_file() else {}
        print(format_summary(summary(read_rows(Path(a.out) if a.out else results_path(Path(a.state))), hc, a.points)))
    elif a.cmd == "homecost":
        body = homecost(a.threads, [m for m in a.models.split(",") if m])
        print(json.dumps(body, indent=1))
    else:
        js = jobs(items=a.items, tiers=[int(t) for t in a.tiers.split(",") if t]) + ([] if a.no_coder else [coder_job(items=a.items)])
        txt = json.dumps(js, indent=1)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf-8")
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
