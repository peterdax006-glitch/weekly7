"""TRAINING MIXES FOR THE SMALL HOME MODELS (h61, 3 Oct 2026; owner: "make sure we are using all the data and also making sure we are using
the data efficently" - collected data is worthless until a model is trained on it, TESTED on held-out items, and improved afterwards).

Loaded ON DEMAND only (python -m creator.trainmix ..., or a GPU-runner 'call' job); never part of the swarm's eager start load.

  inventory  every collected dataset: rows, train/eval split, provenance, licence, privacy / frozen-benchmark / training-exclusion audit,
             and WHICH job consumes it (or UNUSED + why) -> <runtime>/gpuday/DATA_INVENTORY.json + .md
  build      one mix per target model (TARGETS): rows normalised to chat 'messages' (SFT) or prompt/chosen/rejected (preference), then
               exclusion   the coding trust gate's rule (codetrust_heldout.json: nothing from Nupen's own repo/state at/after the cut, no
                           held-out commit), gpuday.private_reason / e-mails, the frozen thinkbench (gpuday.Frozen)
               quality     non-empty answers, the reasoning answer line equals the known answer, code answers parse, length cap
               dedupe      exact + near-duplicate (MinHash over word 5-shingles, LSH) inside the mix AND against every eval / held-out set
                           (EVAL_SETS: ladder items, thinkbench, trust-gate held-out, talk eval, bake-off tasks, the export eval splits)
               dev split   ~10 % by group hash (never the same question / commit / task on both sides): early stopping, not a benchmark
             -> <runtime>/gpuday/trainmix/<target>/{train,dev,pref}.jsonl + sft_train.ids.jsonl + MANIFEST.json
  jobs       the gpupulse ext_job list: upload (audited) -> per target: fine-tune (LoRA, Unsloth; dev-split early stopping; merge + GGUF on
             /dev/shm, the Q4 GGUF into models/) -> register -> effladder on the target's task types -> paired comparison with the base model
             (compare_job; ADOPT only when n >= 50 and the 95% CI of the gain > 0) -> held-out gate (thinkers) -> delete the GGUF from the pod.
             Ordered by expected CPU saving at home.

Nothing here starts or stops a server, and nothing here touches creator/lm (the from-scratch LM)."""
from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1
CODE_SYSTEM = "You are a careful Python programmer. Reply with the complete function(s) in one ```python code block and nothing else."
CHARS_PER_TOKEN = 3.6
MIN_N = 50                         # paired items before a tuned model can be adopted (same rule as creator.gpuselfteach)
NEAR_DUP = 0.8                     # estimated Jaccard of word 5-shingles at or above which two texts are the same item
SHINGLE = 5
N_PERM = 64
BANDS = 16                         # 16 bands x 4 rows: a pair at Jaccard 0.8 is a candidate with probability > 0.99


def runtime() -> Path:
    from creator import device as DEV
    return Path(DEV.runtime_dir())


def gpuday_dir() -> Path:
    return runtime() / "gpuday"


def mix_dir() -> Path:
    return gpuday_dir() / "trainmix"


def gate_path(state: Path) -> Path:
    return Path(state) / "thinking" / "train_gate.jsonl"


# ------------------------------------------------------------------------------------------------ the training exclusion (coding trust gate)
def heldout_file() -> Optional[Path]:
    """codetrust_heldout.json: env NUPEN_CODETRUST_HELDOUT, else creator/ of this checkout, else the trust-gate worktree (h56, not merged)."""
    e = os.environ.get("NUPEN_CODETRUST_HELDOUT")
    for p in ([Path(e)] if e else []) + [ROOT / "creator" / "codetrust_heldout.json", Path.home() / "wt56_trust" / "creator" / "codetrust_heldout.json"]:
        if p.is_file():
            return p
    return None


def load_heldout(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or heldout_file()
    if p is None:
        raise FileNotFoundError("codetrust_heldout.json not found: no training mix is built without the exclusion rule")
    return dict(json.loads(Path(p).read_text(encoding="utf-8")))


def excluded(ts: float, sha: str, h: Mapping[str, Any]) -> bool:
    """Same semantics as creator.codetrust.excluded: at/after the cut, or a held-out commit (prefix match, >= 7 chars)."""
    if ts and ts >= float(h["cut_ts"]):
        return True
    shas = h.get("held_out_commits") or []
    return bool(sha) and len(sha) >= 7 and any(s.startswith(sha) or sha.startswith(s) for s in shas)


# ------------------------------------------------------------------------------------------------ rows, texts, near-duplicates
@dataclass
class Row:
    id: str
    source: str
    group: str                      # dev split + dedupe unit (question, commit, task)
    body: dict[str, Any]            # {"messages": [...]} or {"prompt": [...], "chosen": [...], "rejected": [...]}
    ts: float = 0.0
    sha: str = ""
    licence: str = ""
    kind: str = "sft"               # sft | pref
    meta: dict[str, Any] = field(default_factory=dict)
    own: bool = True                # derived from Nupen's own repo/state: its ts is what the exclusion rule judges (public rows: False)


def _content(msgs: Any) -> list[str]:
    return [str(m.get("content") or "") for m in msgs or [] if isinstance(m, Mapping)]


def prompt_text(body: Mapping[str, Any]) -> str:
    """The last user message (what an eval item would show)."""
    msgs = body.get("messages") or body.get("prompt") or []
    users = [str(m.get("content") or "") for m in msgs if isinstance(m, Mapping) and m.get("role") == "user"]
    return users[-1] if users else ""


def answer_text(body: Mapping[str, Any]) -> str:
    if body.get("messages"):
        m = body["messages"][-1]
        return str(m.get("content") or "") if m.get("role") == "assistant" else ""
    return " ".join(_content(body.get("chosen")))


def full_text(body: Mapping[str, Any]) -> str:
    return "\n".join(_content(body.get("messages")) + _content(body.get("prompt")) + _content(body.get("chosen")) + _content(body.get("rejected")))


_WORD = re.compile(r"\w+")


def norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def exact_key(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()[:20]


_P = 4294967311                    # a prime > 2^32: (a*h + b) mod p with a < 2^31 and h < 2^32 stays inside uint64
_PERM: list[Any] = []


def minhash(text: str) -> Any:
    """N_PERM-value MinHash signature of the word 5-shingles (texts shorter than 5 words: the whole text is one shingle)."""
    import numpy as np
    if not _PERM:
        rng = np.random.default_rng(61)
        _PERM.extend([rng.integers(1, 1 << 31, N_PERM, dtype=np.uint64)[:, None], rng.integers(0, 1 << 31, N_PERM, dtype=np.uint64)[:, None]])
    a, b = _PERM
    w = norm(text).split()
    sh = {" ".join(w[i:i + SHINGLE]) for i in range(max(1, len(w) - SHINGLE + 1))}
    hv = np.fromiter((int.from_bytes(hashlib.blake2b(x.encode("utf-8"), digest_size=4).digest(), "little") for x in sh), dtype=np.uint64,
                     count=len(sh))
    return ((a * hv[None, :] + b) % np.uint64(_P)).min(axis=1)


class NearIndex:
    """LSH over MinHash signatures: add(key, text); match(text) -> the first stored key with estimated Jaccard >= NEAR_DUP (or None)."""

    def __init__(self, threshold: float = NEAR_DUP) -> None:
        self.t = threshold
        self.sigs: dict[str, Any] = {}
        self.buckets: dict[tuple[int, bytes], list[str]] = collections.defaultdict(list)
        self.exact: dict[str, str] = {}

    def _bands(self, sig: Any) -> list[tuple[int, bytes]]:
        r = N_PERM // BANDS
        return [(i, sig[i * r:(i + 1) * r].tobytes()) for i in range(BANDS)]

    def match(self, text: str, sig: Any = None) -> Optional[str]:
        ek = exact_key(text)
        if ek in self.exact:
            return self.exact[ek]
        sig = minhash(text) if sig is None else sig
        seen: set[str] = set()
        for b in self._bands(sig):
            for k in self.buckets.get(b, ()):
                if k in seen:
                    continue
                seen.add(k)
                if float((self.sigs[k] == sig).mean()) >= self.t:
                    return k
        return None

    def add(self, key: str, text: str, sig: Any = None) -> None:
        sig = minhash(text) if sig is None else sig
        self.exact.setdefault(exact_key(text), key)
        self.sigs[key] = sig
        for b in self._bands(sig):
            self.buckets[b].append(key)


# ------------------------------------------------------------------------------------------------ readers
def jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        lines = Path(p).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in lines:
        if ln.strip():
            try:
                v = json.loads(ln)
            except ValueError:
                continue
            if isinstance(v, dict):
                out.append(v)
    return out


def paired(data: Path, ids: Optional[Path] = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(row, ids row) of an export file and its .ids.jsonl (same line order); ids {} when there is none."""
    rows = jsonl(data)
    meta = jsonl(ids) if ids is not None else []
    return [(r, meta[i] if i < len(meta) else {}) for i, r in enumerate(rows)]


def export_dirs() -> dict[str, Path]:
    g = gpuday_dir()
    return {"export": g / "export", "export_plus": g / "export_plus"}


@dataclass
class Ctx:
    state: Path
    repo: Path
    heldout: dict[str, Any]
    questions: Optional[dict[str, Any]] = None      # reasondrills qid -> Question (filled lazily: generating them reads every clone)

    def qs(self) -> dict[str, Any]:
        if self.questions is None:
            from creator import reasondrills as R
            self.questions = {q.qid: q for q in R.generate(R.repos(self.repo), self.state)}
        return self.questions


def _reason_rows(ctx: Ctx, qid: str, trace: str, source: str, drops: collections.Counter[str], styles: Sequence[str]) -> list[Row]:
    """Chat rows for one reasoning question: 'cot' = the verified worked trace; 'plain' = just the answer line (the cheapest home config)."""
    from creator import reasondrills as R
    q = ctx.qs().get(qid)
    if q is None:
        drops["reasoning: question no longer generated (id unknown)"] += 1
        return []
    if q.source == "nupen" and (excluded(float(q.t), "", ctx.heldout) or any(excluded(0.0, k, ctx.heldout) for k in q.keys)):
        drops["exclusion rule: Nupen's own history at/after the cut"] += 1
        return []
    if R.parse_choice(trace) != q.answer:
        drops["reasoning: answer line != known answer"] += 1
        return []
    out = []
    for s in styles:
        msgs = R.build_messages(R.BY_NAME[s], q)
        ans = trace.strip() if s == "cot" else f"ANSWER: {R.LETTERS[q.answer]}"
        out.append(Row(f"{source}:{s}:{qid}", f"{source}_{s}", qid, {"messages": msgs + [{"role": "assistant", "content": ans}]},
                       ts=float(q.t), licence="public repository history (facts: file paths, subjects)", meta={"kind": q.kind, "repo": q.source},
                       own=q.source == "nupen"))
    return out


def src_trace_bank(ctx: Ctx, drops: collections.Counter[str], styles: Sequence[str] = ("cot", "plain")) -> list[Row]:
    from creator import reasondrills as R
    out: list[Row] = []
    for qid, r in sorted(R.trace_bank(ctx.state).items()):
        out += _reason_rows(ctx, qid, str(r.get("trace") or ""), "trace_bank", drops, styles)
    return out


def src_ladder_reasoning(ctx: Ctx, drops: collections.Counter[str], styles: Sequence[str] = ("cot", "plain")) -> list[Row]:
    out: list[Row] = []
    for r in jsonl(gpuday_dir() / "distill" / "ladder_distill.jsonl"):
        m = r.get("meta") or {}
        if m.get("suite") != "reasoning":
            continue
        qid = str(m.get("id") or "").removeprefix("reasoning:")
        out += _reason_rows(ctx, qid, answer_text(r), "ladder_distill", drops, styles)
    return out


def _code_ok(text: str) -> bool:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    try:
        ast.parse(m.group(1) if m else text)
        return True
    except (SyntaxError, ValueError):
        return False


def src_ladder_coding(ctx: Ctx, drops: collections.Counter[str]) -> list[Row]:
    out: list[Row] = []
    for r in jsonl(gpuday_dir() / "distill" / "ladder_distill.jsonl"):
        m = r.get("meta") or {}
        if m.get("suite") != "coding":
            continue
        if m.get("kind") == "nupen_fn":                 # a function of Nupen's CURRENT tree (rl_tasks): after the cut
            drops["exclusion rule: function from Nupen's current tree"] += 1
            continue
        if not _code_ok(answer_text(r)):
            drops["code answer does not parse"] += 1
            continue
        lic = "MBPP CC-BY-4.0 (prompt); answer by Qwen3.6-27B, verified by executing the tests" if m.get("kind") == "mbpp" else "?"
        out.append(Row(f"ladder:{m.get('id')}", "ladder_distill_code", str(m.get("id")), {"messages": r["messages"]}, licence=lic, own=False))
    return out


def src_rl_refs(ctx: Ctx, drops: collections.Counter[str], name: str) -> list[Row]:
    """Function tasks with a reference solution that passed the upstream tests -> 'write this function' SFT rows (train split only)."""
    ex = export_dirs()["export"]
    out: list[Row] = []
    for r, m in paired(ex / f"{name}.jsonl", ex / f"{name}.ids.jsonl"):
        if r.get("split") != "train" or m.get("id") != r.get("id"):
            continue
        ref = str(m.get("reference") or "").strip()
        if not ref or not _code_ok(ref):
            drops[f"{name}: no parsable reference"] += 1
            continue
        msgs = [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": str(r["prompt"])},
                {"role": "assistant", "content": f"```python\n{ref}\n```"}]
        out.append(Row(f"{name}:{r['id']}", name, str(r["id"]), {"messages": msgs}, licence=str(m.get("licence") or ""),
                       meta={"repo": m.get("repo") or m.get("source")}, own=False))
    return out


def src_export_sft(ctx: Ctx, drops: collections.Counter[str], d: str, name: str, keep: Callable[[Mapping[str, Any]], bool] = lambda m: True,
                   own: bool = True) -> list[Row]:
    ex = export_dirs()[d]
    out: list[Row] = []
    for r, m in paired(ex / f"{name}.jsonl", ex / f"{name}.ids.jsonl"):
        if not keep(m) or not r.get("messages"):
            continue
        mine = own and not m.get("repo") and not str(m.get("id", "")).startswith(("hf:", "pub:"))
        if excluded(float(m.get("ts") or 0.0) if mine else 0.0, str(m.get("sha") or ""), ctx.heldout):
            drops["exclusion rule: Nupen's own repo/state at/after the cut or a held-out commit"] += 1
            continue
        out.append(Row(f"{name}:{m.get('id')}", name, str(m.get("group") or m.get("id")), {"messages": r["messages"]},
                       ts=float(m.get("ts") or 0.0), sha=str(m.get("sha") or ""), licence=str(m.get("licence") or ""),
                       meta={"kind": m.get("kind")}, own=mine))
    return out


def src_export_pref(ctx: Ctx, drops: collections.Counter[str], d: str, name: str) -> list[Row]:
    ex = export_dirs()[d]
    out: list[Row] = []
    for r, m in paired(ex / f"{name}.jsonl", ex / f"{name}.ids.jsonl"):
        if not (r.get("prompt") and r.get("chosen") and r.get("rejected")):
            continue
        if excluded(0.0 if m.get("repo") else float(m.get("ts") or 0.0), str(m.get("sha") or m.get("fix") or ""), ctx.heldout):
            drops["exclusion rule: Nupen's own repo/state at/after the cut or a held-out commit"] += 1
            continue
        out.append(Row(f"{name}:{m.get('id')}", name, str(m.get("group") or m.get("id")),
                       {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}, ts=float(m.get("ts") or 0.0),
                       sha=str(m.get("sha") or ""), licence=str(m.get("licence") or ""), kind="pref", meta={"kind": m.get("kind")},
                       own=not m.get("repo")))
    return out


HANDOFF_SHARE = 0.15                                 # handoff rows per in-scope row of the same mix (owner 4 Oct: never act outside the niche)


def src_handoff(ctx: Ctx, drops: collections.Counter[str], scope: Sequence[str], share: float = HANDOFF_SHARE) -> list[Row]:
    """Owner 4 Oct 2026: a specialist never works outside its jurisdiction - anything else is handed to the suited model. The other roles'
    role rows (same prompts the orchestrator would send) answered with exactly 'HANDOFF: <ROLE>'; sampled to `share` of the in-scope rows,
    stable order (sha of the id), so the specialist learns the refusal without the out-of-scope skill."""
    rows = _pm().src_roles(ctx, drops)
    n_in = sum(1 for r in rows if r.meta.get("role") in scope)
    out = [r for r in rows if r.meta.get("role") not in scope and r.body.get("messages")]
    out.sort(key=lambda r: hashlib.sha256(r.id.encode()).hexdigest())
    keep = out[: int(n_in * share)]
    drops["handoff: not sampled"] += len(out) - len(keep)
    res = []
    for r in keep:
        msgs = [m for m in r.body["messages"] if m.get("role") != "assistant"]
        role = str(r.meta.get("role"))
        res.append(dataclasses.replace(r, id="handoff:" + r.id, source="handoff_" + role.lower(),
                                       body={"messages": msgs + [{"role": "assistant", "content": f"HANDOFF: {role}"}]},
                                       meta=dict(r.meta, handoff=True)))
    return res


def trajectory_files(view: str) -> list[Path]:
    """C2 trajectory views (<runtime>/gpuday/trajectories/**): files whose name carries the view (sft | debug | review | pair)."""
    d = gpuday_dir() / "trajectories"
    return sorted(p for p in d.rglob("*.jsonl") if view in p.name.lower() and not p.name.endswith(".ids.jsonl")) if d.is_dir() else []


def src_trajectories(ctx: Ctx, drops: collections.Counter[str], view: str) -> list[Row]:
    held_tasks = {str(t.get("id")) for t in ctx.heldout.get("tasks") or []}
    out: list[Row] = []
    for p in trajectory_files(view):
        for i, r in enumerate(jsonl(p)):
            m = dict(r.get("meta") or {})
            body = ({"messages": r["messages"]} if r.get("messages") else
                    {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} if r.get("chosen") and r.get("rejected") else None)
            if body is None:
                drops[f"trajectories/{view}: no messages or chosen/rejected"] += 1
                continue
            task = str(m.get("task") or m.get("task_id") or r.get("task") or "")
            sha = str(m.get("sha") or r.get("sha") or "")
            if task in held_tasks or excluded(float(m.get("ts") or 0.0), sha, ctx.heldout):
                drops["exclusion rule: trust-gate held-out task / commit"] += 1
                continue
            out.append(Row(f"traj:{view}:{p.stem}:{i}", f"c2_{view}", task or f"{p.stem}:{i}", body, sha=sha,
                           licence="own runs (27B on public/own tasks)", kind="sft" if "messages" in body else "pref", meta={"file": p.name}))
    return out


def talk_files(kind: str) -> list[Path]:
    d = gpuday_dir() / "talk"
    return sorted(d.rglob(f"talk_{kind}_sft.jsonl")) if d.is_dir() else []


def src_talk(ctx: Ctx, drops: collections.Counter[str], kind: str) -> list[Row]:
    out: list[Row] = []
    for p in talk_files(kind):
        for i, r in enumerate(jsonl(p)):
            if r.get("messages"):
                out.append(Row(f"talk:{kind}:{p.parent.name}:{i}", f"talk_{kind}", f"talk:{exact_key(prompt_text(r))}", {"messages": r["messages"]},
                               licence="own (Nupen's records + a big model's grounded answer)",
                               meta={"intent": r.get("intent"), "tag": "voice-only, not gate-eligible", "post_cut": True}))
    return out


# ------------------------------------------------------------------------------------------------ eval / held-out sets (never trained on)
def ladder_items_path() -> Path:
    return runtime() / "gpu" / "ladder" / "items.json"


def eval_sets(ctx: Ctx) -> dict[str, dict[str, Any]]:
    """name -> {"texts": [(id, text)], "ids": set of item ids/shas/qids, "hash": content hash or None}."""
    out: dict[str, dict[str, Any]] = {}

    def put(name: str, texts: list[tuple[str, str]], ids: Iterable[str] = (), h: Optional[str] = None) -> None:
        out[name] = {"texts": [(i, t) for i, t in texts if t and len(t) >= 20], "ids": set(ids), "hash": h}
    try:
        b = json.loads(ladder_items_path().read_text(encoding="utf-8"))
        ev = [it for it in b["items"] if it["split"] == "eval"]
        put("ladder_eval", [(it["id"], prompt_text({"messages": next(iter(it["messages"].values()))})) for it in ev],
            [str(it["id"]).split(":", 1)[1] if it["suite"] == "reasoning" else str(it["id"]) for it in ev], str(b["hash"])[:12])
    except (OSError, ValueError, KeyError):
        put("ladder_eval", [])
    fz = frozen()
    put("thinkbench_frozen", [(f"tb:{i}", t) for i, t in enumerate(fz.texts)], fz.subjects | fz.commits, fz.hash[:12] or None)
    put("trust_gate_heldout", [(str(t.get("id")), str(t.get("task") or "")) for t in ctx.heldout.get("tasks") or []],
        [str(s) for s in ctx.heldout.get("held_out_commits") or []], str(ctx.heldout.get("cut_utc")))
    try:
        from creator import talkeval as TE
        put("talk_eval", [(f"talk:{i}", q.text) for i, q in enumerate(TE.QUESTIONS)])
    except Exception:                                   # noqa: BLE001 - the talk eval is optional
        put("talk_eval", [])
    bt = jsonl(runtime() / "agents" / "repo_tasks.jsonl")
    put("bakeoff_tasks", [(str(t.get("id")), str(t.get("prompt") or "")) for t in bt], [str(t.get("sha") or "") for t in bt if t.get("sha")])
    out["bakeoff_tasks"]["fn_names"] = sorted({d.name.split(".", 1)[1] for d in (runtime() / "agents" / "runs" / "c1" / "aider").glob("fn.*")})
    for d, names in (("export", ("coder_hf_eval", "coder_more_eval", "commit_eval", "handoff_eval", "worked_eval")),
                     ("export_plus", ("commit_eval", "handoff_eval", "pref_eval", "pref_synth_eval", "pref_more_eval"))):
        for n in names:
            rows = paired(export_dirs()[d] / f"{n}.jsonl", export_dirs()[d] / f"{n}.ids.jsonl")
            put(f"{d}/{n}", [(str(m.get("id") or i), prompt_text(r)) for i, (r, m) in enumerate(rows)], [str(m.get("sha") or "") for _r, m in rows if m.get("sha")])
    for n in ("rl_tasks_hf", "rl_tasks_more"):
        ev = [r for r in jsonl(export_dirs()["export"] / f"{n}.jsonl") if r.get("split") == "eval"]
        put(f"export/{n}:eval", [(str(r["id"]), str(r.get("prompt") or "")) for r in ev], [str(r["id"]) for r in ev])   # ids: role rows of eval tasks
    try:                                                                  # R13: the ~200-task eval suite (EXCLUDE.json): ids, requests, def names
        from creator.tools import evalexclude as EX
        ex = EX.load()
        if ex:
            out["eval_suite200"] = ex.eval_set()
    except Exception:                                   # noqa: BLE001 - a broken list must not hide the other evals; the audit below still runs
        pass
    return out


def frozen() -> Any:
    from creator import gpuday as GD
    p = ROOT / "state" / "creator" / "thinkbench" / "items.json"
    alt = Path(os.environ.get("NUPEN_FROZEN_ITEMS") or (Path.home() / "weekly7" / "state" / "creator" / "thinkbench" / "items.json"))
    for c in (p, alt):
        if c.is_file():
            return GD.Frozen.load(c)
    raise FileNotFoundError("frozen thinkbench items.json not found: refusing to build without it")


# ------------------------------------------------------------------------------------------------ the targets (mix plans)
def _pm() -> Any:
    from creator import pipelinemix as PM
    return PM


def _mods() -> Any:
    from creator import trainmods as MODS
    return MODS


SOURCES: dict[str, Callable[[Ctx, collections.Counter[str]], list[Row]]] = {
    "trace_bank": src_trace_bank,
    "ladder_reasoning": src_ladder_reasoning,
    "ladder_coding": src_ladder_coding,
    "rl_refs_hf": lambda c, d: src_rl_refs(c, d, "rl_tasks_hf"),
    "rl_refs_more": lambda c, d: src_rl_refs(c, d, "rl_tasks_more"),
    "coder_hf": lambda c, d: src_export_sft(c, d, "export", "coder_hf_train"),
    "coder_more": lambda c, d: src_export_sft(c, d, "export_plus", "coder_more_train"),
    "coder_nupen": lambda c, d: src_export_sft(c, d, "export_plus", "coder_sft_mix",
                                               keep=lambda m: m.get("source_file") in ("handoff_train", "commit_train")),
    "pref_coder": lambda c, d: src_export_pref(c, d, "export_plus", "pref_train"),
    "c2_sft": lambda c, d: src_trajectories(c, d, "sft"),
    "c2_debug": lambda c, d: src_trajectories(c, d, "debug"),
    "c2_review": lambda c, d: src_trajectories(c, d, "review"),
    "c2_pairs": lambda c, d: src_trajectories(c, d, "pair"),
    "talk_understand": lambda c, d: src_talk(c, d, "understand"),
    "talk_speak": lambda c, d: src_talk(c, d, "speak"),
    "roles": lambda c, d: _pm().src_roles(c, d),
    "locate": lambda c, d: _pm().src_locate(c, d),
    "aider_sft": lambda c, d: _pm().src_aider(c, d, "sft"),
    "aider_pref": lambda c, d: _pm().src_aider(c, d, "pref"),
    "roles_code": lambda c, d: [r for r in _pm().src_roles(c, d) if r.meta.get("role") in ("CODE", "DEBUG")],
    "handoff_code": lambda c, d: src_handoff(c, d, ("CODE", "DEBUG")),
    "calib": lambda c, d: _mods().src_calib(c, d),
    "brevity_sft": lambda c, d: _mods().src_brevity(c, d),
    "brevity_pref": lambda c, d: _mods().src_brevity(c, d, pref=True),
    "judge": lambda c, d: _mods().src_judge(c, d),
}

# Training rates on the RTX 5090 (Unsloth LoRA bf16, no packing; conservative): tokens/s. Overheads: base download + load, merge + GGUF.
TRAIN_TOK_S = {"0.6b": 13000.0, "1.7b": 6500.0, "4b": 3500.0}      # 1.7B measured on the 5090 (3 Oct); 0.6B / 4B scaled from it
STEP_OVERHEAD_S = 0.1
PACKING_SPEEDUP = 1.5                                # packed short rows: ~1.5x fewer padded tokens (only where packing is verified on the GPU)
GROUP_BY_LENGTH_VERIFIED = False                     # set True once a GPU smoke run of --group-by-length passed (4 Oct: packing failed under Unsloth)
TOKEN_BUDGET = {"0.6b": 12e6, "1.7b": 8e6, "4b": 6e6}                # trained tokens per fine-tune (all epochs) before epochs are cut
OVERHEAD_MIN = {"0.6b": 3.0, "1.7b": 5.0, "4b": 9.0}
HF_BASE = {"0.6b": "Qwen/Qwen3-0.6B", "1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B"}
HOME_GGUF = {"0.6b": "Qwen3-0.6B-Q4_K_M.gguf", "1.7b": "Qwen3-1.7B-Q4_K_M.gguf", "4b": "Qwen3-4B-Q4_K_M.gguf"}
BF16_GB = {"0.6b": 1.2, "1.7b": 3.4, "4b": 8.0}
Q4_GB = {"0.6b": 0.4, "1.7b": 1.1, "4b": 2.5}


@dataclass(frozen=True)
class Target:
    name: str
    size: str
    sources: tuple[str, ...]
    why: str
    value: str                                       # expected saving at home (the order)
    max_seq: int = 2048
    epochs: float = 2.0
    lr: float = 1e-4
    batch: int = 8
    accum: int = 2
    min_rows: int = 200
    pref: tuple[str, ...] = ()
    eval_suites: tuple[str, ...] = ()
    eval_configs: Mapping[str, str] = field(default_factory=dict)
    eval_n: Mapping[str, int] = field(default_factory=dict)
    eval_minutes: float = 0.0
    gate: bool = False                               # creator.gpuselfteach held-out gate (reasoning questions)
    vram_mib: int = 12000
    blocked: str = ""                                # non-empty: built and audited, but no GPU job (with the reason)
    also_vs: tuple[str, ...] = ()                    # further base models the tuned one is compared with (their ladder rows exist)
    # Teacher decision 3 Oct 2026: the exclusion rule keeps the CODING trust gate honest; the voice (conversation layer) is never evaluated
    # by that gate, so talk rows built from post-cut records are allowed for the voice adapter ONLY: never in a coder/thinker mix, the adapter
    # is tagged "voice-only, not gate-eligible", and it is evaluated with creator.talkeval (talk_eval_job) instead of the ladder.
    voice_only: bool = False
    roles: bool = False                              # multi-role pipeline adapter (creator.pipelinemix): role balance + per-role held-out
    hf_on_shm: bool = False                          # base download on /dev/shm (the 4B bf16 base does not fit the workspace disk)
    module: str = ""                                 # a training module of creator.trainmods (own held-out files and eval jobs)
    priority: int = 0                                # queue order of a module (lower first)
    filter: Optional[tuple[tuple[str, tuple[str, ...]], ...]] = None   # spec {'role': [...]}: keep only rows whose meta matches
    packing: bool = False                            # pack short rows into max_seq sequences (fewer optimizer steps)
    lora_serve: bool = False                         # no merge: the adapter GGUF is served on the cached base GGUF (llama-server --lora)
    target_minutes: Optional[tuple[int, int]] = None  # size the run to this GPU-minute window (epochs 1-2, train rows sampled down)
    requires: tuple[str, ...] = ()                   # autogen gates: 'adopted:<target>', 'rows:<source>>=N'
    safe_batch: int = 4                              # SAFE MODE = the settings proven on the real GPU (pipeline_17b, 4 Oct): batch 4 x accum 4,
    safe_accum: int = 4                              # no packing, gradient checkpointing on - the automatic retry of a failed fine-tune
    suspended: str = ""                              # autogen backoff: failed twice (after safe mode) -> never queued again (with the error)

    @property
    def base_gguf(self) -> str:
        return HOME_GGUF[self.size]


THINK_EVAL = {"reasoning": "plain@32,cot@256", "thinkbench": "plain@32,cot@256"}
CODER_SFT = ("rl_refs_hf", "rl_refs_more", "ladder_coding", "coder_hf", "coder_more", "coder_nupen")
TARGETS: tuple[Target, ...] = (
    Target("thinker_17b", "1.7b", ("trace_bank", "ladder_reasoning"),
           "Nupen's drills/judgment run constantly on the home 1.7B; a tuned 1.7B answering at plain@32 (5 tokens) or cot@256 replaces "
           "think@1024-2048 (865-1193 tokens) if it matches the 4B/14B on held-out questions",
           "largest: reasoning is the steady home CPU load; up to ~200x fewer generated tokens per question", max_seq=1024, epochs=3, lr=1e-4,
           batch=16, accum=1, eval_suites=("reasoning", "thinkbench"), eval_configs=THINK_EVAL, eval_minutes=4, gate=True, vram_mib=10000,
           also_vs=("Qwen3-4B-Q4_K_M.gguf",)),
    Target("thinker_06b", "0.6b", ("trace_bank", "ladder_reasoning"),
           "the same data on the 0.6B: 2.6x faster decode at home (3.47 vs 1.35 tok/s measured contended), 684 vs 1816 MB RSS",
           "large if it reaches the base 1.7B: same task at ~1/3 of the CPU and RAM", max_seq=1024, epochs=3, lr=2e-4, batch=16, accum=1,
           eval_suites=("reasoning", "thinkbench"), eval_configs=THINK_EVAL, eval_minutes=3, gate=False, vram_mib=7000,
           also_vs=("Qwen3-1.7B-Q4_K_M.gguf",)),
    Target("coder_17b", "1.7b", ("aider_sft", "rl_refs_hf", "rl_refs_more", "ladder_coding", "coder_hf", "coder_more", "coder_nupen", "c2_sft", "c2_debug"),
           "home coder: base 1.7B 0.56 vs 27B 0.815 on the coding ladder (plain@1024); verified references + 27B distill + public edits",
           "medium: fewer escalations to the teacher / pod per coding task", max_seq=4096, epochs=2, lr=1e-4, batch=4, accum=4, min_rows=300,
           pref=("pref_coder", "c2_pairs", "aider_pref"), eval_suites=("coding",), eval_configs={"coding": "plain@1024"}, eval_n={"coding": 200},
           eval_minutes=4, vram_mib=18000),
    # owner goal 4 Oct: small home models run the full coding pipeline on CPU - one multi-role adapter (role tag in the system prompt)
    Target("pipeline_17b", "1.7b", ("roles", "aider_sft", *CODER_SFT), "every pipeline role (SPEC/PLAN/CODE/DEBUG/REVIEW/VALIDATE/LESSON) at home on the 1.7B: "
           "verified C2 role rows (27B first, then 4B, then 1.7B) + the coder SFT, role-balanced",
           "large: the whole pipeline at home instead of the 27B on the pod", max_seq=4096, epochs=2, lr=1e-4, batch=4, accum=4, min_rows=300,
           eval_minutes=8, vram_mib=18000, roles=True, pref=("aider_pref",)),
    Target("pipeline_4b", "4b", ("roles", "aider_sft", *CODER_SFT), "the same pipeline adapter on the 4B (stronger, ~2.3x the home CPU per token)",
           "large if the 1.7B is not good enough at a role", max_seq=4096, epochs=2, lr=1e-4, batch=2, accum=8, min_rows=300,
           eval_minutes=12, vram_mib=26000, roles=True, hf_on_shm=True, pref=("aider_pref",)),
    Target("reviewer_17b", "1.7b", ("c2_review",), "a small reviewer for C2 diffs (REVIEW view of the C2 trajectories)",
           "medium once C2 has run: review at home instead of on the 27B", max_seq=4096, epochs=2, batch=4, accum=4, min_rows=100,
           vram_mib=18000),
    Target("voice_17b", "1.7b", ("talk_understand", "talk_speak"), "the terminal voice (creator.talk): understand + grounded speak rows",
           "small CPU saving; language quality", max_seq=2048, epochs=2, batch=8, accum=2, min_rows=60, vram_mib=12000, voice_only=True),
)


def spec_dirs() -> list[Path]:
    return [ROOT / "creator" / "train_modules.json", mix_dir() / "specs"]


def target_from_spec(d: Mapping[str, Any]) -> Target:
    """One module spec (creator/train_modules.json or <runtime>/gpuday/trainmix/specs/*.json) -> a Target. Unknown keys are refused."""
    fields = {f.name for f in dataclasses.fields(Target)}
    bad = set(d) - fields - {"_doc"}
    if bad:
        raise ValueError(f"module spec {d.get('name')!r}: unknown keys {sorted(bad)}")
    kw: dict[str, Any] = {}
    for k, v in d.items():
        if k == "_doc":
            continue
        kw[k] = tuple(v) if isinstance(v, list) else v
    if kw.get("filter") is not None:
        kw["filter"] = tuple(sorted((str(a), tuple(b)) for a, b in dict(d["filter"]).items()))
    for k in ("sources", "why", "value"):
        kw.setdefault(k, () if k == "sources" else "")
    return Target(**kw)


def load_specs() -> list[Target]:
    """Module specs: the repository's train_modules.json, then the runtime specs folder (autogen); a later spec of the same name wins."""
    out: dict[str, Target] = {}
    for src in spec_dirs():
        files = sorted(src.glob("*.json")) if src.is_dir() else ([src] if src.is_file() else [])
        for f in files:
            try:
                body = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for d in ((body.get("modules") or []) if isinstance(body, dict) and "modules" in body else [body]):
                if isinstance(d, dict) and d.get("name"):
                    out[str(d["name"])] = target_from_spec(d)
    return sorted(out.values(), key=lambda t: (t.priority, t.name))


TARGETS = tuple(t for t in TARGETS if t.name not in {m.name for m in load_specs()}) + tuple(load_specs())
BY_NAME = {t.name: t for t in TARGETS}


def serve_name(t: Target) -> str:
    return {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B", "4b": "Qwen3-4B"}[t.size] + f"-nupen-{t.name.rsplit('_', 1)[0]}-Q4_K_M.gguf"


# ------------------------------------------------------------------------------------------------ build
def tokens(body: Mapping[str, Any]) -> int:
    return int(len(full_text(body)) / CHARS_PER_TOKEN) + 16


def _dev(group: str, frac: float = 0.1) -> bool:
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 1000 < int(frac * 1000)


def screen(rows: Sequence[Row], evals: Mapping[str, Mapping[str, Any]], fz: Any, max_tokens: int,
           drops: collections.Counter[str]) -> list[Row]:
    """Privacy, frozen benchmark, quality, eval overlap (ids + near-duplicate texts), then near-duplicates inside the mix (first kept)."""
    from creator import gpuday as GD
    ev_index = NearIndex()
    ev_ids: set[str] = set()
    fn_names = set()
    for name, e in evals.items():
        ev_ids |= {str(i) for i in e["ids"] if i}
        fn_names |= set(e.get("fn_names") or [])
        for i, t in e["texts"]:
            ev_index.add(f"{name}:{i}", t)
    fn_re = re.compile(r"\bdef (" + "|".join(map(re.escape, sorted(fn_names))) + r")\s*\(") if fn_names else None
    mix = NearIndex()
    out: list[Row] = []
    for r in rows:
        txt = full_text(r.body)
        if GD.private_reason(txt):
            drops["private marker"] += 1
            continue
        if GD.EMAIL_RE.search(txt):
            drops["e-mail address"] += 1
            continue
        if fz.leak(subject=r.group, commit=r.sha, text=txt):
            drops["frozen thinkbench"] += 1
            continue
        if not answer_text(r.body).strip():
            drops["empty answer"] += 1
            continue
        if tokens(r.body) > max_tokens:
            drops[f"longer than {max_tokens} tokens"] += 1
            continue
        if r.group in ev_ids or (r.sha and any(len(r.sha) >= 7 and (s.startswith(r.sha) or r.sha.startswith(s)) for s in ev_ids if len(s) >= 7)):
            drops["eval/held-out item id or commit"] += 1
            continue
        if fn_re is not None and fn_re.search(answer_text(r.body)):
            drops["bake-off function"] += 1
            continue
        p = prompt_text(r.body)
        sig = minhash(p)
        hit = ev_index.match(p, sig)
        if hit:
            drops[f"near-duplicate of eval set {hit.split(':', 1)[0]}"] += 1
            continue
        key = p + "\n" + answer_text(r.body)
        dup = mix.match(key)
        if dup:
            drops["near-duplicate inside the mix"] += 1
            continue
        mix.add(r.id, key)
        out.append(r)
    return out


HELDOUT_PER_ROLE = 200


def _role_mix(t: Target, d: Path, sft: Sequence[Row], drops: collections.Counter[str]) -> tuple[list[Row], dict[str, Any]]:
    """Pipeline targets: rows of held-out tasks never train (any source: an rl reference shares its task id); role rows balanced per role;
    every non-role row (the coder SFT) tagged CODE; per role a held-out eval file (heldout_<ROLE>.jsonl, at most HELDOUT_PER_ROLE rows)."""
    PM = _pm()
    kept = [r for r in sft if PM.split(r.group) != "heldout"]
    drops["pipeline: held-out task (eval side)"] += len(sft) - len(kept)
    role_rows = [r for r in kept if r.meta.get("role") in PM.ROLES and not r.meta.get("aider")]
    aider = [r for r in kept if r.meta.get("aider")]
    other = [r for r in kept if r.meta.get("role") not in PM.ROLES and not r.meta.get("aider")]
    bal, counts = PM.balance(role_rows)
    drops["pipeline: over the per-role cap"] += len(role_rows) - len(bal)
    other = [dataclasses.replace(r, body={"messages": PM.tagged("CODE", r.body["messages"])}) for r in other]   # copies: the source cache is shared
    raw = PM.role_rows_raw()
    held = {str(x["task_id"]) for x in raw if PM.split(str(x["task_id"])) == "heldout"}
    ev = PM.heldout_eval_rows(raw, held, PM.rl_tests())
    for f in d.glob("heldout_*.jsonl"):
        f.unlink()
    nh = {}
    for role, rows in ev.items():
        rows = sorted(rows, key=lambda x: PM._h("ev" + x["id"]))[:HELDOUT_PER_ROLE]
        nh[role] = len(rows)
        with (d / f"heldout_{role}.jsonl").open("w", encoding="utf-8") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    # Aider rows (home default editor): outside the role cap and weighted up - each train-side row repeated AIDER_WEIGHT times (dev keeps one)
    up = [dataclasses.replace(r, id=f"{r.id}#w{k}") for r in aider if PM.split(r.group) == "train" for k in range(1, PM.AIDER_WEIGHT)]
    return bal + other + aider + up, {"train_per_role": counts, "coder_sft_rows": len(other), "aider_rows": len(aider), "aider_weight": PM.AIDER_WEIGHT, "heldout_tasks": len(held), "heldout_rows": nh}


def build_target(t: Target, ctx: Ctx, evals: Mapping[str, Mapping[str, Any]], fz: Any, out_root: Optional[Path] = None,
                 cache: Optional[dict[str, tuple[list[Row], collections.Counter[str]]]] = None) -> dict[str, Any]:
    d = Path(out_root or mix_dir()) / t.name
    d.mkdir(parents=True, exist_ok=True)
    cache = {} if cache is None else cache
    by_src: dict[str, int] = {}
    raw: list[Row] = []
    pref_raw: list[Row] = []
    drops: collections.Counter[str] = collections.Counter()
    for s in t.sources + t.pref:
        if s.startswith("talk_") and not t.voice_only:
            raise ValueError(f"{t.name}: talk rows are allowed in the voice-only mix only")
        if s not in cache:
            dr: collections.Counter[str] = collections.Counter()
            cache[s] = (SOURCES[s](ctx, dr), dr)
        rows, dr = cache[s]
        drops.update({f"{s}: {k}": v for k, v in dr.items()})
        by_src[s] = len(rows)
        for r in rows:
            (pref_raw if r.kind == "pref" else raw).append(r)
    if t.filter:                                    # spec filter, e.g. {'role': ['DEBUG']}: only the matching rows
        flt = dict(t.filter)
        keep_f = lambda r: all(str(r.meta.get(k)) in vs for k, vs in flt.items())  # noqa: E731
        drops["spec filter"] += sum(1 for r in raw + pref_raw if not keep_f(r))
        raw, pref_raw = [r for r in raw if keep_f(r)], [r for r in pref_raw if keep_f(r)]
    sd: collections.Counter[str] = collections.Counter()
    sft = screen(raw, evals, fz, t.max_seq, sd)
    pd: collections.Counter[str] = collections.Counter()
    pref = screen(pref_raw, evals, fz, t.max_seq, pd) if pref_raw else []
    drops.update({f"screen: {k}": v for k, v in sd.items()})
    drops.update({f"screen pref: {k}": v for k, v in pd.items()})
    role_counts: dict[str, Any] = {}
    if t.module and t.module != "roles":
        sft, pref, role_counts = _mods().prepare(t, d, sft, pref, drops)
        train = [r for r in sft if not _dev(r.group)]
        dev = [r for r in sft if _dev(r.group)]
    elif t.roles or t.module == "roles":
        sft, role_counts = _role_mix(t, d, sft, drops)
        PM = _pm()
        pref = [r for r in pref if PM.split(r.group) != "heldout"]        # a held-out task never trains, in any form
        train = [r for r in sft if PM.split(r.group) == "train"]
        dev = [r for r in sft if PM.split(r.group) == "dev"]
    else:
        train = [r for r in sft if not _dev(r.group)]
        dev = [r for r in sft if _dev(r.group)]
    if t.target_minutes:                             # one epoch over the window's upper bound: a deterministic sample of the training rows
        long_f = 1.3 if t.max_seq > 2048 else 1.0
        tok_all = sum(tokens(r.body) for r in train)
        budget = (t.target_minutes[1] - OVERHEAD_MIN[t.size] - t.eval_minutes) * 60.0 * TRAIN_TOK_S[t.size] / long_f * (PACKING_SPEEDUP if t.packing else 1.0)
        if tok_all > budget > 0:
            frac = budget / tok_all
            kept = [r for r in train if (int(hashlib.sha256(("fit" + r.id).encode()).hexdigest()[:8], 16) % 10000) < frac * 10000]
            drops[f"sized to {t.target_minutes[1]} GPU min (sampled)"] += len(train) - len(kept)
            train = kept
    files = {"train.jsonl": train, "dev.jsonl": dev, "pref.jsonl": pref}
    for n, rows in files.items():
        with (d / n).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r.body, ensure_ascii=False) + "\n")
    dev_ids = {r.id for r in dev}
    for n, rows in (("sft_train.ids.jsonl", train + dev), ("pref_train.ids.jsonl", pref)):
        with (d / n).open("w", encoding="utf-8") as f:
            for r in rows:
                # 'ts' is what the trust gate's exclusion rule (codetrust.audit_export) judges: set for rows derived from Nupen's own
                # repo/state; public-repository rows carry their commit time as 'source_ts' (the rule is about Nupen's history only)
                f.write(json.dumps({"id": r.id, "source": r.source, "group": r.group, "ts": r.ts if r.own else 0.0, "source_ts": r.ts,
                                    "own": r.own, "sha": r.sha if r.own else "", "source_sha": r.sha, "licence": r.licence,
                                    "split": "dev" if r.id in dev_ids else "train", **r.meta}, sort_keys=True) + "\n")
    qids = sorted({r.group for r in train + dev if r.source.startswith(("trace_bank", "ladder_distill_cot", "ladder_distill_plain"))})
    (d / "train_qids.json").write_text(json.dumps(qids), encoding="utf-8")
    tok = sum(tokens(r.body) for r in train)
    used = collections.Counter(r.source for r in train + dev + pref)
    man = {"target": t.name, "base": HF_BASE[t.size], "serve_as": serve_name(t), "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rows": {"train": len(train), "dev": len(dev), "pref": len(pref)}, "tokens_train": tok, "by_source_raw": by_src,
           "by_source_kept": dict(used), "drops": dict(sorted(drops.items())), "epochs": t.epochs, "max_seq": t.max_seq,
           "eval_sets": {k: {"n": len(v["texts"]), "hash": v["hash"]} for k, v in evals.items()},
           "heldout_cut": ctx.heldout.get("cut_utc"), "blocked": t.blocked,
           "tag": "voice-only, not gate-eligible" if t.voice_only else "gate-eligible (exclusion rule applied)", "roles": role_counts,
           "files": {n: {"rows": sum(1 for _ in (d / n).open(encoding="utf-8")), "sha256": hashlib.sha256((d / n).read_bytes()).hexdigest()}
                     for n in ("train.jsonl", "dev.jsonl", "pref.jsonl")}}
    man["gpu_minutes"] = minutes(t, man)
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def epochs_for(t: Target, man: Mapping[str, Any]) -> float:
    """Epochs sized to the data: at most t.epochs, and no more than TOKEN_BUDGET trained tokens in total (at least one epoch). A large
    mix (e.g. the grown trace bank: 87k rows, 19.5M tokens) gets one pass; a small one its full epochs. Dev early stopping only shortens it."""
    tok = max(1, int(man.get("tokens_train") or 0))
    if t.target_minutes:                            # sized to the GPU window instead: 2 epochs when they fit the upper bound, else 1
        per_epoch = tok * (1.3 if t.max_seq > 2048 else 1.0) / TRAIN_TOK_S[t.size] / 60.0 / (PACKING_SPEEDUP if t.packing else 1.0)
        return 2.0 if t.epochs >= 2 and 2 * per_epoch + OVERHEAD_MIN[t.size] <= t.target_minutes[1] else 1.0
    return float(max(1, min(t.epochs, math.floor(TOKEN_BUDGET[t.size] / tok))))


def steps_for(t: Target, man: Mapping[str, Any]) -> int:
    return math.ceil(int(man["rows"]["train"]) / max(1, t.batch * t.accum)) * int(math.ceil(epochs_for(t, man)))


def minutes(t: Target, man: Mapping[str, Any]) -> dict[str, float]:
    """Expected GPU minutes, calibrated on the MEASURED 5090 run (3 Oct, ft_thinker_17b: 3175 tokens/step at 2.1 it/s -> ~6.5k tok/s
    including logging): trained tokens / rate x a long-sequence factor + a per-step overhead + dev evaluations (~4 per epoch, forward only)
    + DPO + load/merge/GGUF. The runner's cap (cap_minutes) is twice this plus 10."""
    ep = epochs_for(t, man)
    tok = float(man["tokens_train"]) * ep
    long_f = 1.3 if t.max_seq > 2048 else 1.0
    train_s = tok / TRAIN_TOK_S[t.size] * long_f / (PACKING_SPEEDUP if t.packing else 1.0) + steps_for(t, man) * STEP_OVERHEAD_S
    dev_tok = float(man.get("tokens_dev") or man["tokens_train"] * man["rows"]["dev"] / max(1, man["rows"]["train"]))
    dev_s = dev_tok * 4 * math.ceil(ep) / (3 * TRAIN_TOK_S[t.size])
    pref_s = float(man["rows"]["pref"]) * 3 * t.max_seq / 4 / TRAIN_TOK_S[t.size] * 2 if man["rows"]["pref"] >= 20 else 0.0
    tr = (train_s + dev_s + pref_s) / 60.0 + OVERHEAD_MIN[t.size]
    basis = "trainmix"
    meas = _measured_train_s(t, man, ep)                         # P1.5: the measured-runs model (creator.gpueff), the table above is its fallback
    if meas is not None:
        tr = (meas[0] + dev_s + pref_s) / 60.0 + max(0.0, OVERHEAD_MIN[t.size] - meas[1] / 60.0)
        basis = "gpueff"
    return {"train": round(tr, 1), "eval": float(t.eval_minutes), "gate": 12.0 if t.gate else 0.0, "epochs": ep, "steps": steps_for(t, man),
            "total": round(tr + t.eval_minutes + (12.0 if t.gate else 0.0) + 1.0, 1), "basis": basis}


_EFF: list[Any] = []                                             # cached creator.gpueff.Eff (built once per process, from the recorded GPU runs)


def _measured_train_s(t: Target, man: Mapping[str, Any], epochs: float) -> Optional[tuple[float, float]]:
    """(train seconds, load seconds) of this module from the measured runs of the same model size, or None to use the calibrated table.
    Needs >= 2 measured runs of the size; NUPEN_JOBCOST=off disables it. Any failure falls back silently (an estimate never breaks a build)."""
    if os.environ.get("NUPEN_JOBCOST", "").lower() in ("off", "0", "no"):
        return None
    try:
        from creator import gpueff as GE
        if not _EFF:
            _EFF.append(GE.Eff(GE.load_runs()))
        eff = _EFF[0]
        if t.size not in eff.rate or sum(1 for r in eff.runs if r["size"] == t.size) < 2:
            return None
        padded = GE.padded_tokens(float(man["rows"]["train"]) * epochs, None, float(man["tokens_train"]) * epochs)
        return eff.train_s(t.size, padded), eff.load_s.get(t.size, 60.0)
    except Exception:                                            # noqa: BLE001
        return None


def cap_minutes(train_minutes: float) -> float:
    return round(train_minutes * 2 + 10, 1)


def build(state: Path, repo: Path, targets: Sequence[str] = (), out_root: Optional[Path] = None, say: Callable[[str], None] = print) -> dict[str, Any]:
    ctx = Ctx(Path(state), Path(repo), load_heldout())
    evals = eval_sets(ctx)
    fz = frozen()
    say("eval sets: " + ", ".join(f"{k} {len(v['texts'])}" for k, v in evals.items()))
    cache: dict[str, tuple[list[Row], collections.Counter[str]]] = {}
    res = {}
    for t in TARGETS:
        if targets and t.name not in targets:
            continue
        m = build_target(t, ctx, evals, fz, out_root, cache)
        say(f"{t.name}: train {m['rows']['train']} dev {m['rows']['dev']} pref {m['rows']['pref']}  {m['tokens_train']} tokens  "
            f"~{m['gpu_minutes']['total']} GPU min{'  BLOCKED' if t.blocked else ''}")
        res[t.name] = m
    return res


def audit_mix(d: Path, heldout: Mapping[str, Any]) -> list[str]:
    """Before upload: gpuday.audit_export (private markers, e-mails, frozen texts) + the trust-gate exclusion on every ids row."""
    from creator import gpuday as GD
    bad = GD.audit_export(d, frozen())
    t = BY_NAME.get(Path(d).name)
    if t is not None and t.voice_only:              # voice-only: post-cut talk rows allowed, but nothing else may be in it
        for ids in sorted(Path(d).glob("*_train.ids.jsonl")):
            bad += [f"{d.name}/{ids.name}: {r.get('id')} is not a talk row (voice-only mix)" for r in jsonl(ids)
                    if not str(r.get("source", "")).startswith("talk_")]
        return bad
    for ids in sorted(Path(d).glob("*_train.ids.jsonl")):
        for r in jsonl(ids):
            if excluded(float(r.get("ts") or 0.0), str(r.get("sha") or ""), heldout):
                bad.append(f"{d.name}/{ids.name}: {r.get('id')} violates the exclusion rule")
    return bad


# ------------------------------------------------------------------------------------------------ inventory
def _count(p: Path) -> int:
    try:
        with Path(p).open(encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def inventory(state: Path, mixes: Optional[Mapping[str, Mapping[str, Any]]] = None) -> list[dict[str, Any]]:
    """Every collected dataset with rows, split, provenance, licence, audit and consumer. `mixes` = build() result (kept rows per source)."""
    g, th = gpuday_dir(), Path(state) / "thinking"
    ex, ep = g / "export", g / "export_plus"
    kept: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for tname, m in (mixes or {}).items():
        for s, n in (m.get("by_source_kept") or {}).items():
            kept[s][tname] = n
    def used_by(*srcs: str) -> dict[str, int]:
        c: collections.Counter[str] = collections.Counter()
        for s in srcs:
            c.update(kept.get(s, {}))
        return dict(c)
    traj = sum(_count(p) for v in ("sft", "debug", "review", "pair") for p in trajectory_files(v))
    talk_rows = sum(_count(p) for k in ("understand", "speak") for p in talk_files(k))
    E = []

    def add(name: str, path: str, rows: Any, split: str, prov: str, lic: str, audit: str, consumer: str, kept_in: Optional[Mapping[str, int]] = None) -> None:
        E.append({"name": name, "path": path, "rows": rows, "split": split, "provenance": prov, "licence": lic, "audit": audit,
                  "consumer": consumer, "kept_in_mix": dict(kept_in or {})})
    add("trace_bank", str(th / "trace_bank.jsonl"), _count(th / "trace_bank.jsonl"), "all train (dev 10% by qid inside the mix)",
        "Qwen3-14B worked traces on reasondrills questions (public repo histories; 2 Nupen-history rows)", "public repository facts",
        "answer line re-checked against the known answer; Nupen rows filtered by the cut; frozen/eval overlap removed",
        "thinker_17b + thinker_06b SFT (cot + plain rows); retrieval examples at home (reasondrills)", used_by("trace_bank_cot", "trace_bank_plain"))
    add("ladder_distill", str(g / "distill" / "ladder_distill.jsonl"), _count(g / "distill" / "ladder_distill.jsonl"), "distill pool (never eval)",
        "verified big-model answers to the effladder DISTILL pool (27B coding now; 14B reasoning when its ladder lands)",
        "MBPP CC-BY-4.0 prompts; model answers", "verified by executed tests / known answers; nupen_fn rows dropped (current tree)",
        "coder_17b (coding rows); thinker_* (reasoning rows)", used_by("ladder_distill_code", "ladder_distill_cot", "ladder_distill_plain"))
    add("effladder results", str(th / "effladder.jsonl"), _count(th / "effladder.jsonl"), "eval only (items hash ade390bee97a)",
        "every ladder model x config on the frozen ladder items", "own measurements", "eval rows: never trained on",
        "router / cascade choice (effladder summary + homecost); the BASE side of every compare_job here", None)
    add("ladder items", str(ladder_items_path()), "see counts", "eval + distill pool", "reasondrills/judgment/thinkbench/MBPP/HumanEval/rl",
        "MBPP CC-BY-4.0, HumanEval MIT, public histories", "eval side is an EVAL SET for every dedupe here",
        "effladder eval of every tuned model; dedupe", None)
    for d, names in ((ex, ("coder_hf_train", "coder_hf_eval", "rl_tasks_hf", "rl_tasks_more", "coder_sft_mix", "commit_train", "commit_eval",
                           "embed_docs")),
                     (ep, ("coder_sft_mix", "coder_more_train", "coder_more_eval", "commit_train", "commit_eval", "handoff_train", "handoff_eval",
                           "handoff_unjudged", "pref_train", "pref_eval", "pref_synth_train", "pref_more_train", "rl_tasks", "worked_train"))):
        for n in names:
            add(f"{d.name}/{n}", str(d / f"{n}.jsonl"), _count(d / f"{n}.jsonl"), *INV_FACTS.get(f"{d.name}/{n}", INV_FACTS.get(n, ("?", "?", "?", "?", "?"))),
                kept_in=used_by(*INV_SOURCES.get(f"{d.name}/{n}", ())))
    add("C2 trajectories", str(g / "trajectories"), traj, "views SFT / DEBUG / REVIEW / pairs", "27B agent runs (C2), later today",
        "own runs", "rows on trust-gate held-out tasks/commits dropped; bake-off/eval overlap removed",
        "coder_17b (SFT+DEBUG), reviewer_17b (REVIEW), coder DPO (pairs) - the jobs skip themselves until the rows exist",
        used_by("c2_sft", "c2_debug", "c2_review", "c2_pairs"))
    add("talk rows", str(g / "talk"), talk_rows, "understand / speak SFT", "big model on Nupen's docs + records (creator.talkdata)",
        "own", "post-cut (current docs/records): allowed for the VOICE adapter only (teacher, 3 Oct) - never in a coder/thinker mix",
        "voice_17b SFT (tag 'voice-only, not gate-eligible'), evaluated with creator.talkeval", used_by("talk_understand", "talk_speak"))
    add("bake-off transcripts", str(runtime() / "agents" / "runs" / "c1"), _count(runtime() / "agents" / "runs" / "c1" / "results.jsonl"),
        "eval (bake-off task set)", "4 agents x 33 tasks", "own runs", "task prompts/shas/function names are an EVAL SET for every dedupe",
        "agent bake-off verdict; UNUSED for training on purpose (the bake-off is a benchmark; training on it would end it)", None)
    for pi in sorted(runtime().glob("pubindex*")):
        add(f"public-code index {pi.name}", str(pi), sum(1 for _ in pi.rglob("*") if _.is_file()), "index (files)", "embedded public code",
            "public repositories (licences per repo)", "retrieval only", "retrieval at home (code search); not a training set", None)
    add("judgment records", str(th / "judgment.jsonl"), _count(th / "judgment.jsonl"), "records", "home judgment rounds", "own",
        "outcome labels, no worked answers", "UNUSED for fine-tuning: no verified worked answers yet (a 0.6B judgment adapter is the next candidate: "
        "0.6B retrieval@64 0.738 > 1.7B)", None)
    return E


# name -> (split, provenance, licence, audit, consumer)
INV_FACTS: dict[str, tuple[str, str, str, str, str]] = {
    "export/coder_hf_train": ("train (eval = 1 in 20 repos)", "bigcode/commitpackft Python commits -> SEARCH/REPLACE", "per-row permissive repo licence",
                              "h49 export audit; public (no cut)", "coder_17b SFT"),
    "export/coder_hf_eval": ("eval", "commitpackft held-out repos", "permissive", "EVAL SET (dedupe)", "eval only (dedupe; harness)"),
    "export/rl_tasks_hf": ("train 1354 / eval 146", "nvidia/OpenCodeInstruct functions + tests", "CC-BY-4.0 (attribution NVIDIA)",
                           "upstream tests pass; eval side = EVAL SET", "coder_17b SFT (train references); GRPO tasks (rl_grpo, later)"),
    "export/rl_tasks_more": ("train / eval", "public repo functions + generated tests", "per-repo (BSD/MIT/PSF...)", "eval side = EVAL SET",
                             "coder_17b SFT (train references)"),
    "export/coder_sft_mix": ("train", "older export (Nupen handoffs + commits)", "own", "superseded by export_plus/coder_sft_mix", "UNUSED: superseded"),
    "export/commit_train": ("train", "Nupen commits (older export)", "own", "36 rows at/after the cut: must NOT be trained on", "UNUSED: superseded by export_plus (cut-clean)"),
    "export/commit_eval": ("eval", "Nupen commits", "own", "EVAL SET", "eval only (dedupe; coder harness)"),
    "export/embed_docs": ("index docs", "Nupen commits/lessons for the embedding index", "own", "private scrub", "embedding index (retrieval)"),
    "export_plus/coder_sft_mix": ("train", "Nupen handoffs + commits + public commits", "own + public", "exclusion rule applied per row (ts/sha)",
                                  "coder_17b SFT (Nupen rows only; public rows come from coder_more)"),
    "export_plus/coder_more_train": ("train", "public repo commits with tests", "per-repo licence", "public", "coder_17b SFT"),
    "export_plus/coder_more_eval": ("eval", "public repo commits", "per-repo", "EVAL SET", "eval only (dedupe)"),
    "export_plus/commit_train": ("train", "Nupen commits", "own", "cut-clean (max ts before the cut); exclusion re-checked per row", "coder_17b via coder_sft_mix"),
    "export_plus/commit_eval": ("eval", "Nupen commits", "own", "EVAL SET", "eval only (dedupe; coder harness)"),
    "export_plus/handoff_train": ("train", "Nupen handoffs (adopted)", "own", "exclusion rule", "coder_17b via coder_sft_mix"),
    "export_plus/handoff_eval": ("eval", "Nupen handoffs", "own", "EVAL SET", "eval only"),
    "export_plus/handoff_unjudged": ("unjudged", "interrupted handoffs", "own", "no verdict", "UNUSED: no outcome label (not trusted as SFT)"),
    "export_plus/pref_train": ("train", "2 real Nupen pairs + 9 public fixes + 194 synthetic 'tests omitted'", "own + public", "exclusion rule",
                               "coder_17b DPO stage (>= 20 pairs)"),
    "export_plus/pref_eval": ("eval", "synthetic pairs", "public", "EVAL SET", "eval only"),
    "export_plus/pref_synth_train": ("train", "synthetic (inside pref_train)", "public", "-", "via pref_train"),
    "export_plus/pref_more_train": ("train", "public fix pairs (inside pref_train)", "public", "-", "via pref_train"),
    "export_plus/rl_tasks": ("train / eval", "Nupen's own functions (current tree)", "own", "BLOCKED for training: read from the tree after the cut",
                             "effladder nupen_fn eval items only"),
    "export_plus/worked_train": ("empty", "gpuday worked bank (never written)", "-", "-", "UNUSED: empty - the trace bank replaces it (thinker mixes)"),
}
INV_SOURCES: dict[str, tuple[str, ...]] = {
    "export/coder_hf_train": ("coder_hf_train",), "export/rl_tasks_hf": ("rl_tasks_hf",), "export/rl_tasks_more": ("rl_tasks_more",),
    "export_plus/coder_sft_mix": ("coder_sft_mix",), "export_plus/coder_more_train": ("coder_more_train",), "export_plus/pref_train": ("pref_train",),
}


def inventory_md(E: Sequence[Mapping[str, Any]], mixes: Mapping[str, Mapping[str, Any]]) -> str:
    L = ["# Data inventory (h61)", "", f"Built {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by `python -m creator.trainmix inventory`.",
         "Training exclusion: no row from Nupen's own repo/state at/after 2026-10-01T21:58:30Z or a trust-gate held-out commit/task.", "",
         "| dataset | rows | split | provenance | licence | audit | consumer | kept in mix |", "|---|---|---|---|---|---|---|---|"]
    for e in E:
        k = ", ".join(f"{t} {n}" for t, n in e["kept_in_mix"].items()) or "-"
        L.append(f"| {e['name']} | {e['rows']} | {e['split']} | {e['provenance']} | {e['licence']} | {e['audit']} | {e['consumer']} | {k} |")
    L += ["", "## Mixes", "", "| target | train | dev | pref | train tokens | GPU min | kept by source | blocked |", "|---|---|---|---|---|---|---|---|"]
    for n, m in mixes.items():
        L.append(f"| {n} | {m['rows']['train']} | {m['rows']['dev']} | {m['rows']['pref']} | {m['tokens_train']} | {m['gpu_minutes']['total']} | "
                 f"{', '.join(f'{s} {c}' for s, c in m['by_source_kept'].items()) or '-'} | {m['blocked'] or '-'} |")
    L += ["", "## Drops per mix", ""]
    for n, m in mixes.items():
        L.append(f"- **{n}**: " + "; ".join(f"{k} {v}" for k, v in m["drops"].items()))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ GPU jobs
POD_DIR = "gpuday"
SHM = "/dev/shm/nupen_train"
HF_CACHE = f"{POD_DIR}/hf_cache"                     # on /workspace (the merge scratch goes to /dev/shm)


def upload_bundle(root: Path, names: Sequence[str]) -> tuple[bytes, dict[str, Any]]:
    """gzip tar: the (patched) finetune.py + every target's mix, extracted under gpuday/ on the pod. Refused whole on any audit finding."""
    import io
    import tarfile
    h = load_heldout()
    members: list[tuple[str, bytes]] = [("finetune.py", (ROOT / "scripts" / "gpuday" / "finetune.py").read_bytes())]
    for n in names:
        d = Path(root) / n
        bad = audit_mix(d, h)
        if bad:
            raise ValueError(f"mix {n} failed the audit, nothing is uploaded: {bad[:5]}")
        for f in ("train.jsonl", "dev.jsonl", "pref.jsonl", "MANIFEST.json", "spec_prompts.jsonl"):
            if (d / f).is_file():
                members.append((f"trainmix/{n}/{f}", (d / f).read_bytes()))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            tf.addfile(ti, io.BytesIO(data))
    blob = buf.getvalue()
    return blob, {"files": [m[0] for m in members], "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def ft_remote(t: Target, man: Mapping[str, Any], shm: str = SHM, wait_for: str = "", vram_wait_s: int = 1800) -> str:
    """Pod script of one fine-tune: row/VRAM/disk checks (skip with the reason, exit 0) -> finetune.py pipeline with dev early stopping,
    merge + f16 + Q4 GGUF on /dev/shm -> the Q4 GGUF moved into models/ (served after register) -> adapter + result.json kept under
    gpuday/runs/<job>/ (they come home) -> /dev/shm scratch removed."""
    name, data = f"ft_{t.name}", f"{POD_DIR}/trainmix/{t.name}"
    w, keep = f"{shm}/{name}", f"{POD_DIR}/runs/{name}"
    ws_gb = math.ceil(BF16_GB[t.size] + Q4_GB[t.size] + 0.5)                 # HF base cache + the served GGUF
    shm_gb = math.ceil(2 * BF16_GB[t.size] + Q4_GB[t.size] + 0.5)            # merged + f16 GGUF + Q4 GGUF
    hf_home = f"$PWD/{HF_CACHE}"
    if t.hf_on_shm:                                  # the base download moves to the RAM disk too: only the served GGUF stays on the workspace
        ws_gb, shm_gb, hf_home = math.ceil(Q4_GB[t.size] + 0.5), math.ceil(3 * BF16_GB[t.size] + Q4_GB[t.size] + 0.5), f"{shm}/hf_cache"
    if t.lora_serve:                                 # no merge / f16 / Q4: only the base download (cached) and the adapter
        ws_gb, shm_gb = (1 if t.hf_on_shm else math.ceil(BF16_GB[t.size] + 0.5)), (math.ceil(BF16_GB[t.size] + 1) if t.hf_on_shm else 2)
    pref = man["rows"]["pref"] >= 20
    no_ckpt, need_gb = _mods().ckpt_plan(t)
    need_mib = max(int(t.vram_mib), int(need_gb * 1024))
    opts = (f"--base {HF_BASE[t.size]} --data {data}/train.jsonl --eval-data {data}/dev.jsonl --out {w} --max-seq {t.max_seq} "
            f"--epochs {epochs_for(t, man)} --batch {t.batch} --accum {t.accum} --lr {t.lr} --patience 2 "
            f"--llama-cpp {POD_DIR}/llama.cpp --quantize \"$(cat {POD_DIR}/quantize_path)\" --quant Q4_K_M --adapter-gguf"
            + (f" --pref {data}/pref.jsonl --method dpo" if pref else "") + (" --packing" if t.packing else "")
            + (" --no-merge" if t.lora_serve else "") + (" --no-grad-ckpt" if no_ckpt else "")
            + (" --group-by-length" if GROUP_BY_LENGTH_VERIFIED and not t.packing else ""))
    skip = lambda why: f"printf '@@result={{\"skipped\": \"%s\"}}\\n' \"{why}\"; exit 0"  # noqa: E731
    sv = serve_name(t)
    safe_opts = re.sub(r" --(packing|no-grad-ckpt|group-by-length)\b", "", opts)
    safe_opts = re.sub(r"--batch \d+ --accum \d+", f"--batch {t.safe_batch} --accum {t.safe_accum}", safe_opts)
    # The pod's Python / llama.cpp: gpuday/python (pod_setup.sh) when present, else the image's venv /venv/main (Unsloth stack), else python3
    # (h61 first run: gpuday/python and gpuday/quantize_path were missing - the setup had run with $HOME as its cwd - and the system python3
    # has no 'datasets'). Converter: gpuday/llama.cpp or $HOME/gpuday/llama.cpp; quantizer: the recorded path, PATH, /opt/llama.cpp.
    find_py = ("PY=\"$(cat gpuday/python 2>/dev/null || cat \"$HOME/gpuday/python\" 2>/dev/null || true)\"; "
               "[ -n \"$PY\" ] && [ -x \"$PY\" ] || PY=/venv/main/bin/python; [ -x \"$PY\" ] || PY=$(command -v python3 || command -v python)")
    find_llama = ("LC=gpuday/llama.cpp; [ -f $LC/convert_hf_to_gguf.py ] || LC=\"$HOME/gpuday/llama.cpp\"; "
                  "[ -f \"$LC/convert_hf_to_gguf.py\" ] || { printf '@@result={\"skipped\": \"no llama.cpp convert_hf_to_gguf.py\"}\\n'; exit 0; }; "
                  "Q=\"$(cat gpuday/quantize_path 2>/dev/null || cat \"$HOME/gpuday/quantize_path\" 2>/dev/null || true)\"; "
                  "[ -n \"$Q\" ] && [ -x \"$Q\" ] || Q=$(command -v llama-quantize || true); [ -n \"$Q\" ] && [ -x \"$Q\" ] || Q=/opt/llama.cpp/llama-quantize; "
                  "[ -x \"$Q\" ] || { printf '@@result={\"skipped\": \"no llama-quantize\"}\\n'; exit 0; }")
    opts = opts.replace(f"--llama-cpp {POD_DIR}/llama.cpp --quantize \"$(cat {POD_DIR}/quantize_path)\"", "--llama-cpp \"$LC\" --quantize \"$Q\"")
    return "\n".join([
        "set -e", find_py,
        "\"$PY\" -c 'import datasets, trl, peft' 2>/dev/null || { printf '@@result={\"skipped\": \"%s has no datasets/trl/peft\"}\\n' \"$PY\"; exit 0; }",
        find_llama,
        f"n=$(wc -l < {data}/train.jsonl 2>/dev/null || echo 0)",
        f"if [ \"$n\" -lt {t.min_rows} ]; then {skip(f'$n training rows, needs {t.min_rows}')}; fi",
        *([f"for i in $(seq 1 120); do grep -q -E \"'loss'|it/s\" {wait_for} 2>/dev/null && break; sleep 5; done   # paired run: the partner holds its VRAM first"]
          if wait_for else []),
        # wait (an overlapping eval server or a paired trainer may still be loading/finishing) - then skip with the reason
        f"need={need_mib}; t0=$(date +%s); while :; do vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); "
        f"vfree=${{vfree:-0}}; [ \"$vfree\" -ge \"$need\" ] && break; [ $(( $(date +%s) - t0 )) -ge {int(vram_wait_s)} ] && break; sleep 15; done",
        f"if [ \"$vfree\" -lt \"$need\" ]; then {skip(f'VRAM: $vfree MiB free after {int(vram_wait_s)} s, needs {need_mib}')}; fi",
        f"wsf=$(df -Pk . | awk 'NR==2 {{print int($4/1048576)}}'); shf=$(df -Pk {posixpath.dirname(shm)} | awk 'NR==2 {{print int($4/1048576)}}')",
        f"if [ \"$wsf\" -lt {ws_gb} ]; then {skip(f'disk: $wsf GB free on the workspace, needs {ws_gb}')}; fi",
        f"if [ \"$shf\" -lt {shm_gb} ]; then {skip(f'disk: $shf GB free on {posixpath.dirname(shm)}, needs {shm_gb}')}; fi",
        f"export HF_HOME=\"{hf_home}\" PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false",
        f"for i in $(seq 1 360); do ls \"$HF_HOME\"/.prestage_*.lock >/dev/null 2>&1 || break; sleep 5; done   # a base pre-staged in the background: wait",
        f"rm -rf {w}; mkdir -p {w} {keep} models",
        # SAFE MODE: a failed fine-tune is retried ONCE with the proven settings (no packing / no speed flags, safe batch); the reason is kept
        f"if ! \"$PY\" {POD_DIR}/finetune.py pipeline {opts} > {keep}/train.log 2>&1; then "
        f"why=$(grep -a -E 'Error|error' {keep}/train.log | tail -1 | tr -d '\"\\\\' | cut -c1-300); cp -f {keep}/train.log {keep}/train_failed.log; "
        f"echo \"safe mode retry: $why\"; rm -rf {w}; mkdir -p {w}; "
        f"\"$PY\" {POD_DIR}/finetune.py pipeline {safe_opts} > {keep}/train.log 2>&1 || {{ tail -40 {keep}/train.log; rm -rf {w}; exit 5; }}; "
        f"\"$PY\" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d[\"safe_mode\"]={{\"why\": sys.argv[2]}}; json.dump(d,open(p,\"w\"))' "
        f"{w}/result.json \"$why\"; fi",
        f"g={w}/model-Q4_K_M.gguf; if [ -f \"$g\" ]; then mv -f \"$g\" models/{sv}; sha256sum models/{sv} | cut -d' ' -f1 > models/{sv}.ok; "
        f"echo \"@@served_as={sv}\"; fi",
        f"ad={w}/adapter; [ -d {w}/dpo/adapter ] && ad={w}/dpo/adapter; cp -r \"$ad\" {keep}/adapter; cp -f {w}/result.json {keep}/; "
        f"[ -f {w}/adapter.gguf ] && cp -f {w}/adapter.gguf {keep}/ || true",
        *([f"if [ -f {w}/adapter.gguf ]; then cp -f {w}/adapter.gguf models/{lora_name(t)}; echo \"@@lora={lora_name(t)}\"; fi"]
          if t.lora_serve else []),
        f"rm -rf {w}",
        f"cat {keep}/result.json | tr -d '\\r\\n' | sed 's/^/@@result=/'", "echo", ""])


def cleanup_remote() -> str:
    """End of a training list: remove the merge/f16 scratch (/dev/shm/nupen_train/ft_*) and any tuned GGUF left in models/ (its delete job
    normally did that). The base-model download caches ({HF_CACHE}, {SHM}/hf_cache) are KEPT: deleting them made every later run download
    the base again while the GPU idled (teacher, 4 Oct); the guardian's disk guard protects them."""
    return (f"rm -rf {SHM}/ft_*; rm -f models/*-nupen-*.gguf models/*-nupen-*.gguf.ok; "
            "echo '@@result={\"cleaned\": true, \"kept\": \"hf_cache\"}'\n")


def stop_model_servers_remote(models: Sequence[str]) -> str:
    """Stop only the runner's servers that serve these models (run/<port>.sig names the model): the eval server leaves the VRAM to the next
    trainer; a guardian filler or another model's server is not touched."""
    pats = " ".join(f"'{m}|'" for m in models)
    return ("cd run 2>/dev/null || { echo '@@result={\"stopped\": []}'; exit 0; }\nst=\"\"\n"
            "for sig in *.sig; do [ -f \"$sig\" ] || continue; p=${sig%.sig}; for m in " + pats + "; do "
            "case \"$(cat $sig)\" in \"$m\"*) kill \"$(cat $p.pid 2>/dev/null)\" 2>/dev/null; rm -f $p.pid $p.sig; st=\"$st $p\";; esac; done; done\n"
            "echo \"@@result={\\\"stopped\\\": \\\"$st\\\"}\"\n")


def stage_of(job: Mapping[str, Any]) -> str:
    """'train' (upload, fetch, fine-tune, register, prestage) or 'eval' (the held-out evals, stop the eval server, delete): the module runner
    overlaps module N's eval stage with module N+1's train stage."""
    n = str(job.get("name") or "")
    if job.get("stage"):
        return str(job["stage"])
    return "train" if n.startswith(("trainmix_upload", "ft_", "register_", "fetch_", "prestage_")) else "eval"


def delete_remote(t: Target) -> str:
    sv = serve_name(t)
    lf = lora_name(t)
    return f"rm -f models/{sv} models/{sv}.ok models/{lf} models/{lf}.ok; echo '@@result={{\"deleted\": \"{sv}\"}}'\n"


def _keep_on_pod() -> tuple[str, ...]:
    from creator import effladder as EL
    return tuple(EL.KEEP_ON_POD)


def lora_name(t: Target) -> str:
    return serve_name(t).replace("-Q4_K_M.gguf", "-lora.gguf")


def register_lora_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job after a lora_serve fine-tune: the adapter GGUF's bytes + sha256 (from the ft job's result) go into the extra_models
    file as an ADAPTER entry {bytes, sha256, base, lora}: the runner serves `serve_as` as llama-server -m <base> --lora <adapter> (no merged
    model). The runner re-hashes the adapter on the pod before serving."""
    from creator import gpuday as GD
    from creator import gpupulse as GP
    a = dict(ctx.get("args") or {})
    src, name, base, lora = (str(a.get(k) or "") for k in ("source", "serve_as", "base", "lora"))
    pulse = str(ctx.get("pulse") or "")
    row = None
    for r in jsonl(Path(str(ctx["state"])) / "thinking" / "gpu_pulse_runs.jsonl"):
        if r.get("job") == f"ext:{src}" and (not pulse or str(r.get("gpu_pulse") or "") == pulse):
            row = r
    res = (row or {}).get("result") if isinstance((row or {}).get("result"), Mapping) else {}
    f = (((res or {}).get("lora_gguf") or {}).get("files") or {}).get("adapter.gguf") if isinstance(res, Mapping) else None
    if row is None or row.get("rc") not in (0, None) or not isinstance(f, Mapping) or not f.get("sha256"):
        return {"registered": False, "reason": f"{src} made no adapter.gguf" + (f" (rc {row.get('rc')})" if row else " (no run)")}
    path = Path(str(a.get("extra_models_file") or ctx.get("extra_models_file") or GP.extra_models_file()))
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    cur = cur if isinstance(cur, dict) else {}
    cur[name] = {"bytes": int(f["bytes"]), "sha256": str(f["sha256"]).lower(), "base": base, "lora": lora}
    GP.extra_models({"extra_models": cur, "extra_models_file": str(path.with_name(path.name + ".absent"))})      # validate before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    del GD
    return {"registered": True, "model": name, "base": base, "lora": lora, "bytes": int(f["bytes"]), "file": str(path)}


UNSLOTH_REPO = {"0.6b": "unsloth/Qwen3-0.6B", "1.7b": "unsloth/Qwen3-1.7B", "4b": "unsloth/Qwen3-4B"}   # what Unsloth fetches for Qwen/Qwen3-*


def prestage_remote(t: Target, shm: str = SHM) -> str:
    """Background pre-download of a module's base weights into the HF cache it trains from (parallel ranged curl, the /root/pdl.sh pattern:
    HF throttles single connections from the pod). Holds $HF_HOME/.prestage_<size>.lock while running; ft_remote waits for it. Returns at once."""
    hf_home = f"{shm}/hf_cache" if t.hf_on_shm else f"$PWD/{HF_CACHE}"
    repo = UNSLOTH_REPO[t.size]
    py = ("import os,sys,subprocess,json\n"
          "from huggingface_hub import HfApi, snapshot_download\n"
          "repo=sys.argv[1]\n"
          "d=snapshot_download(repo, allow_patterns=['*.json','*.txt','*.jinja','*.model'])\n"
          "info=HfApi().model_info(repo, files_metadata=True)\n"
          "for s in info.siblings:\n"
          "    if not s.rfilename.endswith('.safetensors') or not s.size: continue\n"
          "    f=os.path.join(d,s.rfilename)\n"
          "    if os.path.isfile(f) and os.path.getsize(f)==s.size: continue\n"
          "    url=f'https://huggingface.co/{repo}/resolve/main/{s.rfilename}'\n"
          "    n=8; ch=(s.size+n-1)//n; ps=[]\n"
          "    for i in range(n):\n"
          "        a=i*ch; b=min(s.size,(i+1)*ch)-1\n"
          "        ps.append(subprocess.Popen(['bash','-c',f'until curl -fL -s -r {a}-{b} -o {f}.p{i} --speed-limit 50000 --speed-time 60 {url} "
          "&& [ $(stat -c %s {f}.p{i}) = {b-a+1} ]; do sleep 3; done']))\n"
          "    [p.wait() for p in ps]\n"
          "    with open(f+'.tmp','wb') as o:\n"
          "        for i in range(n):\n"
          "            o.write(open(f'{f}.p{i}','rb').read()); os.remove(f'{f}.p{i}')\n"
          "    if os.path.getsize(f+'.tmp')==s.size: os.replace(f+'.tmp',f)\n"
          "print('prestaged',repo)\n")
    lock = f"\"$HF_HOME\"/.prestage_{t.size}.lock"
    return "\n".join([
        f"export HF_HOME=\"{hf_home}\"; mkdir -p \"$HF_HOME\"",
        "PY=\"$(cat gpuday/python 2>/dev/null || true)\"; [ -n \"$PY\" ] && [ -x \"$PY\" ] || PY=/venv/main/bin/python",
        f"cat > /tmp/prestage_{t.size}.py <<'PRESTAGE_PY'\n{py}PRESTAGE_PY",
        f"if [ -f {lock} ]; then echo '@@result={{\"prestage\": \"already running\"}}'; exit 0; fi",
        f"touch {lock}; nohup bash -c \"\\\"$PY\\\" /tmp/prestage_{t.size}.py {repo} > /tmp/prestage_{t.size}.log 2>&1; rm -f {lock}\" >/dev/null 2>&1 &",
        f"echo '@@result={{\"prestage\": \"{repo}\", \"background\": true}}'", ""])


def jobs(cfg: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None, targets: Sequence[str] = (),
         cleanup: bool = True, prestage: Optional[Target] = None) -> list[dict[str, Any]]:
    """The GPU-runner job list (gpu_pulse.py run --jobs-from creator.trainmix:jobs). Reads the built mixes (build first)."""
    c = dict(cfg or {})
    root = Path(root or c.get("trainmix_root") or mix_dir())
    want = list(targets or c.get("trainmix_targets") or [t.name for t in TARGETS])
    plan: list[tuple[Target, dict[str, Any]]] = []
    for t in TARGETS:
        if t.name not in want or t.blocked:
            continue
        mp = root / t.name / "MANIFEST.json"
        if not mp.is_file():
            continue
        plan.append((t, json.loads(mp.read_text(encoding="utf-8"))))
    blob, info = upload_bundle(root, [t.name for t, _m in plan])
    from creator import gpuday as GD
    out: list[dict[str, Any]] = [{"name": "trainmix_upload", "remote": GD.upload_script(blob, info), "minutes": 1, "low_util_abort_minutes": 0}]
    for t, man in plan:
        mins = minutes(t, man)                       # always from the current estimator (a manifest's figure may predate a calibration)
        sv = serve_name(t)
        out.append({"name": f"ft_{t.name}", "remote": ft_remote(t, man), "free_gpu": True, "minutes": mins["train"], "stage": "train",
                    "max_minutes": cap_minutes(mins["train"]), "low_util_abort_minutes": 0,
                    "outputs": [f"{POD_DIR}/runs/ft_{t.name}/result.json", f"{POD_DIR}/runs/ft_{t.name}/adapter",
                                f"{POD_DIR}/runs/ft_{t.name}/adapter.gguf", f"{POD_DIR}/runs/ft_{t.name}/train.log"]})
        fetched = t.lora_serve and t.base_gguf not in _keep_on_pod()
        if fetched:                                  # the adapter is served on the base GGUF: fetch it when the pod does not keep it
            from creator import effladder as EL
            out.append({"name": f"fetch_{t.name}_basegguf", "remote": EL.fetch_script(".", t.base_gguf), "minutes": 0.5, "low_util_abort_minutes": 0})
        if t.lora_serve:
            out.append({"name": f"register_{t.name}", "call": "creator.trainmix:register_lora_job", "minutes": 1, "low_util_abort_minutes": 0,
                        "args": {"source": f"ft_{t.name}", "serve_as": sv, "base": t.base_gguf, "lora": lora_name(t)}})
        else:
            out.append({"name": f"register_{t.name}", "call": "creator.gpuday:register_tuned_job", "args": {"source": f"ft_{t.name}", "serve_as": sv},
                        "minutes": 1, "low_util_abort_minutes": 0})
        if t.eval_suites:
            out.append({"name": f"eval_{t.name}", "call": "creator.trainmix:eval_job", "model": sv, "minutes": t.eval_minutes,
                        "max_minutes": round(t.eval_minutes * 2.5, 1), "low_util_abort_minutes": 0,
                        "args": {"target": t.name, "suites": list(t.eval_suites), "configs": dict(t.eval_configs), "n": dict(t.eval_n),
                                 "base": ",".join([t.base_gguf, *t.also_vs]), "run": f"train-{t.name}"}})
        if t.roles:                                  # per-role held-out eval: the base model, then the adapter (same rows, same pulse)
            for m, role in ((t.base_gguf, "base"), (sv, "tuned")):
                out.append({"name": f"roleeval_{t.name}_{role}", "call": "creator.pipelinemix:pipeline_eval_job", "model": m,
                            "minutes": t.eval_minutes, "max_minutes": round(t.eval_minutes * 2.5, 1), "low_util_abort_minutes": 0,
                            "args": {"target": t.name, "role": role, "dir": str(root / t.name)}})
        if t.module:                                 # a training module's own held-out eval (creator.trainmods)
            out += _mods().eval_jobs(t, root, sv)
        if t.voice_only:                             # talk eval: the base voice, then the adapter (same questions, same pulse)
            for m, role in ((t.base_gguf, "base"), (sv, "tuned")):
                out.append({"name": f"talkeval_{t.name}_{role}", "call": "creator.trainmix:talk_eval_job", "model": m, "minutes": 6,
                            "max_minutes": 15, "low_util_abort_minutes": 0,
                            "args": {"target": t.name, "role": role, "base": t.base_gguf, "tuned": sv}})
        if t.gate:
            out.append({"name": f"gate_{t.name}", "call": "creator.gpuselfteach:heldout_gate_job", "model": sv, "minutes": 12, "max_minutes": 25,
                        "low_util_abort_minutes": 0, "args": {"n": 400, "home_model": t.base_gguf, "tuned": sv,
                                                              "train_qids": str(root / t.name / "train_qids.json")}})
        if fetched:
            from creator import effladder as EL
            out.append({"name": f"delete_{t.name}_basegguf", "remote": EL.delete_script(".", t.base_gguf), "minutes": 0.1, "low_util_abort_minutes": 0})
        out.append({"name": f"stopeval_{t.name}", "remote": stop_model_servers_remote([sv, t.base_gguf]), "minutes": 0.1,
                    "low_util_abort_minutes": 0})
        out.append({"name": f"delete_{t.name}", "remote": delete_remote(t), "minutes": 0.1, "low_util_abort_minutes": 0})
    if prestage is not None:                         # the NEXT module's base, downloaded in the background while this file's evals run
        out.append({"name": f"prestage_{prestage.name}", "remote": prestage_remote(prestage), "minutes": 0.1, "low_util_abort_minutes": 0})
    if cleanup:
        out.append({"name": "trainmix_cleanup", "remote": cleanup_remote(), "minutes": 0.1, "low_util_abort_minutes": 0})
    return out


# ------------------------------------------------------------------------------------------------ evaluation: effladder + paired comparison
def compare(rows: Sequence[Mapping[str, Any]], tuned: str, base: str, suites: Sequence[str], items_hash: str = "") -> dict[str, Any]:
    """Paired, per suite: every (tuned config) vs every (base config) on the SAME items (effladder rows of one items hash). gain = mean of
    tuned correct - base correct, 95% CI; ADOPT for a pair only when n >= MIN_N and the CI lower bound > 0. Also tokens out per answer
    (the home cost): the efficiency claim is a cheap tuned config >= an expensive base config."""
    from creator import gpuselfteach as GS
    ih = items_hash or next((str(r["items_hash"]) for r in rows if r.get("model") == tuned), "")
    by: dict[tuple[str, str, str], dict[str, tuple[int, float]]] = collections.defaultdict(dict)
    for r in rows:
        if r.get("items_hash") != ih or r.get("model") not in (tuned, base) or r.get("suite") not in suites or r.get("split") != "eval":
            continue
        by[(str(r["model"]), str(r["suite"]), str(r["config"]))].setdefault(str(r["item"]), (int(bool(r.get("correct"))), float(r.get("tok_out") or 0)))
    out: list[dict[str, Any]] = []
    for s in suites:
        tc = sorted(c for (m, ss, c) in by if m == tuned and ss == s)
        bc = sorted(c for (m, ss, c) in by if m == base and ss == s)
        for a in tc:
            for b in bc:
                ta, tb = by[(tuned, s, a)], by[(base, s, b)]
                common = sorted(set(ta) & set(tb))
                d = [float(ta[i][0] - tb[i][0]) for i in common]
                m, lo, hi = GS.mean_ci(d)
                ok = len(d) >= MIN_N and lo > 0
                out.append({"suite": s, "tuned_config": a, "base_config": b, "n": len(d),
                            "tuned_acc": round(sum(ta[i][0] for i in common) / len(common), 4) if common else None,
                            "base_acc": round(sum(tb[i][0] for i in common) / len(common), 4) if common else None,
                            "gain": round(m, 4), "gain_ci95": [None if not math.isfinite(lo) else round(lo, 4), None if not math.isfinite(hi) else round(hi, 4)],
                            "tuned_tok_out": round(sum(ta[i][1] for i in common) / len(common), 1) if common else None,
                            "base_tok_out": round(sum(tb[i][1] for i in common) / len(common), 1) if common else None,
                            "verdict": "ADOPT" if ok else "KEEP_BASE",
                            "not_worse": bool(d) and len(d) >= MIN_N and lo > -0.05})
    same = [p for p in out if p["tuned_config"] == p["base_config"]]
    adopt = [p for p in same if p["verdict"] == "ADOPT"]
    cheaper = [p for p in out if p["not_worse"] and (p["tuned_tok_out"] or 0) * 2 <= (p["base_tok_out"] or 0)]
    return {"tuned": tuned, "base": base, "items_hash": ih, "pairs": out,
            "verdict": "ADOPT" if adopt else "KEEP_BASE",
            "why": (f"significantly better at {', '.join(p['suite'] + ' ' + p['tuned_config'] for p in adopt)}" if adopt else
                    f"no same-config gain with n >= {MIN_N} and CI lower bound > 0"),
            "cheaper_not_worse": [f"{p['suite']}: tuned {p['tuned_config']} vs base {p['base_config']} (gain {p['gain']}, CI {p['gain_ci95']})"
                                  for p in cheaper]}


def eval_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job: the tuned model (served by this pulse) answers the target's effladder items with the same configs as the base
    model's recorded rows, then compare() writes one record to <state>/thinking/train_gate.jsonl. The record is a verdict, never a switch."""
    from creator import effladder as EL
    a = dict(ctx.get("args") or {})
    tuned = str(a.get("tuned") or ctx.get("model") or "")
    suites = list(a.get("suites") or [])
    sub = dict(ctx, args={"model": tuned, "suites": suites, "configs": a.get("configs") or {}, "n": a.get("n") or {}, "run": a.get("run") or "",
                          **({"port": a["port"]} if a.get("port") else {}), **({"items": a["items"]} if a.get("items") else {})})
    res = EL.ladder_job(sub)
    recs = [record(Path(str(ctx["state"])), tuned, b, suites, str(res.get("items_hash") or ""), str(a.get("target") or ""),
                   str(ctx.get("pulse") or ""), ladder=res) for b in str(a.get("base") or "").split(",") if b]
    return {"ladder": res, "verdicts": {r["base"]: r["verdict"] for r in recs}, "cheaper_not_worse": {r["base"]: r["cheaper_not_worse"] for r in recs},
            "n": {r["base"]: max((p["n"] for p in r["pairs"]), default=0) for r in recs}}


def record(state: Path, tuned: str, base: str, suites: Sequence[str], items_hash: str, target: str, pulse: str = "",
           ladder: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    from creator import effladder as EL
    cmp_ = compare(EL.read_rows(EL.results_path(state)), tuned, base, suites, items_hash)
    rec = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "pulse": pulse, "target": target, "ladder": dict(ladder or {}),
           **cmp_}
    p = gate_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def talk_compare(base_rows: Sequence[Mapping[str, Any]], tuned_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired by question text: intent accuracy, groundedness, answered; tuned - base with 95% CI. ADOPT needs n >= MIN_N, both intent and
    grounded CI lower bounds >= 0 and one > 0. The talk eval has ~45 questions: below MIN_N the verdict is INSUFFICIENT_N (never adopted)."""
    from creator import gpuselfteach as GS
    b = {str(r["q"]): r for r in base_rows}
    common = [r for r in tuned_rows if str(r["q"]) in b]
    out: dict[str, Any] = {"n": len(common)}
    los = []
    for k in ("intent_ok", "grounded", "answered"):
        d = [float(int(bool(r[k])) - int(bool(b[str(r["q"])][k]))) for r in common]
        m, lo, hi = GS.mean_ci(d)
        los.append(lo)
        out[k] = {"gain": round(m, 4), "ci95": [None if not math.isfinite(lo) else round(lo, 4), None if not math.isfinite(hi) else round(hi, 4)]}
    out["tokens_out_mean"] = {"base": round(sum(float(b[str(r["q"])].get("tokens_out") or 0) for r in common) / max(1, len(common)), 1),
                              "tuned": round(sum(float(r.get("tokens_out") or 0) for r in common) / max(1, len(common)), 1)}
    if len(common) < MIN_N:
        out["verdict"] = "INSUFFICIENT_N"
    else:
        out["verdict"] = "ADOPT" if min(los[:2]) >= 0 and max(los[:2]) > 0 else "KEEP_BASE"
    return out


def talk_eval_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job: creator.talkeval's held-out questions answered by the voice this pulse serves (the base voice, then the voice-only
    adapter). Each run is one record in <state>/thinking/train_gate.jsonl; the tuned run carries the paired comparison with the base run of
    the same pulse. Tag: voice-only, not gate-eligible."""
    from creator import gpupulse as GP
    from creator import talk as T
    from creator import talkeval as TE
    a = dict(ctx.get("args") or {})
    model, role = str(ctx.get("model") or ""), str(a.get("role") or "tuned")
    got = GP.attach(Path(str(ctx["tunnel_file"])), model)
    if got is None:
        return {"verdict": "NOT_RUN", "why": f"the pulse does not serve {model}"}
    port, pulse_id = got
    voice = T.Voice(Path(model), factory=lambda: GP.PodLLM(port, model, pulse_id))
    try:
        res = TE.run_layer(Path(str(ctx["repo"])), "voice", list(TE.QUESTIONS), voice)
    finally:
        voice.close()
    state, pulse = Path(str(ctx["state"])), str(ctx.get("pulse") or "")
    rec: dict[str, Any] = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "pulse": pulse, "target": a.get("target"),
                           "kind": "talkeval", "role": role, "model": model, "tag": "voice-only, not gate-eligible",
                           **{k: v for k, v in res.items() if k != "rows"}, "rows": res["rows"]}
    if role == "tuned":
        base = [r for r in jsonl(gate_path(state)) if r.get("kind") == "talkeval" and r.get("role") == "base" and r.get("pulse") == pulse
                and r.get("target") == a.get("target")]
        rec["compare"] = talk_compare(base[-1]["rows"], res["rows"]) if base else {"verdict": "NOT_RUN", "why": "no base run in this pulse"}
    p = gate_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return {k: v for k, v in rec.items() if k != "rows"}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m creator.trainmix", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "inventory", "jobs", "check", "autogen"))
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--targets", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--loop", action="store_true", help="autogen: keep the queue filled every --interval seconds (IDLE priority)")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--dry", action="store_true", help="autogen: build and validate, write no queue file")
    a = ap.parse_args(argv)
    targets = [t for t in a.targets.split(",") if t]
    if a.cmd == "autogen":
        MODS = _mods()

        def say(m: str) -> None:
            line = f"{dt.datetime.now().strftime('%H:%M:%S')} {m}"
            print(line, flush=True)
            try:
                lg = MODS.queue_dir() / "logs"
                lg.mkdir(parents=True, exist_ok=True)
                with (lg / "autogen.log").open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
        if a.loop:
            MODS.autogen_loop(Path(a.state), Path(a.repo), a.interval, say)
        else:
            MODS.autogen_once(Path(a.state), Path(a.repo), say, dry=a.dry)
        return 0
    if a.cmd in ("build", "inventory"):
        mixes = build(Path(a.state), Path(a.repo), targets)
        E = inventory(Path(a.state), mixes)
        g = gpuday_dir()
        (g / "DATA_INVENTORY.json").write_text(json.dumps({"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                           "datasets": E, "mixes": mixes}, indent=1, default=str), encoding="utf-8")
        (g / "DATA_INVENTORY.md").write_text(inventory_md(E, mixes), encoding="utf-8")
        print(f"inventory: {len(E)} datasets -> {g / 'DATA_INVENTORY.md'}")
        return 0
    js = jobs({}, targets=targets)
    from creator import gpupulse as GP
    for j in js:
        GP.ext_job(j)
    if a.cmd == "jobs":
        out = Path(a.out or mix_dir() / "jobs_train.json")
        out.write_text(json.dumps(js, indent=1), encoding="utf-8")
        print(f"{len(js)} jobs -> {out} ({out.stat().st_size // 1024} KB)")
    tot = sum(float(j["minutes"]) for j in js)
    for j in js:
        print(f"  {j['name']:<28} {j['minutes']:>6} min  {j.get('model') or ''}")
    print(f"total ~{round(tot, 1)} GPU-runner minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
