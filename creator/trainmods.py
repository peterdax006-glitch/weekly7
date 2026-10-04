"""TRAINING MODULES (h61, owner 4 Oct 2026: "a bank of training modules ready so the GPU always has compute-heavy useful work"). On demand only.

Each module is a creator.trainmix Target (module=<name>) with its own data (sources here), its own held-out eval rows (heldout_<ROLE>.jsonl,
scored by creator.pipelinemix) and its own adoption rule (paired, n >= 50, 95% CI of the gain > 0). Order = value:

  calib_17b       honest CONFIDENCE. Rows: a role prompt + a PROPOSED answer (REVIEW / VALIDATE / CODE rows of every model, verified or not) ->
                  'CONFIDENCE: p' with p = the empirical pass rate of that kind of answer (role x model x claim x length, measured on training
                  tasks only). Eval: CALIB rows of held-out tasks, Brier gain vs the base model (ECE reported; target <= 0.10).
                  Trust-gate (codetrust) results are held-out tasks: never training rows (counted in the inventory).
  brevity_17b     shortest-correct. SFT on the SHORTEST verified answer per task x role (any model) + DPO short-verified (chosen) vs
                  long-verified (rejected, >= 1.5x the tokens). Eval: held-out role rows; ADOPT when output tokens drop (CI > 0) at an equal pass
                  rate (accuracy CI lower bound >= -0.05).
  promptbake_17b  role instructions baked in: trained with the 1-line role tag; eval = the BASE model with the long TEACHER_BRIEF role text vs
                  the tuned model with the short tag on the same held-out rows (accuracy paired; prefill tokens saved reported).
  draft_06b       speculative-decoding draft: Qwen3-0.6B distilled on the pipeline answers (role rows: what pipeline_17b is trained to emit).
                  Eval ON THE POD: llama-server -m <1.7B target> -md <draft>, held-out role prompts, accepted / drafted tokens per prompt for the
                  base 0.6B draft vs the tuned draft (paired; tokens/s reported).
  locate_17b      LOCATE from real public commits (pipelinemix.src_locate: task -> changed files); eval = file-set exact match + precision/recall.
  aider_17b       Aider-format coder: aider_distill rows (+ verified CODE role rows); eval = held-out tasks' Aider prompts, edits applied to the task
                  stub, tests run (pass rate).
  judge_06b       0.6B judgment adapter: public-commit judgment cases (creator.judgment pub topics), TIME-ORDERED (training cases resolved
                  before the first effladder judgment eval case was created; eval subjects never trained); eval = effladder judgment items.
  voice_17b       (creator.trainmix, voice-only): talk rows, eval creator.talkeval."""
from __future__ import annotations

import bisect
import collections
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

LONG_RATIO = 1.5                   # brevity DPO: the rejected verified answer has at least this many times the chosen one's characters
CALIB_ROLES = ("REVIEW", "VALIDATE", "CODE")
CALIB_ASK = ("\n\nProposed answer:\n{answer}\n\nHow likely is this proposed answer to be confirmed when it is actually checked (tests run, "
             "verdict compared with the real outcome)? Reply with one line 'CONFIDENCE: <0.00-1.00>'.")
SPEC_PROMPTS = 60                  # held-out prompts for the speculative-decoding eval
DRAFT_PORT = 18990


def _pm() -> Any:
    from creator import pipelinemix as PM
    return PM


def _tm() -> Any:
    from creator import trainmix as TM
    return TM


def _len_bin(text: str) -> int:
    n = len(text)
    return 0 if n < 400 else 1 if n < 1500 else 2


def _claim(role: str, out: str) -> str:
    PM = _pm()
    if role == "REVIEW":
        m = PM.VERDICT_RE.search(out)
        return m.group(1).lower() if m else "?"
    if role == "VALIDATE":
        m = PM.MEETS_RE.search(out)
        return m.group(1).lower() if m else "?"
    return "edit" if "<<<<<<< SEARCH" in out else "other"


def _outcome(r: Mapping[str, Any]) -> Optional[int]:
    v = r.get("verified")
    return int(v) if isinstance(v, bool) else None


def calib_key(r: Mapping[str, Any]) -> tuple[str, str, str, int]:
    role = _pm().role_of(r)
    return (role, str(r.get("model")), _claim(role, str(r.get("output") or "")), _len_bin(str(r.get("output") or "")))


def calib_rates(raw: Sequence[Mapping[str, Any]], prior: float = 2.0) -> dict[tuple[str, str, str, int], float]:
    """Empirical pass rate per (role, model, claim, length bin) on TRAINING tasks only, shrunk toward the role's rate (prior pseudo-counts)."""
    PM = _pm()
    by: dict[Any, list[int]] = collections.defaultdict(list)
    role_all: dict[str, list[int]] = collections.defaultdict(list)
    for r in raw:
        y = _outcome(r)
        if y is None or PM.split(str(r["task_id"])) == "heldout" or PM.role_of(r) not in CALIB_ROLES:
            continue
        by[calib_key(r)].append(y)
        role_all[PM.role_of(r)].append(y)
    base = {k: sum(v) / len(v) for k, v in role_all.items()}
    return {k: (sum(v) + prior * base[k[0]]) / (len(v) + prior) for k, v in by.items()}


def calib_messages(r: Mapping[str, Any]) -> list[dict[str, str]]:
    PM = _pm()
    role = PM.role_of(r)
    return [{"role": "system", "content": "[ROLE: CALIB] You are Nupen's self-check: estimate honestly how likely a proposed answer is right."},
            {"role": "user", "content": f"ROLE OF THE ANSWER: {role}\n" + str(r["input"]) + CALIB_ASK.format(answer=str(r.get("output") or ""))}]


def src_calib(ctx: Any, drops: collections.Counter[str], raw: Optional[Sequence[Mapping[str, Any]]] = None) -> list[Any]:
    PM, TM = _pm(), _tm()
    raw = PM.role_rows_raw() if raw is None else raw
    rates = calib_rates(raw)
    out: list[Any] = []
    for i, r in enumerate(raw):
        if PM.role_of(r) not in CALIB_ROLES or _outcome(r) is None or not str(r.get("task_id", "")).startswith(("hf:", "pub:")):
            continue
        k = calib_key(r)
        if k not in rates or PM.split(str(r["task_id"])) == "heldout":         # held-out tasks are the eval side only
            continue
        msgs = calib_messages(r) + [{"role": "assistant", "content": f"CONFIDENCE: {rates[k]:.2f}"}]
        out.append(TM.Row(f"calib:{r['task_id']}:{i}", "calib", str(r["task_id"]), {"messages": msgs}, licence="public task",
                          meta={"role": "CALIB_TRAIN", "task": r["task_id"]}, own=False))
    return out


def calib_heldout(raw: Sequence[Mapping[str, Any]], n: int = 400) -> list[dict[str, Any]]:
    PM = _pm()
    rows = []
    for i, r in enumerate(raw):
        if PM.role_of(r) in CALIB_ROLES and _outcome(r) is not None and PM.split(str(r["task_id"])) == "heldout":
            msgs = calib_messages(r) + [{"role": "assistant", "content": ""}]
            rows.append({"id": f"CALIB:{r['task_id']}:{i}", "messages": msgs, "meta": {"y": _outcome(r), "task": r["task_id"], "model": r.get("model")}})
    return sorted(rows, key=lambda x: PM._h("cal" + str(x["id"])))[:n]


def src_brevity(ctx: Any, drops: collections.Counter[str], raw: Optional[Sequence[Mapping[str, Any]]] = None, pref: bool = False) -> list[Any]:
    """Per (task, role): the shortest VERIFIED answer (SFT); with pref, (shortest verified, longest verified) when long >= LONG_RATIO x short."""
    PM, TM = _pm(), _tm()
    by: dict[tuple[str, str], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for r in (PM.role_rows_raw() if raw is None else raw):
        if r.get("verified") is True and str(r.get("task_id", "")).startswith(("hf:", "pub:")):
            by[(str(r["task_id"]), PM.role_of(r))].append(r)
    out: list[Any] = []
    for (tid, role), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda x: (len(str(x["output"])), PM._rank(x)))
        short, long_ = rs[0], rs[-1]
        sysm = PM.tag(role)
        if not pref:
            out.append(TM.Row(f"brief:{role}:{tid}", "brevity_sft", tid, {"messages": [{"role": "system", "content": sysm},
                       {"role": "user", "content": str(short["input"])}, {"role": "assistant", "content": str(short["output"]).strip()}]},
                       licence="public task", meta={"role": role, "task": tid}, own=False))
        elif len(str(long_["output"])) >= LONG_RATIO * max(1, len(str(short["output"]))) and str(long_["input"]) == str(short["input"]):
            out.append(TM.Row(f"briefpref:{role}:{tid}", "brevity_pref", tid,
                              {"prompt": [{"role": "system", "content": sysm}, {"role": "user", "content": str(short["input"])}],
                               "chosen": [{"role": "assistant", "content": str(short["output"]).strip()}],
                               "rejected": [{"role": "assistant", "content": str(long_["output"]).strip()}]},
                              licence="public task", kind="pref", meta={"role": role, "task": tid}, own=False))
    return out


def judge_cases(state: Path, repo: Path) -> tuple[list[Any], dict[str, float], set[str]]:
    """(training cases, per-topic time cut, eval subjects): public cases resolved BEFORE the first effladder judgment eval case was created."""
    from creator import judgment as J
    TM = _tm()
    try:
        items = json.loads(TM.ladder_items_path().read_text(encoding="utf-8"))["items"]
    except (OSError, ValueError, KeyError):
        items = []
    ev = [it for it in items if it.get("suite") == "judgment" and it.get("split") == "eval"]
    ev_subj = {str(it["id"]).split(":", 2)[2] for it in ev}
    train: list[Any] = []
    cuts: dict[str, float] = {}
    for topic in J.PUB_TOPICS:
        cases = J.load_cases(topic, state, repo)
        evc = [c for c in cases if c.subject in ev_subj]
        cut = min((c.created for c in evc), default=math.inf)
        cuts[topic] = cut
        train += [c for c in cases if c.resolved < cut and c.subject not in ev_subj]
    return train, cuts, ev_subj


def src_judge(ctx: Any, drops: collections.Counter[str]) -> list[Any]:
    """'PROBABILITY' targets from the past only: half the case kind's event rate in the resolved history before the case, half its outcome
    (clipped to [0.05, 0.95]); the prompt is creator.judgment's own (retrieval@64 strategy: hint + shots), which never shows the outcome."""
    from creator import judgment as J
    TM = _tm()
    train, _cuts, _ev = judge_cases(Path(ctx.state), Path(ctx.repo))
    by_topic: dict[str, list[Any]] = collections.defaultdict(list)
    for c in train:
        by_topic[c.topic].append(c)
    out: list[Any] = []
    strat = {"shots": 5, "hint": 1}
    for topic, cases in by_topic.items():
        cases.sort(key=lambda c: c.created)
        resolved = sorted(c.resolved for c in cases)
        for c in cases:
            if bisect.bisect_left(resolved, c.created) < 20:            # too little history before it: the record says nothing yet
                drops["judge: under 20 resolved before"] += 1
                continue
            hist = [x for x in J.history(cases, c) if x.group == c.group] or J.history(cases, c)
            rate = sum(x.y for x in hist) / max(1, len(hist))
            p = min(0.95, max(0.05, 0.5 * rate + 0.5 * c.y))
            msgs = J.build_prompt(strat, cases, c) + [{"role": "assistant", "content": f"Kind '{c.group}' had event rate {rate:.2f} in "
                                                                                       f"{len(hist)} past cases.\nPROBABILITY: {p:.2f}"}]
            out.append(TM.Row(f"judge:{topic}:{c.subject}", "judge", f"{topic}:{c.subject}", {"messages": msgs}, ts=float(c.created),
                              licence="public repository history", meta={"topic": topic}, own=False))
    return out


# ------------------------------------------------------------------------------------------------ build hooks (trainmix.build_target)
def prepare(t: Any, d: Path, sft: list[Any], pref: list[Any], drops: collections.Counter[str]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Per module: drop held-out tasks from training, write heldout_<ROLE>.jsonl; returns (sft, pref, info)."""
    PM = _pm()
    for f in Path(d).glob("heldout_*.jsonl"):
        f.unlink()
    m = t.module
    info: dict[str, Any] = {}

    def keep(rows: list[Any]) -> list[Any]:
        k = [r for r in rows if PM.split(r.group) != "heldout"]
        drops[f"{m}: held-out task (eval side)"] += len(rows) - len(k)
        return k

    def write(role: str, rows: Sequence[Mapping[str, Any]], n: int = 200) -> None:
        rows = sorted(rows, key=lambda x: PM._h("ev" + str(x["id"])))[:n]
        info.setdefault("heldout_rows", {})[role] = len(rows)
        with (Path(d) / f"heldout_{role}.jsonl").open("w", encoding="utf-8") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    if m in ("calib", "brevity", "promptbake", "draft", "aider"):
        sft, pref = keep(sft), keep(pref)
        raw = PM.role_rows_raw()
        held = {str(x["task_id"]) for x in raw if PM.split(str(x["task_id"])) == "heldout"}
        if m == "calib":
            write("CALIB", calib_heldout(raw), 400)
        elif m == "aider":
            ev = PM.heldout_eval_rows(raw, held, PM.rl_tests())
            write("CODE", aider_heldout(held, PM.rl_tests()) or ev.get("CODE", []))
        else:
            for role, rows in PM.heldout_eval_rows(raw, held, PM.rl_tests()).items():
                write(role, rows)
        if m == "draft":
            prompts = [{"id": x["id"], "messages": x["messages"][:-1]} for role, rows in PM.heldout_eval_rows(raw, held, PM.rl_tests()).items()
                       for x in rows]
            prompts = sorted(prompts, key=lambda x: PM._h("spec" + x["id"]))[:SPEC_PROMPTS]
            (Path(d) / "spec_prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8")
            info["spec_prompts"] = len(prompts)
        info["heldout_tasks"] = len(held)
    elif m == "locate":
        held_rows = [r for r in sft if PM.split(r.group) == "heldout"]
        sft = keep(sft)
        write("LOCATE", [{"id": r.id, "messages": r.body["messages"], "meta": {"changed": r.meta.get("changed"), "repo": r.meta.get("repo")}}
                         for r in held_rows])
    elif m == "judge":
        info["note"] = "eval = effladder judgment items (time-ordered: training cases resolved before the first eval case was created)"
    return sft, pref, info


def aider_heldout(held: set[str], tests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Held-out Aider rows (first turn of a held-out task): the edit is applied to the task's stub file and its tests are run."""
    PM, TM = _pm(), _tm()
    out = []
    for i, r in enumerate(TM.jsonl(PM.aider_dir() / "sft.jsonl")):
        tid = PM._aider_task(r, i)
        msgs = r.get("messages")
        if tid not in held or tid not in tests or not isinstance(msgs, list) or str(r.get("turn", "1")) != "1":
            continue
        name = str(tests[tid]["name"])
        shown = [mm.group(1) for m_ in msgs if m_.get("role") == "user"
                 for mm in re.finditer(r"solution\.py\s*\n```[a-z]*\n(.*?)```", str(m_.get("content") or ""), re.S)]
        current = shown[-1] if shown else f"def {name}(*args, **kwargs):\n    raise NotImplementedError\n"   # the file as Aider showed it
        if msgs and msgs[-1].get("role") == "user":           # Aider log row: prompt messages + 'reply' (the reference, never shown)
            msgs = list(msgs) + [{"role": "assistant", "content": str(r.get("reply") or "")}]
        out.append({"id": f"AIDER:{tid}:{i}", "messages": PM.tagged("CODE", msgs[:-1]) + [msgs[-1]],
                    "meta": {"check": "py", "test": tests[tid], "current": current,
                             "task": tid}})
    return out


# ------------------------------------------------------------------------------------------------ eval job specs (trainmix.jobs)
def eval_jobs(t: Any, root: Path, sv: str) -> list[dict[str, Any]]:
    m = t.module
    d = str(Path(root) / t.name)
    if m == "voice":                                 # trainmix adds the talk evals for voice-only targets
        return []
    if m == "judge":
        return [{"name": f"eval_{t.name}", "call": "creator.trainmix:eval_job", "model": sv, "minutes": t.eval_minutes,
                 "max_minutes": round(t.eval_minutes * 2.5, 1), "low_util_abort_minutes": 0,
                 "args": {"target": t.name, "suites": ["judgment"], "configs": {"judgment": "plain@64,retrieval@64"}, "base": t.base_gguf,
                          "run": f"train-{t.name}"}}]
    if m == "draft":
        return draft_eval_jobs(t, sv)
    mode = {"calib": "calibration", "brevity": "brevity"}.get(m, "accuracy")
    out = []
    for model, role in ((t.base_gguf, "base"), (sv, "tuned")):
        args: dict[str, Any] = {"target": t.name, "role": role, "dir": d, "mode": mode}
        if m == "promptbake" and role == "base":
            args["long_system"] = True
        out.append({"name": f"roleeval_{t.name}_{role}", "call": "creator.pipelinemix:pipeline_eval_job", "model": model,
                    "minutes": t.eval_minutes, "max_minutes": round(t.eval_minutes * 2.5, 1), "low_util_abort_minutes": 0, "args": args})
    return out


def spec_remote(target_model: str, draft: str, label: str, prompts: str, port: int = DRAFT_PORT, n_predict: int = 256) -> str:
    """Pod script: its OWN llama-server (target + draft, own port; killed at the end, nothing else touched) answers every prompt; per prompt
    the server's draft_n / draft_n_accepted timings -> @@result with the per-prompt acceptance."""
    py = ("import json,sys,urllib.request\n"
          "rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]\nout=[]\n"
          "for r in rows:\n"
          "    b=json.dumps({'messages':r['messages'],'max_tokens':%d,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode()\n"
          "    try:\n"
          "        q=urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})\n"
          "        d=json.loads(urllib.request.urlopen(q,timeout=300).read())\n"
          "    except Exception as e:\n"
          "        continue\n"
          "    t=d.get('timings') or {}\n"
          "    out.append({'id':r['id'],'draft_n':t.get('draft_n'),'accepted':t.get('draft_n_accepted'),'tps':t.get('predicted_per_second')})\n"
          "print('@@result='+json.dumps({'label':sys.argv[2],'rows':out}))\n") % (n_predict, port)
    return "\n".join([
        "set -e", f"T=models/{target_model}; D=models/{draft}",
        "[ -f \"$T\" ] && [ -f \"$D\" ] || { printf '@@result={\"skipped\": \"missing %s or %s\"}\\n' \"$T\" \"$D\"; exit 0; }",
        "SRV=$(dirname \"$(cat run/exe 2>/dev/null || echo /opt/llama.cpp/llama-server)\")/llama-server; [ -x \"$SRV\" ] || SRV=/opt/llama.cpp/llama-server",
        f"\"$SRV\" -m \"$T\" -md \"$D\" -ngl 99 -ngld 99 --draft-max 16 --draft-min 1 -c 8192 -np 1 --port {port} > /tmp/spec_{label}.log 2>&1 &",
        "PID=$!; trap 'kill $PID 2>/dev/null || true' EXIT",
        f"for i in $(seq 1 120); do curl -s 127.0.0.1:{port}/health | grep -q ok && break; sleep 1; done",
        f"PY=\"$(cat gpuday/python 2>/dev/null || echo /venv/main/bin/python)\"; [ -x \"$PY\" ] || PY=python3",
        f"cat > /tmp/spec_eval.py <<'SPEC_PY'\n{py}SPEC_PY",
        f"\"$PY\" /tmp/spec_eval.py {prompts} {label}", ""])


def draft_eval_jobs(t: Any, sv: str) -> list[dict[str, Any]]:
    """Fetch the base 0.6B (not kept on the pod), measure base draft then tuned draft against the 1.7B target, record, delete the base copy."""
    from creator import effladder as EL
    TM = _tm()
    prompts = f"{TM.POD_DIR}/trainmix/{t.name}/spec_prompts.jsonl"
    target = "Qwen3-1.7B-Q4_K_M.gguf"
    out: list[dict[str, Any]] = [{"name": f"fetch_{t.name}_base", "remote": EL.fetch_script(".", t.base_gguf), "minutes": 0.5, "low_util_abort_minutes": 0}]
    for draft, label in ((t.base_gguf, "base"), (sv, "tuned")):
        out.append({"name": f"spec_{t.name}_{label}", "remote": spec_remote(target, draft, label, prompts), "free_gpu": True, "minutes": 4,
                    "max_minutes": 12, "low_util_abort_minutes": 0})
    out.append({"name": f"specgate_{t.name}", "call": "creator.trainmods:spec_gate_job", "minutes": 0.5, "low_util_abort_minutes": 0,
                "args": {"target": t.name, "base": f"spec_{t.name}_base", "tuned": f"spec_{t.name}_tuned"}})
    out.append({"name": f"delete_{t.name}_base", "remote": EL.delete_script(".", t.base_gguf), "minutes": 0.1, "low_util_abort_minutes": 0})
    return out


def spec_compare(base: Sequence[Mapping[str, Any]], tuned: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from creator import gpuselfteach as GS
    TM = _tm()

    def rate(r: Mapping[str, Any]) -> Optional[float]:
        n = r.get("draft_n")
        return float(r.get("accepted") or 0) / float(n) if n else None
    b = {r["id"]: rate(r) for r in base}
    pairs = [(rate(r), b[r["id"]]) for r in tuned if r["id"] in b and rate(r) is not None and b[r["id"]] is not None]
    d = [float(x) - float(y) for x, y in pairs if x is not None and y is not None]
    m, lo, hi = GS.mean_ci(d)
    tps = lambda rs: round(sum(float(r.get("tps") or 0) for r in rs) / max(1, len(rs)), 1)  # noqa: E731
    return {"n": len(d), "acceptance_gain": round(m, 4),
            "acceptance_gain_ci95": [None if not math.isfinite(lo) else round(lo, 4), None if not math.isfinite(hi) else round(hi, 4)],
            "acceptance_base": round(sum(y for _x, y in pairs if y is not None) / max(1, len(pairs)), 4),
            "acceptance_tuned": round(sum(x for x, _y in pairs if x is not None) / max(1, len(pairs)), 4),
            "tok_s_base_pod": tps(base), "tok_s_tuned_pod": tps(tuned),
            "verdict": "ADOPT" if len(d) >= TM.MIN_N and lo > 0 else "KEEP_BASE",
            "home_note": "CPU speed-up ~ (1 + a*k) / (1 + k*c) per verified step (a = acceptance, k = drafted tokens, c = draft/target cost ~0.35)"}


def spec_gate_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job: reads this pulse's two spec_* runner rows and records the paired acceptance comparison (train_gate.jsonl)."""
    import datetime as dt
    TM = _tm()
    a = dict(ctx.get("args") or {})
    state, pulse = Path(str(ctx["state"])), str(ctx.get("pulse") or "")
    got: dict[str, Any] = {}
    for r in TM.jsonl(state / "thinking" / "gpu_pulse_runs.jsonl"):
        for k in ("base", "tuned"):
            if r.get("job") == f"ext:{a.get(k)}" and (not pulse or str(r.get("gpu_pulse") or "") == pulse) and isinstance(r.get("result"), Mapping):
                got[k] = r["result"]
    if not all(isinstance(got.get(k), Mapping) and got[k].get("rows") for k in ("base", "tuned")):
        return {"verdict": "NOT_RUN", "why": f"missing spec runs: {sorted(got)}"}
    rec = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "pulse": pulse, "target": a.get("target"), "kind": "spec_draft",
           **spec_compare(got["base"]["rows"], got["tuned"]["rows"])}
    p = TM.gate_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


_ = re


# ------------------------------------------------------------------------------------------------ AUTOGEN: keep the module queue filled
QUEUE_MIN = 3


def queue_dir() -> Path:
    return _tm().mix_dir() / "queue"


def _queued_names(q: Path) -> set[str]:
    out = set()
    for sub in ("", "done", "failed"):
        d = q / sub if sub else q
        for f in d.glob("*.json") if d.is_dir() else []:
            out.add(f.stem.split("_", 1)[1] if "_" in f.stem and f.stem.split("_", 1)[0].isdigit() else f.stem)
    return out


def _adopted(state: Path, target: str) -> bool:
    for r in _tm().jsonl(_tm().gate_path(state)):
        if r.get("target") == target and (r.get("verdict") == "ADOPT" or (r.get("compare") or {}).get("adopt_roles")):
            return True
    return False


def gates_ok(t: Any, ctx: Any, state: Path) -> tuple[bool, str]:
    TM = _tm()
    for g in t.requires:
        kind, _, arg = str(g).partition(":")
        if kind == "adopted" and not _adopted(state, arg):
            return False, f"waits for {arg} to be adopted"
        if kind == "rows":
            src, _, n = arg.partition(">=")
            have = len(TM.SOURCES[src](ctx, collections.Counter()))
            if have < int(n or 0):
                return False, f"{src} has {have} rows, needs {n}"
    return True, ""


def weak_spot_specs(state: Path) -> list[dict[str, Any]]:
    """Specs for the weakest roles of the latest tuned pipeline eval (tuned accuracy < 0.5 with n >= 50): a role-focused adapter on every
    verified row of that role (+ Aider rows for CODE/DEBUG). Written to <runtime>/gpuday/trainmix/specs/ (config, not code)."""
    TM = _tm()
    recs = [r for r in TM.jsonl(TM.gate_path(state)) if r.get("kind") == "pipeline_eval" and r.get("role") == "tuned" and r.get("compare")]
    if not recs:
        return []
    out = []
    for role, v in sorted(((recs[-1]["compare"] or {}).get("roles") or {}).items()):
        if v.get("n", 0) >= 50 and v.get("tuned_acc") is not None and float(v["tuned_acc"]) < 0.5:
            out.append({"name": f"role_{role.lower()}_17b", "priority": 15, "size": "1.7b",
                        "sources": ["roles", "aider_sft"] if role in ("CODE", "DEBUG") else ["roles"], "filter": {"role": [role]},
                        "module": "roles", "why": f"weak spot: {role} tuned accuracy {v['tuned_acc']} on held-out tasks",
                        "value": "high: the weakest pipeline step", "max_seq": 4096, "epochs": 2, "batch": 8, "accum": 2, "packing": True,
                        "lora_serve": True, "target_minutes": [45, 60], "min_rows": 200, "eval_minutes": 6, "vram_mib": 20000})
    return out


def pipeline_busy() -> bool:
    """The teacher's rule: no module file while a pipeline training run is still going (its log lacks the exit marker)."""
    f = _tm().runtime() / "gpu" / "train_pipeline.out"
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "pipeline training" in text and "pipeline training exit" not in text


def autogen_once(state: Path, repo: Path, say: Any = print, queue_min: int = QUEUE_MIN, dry: bool = False) -> list[str]:
    """Fill the queue to `queue_min` pending module files: weak-spot specs written first, then every spec by priority whose gates pass and
    that was never queued (pending, done or failed): build -> audit (inside jobs) -> job spec validation -> queue/<priority>_<name>.json."""
    import json as _json
    TM = _tm()
    q = queue_dir()
    q.mkdir(parents=True, exist_ok=True)
    sd = TM.mix_dir() / "specs"
    sd.mkdir(parents=True, exist_ok=True)
    for spec in weak_spot_specs(state):
        f = sd / f"{spec['name']}.json"
        if not f.is_file():
            f.write_text(_json.dumps(spec, indent=1), encoding="utf-8")
            say(f"autogen: new weak-spot spec {spec['name']} ({spec['why']})")
    if pipeline_busy():
        say("autogen: pipeline training still running - no new file")
        return []
    pending = sorted(q.glob("*.json"))
    if len(pending) >= queue_min:
        return []
    import importlib
    TM = importlib.reload(TM)                       # specs written since the last pass are part of TARGETS now
    ctx = TM.Ctx(Path(state), Path(repo), TM.load_heldout())
    seen = _queued_names(q)
    cands = [t for t in TM.TARGETS if t.module and t.name not in seen and not t.blocked]
    written: list[str] = []
    from creator import gpupulse as GP
    for i, t in enumerate(cands):
        if len(pending) + len(written) >= queue_min:
            break
        ok, why = gates_ok(t, ctx, state)
        if not ok:
            say(f"autogen: {t.name} not yet ({why})")
            continue
        man = TM.build(Path(state), Path(repo), [t.name], say=lambda m: None)[t.name]
        if man["rows"]["train"] < t.min_rows:
            say(f"autogen: {t.name} has {man['rows']['train']} training rows < {t.min_rows}: skipped this pass")
            continue
        nxt = next((c for c in cands[i + 1:] if gates_ok(c, ctx, state)[0]), None)
        js = TM.jobs({}, targets=[t.name], cleanup=False, prestage=nxt)
        for j in js:
            GP.ext_job(j)
        out = q / f"{t.priority:02d}_{t.name}.json"
        if not dry:
            tmp = out.with_suffix(".tmp")
            tmp.write_text(_json.dumps(js, indent=1), encoding="utf-8")
            tmp.replace(out)
        m = TM.minutes(t, man)
        say(f"autogen: queued {out.name}: train {man['rows']['train']} rows, ~{m['train']} min train + {m['eval']} eval"
            + (f"; prestages {nxt.name}" if nxt else ""))
        written.append(out.name)
    return written


def autogen_loop(state: Path, repo: Path, interval_s: float = 300.0, say: Any = print) -> None:
    import time
    from creator import gpuday as GD
    GD._idle_priority()
    while True:
        try:
            autogen_once(state, repo, say)
        except Exception as e:                       # noqa: BLE001 - one failed pass is logged; the loop goes on
            say(f"autogen: pass failed: {type(e).__name__}: {str(e)[:300]}")
        time.sleep(interval_s)
