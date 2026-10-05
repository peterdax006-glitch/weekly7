"""PIPELINE ADAPTER DATA (h61, owner goal 4 Oct 2026: small home models run the FULL coding pipeline on CPU). On demand only.

One multi-role LoRA adapter (Qwen3-1.7B and Qwen3-4B variants, creator.trainmix targets pipeline_17b / pipeline_4b) trained on ROLE-TAGGED
rows: the role is named in the system prompt ("[ROLE: LOCATE] ..."), so one adapter serves every step of the pipeline at home.

  roles     SPEC, LOCATE, PLAN, CODE, DEBUG, REVIEW, VALIDATE, LESSON
  sources   C2 trajectories (27B runs): any <runtime>/gpuday/trajectories/**/*.jsonl row with a 'role' (top level or meta) - expected later;
            LOCATE now: real commits -> (task = the commit subject, the parent tree's file list) -> the changed files. Public clones under
            <runtime>/public_repos (no cut: not Nupen's history) and Nupen's own commits BEFORE the trust-gate cut (never a held-out commit).
  balance   at most ROLE_CAP rows per role (a deterministic hash sample), so one plentiful role cannot drown the others
  splits    per role: held-out (~10 %, by group) for the per-role eval, dev (~5 %) for early stopping, the rest train - never the same commit /
            task on two sides; trainmix.screen removes eval/held-out overlap and private/frozen rows as for every mix
  eval      pipeline_eval_job (gpupulse 'call'): the served model answers each role's held-out rows; scored per role -
              LOCATE      exact match of the file SET (and recall)
              REVIEW / VALIDATE   the label (approve/reject, pass/fail) against the test-derived label in the row (meta.label / meta.passed)
              CODE / DEBUG        pass rate by executing the row's tests when it carries them (effladder item format: meta.check 'py' +
                                  meta.test); rows without tests are reported as needing the repository harness (creator.gpuday), not scored
              SPEC / PLAN / LESSON  no automatic metric: dev loss only (reported, never adopted on)
            base and tuned runs of the same pulse are paired per role: ADOPT only when n >= 50 and the 95% CI of the gain > 0."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROLES = ("SPEC", "LOCATE", "PLAN", "CODE", "DEBUG", "REVIEW", "VALIDATE", "LESSON")
EXTRA_ROLES = ("CALIB",)           # eval-only pseudo-role: the CONFIDENCE of a proposed answer (creator.trainmods calib)
ROLE_CAP = 1500
PER_REPO = 30                      # LOCATE rows per public repository (spread over the whole history)
MAX_CHANGED = 6
MAX_CANDIDATES = 80
ROLE_SYSTEM = {
    "SPEC": "Turn the request into a short, testable specification: inputs, outputs, edge cases, acceptance tests.",
    "LOCATE": "Given a task and the project's file list, name the files that must change. Reply with one path per line and nothing else.",
    "PLAN": "Write a short numbered plan of the code changes for the task, file by file.",
    "CODE": "Write the code change for the task exactly in the requested format.",
    "DEBUG": "A test failed. Find the cause from the output and write the fix in the requested format.",
    "REVIEW": "Review the candidate. First line 'VERDICT: correct' or 'VERDICT: incorrect', then ISSUES and CONFIDENCE.",
    "VALIDATE": "Decide whether the final solution meets the spec. 'MEETS SPEC: yes|no', then GAPS and BROKE ANYTHING ELSE.",
    "LESSON": "State the reusable lesson of this run in one or two sentences.",
}
MAX_TOKENS = {"LOCATE": 256, "REVIEW": 512, "VALIDATE": 384, "CODE": 2048, "DEBUG": 2048, "SPEC": 512, "PLAN": 512, "LESSON": 128}


def tag(role: str) -> str:
    return f"[ROLE: {role}] You are Nupen's {role.lower()} step. {ROLE_SYSTEM[role]}"


def tagged(role: str, msgs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The role tag at the start of the system prompt (a system message is added when the row has none)."""
    out = [dict(m) for m in msgs]
    if out and out[0].get("role") == "system":
        if not str(out[0].get("content", "")).startswith("[ROLE:"):
            out[0]["content"] = tag(role) + "\n" + str(out[0].get("content") or "")
    else:
        out.insert(0, {"role": "system", "content": tag(role)})
    return out


def role_of(r: Mapping[str, Any]) -> str:
    v = str(r.get("role") or (r.get("meta") or {}).get("role") or "").upper()
    return v if v in ROLES else ""


# ------------------------------------------------------------------------------------------------ LOCATE from real commits
def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return r.stdout if r.returncode == 0 else ""


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def candidates(tree: Sequence[str], changed: Sequence[str], key: str, n: int = MAX_CANDIDATES) -> list[str]:
    """The changed files plus the files nearest to them (same directory, then same top directory, then a hash sample), sorted: the model
    must pick, not copy. Deterministic."""
    ch = set(changed)
    dirs = {p.rsplit("/", 1)[0] if "/" in p else "" for p in ch}
    tops = {p.split("/", 1)[0] for p in ch}
    rest = [p for p in tree if p not in ch]
    near = sorted(rest, key=lambda p: (0 if (p.rsplit("/", 1)[0] if "/" in p else "") in dirs else 1 if p.split("/", 1)[0] in tops else 2,
                                       _h(key + p)))
    return sorted(ch | set(near[: max(0, n - len(ch))]))


def locate_row(repo_name: str, subject: str, changed: Sequence[str], tree: Sequence[str], key: str) -> dict[str, Any]:
    files = candidates(tree, changed, key)
    user = f"Repository '{repo_name}'. Task: {subject}\n\nFiles:\n" + "\n".join(files)
    return {"messages": [{"role": "system", "content": tag("LOCATE")}, {"role": "user", "content": user},
                         {"role": "assistant", "content": "\n".join(sorted(changed))}]}


def src_locate(ctx: Any, drops: collections.Counter[str], per_repo: int = PER_REPO) -> list[Any]:
    from creator import gpuday as GD
    from creator import reasondrills as R
    from creator import trainmix as TM
    out: list[Any] = []
    for name, path in R.repos(ctx.repo):
        try:
            hist = R.history(path)
        except Exception:                                   # noqa: BLE001 - a broken clone is skipped
            drops["locate: unreadable clone"] += 1
            continue
        own = name == "nupen"
        pool = [c for c in hist if 1 <= len({p for _s, p in c.changes}) <= MAX_CHANGED and R._ok_subject(c.s)
                and all(st in "MD" for st, _p in c.changes)]           # modified files only: a new file cannot be picked from a list
        if own:
            pool = [c for c in pool if not TM.excluded(float(c.t), c.h, ctx.heldout)]
        pool.sort(key=lambda c: _h(name + c.h))
        got = 0
        for c in pool:
            if got >= per_repo:
                break
            changed = sorted({p for st, p in c.changes if st == "M"})           # files to change (deleted files are not 'located')
            if not changed:
                continue
            tree = [p for p in _git(path, "ls-tree", "-r", "--name-only", f"{c.h}^").splitlines() if p]
            if not tree or not set(changed) <= set(tree):
                drops["locate: no parent tree"] += 1
                continue
            subj = GD.scrub(c.s)
            body = locate_row(name, subj, changed, tree, c.h)
            out.append(TM.Row(f"locate:{name}:{c.h[:12]}", "locate", f"{name}:{c.h[:12]}", body, ts=float(c.t), sha=c.h[:12],
                              licence="public repository history" if not own else "own (before the cut)",
                              meta={"role": "LOCATE", "repo": name, "changed": changed}, own=own))
            got += 1
    return out


MODEL_RANK = ("Qwen3.6-27B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf")      # prefer the strongest teacher's row


def role_files() -> list[Path]:
    from creator import trainmix as TM
    d = TM.gpuday_dir() / "trajectories" / "roles"
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def role_rows_raw() -> list[dict[str, Any]]:
    from creator import trainmix as TM
    out: list[dict[str, Any]] = []
    for p in role_files():
        out += [r for r in TM.jsonl(p) if role_of(r) and r.get("task_id") and r.get("input") and r.get("output") is not None]
    return out


def _rank(r: Mapping[str, Any]) -> int:
    m = str(r.get("model") or "")
    return MODEL_RANK.index(m) if m in MODEL_RANK else len(MODEL_RANK)


def src_roles(ctx: Any, drops: collections.Counter[str], raw: Optional[Sequence[Mapping[str, Any]]] = None) -> list[Any]:
    """C2 role rows (<runtime>/gpuday/trajectories/roles/<model>.jsonl, TEACHER_BRIEF formats): VERIFIED rows only; per (task, role) the row
    of the strongest model (27B, then 4B, then 1.7B). Public tasks only (hf:/pub: ids); a held-out task's rows never reach training (split by
    task in trainmix). The role tag goes into the system prompt."""
    from creator import trainmix as TM
    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in (role_rows_raw() if raw is None else raw):
        if r.get("verified") is not True:
            drops["roles: not verified"] += 1
            continue
        tid = str(r["task_id"])
        if not tid.startswith(("hf:", "pub:")):
            drops["roles: not a public task"] += 1
            continue
        k = (tid, role_of(r))
        if k not in best or _rank(r) < _rank(best[k]):
            best[k] = r
    out: list[Any] = []
    for (tid, role), r in sorted(best.items()):
        msgs = [{"role": "system", "content": tag(role)}, {"role": "user", "content": str(r["input"])},
                {"role": "assistant", "content": str(r["output"]).strip()}]
        out.append(TM.Row(f"role:{role}:{tid}", f"role_{role.lower()}", tid, {"messages": msgs}, licence="public task; answer by " + str(r.get("model")),
                          meta={"role": role, "task": tid, "model": r.get("model")}, own=False))
    return out


AIDER_WEIGHT = 2                   # Aider is the home default editor: its exact-prompt rows count twice in training (CODE/DEBUG)


def aider_dir() -> Path:
    from creator import trainmix as TM
    return TM.gpuday_dir() / "aider_distill"


def _aider_task(r: Mapping[str, Any], i: int) -> str:
    m = r.get("meta") or {}
    return str(r.get("task_id") or m.get("task_id") or m.get("task") or r.get("task") or f"aider:{i}")


def src_aider(ctx: Any, drops: collections.Counter[str], kind: str = "sft") -> list[Any]:
    """Aider distill (<runtime>/gpuday/aider_distill/{sft,pref}.jsonl, written while the 27B runs): exact Aider prompts -> 27B diff replies
    from PASSED public tasks. Whatever exists at build time is used (rebuild before a run). SFT rows keep their role (DEBUG when tagged so,
    else CODE); a row that is not marked passed/verified (when the field exists) is dropped; only public task ids (hf:/pub:)."""
    from creator import trainmix as TM
    out: list[Any] = []
    for i, r in enumerate(TM.jsonl(aider_dir() / f"{kind}.jsonl")):
        m = dict(r.get("meta") or {})
        ok = [r.get(k, m.get(k)) for k in ("passed", "verified") if k in r or k in m]
        if ok and not all(v is True or str(v).lower() == "true" for v in ok):
            drops["aider: not passed"] += 1
            continue
        tid = _aider_task(r, i)
        if not tid.startswith(("hf:", "pub:")):
            drops["aider: not a public task id"] += 1
            continue
        if kind == "sft":
            if not isinstance(r.get("messages"), list) or r.get("kind", "edit") != "edit":
                continue
            if r.get("reply") is not None and r["messages"] and r["messages"][-1].get("role") == "user":
                r = dict(r, messages=list(r["messages"]) + [{"role": "assistant", "content": str(r["reply"])}])   # Aider log: prompt + reply
            role = role_of(r) if role_of(r) in ("CODE", "DEBUG") else "CODE"
            out.append(TM.Row(f"aider:{role}:{tid}:{i}", "aider_sft", tid, {"messages": tagged(role, r["messages"])},
                              licence="public task; 27B reply through Aider's prompts", meta={"role": role, "task": tid, "aider": True},
                              own=False))
        elif (r.get("prompt") or r.get("messages")) and r.get("chosen") and r.get("rejected"):
            def turn(v: Any) -> list[dict[str, Any]]:
                return v if isinstance(v, list) else [{"role": "assistant", "content": str(v)}]
            out.append(TM.Row(f"aiderpref:{tid}:{i}", "aider_pref", tid,
                              {"prompt": tagged("CODE", r.get("prompt") or r["messages"]), "chosen": turn(r["chosen"]), "rejected": turn(r["rejected"])},
                              licence="public task", kind="pref", meta={"task": tid}, own=False))
    return out


def rl_tests() -> dict[str, dict[str, Any]]:
    """task id -> {'name', 'cases'} from the export's RL task files (the tests the C2 runs were checked with)."""
    from creator import trainmix as TM
    out: dict[str, dict[str, Any]] = {}
    for d in TM.export_dirs().values():
        for n in ("rl_tasks_hf", "rl_tasks_more"):
            for r in TM.jsonl(d / f"{n}.jsonl"):
                if r.get("id") and r.get("tests"):
                    out.setdefault(str(r["id"]), {"name": r["name"], "cases": r["tests"]})
    return out


SOL_RE = re.compile(r"(?:Current|Final|Candidate) solution\.py:\s*```(?:python)?\n(.*?)```", re.S)


def heldout_eval_rows(raw: Sequence[Mapping[str, Any]], heldout_tasks: set[str], tests: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per role, the held-out tasks' rows that can be SCORED (any model's prompt; the label/tests come from the record, not the answer):
    CODE/DEBUG with the task's tests and the starting solution.py; REVIEW with real_pass; VALIDATE with truth. One row per (task, role, prompt)."""
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for r in sorted(raw, key=_rank):
        role, tid = role_of(r), str(r.get("task_id") or "")
        if tid not in heldout_tasks:
            continue
        k = (tid, role, hashlib.sha256(str(r["input"]).encode("utf-8")).hexdigest()[:16])
        if k in seen:
            continue
        meta: dict[str, Any] = {"task": tid, "prompt_by": r.get("model")}
        if role in ("CODE", "DEBUG"):
            m = SOL_RE.search(str(r["input"]))
            if tid not in tests or m is None:
                continue
            meta.update(check="py", test=tests[tid], current=m.group(1))
        elif role == "REVIEW" and isinstance(r.get("real_pass"), bool):
            meta["label"] = r["real_pass"]
        elif role == "VALIDATE" and isinstance(r.get("truth"), bool):
            meta["label"] = r["truth"]
        else:
            continue
        seen.add(k)
        out[role].append({"id": f"{role}:{tid}:{k[2][:8]}", "messages": [{"role": "system", "content": tag(role)},
                                                                       {"role": "user", "content": str(r["input"])},
                                                                       {"role": "assistant", "content": str(r["output"])}], "meta": meta})
    return dict(out)


EDIT_RE = re.compile(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)


def apply_edits(current: str, reply: str) -> Optional[str]:
    """SEARCH/REPLACE blocks applied to the starting solution.py; a ```python block alone is taken as the whole file; None when a SEARCH
    text is not found (the edit would fail at home too)."""
    blocks = EDIT_RE.findall(reply)
    if not blocks:
        m = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
        return m.group(1) if m else None
    text = current
    for old, new in blocks:
        if old not in text:
            return None
        text = text.replace(old, new, 1)
    return text


def src_c2_roles(ctx: Any, drops: collections.Counter[str]) -> list[Any]:
    """Every role-tagged C2 row (SFT format); the trust-gate exclusion as in trainmix.src_trajectories."""
    from creator import trainmix as TM
    held_tasks = {str(t.get("id")) for t in ctx.heldout.get("tasks") or []}
    d = TM.gpuday_dir() / "trajectories"
    out: list[Any] = []
    for p in sorted(d.rglob("*.jsonl")) if d.is_dir() else []:
        if p.name.endswith(".ids.jsonl"):
            continue
        for i, r in enumerate(TM.jsonl(p)):
            role = role_of(r)
            if not role or not r.get("messages"):
                continue
            m = dict(r.get("meta") or {})
            task = str(m.get("task") or m.get("task_id") or r.get("task") or "")
            sha = str(m.get("sha") or r.get("sha") or "")
            if task in held_tasks or TM.excluded(float(m.get("ts") or 0.0), sha, ctx.heldout):
                drops["exclusion rule: trust-gate held-out task / commit"] += 1
                continue
            keep = {k: m[k] for k in ("label", "passed", "check", "test", "task", "agent") if k in m}
            out.append(TM.Row(f"c2:{role}:{p.stem}:{i}", f"c2_{role.lower()}", task or f"{p.stem}:{i}", {"messages": tagged(role, r["messages"])},
                              sha=sha, licence="own runs (27B)", meta={"role": role, **keep}, own=bool(sha)))
    return out


# ------------------------------------------------------------------------------------------------ balance + per-role splits
def balance(rows: Sequence[Any], cap: int = ROLE_CAP) -> tuple[list[Any], dict[str, int]]:
    by: dict[str, list[Any]] = collections.defaultdict(list)
    for r in rows:
        by[str(r.meta.get("role") or "")].append(r)
    out: list[Any] = []
    for role, rs in sorted(by.items()):
        out += sorted(rs, key=lambda r: _h("bal" + r.id))[:cap]
    return out, {k: min(len(v), cap) for k, v in sorted(by.items())}


def split(group: str) -> str:
    v = _h("split" + group) % 1000
    return "heldout" if v < 100 else "dev" if v < 150 else "train"


# ------------------------------------------------------------------------------------------------ per-role scoring
CONF_RE = re.compile(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?|\.\d+)", re.I)


def brief_sections(path: Optional[Path] = None) -> dict[str, str]:
    """ROLE -> its long instruction text from TEACHER_BRIEF.md ('### ROLE (...)' up to the next heading): what promptbake replaces."""
    from creator import trainmix as TM
    try:
        text = Path(path or TM.gpuday_dir() / "TEACHER_BRIEF.md").read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    intro = text.split("\n### ", 1)[0].strip()       # the brief's shared part (vision, efficiency, honesty) goes with every role, as the teacher got it
    for m in re.finditer(r"^### ([A-Z]+)\b[^\n]*\n(.*?)(?=^#{2,3} |\Z)", text, re.S | re.M):
        if m.group(1) in ROLES:
            out[m.group(1)] = intro + "\n\n### " + m.group(1) + "\n" + m.group(2).strip()
    return out


VERDICT_RE = re.compile(r"VERDICT\s*:\s*\**\s*(correct|incorrect)", re.I)
MEETS_RE = re.compile(r"MEETS SPEC\s*:\s*\**\s*(yes|no)", re.I)
POS = {"APPROVE", "ACCEPT", "LGTM", "PASS", "PASSED", "OK"}
NEG = {"REJECT", "REQUEST_CHANGES", "FAIL", "FAILED"}
LABEL_RE = re.compile(r"\b(APPROVE|ACCEPT|LGTM|REJECT|REQUEST_CHANGES|PASS(?:ED)?|FAIL(?:ED)?)\b")


def label_of(text: str) -> Optional[bool]:
    found = LABEL_RE.findall(text.upper())
    if not found:
        return None
    return found[-1] in POS                                    # the LAST verdict word decides


def paths_of(text: str) -> set[str]:
    out = set()
    for ln in re.sub(r"<think>.*?(?:</think>|\Z)", "", text, flags=re.S).splitlines():
        s = ln.strip().strip("`*-• ").strip()
        if s and " " not in s and ("/" in s or "." in s):
            out.add(s)
    return out


def score(role: str, meta: Mapping[str, Any], reference: str, reply: str) -> Optional[dict[str, Any]]:
    """{'correct': bool, ...} or None when the role has no automatic metric for this row."""
    if role == "LOCATE":
        want = set(meta.get("changed") or paths_of(reference))
        got = paths_of(reply)
        return {"correct": bool(want) and got == want, "recall": round(len(got & want) / max(1, len(want)), 3),
                "precision": round(len(got & want) / len(got), 3) if got else 0.0}
    if role == "CALIB" and "y" in meta:
        mc = CONF_RE.search(re.sub(r"<think>.*?(?:</think>|\Z)", "", reply, flags=re.S))
        p = min(1.0, max(0.0, float(mc.group(1)))) if mc else 0.5            # no number = the uninformative 0.5
        y = int(bool(meta["y"]))
        return {"correct": (p >= 0.5) == bool(y), "p": p, "y": y, "brier": round((p - y) ** 2, 4), "parsed": mc is not None}
    if role in ("REVIEW", "VALIDATE") and isinstance(meta.get("label"), bool):
        m = (VERDICT_RE if role == "REVIEW" else MEETS_RE).search(re.sub(r"<think>.*?(?:</think>|\Z)", "", reply, flags=re.S))
        verdict = None if m is None else m.group(1).lower() in ("correct", "yes")
        return {"correct": verdict is not None and verdict == meta["label"], "parsed": verdict is not None}
    if role in ("CODE", "DEBUG") and meta.get("check") == "py" and meta.get("test") and "current" in meta:
        from creator import effladder as EL
        code = apply_edits(str(meta["current"]), reply)
        if code is None:
            return {"correct": False, "applied": False}
        return dict(EL.score({"check": "py", "test": meta["test"]}, f"```python\n{code}\n```"), applied=True)
    if role in ("REVIEW", "VALIDATE"):
        lab = meta.get("label")
        truth = (lab if isinstance(lab, bool) else label_of(str(lab))) if lab is not None else (bool(meta["passed"]) if "passed" in meta else label_of(reference))
        if truth is None:
            return None
        return {"correct": label_of(reply) == truth}
    if role in ("CODE", "DEBUG") and meta.get("check") == "py" and meta.get("test"):
        from creator import effladder as EL
        return EL.score({"check": "py", "test": meta["test"]}, reply)
    return None


def heldout_rows(d: Path) -> list[dict[str, Any]]:
    from creator import trainmix as TM
    out: list[dict[str, Any]] = []
    for role in ROLES + EXTRA_ROLES:
        out += [dict(r, role=role) for r in TM.jsonl(Path(d) / f"heldout_{role}.jsonl")]
    return out


POD_EXEC_WORKERS = 8               # test programs run on the POD's CPU (public tasks only): <= 8 at a time, nice 10


def program_for(role: str, meta: Mapping[str, Any], reply: str) -> Optional[str]:
    """The test program of a CODE / DEBUG answer (edits applied to the task's starting file), or None when there is nothing to run."""
    if role not in ("CODE", "DEBUG") or meta.get("check") != "py" or not meta.get("test") or "current" not in meta:
        return None
    code = apply_edits(str(meta["current"]), reply)
    if code is None:
        return None
    from creator import effladder as EL
    return EL.program(meta["test"], code)


POD_RUNNER = r"""import json, subprocess, sys, tempfile, os, base64
from concurrent.futures import ThreadPoolExecutor
progs = json.loads(base64.b64decode(sys.stdin.readline()).decode())
RUN = %r
def one(item):
    k, prog = item
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run(["nice", "-n", "10", sys.executable, "-I", "-c", RUN], input=prog.encode(), capture_output=True, timeout=10, cwd=td)
        except subprocess.TimeoutExpired:
            return k, False
    return k, p.stdout.decode("utf-8", "replace").strip().startswith("PASS")
with ThreadPoolExecutor(%d) as ex:
    res = dict(ex.map(one, progs.items()))
print("@@result=" + json.dumps(res))
"""


def pod_exec(progs: Mapping[str, str], shell: Any = None, cfg: Optional[Mapping[str, Any]] = None) -> dict[str, bool]:
    """Run the test programs on the pod's CPU (one ssh call; <= POD_EXEC_WORKERS at once under nice 10) -> {id: passed}. The programs are
    public-task code + tests only (the caller filters); the private-marker guard runs before anything leaves the PC."""
    import base64
    from creator import effladder as EL
    from creator import gpupulse as GP
    if not progs:
        return {}
    GP.outbound_ok([{"content": v} for v in progs.values()])
    sh = shell or GP.shell_for(cfg or GP.load_config())
    blob = base64.b64encode(json.dumps(dict(progs)).encode("utf-8")).decode("ascii")
    script = ("PY=\"$(cat gpuday/python 2>/dev/null || true)\"; [ -n \"$PY\" ] && [ -x \"$PY\" ] || PY=$(command -v python3 || command -v python)\n"
              f"cat > /tmp/nupen_pod_exec.py <<'POD_EXEC'\n{POD_RUNNER % (EL.RUNNER, POD_EXEC_WORKERS)}POD_EXEC\n"
              f"echo '{blob}' | \"$PY\" /tmp/nupen_pod_exec.py\n")
    rc, out = sh.run(script, timeout=1800, check=False)
    for ln in out.splitlines():
        if ln.startswith("@@result="):
            return {str(k): bool(v) for k, v in json.loads(ln[len("@@result="):]).items()}
    raise RuntimeError(f"pod exec gave no result (rc {rc}): {out[-300:]}")


def run_rows(ask: Any, rows: Sequence[Mapping[str, Any]], workers: int = 8, code_exec: Any = None) -> list[dict[str, Any]]:
    """Ask every held-out row (prompt = its messages without the reference answer); score it per role. `code_exec` ({id: program} ->
    {id: passed}, e.g. pod_exec): CODE / DEBUG programs of PUBLIC tasks run there in one batch after all answers are in (the busy PC only
    applies the edits); anything it cannot answer is scored here as before."""
    import concurrent.futures as cf
    if code_exec is not None:
        progs: dict[str, str] = {}
        raw: list[tuple[Mapping[str, Any], Any]] = []

        def ask_only(r: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
            try:
                return r, ask(list(r["messages"])[:-1], MAX_TOKENS.get(str(r["role"]), 512))
            except Exception as e:                        # noqa: BLE001
                return r, e
        with cf.ThreadPoolExecutor(max(1, workers)) as ex:
            raw = list(ex.map(ask_only, rows))
        for r, got in raw:
            if isinstance(got, Exception):
                continue
            reply = str(got.get("content") or "") if isinstance(got, Mapping) else str(got)
            pg = program_for(str(r["role"]), r.get("meta") or {}, reply)
            if pg is not None and str((r.get("meta") or {}).get("task", "")).startswith(("hf:", "pub:")):
                progs[str(r["id"])] = pg
        try:
            passed = code_exec(progs) if progs else {}
        except Exception:                                 # noqa: BLE001 - the pod is unreachable: score on this PC
            passed = {}
        out: list[dict[str, Any]] = []
        for r, got in raw:
            if isinstance(got, Exception):
                out.append({"id": r["id"], "role": r["role"], "error": f"{type(got).__name__}: {str(got)[:120]}"})
                continue
            reply = str(got.get("content") or "") if isinstance(got, Mapping) else str(got)
            cost = {k: got[k] for k in ("tok_in", "tok_out") if k in got} if isinstance(got, Mapping) else {}
            if str(r["id"]) in passed:
                sc: Optional[dict[str, Any]] = {"correct": passed[str(r["id"])], "applied": True, "exec": "pod"}
            else:
                sc = score(str(r["role"]), r.get("meta") or {}, str(r["messages"][-1].get("content") or ""), reply)
            out.append({"id": r["id"], "role": r["role"], "scored": sc is not None, **cost, **(sc or {})})
        return out

    def one(r: Mapping[str, Any]) -> dict[str, Any]:
        msgs = list(r["messages"])
        ref = str(msgs[-1].get("content") or "")
        try:
            got = ask(msgs[:-1], MAX_TOKENS.get(str(r["role"]), 512))
        except Exception as e:                                # noqa: BLE001 - a failed call is a missing pair, never a guess
            return {"id": r["id"], "role": r["role"], "error": f"{type(e).__name__}: {str(e)[:120]}"}
        reply = str(got.get("content") or "") if isinstance(got, Mapping) else str(got)
        cost = {k: got[k] for k in ("tok_in", "tok_out") if k in got} if isinstance(got, Mapping) else {}
        sc = score(str(r["role"]), r.get("meta") or {}, ref, reply)
        return {"id": r["id"], "role": r["role"], "scored": sc is not None, **cost, **(sc or {})}
    with cf.ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, rows))


def _ci(xs: Sequence[float]) -> tuple[float, list[Optional[float]]]:
    from creator import gpuselfteach as GS
    m, lo, hi = GS.mean_ci(xs)
    return round(m, 4), [None if not math.isfinite(lo) else round(lo, 4), None if not math.isfinite(hi) else round(hi, 4)]


def ece(ps: Sequence[float], ys: Sequence[int], bins: int = 10) -> Optional[float]:
    if not ps:
        return None
    tot = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(ps) if (b / bins <= p < (b + 1) / bins) or (b == bins - 1 and p == 1.0)]
        if idx:
            tot += len(idx) * abs(sum(ps[i] for i in idx) / len(idx) - sum(ys[i] for i in idx) / len(idx))
    return round(tot / len(ps), 4)


def compare(base: Sequence[Mapping[str, Any]], tuned: Sequence[Mapping[str, Any]], mode: str = "accuracy") -> dict[str, Any]:
    """Paired per role. mode 'accuracy': gain in correct (ADOPT: n >= 50, CI > 0). 'calibration': Brier improvement (base - tuned; ADOPT:
    n >= 50, CI > 0), ECE reported (target <= 0.10). 'brevity': output tokens saved per row (ADOPT: n >= 50, CI of the saving > 0 AND the
    accuracy change's CI lower bound >= -0.05: equal pass rate within noise)."""
    from creator import trainmix as TM
    b = {r["id"]: r for r in base if r.get("scored")}
    if mode in ("calibration", "brevity"):
        per2: dict[str, Any] = {}
        for role in ROLES + EXTRA_ROLES:
            pairs = [(r, b[r["id"]]) for r in tuned if r.get("scored") and r["role"] == role and r["id"] in b]
            if not pairs:
                continue
            acc, acc_ci = _ci([float(int(bool(t["correct"])) - int(bool(x["correct"]))) for t, x in pairs])
            e: dict[str, Any] = {"n": len(pairs), "acc_gain": acc, "acc_gain_ci95": acc_ci}
            if mode == "calibration":
                g, ci = _ci([float(x.get("brier", 0.25)) - float(t.get("brier", 0.25)) for t, x in pairs])
                e.update(brier_gain=g, brier_gain_ci95=ci,
                         ece_tuned=ece([float(t.get("p", 0.5)) for t, _x in pairs], [int(t.get("y", 0)) for t, _x in pairs]),
                         ece_base=ece([float(x.get("p", 0.5)) for _t, x in pairs], [int(x.get("y", 0)) for _t, x in pairs]))
                ok = len(pairs) >= TM.MIN_N and (ci[0] or 0) > 0
            else:
                g, ci = _ci([float(x.get("tok_out", 0)) - float(t.get("tok_out", 0)) for t, x in pairs])
                e.update(tokens_saved=g, tokens_saved_ci95=ci,
                         tok_out_tuned=round(sum(float(t.get("tok_out", 0)) for t, _x in pairs) / len(pairs), 1),
                         tok_out_base=round(sum(float(x.get("tok_out", 0)) for _t, x in pairs) / len(pairs), 1))
                ok = len(pairs) >= TM.MIN_N and (ci[0] or 0) > 0 and (acc_ci[0] if acc_ci[0] is not None else -1) >= -0.05
            e["verdict"] = "ADOPT" if ok else "KEEP_BASE"
            per2[role] = e
        return {"mode": mode, "roles": per2, "adopt_roles": sorted(k for k, v in per2.items() if v["verdict"] == "ADOPT")}
    from creator import gpuselfteach as GS
    per: dict[str, Any] = {}
    for role in ROLES + EXTRA_ROLES:
        pairs = [(r, b[r["id"]]) for r in tuned if r.get("scored") and r["role"] == role and r["id"] in b]
        unscored = sum(1 for r in tuned if r["role"] == role and not r.get("scored"))
        if not pairs and not unscored:
            continue
        d = [float(int(bool(t["correct"])) - int(bool(x["correct"]))) for t, x in pairs]
        m, lo, hi = GS.mean_ci(d)
        per[role] = {"n": len(d), "unscored": unscored, "gain": round(m, 4),
                     "tuned_acc": round(sum(bool(t["correct"]) for t, _x in pairs) / len(pairs), 4) if pairs else None,
                     "base_acc": round(sum(bool(x["correct"]) for _t, x in pairs) / len(pairs), 4) if pairs else None,
                     "gain_ci95": [None if not math.isfinite(lo) else round(lo, 4), None if not math.isfinite(hi) else round(hi, 4)],
                     "verdict": "ADOPT" if len(d) >= TM.MIN_N and lo > 0 else ("NO_METRIC" if not pairs else "KEEP_BASE")}
    return {"roles": per, "adopt_roles": sorted(k for k, v in per.items() if v["verdict"] == "ADOPT")}


def pipeline_eval_job(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """gpupulse 'call' job: the served model (args role 'base' or 'tuned') answers the target's held-out rows of every role. Each run is one
    record in <state>/thinking/train_gate.jsonl; the tuned run carries the per-role paired comparison with this pulse's base run."""
    from creator import effladder as EL
    from creator import gpupulse as GP
    from creator import trainmix as TM
    a = dict(ctx.get("args") or {})
    model, which, target = str(ctx.get("model") or ""), str(a.get("role") or "tuned"), str(a.get("target") or "")
    got = GP.attach(Path(str(ctx["tunnel_file"])), model)
    if got is None:
        return {"verdict": "NOT_RUN", "why": f"the pulse does not serve {model}"}
    ep = EL.Endpoint(got[0], model)
    rows = heldout_rows(Path(str(a.get("dir") or TM.mix_dir() / target)))
    if a.get("long_system"):                         # promptbake: the base model gets the LONG role instructions (TEACHER_BRIEF section)
        brief = brief_sections()
        rows = [dict(r, messages=[{"role": "system", "content": brief.get(str(r["role"]), tag(str(r["role"])))}] + list(r["messages"][1:]))
                for r in rows]
    pod = (lambda progs: pod_exec(progs)) if a.get("exec", "pod") == "pod" else None      # test programs on the pod's CPU, not this busy PC
    res = run_rows(lambda msgs, n: ep.call(msgs, n, False), rows, int(ctx.get("workers") or 8), code_exec=pod)
    state, pulse = Path(str(ctx["state"])), str(ctx.get("pulse") or "")
    rec: dict[str, Any] = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "pulse": pulse, "target": target,
                           "kind": "pipeline_eval", "role": which, "model": model, "n": len(res), "rows": res}
    if which == "tuned":
        base = [r for r in TM.jsonl(TM.gate_path(state)) if r.get("kind") == "pipeline_eval" and r.get("role") == "base"
                and r.get("pulse") == pulse and r.get("target") == target]
        rec["compare"] = (compare(base[-1]["rows"], res, str(a.get("mode") or "accuracy")) if base else
                          {"verdict": "NOT_RUN", "why": "no base run in this pulse"})
        if base:
            tin = [float(r.get("tok_in", 0)) for r in res if "tok_in" in r]
            bin_ = [float(r.get("tok_in", 0)) for r in base[-1]["rows"] if "tok_in" in r]
            rec["compare"]["prefill_tokens_mean"] = {"base": round(sum(bin_) / max(1, len(bin_)), 1), "tuned": round(sum(tin) / max(1, len(tin)), 1)}
    p = TM.gate_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return {k: v for k, v in rec.items() if k != "rows"}


def export_skills(state: Path, cmp: Mapping[str, Any], targets: Optional[Mapping[str, float]] = None, default_target: float = 0.8,
                  uses_per_day: Optional[Mapping[str, float]] = None, fail_cost: Optional[Mapping[str, float]] = None) -> int:
    """Write per-role accuracies of a `compare()` result (tuned_acc, else base_acc) to metrics/skills.jsonl for the weak_skill detector.
    uses_per_day / fail_cost (CPU-seconds a failed use costs) come from the metrics bus; absent -> 0, which values the candidate at 0."""
    rows = []
    for role, v in (cmp.get("roles") or {}).items():
        acc = v.get("tuned_acc") if v.get("tuned_acc") is not None else v.get("base_acc")
        if acc is None:
            continue
        rows.append({"role": role, "score": acc, "target": (targets or {}).get(role, default_target), "n": v.get("n"),
                     "uses_per_day": (uses_per_day or {}).get(role, 0.0), "fail_cost": (fail_cost or {}).get(role, 0.0)})
    d = Path(state) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "skills.jsonl").open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + chr(10))
    return len(rows)
