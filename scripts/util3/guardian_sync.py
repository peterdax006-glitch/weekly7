"""h64 GPU guardian, PC side: stage questions on the pod, pull the pod's answers back into the trace bank.

The guardian (scripts/util3/guardian/*.sh + feed.py) lives entirely on the pod: it scales filler llama-servers into idle GPU and its feeder answers
staged questions there, so nothing depends on this PC, SSH or an agent while it runs. This script is the slow, PC-side half:
  stage  - every PUBLIC-repository reasoning question (reasondrills.generate_stream; frozen items already dropped; Nupen's own repository is
           never staged: its history stays on this PC) as {qid, kind, source, t, messages, answer} -> pod /root/guardian/q/questions.jsonl.gz.
           The answer travels so the pod can pick what to retry; the prompt never contains it (outcome-blind; checked here).
  pull   - fetch the new bytes of the pod's answer logs (/root/guardian/out/*.jsonl, byte offsets kept here) and bank every answer that,
           re-checked against THIS PC's own question, is correct and an outcome-blind trace (reasondrills.trace_ok) and whose qid is not banked
           yet (reasondrills.bank_add; the row records the model and gpu_pulse 'guardian-<model>').
  loop   - pull every --every seconds, re-stage when the questions change (new repositories) - run it at idle priority.
    python scripts/util3/guardian_sync.py stage|pull|loop --host H --sshport P [--relay host:port]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RT = Path.home() / ("creator_runtime/util3/guardian")


def ssh_base(a, route) -> list[str]:
    return ["ssh", "-i", a.key, "-p", route[1], "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", f"root@{route[0]}"]


def routes(a) -> list[tuple[str, str]]:
    return [(a.host, a.sshport)] + ([tuple(a.relay.split(":"))] if a.relay else [])


def run_ssh(a, cmd: str, stdin: bytes | None = None, timeout: float = 600) -> bytes:
    last = None
    for r in routes(a):
        try:
            p = subprocess.run(ssh_base(a, r) + [cmd], input=stdin, capture_output=True, timeout=timeout)
            if p.returncode == 0:
                return p.stdout
            last = p.stderr.decode(errors="replace")[-300:]
        except subprocess.TimeoutExpired as e:
            last = repr(e)
    raise RuntimeError(f"ssh failed on every route: {last}")


def questions(main: Path):
    from creator import reasondrills as R
    state = main / "state" / "creator"
    out = []
    for part in R.generate_stream(R.repos(main), state):
        out += [q for q in R.order(part) if R.REVISIT not in q.qid and q.source != "nupen"]
    return out


def stage(a, main: Path) -> int:
    from creator import gpupulse as GP
    from creator import generator as G
    from creator import reasondrills as R
    st = R.Strategy("trace", "cot", max_tokens=R.TRACE_TOKENS)
    qs = questions(main)
    have = set(R.trace_bank(main / "state" / "creator"))
    asked = {f.name[len("asked_"):-len(".txt")]: set(f.read_text(encoding="utf-8").split()) for f in RT.parent.glob("asked_*.txt")}
    RT.mkdir(parents=True, exist_ok=True)
    tmp = RT / "questions.jsonl.gz"
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for q in qs:
            msgs = G.prepare_messages(R.build_messages(st, q), "qwen3")
            if any("Correct answer" in m["content"] for m in msgs):
                continue
            GP.outbound_ok(msgs)                                         # private-marker guard, as for every prompt that leaves the PC
            f.write(json.dumps({"qid": q.qid, "kind": q.kind, "source": q.source, "t": q.t, "messages": msgs, "answer": q.answer,
                                "banked": q.qid in have, "asked_by": sorted(m for m, ids in asked.items() if q.qid in ids)}) + "\n")
            n += 1
    run_ssh(a, "mkdir -p /root/guardian/q && cat > /root/guardian/q/questions.jsonl.gz.part && mv -f /root/guardian/q/questions.jsonl.gz.part "
               "/root/guardian/q/questions.jsonl.gz", stdin=tmp.read_bytes(), timeout=1800)
    (RT / "staged.json").write_text(json.dumps({"n": n, "at": time.time()}), encoding="utf-8")
    print(f"staged {n} questions ({tmp.stat().st_size // 1024} KiB)", flush=True)
    return n


def pull(a, main: Path, qmap: dict) -> dict:
    from creator import reasondrills as R
    state = main / "state" / "creator"
    offs_p = RT / "offsets.json"
    offs = json.loads(offs_p.read_text()) if offs_p.exists() else {}
    names = run_ssh(a, "ls /root/guardian/out/ 2>/dev/null | grep 'jsonl$' || true").decode().split()
    have = set(R.trace_bank(state))
    stats = {"rows": 0, "banked": 0, "rejected": 0}
    for name in names:
        off = int(offs.get(name, 0))
        data = run_ssh(a, f"tail -c +{off + 1} /root/guardian/out/{name} | head -c 50000000", timeout=900)
        cut = data.rfind(b"\n") + 1                                      # only whole lines; the rest comes next time
        lines = data[:cut].decode("utf-8", errors="replace").splitlines()
        model = name[:-len(".jsonl")]
        asked = []
        for ln in lines:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            stats["rows"] += 1
            q = qmap.get(r.get("qid"))
            if q is None:
                stats["rejected"] += 1
                continue
            asked.append(q.qid)
            reply = str(r.get("reply") or "")
            if q.qid in have or R.parse_choice(reply) != q.answer or not R.trace_ok(reply):
                continue
            R.bank_add(state, q, reply, str(r.get("model") or model), f"guardian-{model}")
            have.add(q.qid)
            stats["banked"] += 1
        if asked:
            with (RT.parent / f"asked_{model}.txt").open("a", encoding="utf-8") as f:
                f.write("".join(x + "\n" for x in asked))
        offs[name] = off + cut
        offs_p.write_text(json.dumps(offs), encoding="utf-8")
    return stats


CODING_SOURCES = ("rl_tasks_hf.jsonl", "rl_tasks_more.jsonl")            # PUBLIC only (OpenCodeInstruct + mined public repos); never rl_tasks.jsonl
EXPORT = Path.home() / ("creator_runtime/gpuday/export_plus")
BRIEF = Path.home() / ("creator_runtime/gpuday/TEACHER_BRIEF.md")
ROLES = Path.home() / ("creator_runtime/gpuday/trajectories/roles")


def stage_coding(a) -> int:
    """Public coding tasks (train split only) + the teacher brief -> the pod's coder.py. Every prompt passes the private-marker guard."""
    from creator import gpupulse as GP
    import ast
    rows = []
    for name in CODING_SOURCES:
        for ln in (EXPORT / name).read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            if r.get("split") != "train" or not str(r.get("id", "")).startswith(("hf:", "pub:")):
                continue
            GP.outbound_ok([{"content": r["prompt"]}, {"content": str(r["tests"])}])
            ast.literal_eval(r["tests"]) if isinstance(r["tests"], str) else None
            rows.append({"id": r["id"], "name": r["name"], "prompt": r["prompt"],
                         "tests": r["tests"] if isinstance(r["tests"], str) else repr(r["tests"])})
    brief = BRIEF.read_text(encoding="utf-8")
    GP.outbound_ok([{"content": brief}])
    RT.mkdir(parents=True, exist_ok=True)
    tmp = RT / "coding_tasks.jsonl.gz"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    run_ssh(a, "mkdir -p /root/guardian/q && cat > /root/guardian/q/coding_tasks.jsonl.gz", stdin=tmp.read_bytes())
    run_ssh(a, "cat > /root/guardian/q/TEACHER_BRIEF.md", stdin=brief.encode("utf-8"))
    print(f"staged {len(rows)} public coding tasks + the teacher brief", flush=True)
    return len(rows)


def pull_roles(a) -> dict:
    """New bytes of the pod's role rows (out_roles/<model>.jsonl) -> trajectories/roles/<model>.jsonl here (whole lines only)."""
    offs_p = RT / "offsets_roles.json"
    offs = json.loads(offs_p.read_text()) if offs_p.exists() else {}
    names = run_ssh(a, "ls /root/guardian/out_roles/ 2>/dev/null | grep 'jsonl$' || true").decode().split()
    ROLES.mkdir(parents=True, exist_ok=True)
    got = 0
    for name in names:
        off = int(offs.get(name, 0))
        data = run_ssh(a, f"tail -c +{off + 1} /root/guardian/out_roles/{name} | head -c 50000000", timeout=900)
        cut = data.rfind(b"\n") + 1
        if cut:
            with (ROLES / name).open("ab") as f:
                f.write(data[:cut])
            got += data[:cut].count(b"\n")
        offs[name] = off + cut
        offs_p.write_text(json.dumps(offs), encoding="utf-8")
    return {"role_rows": got}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("stage", "stage-coding", "pull", "loop"))
    ap.add_argument("--main", default=str(Path.home() / "weekly7"))
    ap.add_argument("--host", required=True)
    ap.add_argument("--sshport", required=True)
    ap.add_argument("--relay", default="")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "nupen_vast"))
    ap.add_argument("--every", type=float, default=300)
    a = ap.parse_args()
    main_dir = Path(a.main)
    sys.path.insert(0, str(main_dir))
    sys.path.insert(0, str(main_dir / "scripts"))
    os.chdir(main_dir)
    from lowprio import lower_own_priority, IDLE_PRIORITY_CLASS
    lower_own_priority(IDLE_PRIORITY_CLASS)
    RT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "stage-coding":
        stage_coding(a)
        return 0
    if a.cmd == "stage":
        stage(a, main_dir)
        return 0
    qmap = {q.qid: q for q in questions(main_dir)}
    if a.cmd == "pull":
        print(json.dumps(pull(a, main_dir, qmap)))
        return 0
    staged_n = len(qmap)
    last_q = time.monotonic()
    while not (RT / "STOP").exists():
        try:
            print(time.strftime("%H:%M:%S"), json.dumps(dict(pull(a, main_dir, qmap), **pull_roles(a))), flush=True)
            if time.monotonic() - last_q > 3600:                         # new repositories / commits: re-stage when the set grew
                qmap = {q.qid: q for q in questions(main_dir)}
                last_q = time.monotonic()
                if len(qmap) > staged_n + 500:
                    staged_n = stage(a, main_dir)
        except Exception as e:                                           # noqa: BLE001 - a flaky pod link: try again next round
            print(time.strftime("%H:%M:%S"), "error", repr(e)[:300], flush=True)
        time.sleep(a.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
