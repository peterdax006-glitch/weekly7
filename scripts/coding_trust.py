"""CODING TRUST GATE runner (creator.codetrust): what may Nupen change on its own, measured on held-out tasks from its own later history.

  python scripts/lowprio.py --idle python scripts/coding_trust.py run --url http://127.0.0.1:18400 [--model NAME] [--limit 5] [--classes bugfix ...]
  python scripts/coding_trust.py run --tunnel Qwen3-14B-Q4_K_M.gguf          the 14B through ~/creator_runtime/gpu/tunnel.json
  python scripts/coding_trust.py run --agent-cmd "aider --yes --message-file {task_file}" --url ... --name aider-27b
  python scripts/coding_trust.py run --stub reference|null|server            dry runs (server = a local OpenAI-compatible stub that
                                                                             answers with the reference change: the HTTP path end to end)
  python scripts/coding_trust.py tasks                                       the held-out set: counts per class, the exclusion rule
  python scripts/coding_trust.py freeze --n-recent 300                       (teacher only) freeze the held-out set ONCE
  python scripts/coding_trust.py report RESULTS.jsonl                        gate report (json + markdown) from attempt records
  python scripts/coding_trust.py safeloop                                    does the kernel already enforce the safe loop?
  python scripts/coding_trust.py estimate [--gen-tps 40 --prefill-tps 2500 --parallel 4]   GPU minutes for a full run

Everything runs in sandboxes of a SEPARATE clone (default ~/creator_runtime/codetrust/repo, made from ~/weekly7 on first use): the main
repository's refs and files are never touched and nothing is merged. Output: ~/creator_runtime/codetrust/<setup>/ (results-*.jsonl,
report.json, report.md). The process drops itself to IDLE priority."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import codetrust as CT  # noqa: E402

RUNTIME = Path.home() / "creator_runtime" / "codetrust"


def _tasks(a: argparse.Namespace) -> list[CT.Task]:
    h = CT.load_heldout(Path(a.heldout))
    ts = [CT.Task.from_dict(t) for t in h["tasks"]]
    if not a.no_derived:
        ts = ts + CT.derive_tests_tasks(ts)
    if a.classes:
        ts = [t for t in ts if t.cls in a.classes]
    if a.task_ids:
        ts = [t for t in ts if t.id in a.task_ids or t.sha[:12] in a.task_ids]
    if a.spread and a.limit:                               # one of each class first, then the rest (dry runs see every check)
        first = []
        for c in CT.CLASSES:
            first += [t for t in ts if t.cls == c][:1]
        ts = first + [t for t in ts if t not in first]
    if not a.spread:
        ts = sorted(ts, key=lambda t: (t.ts, t.id))                   # classes interleaved in time order (futility needs early samples)
    return ts[: a.limit] if a.limit else ts


def tunnel_url(model: str, path: Path = Path.home() / "creator_runtime" / "gpu" / "tunnel.json") -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    ports = (d.get("models") or {}).get(model) or []
    if not ports:
        raise SystemExit(f"{model} is not in {path}: {sorted(d.get('models') or {})}")
    return f"http://127.0.0.1:{ports[0]}"


def _write_report(out: Path, recs: list[dict[str, Any]], setup: dict[str, Any], g: CT.Gate) -> dict[str, Any]:
    rep = CT.report(recs, g, setup)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "report.md").write_text(CT.markdown(rep), encoding="utf-8")
    return rep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "tasks", "freeze", "report", "safeloop", "estimate"))
    ap.add_argument("results", nargs="?", default="")
    ap.add_argument("--heldout", default=str(CT.HELDOUT_FILE))
    ap.add_argument("--url", default="")
    ap.add_argument("--tunnel", default="", help="model file name in ~/creator_runtime/gpu/tunnel.json (e.g. Qwen3-14B-Q4_K_M.gguf)")
    ap.add_argument("--model", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--agent-cmd", default="")
    ap.add_argument("--agent-timeout", type=float, default=1800.0)
    ap.add_argument("--stub", choices=("", "reference", "null", "server"), default="")
    ap.add_argument("--name", default="", help="setup name (output directory and results file)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spread", action="store_true", help="with --limit: one task of each class first")
    ap.add_argument("--classes", nargs="*", default=[])
    ap.add_argument("--task-ids", nargs="*", default=[])
    ap.add_argument("--no-derived", action="store_true", help="leave out the derived tests-only tasks")
    ap.add_argument("--repo", default=str(RUNTIME / "repo"))
    ap.add_argument("--source", default=str(Path.home() / "weekly7"))
    ap.add_argument("--out", default="")
    ap.add_argument("--suite", nargs="*", default=None, help="protected suite override (default: codetrust.PROTECTED_SUITE)")
    ap.add_argument("--test-timeout", type=float, default=1800.0)
    ap.add_argument("--check-host", choices=("local", "pod"), default="local",
                    help="where the sandbox tests run: this PC (xdist on free cores) or the GPU pod's CPUs over ssh (public code only)")
    ap.add_argument("--workers", type=int, default=0, help="local xdist workers (0 = free cores, max 8; 1 = serial)")
    ap.add_argument("--pod-cores", type=int, default=4)
    ap.add_argument("--pod-ssh", default="", help="ssh command for the pod, e.g. 'ssh -i <key> -p <port> <user>@<host>' (never commit a real one)")
    ap.add_argument("--futility", type=int, default=10,
                    help="stop attempting a class once it has this many scored tasks and its Wilson UPPER bound is below the gate (0 = off)")
    ap.add_argument("--tasks-parallel", type=int, default=1, help="tasks attempted at the same time")
    ap.add_argument("--affected-only", action="store_true", help="skip the full protected suite on passing attempts (dry runs only)")
    ap.add_argument("--n-recent", type=int, default=300)
    ap.add_argument("--min-n", type=int, default=CT.MIN_N)
    ap.add_argument("--min-lower", type=float, default=CT.MIN_LOWER)
    ap.add_argument("--gen-tps", type=float, default=40.0)
    ap.add_argument("--prefill-tps", type=float, default=2500.0)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args(argv)
    g = CT.Gate(min_n=a.min_n, min_lower=a.min_lower)
    if a.cmd == "freeze":
        man = CT.freeze(Path(a.source), a.n_recent, path=Path(a.heldout))
        print(json.dumps({k: man[k] for k in ("cut_utc", "head", "by_class", "skipped", "rule")}, indent=1))
        return 0
    if a.cmd == "tasks":
        h = CT.load_heldout(Path(a.heldout))
        ts = _tasks(a)
        print(json.dumps({**{k: h[k] for k in ("cut_utc", "head", "skipped", "rule")}, "frozen_by_class": h["by_class"],
                          "with_derived_by_class": {c: sum(1 for t in ts if t.cls == c) for c in CT.CLASSES},
                          "derived_tests_only": sum(1 for t in ts if t.start_overlay)}, indent=1))
        return 0
    if a.cmd == "safeloop":
        print(json.dumps(CT.safe_loop_check(), indent=1))
        return 0
    if a.cmd == "estimate":
        ts = _tasks(a)
        src = Path(a.source)
        chars = {t.id: sum(len(CT._git(src, "show", f"{t.base}:{p}")[: CT.MAX_FILE_CHARS]) for p in CT.context_files(t)) + len(t.task)
                 for t in ts}
        print(json.dumps(CT.estimate_gpu_minutes(ts, chars, gen_tps=a.gen_tps, prefill_tps=a.prefill_tps, parallel=a.parallel), indent=1))
        return 0
    if a.cmd == "report":
        recs = CT._jsonl(Path(a.results))
        rep = CT.report(recs, g, (recs[0].get("setup") if recs else {}) or {})
        print(CT.markdown(rep))
        return 0
    from creator import gpuday as GD
    GD._idle_priority()
    repo = CT.eval_clone(Path(a.repo).expanduser(), Path(a.source).expanduser())
    tasks = _tasks(a)
    checker: Any = CT.LocalChecker(workers=a.workers)
    if a.check_host == "pod":
        ssh: tuple[str, ...] = CT.POD_SSH
        if a.pod_ssh:
            parts = CT.split_command(a.pod_ssh)
            ssh = (parts[0], "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                   *[str(Path(x).expanduser()) if x.startswith("~") else x for x in parts[1:]])
        checker = CT.RemoteChecker(ssh=ssh, cores=a.pod_cores)
        print(checker.setup())
    agent: CT.Agent
    url = tunnel_url(a.tunnel) if a.tunnel else a.url
    if a.stub == "server":
        url_stub, _srv = CT.stub_server({t.task: CT.reference_edits(repo, t) for t in tasks})
        agent = CT.HttpAgent(url_stub, model="stub-reference")
        agent.name = "http-stub-reference"
    elif a.stub:
        agent = CT.StubAgent(a.stub, 0.9 if a.stub == "reference" else 0.1)
    elif a.agent_cmd:
        agent = CT.CommandAgent(a.agent_cmd, url=url, model=a.model or a.tunnel, timeout=a.agent_timeout, name=a.name)
    elif url:
        agent = CT.HttpAgent(url, model=a.model or a.tunnel, think=a.think)
    else:
        raise SystemExit("give --url, --tunnel, --agent-cmd or --stub")
    if a.name:
        agent.name = a.name
    out = Path(a.out).expanduser() if a.out else RUNTIME / CT.slug(agent.name)
    run = CT.Runner(repo, out, suite=tuple(a.suite) if a.suite is not None else CT.PROTECTED_SUITE, test_timeout=a.test_timeout,
                    full_suite=not a.affected_only, cache=RUNTIME / "validation" / checker.name, checker=checker)
    run.gate_lower = a.min_lower
    run.run(tasks, agent, parallel=a.tasks_parallel, futility=a.futility)
    recs = CT._jsonl(out / f"results-{CT.slug(agent.name)}.jsonl")
    rep = _write_report(out, recs, agent.describe(), g)
    print(CT.markdown(rep))
    print(f"results: {out}")
    try:
        from creator import notify as N
        N.queue_finished("coding trust run", len(recs))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
