"""Known-fix table of the GPU operator (MASTER_BLUEPRINT 8.7, PHASE0 P0.6): every real incident of 3-4 Oct 2026 as an entry
{id, signature (regexes over log lines and/or a metric condition on a health record), diagnosis, ranked remedies, verification}.

`match_line` / `match_metrics` recognise an incident, `replay` scans a set of logs and reports each incident with its timestamp,
`remedies_for` returns the ranked action list. Pure and deterministic: nothing here touches the pod; acting on a remedy is the
operator's job (and passes the diff checker and the budget). Nothing host-, key- or path-specific is stored in the table."""
from __future__ import annotations

import dataclasses
import datetime as dt
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

_TS = re.compile(r"^(?:HEALTH\s+)?(\d{2}:\d{2}:\d{2})\b")
_ISO = re.compile(r'"at": "(\d{4}-\d{2}-\d{2}T[\d:]+)')


@dataclasses.dataclass(frozen=True)
class Remedy:
    action: str
    expected_effect: str
    risk: str                               # "low" | "medium" | "high"


@dataclasses.dataclass(frozen=True)
class KnownFix:
    id: str
    kind: str
    signature: tuple[str, ...]              # regexes over single log lines (case-insensitive)
    diagnosis: str
    remedies: tuple[Remedy, ...]            # ranked best first (expected gain, then low risk)
    verification: str                       # what to measure after, and the pass condition
    metric: Optional[Callable[[dict], bool]] = None     # condition on a parsed health record
    in_logs: bool = True                    # False: the incident is documented but not present in the 3-4 Oct logs

    def compiled(self) -> tuple["re.Pattern[str]", ...]:
        return tuple(re.compile(s, re.I) for s in self.signature)


R = Remedy
TABLE: tuple[KnownFix, ...] = (
    KnownFix("vram-starved-trainer", "vram_starvation", (r"skipped\W+VRAM:\s*\d+\s*MiB free after \d+ s,? needs \d+",),
             "A fine-tune could not start: filler models held the VRAM it needs and never released it within the wait window.",
             (R("evict or shrink the filler models (stop the largest filler) until free VRAM >= need", "trainer starts within a minute", "low"),
              R("hold the filler feeder (guardian hold) before the trainer's start step, release after the adapter is saved", "no recurrence", "low"),
              R("pair a smaller job that fits the free VRAM instead of waiting", "GPU stays busy while the trainer waits", "medium")),
             "free VRAM >= job need within 120 s of the remedy; the ft job reports steps and no 'skipped' key"),
    KnownFix("smoke-release-race", "smoke_release_race", (r"mv: cannot stat .*queue/[\w.\-]+\.json", r"FAILED rc=[1-9]\d*\D.*ext:smoke_\w+"),
             "The queue file was moved or released while the smoke job still ran: the runner raced the guardian release and lost its queue entry.",
             (R("release the guardian only after the smoke job's result is written", "no lost queue file", "low"),
              R("re-queue the module from its autogen spec and rerun the smoke", "module runs again", "low"),
              R("skip the smoke when the trainer and base are unchanged", "removes the race window", "medium")),
             "the module reaches DONE; queue holds exactly one file for it"),
    KnownFix("feeder-stall", "feeder_stall", (r"feeder.*(stall|stuck|idle|no (new )?work)",),
             "Filler models are loaded but the GPU is idle: the PC-side feeder stopped sending requests.",
             (R("restart the feeder and confirm requests in flight > 0", "util back above 60% in 2-3 minutes", "low"),
              R("move the eval/feeder client onto the pod", "removes the home link as a bottleneck", "medium"),
              R("raise workers/slots on the feeder", "more requests in flight", "low")),
             "GPU util >= 60% for 3 consecutive samples after the remedy",
             metric=lambda m: m.get("util") is not None and m["util"] <= 5 and (m.get("guard") or 0) >= 1 and m.get("stop") == 0 and (m.get("runner") or 0) > 0),
    KnownFix("transfer-stall", "transfer_stall", (r"closed by remote host", r"sync \S+ (\d+)/(\d+)\s*$"),
             "A relayed file transfer stalled or dropped partway (seen at about 45 MB on the relay); the copy is partial.",
             (R("retry through the direct address with resume (rsync partial)", "completes at line speed", "low"),
              R("split the file into chunks below the stall size and verify each hash", "completes despite the relay", "low"),
              R("move the file by the pod's own uploader to a storage bucket and pull from there", "bypasses the relay", "medium")),
             "remote size == local size and the hash matches; sync_age drops below 30 min"),
    KnownFix("missing-lora", "missing_lora", (r"is not on the pod - refusing to serve it",),
             "The merged model's LoRA file was never copied to the pod (or was deleted), so the eval server refuses to serve it.",
             (R("re-upload the adapter from the PC and re-register the module", "model serves", "low"),
              R("re-run the ft job if the adapter exists nowhere", "adapter rebuilt", "medium"),
              R("skip the tuned eval and score it later at home", "lets the queue move on", "low")),
             "the server for the tuned model starts and the tuned eval returns n results"),
    KnownFix("unseen-kind-vram-reserve", "unseen_kind_vram", (r"unseen[ _-]kind.*(vram|reserve)", r"no vram (reserve|estimate) for kind"),
             "The VRAM planner had no measurement for this job kind and reserved too little (or too much).",
             (R("reserve the largest measured peak of the same family plus 20%", "job fits first time", "low"),
              R("run it alone once, record the peak, then pack", "adds the kind to the efficiency table", "low")),
             "recorded peak_vram_gb exists for the kind; no OOM", in_logs=False),
    KnownFix("template-download-disk", "disk_full", (r"no space left on device", r"\bENOSPC\b", r"disk-low\(\d+G\)"),
             "The pod disk filled (a template or base model auto-downloaded onto it); jobs fail or the pod stalls.",
             (R("delete unused template/base downloads and caches, keep only prestaged bases in use", "frees tens of GB", "low"),
              R("point the download cache at the larger volume", "no recurrence", "medium"),
              R("recreate the instance with a bigger disk", "permanent fix, costs a restart", "high")),
             "free disk >= 10 GB after the remedy and stays so for an hour",
             metric=lambda m: m.get("disk_g") is not None and m["disk_g"] <= 3),
    KnownFix("pkill-self-match", "pkill_self_match", (r"self-?match", r"pkill -f .*pkill", r"killed (its own|the calling) (shell|script|command)"),
             "A process-name match (pkill -f / pgrep -f) matched the command line of the very command running it and killed it.",
             (R("match by PID file or a pattern in brackets ([n]upen) so the command line does not match itself", "no self-kill", "low"),
              R("use the guardian's stop file instead of killing", "clean stop", "low")),
             "the swap/stop script exits 0 and the target process is gone", in_logs=False),
    KnownFix("autogen-without-state", "autogen_no_state", (r"autogen:.*(empty state|state (is )?(empty|missing)|no state)",),
             "autogen was started without --state, read an empty state, and queued nothing or the wrong things.",
             (R("restart autogen with --state pointing at the live state directory", "queue fills from real weak spots", "low"),
              R("make --state required in the launcher", "no recurrence", "low")),
             "the next autogen pass logs 'queued' lines with non-zero train rows", in_logs=False),
    KnownFix("deadman-not-armed", "deadman_unarmed", (r"deadman.*(not armed|unarmed|missing|disarmed|no timer)",),
             "The pod's stop timer was not armed: a crashed controller would leave the pod running and billing.",
             (R("arm the deadman stop timer now (stop in N minutes, re-armed by each healthy pulse)", "bounded spend", "low"),
              R("stop the instance after syncing everything home", "no spend", "medium")),
             "the pod reports a deadman time in the future after every pulse", in_logs=False),
    KnownFix("base-download-wait", "base_download_wait", (r"waiting for (the )?base\b", r"base (model )?download(ing)? .*(wait|slow|in progress)"),
             "A job waited on a base model that was not yet prestaged.",
             (R("prestage the next job's base while the current job runs", "wait becomes zero", "low"),
              R("reorder the queue to run jobs whose base is present", "GPU stays busy", "low")),
             "job start delay < 60 s after the base is requested", in_logs=False),
    KnownFix("eval-too-small", "eval_too_small", (r'"n": ([0-9]|[1-4][0-9])\s*,?\s*$',),
             "A held-out eval ran on fewer than 50 items; its pass rate cannot support an adopt decision.",
             (R("rerun the eval on the full held-out set (n >= 200)", "interval narrow enough to decide", "low"),
              R("do not adopt on this eval; mark the verdict inconclusive", "no false adoption", "low")),
             "n >= 50 (target 200) and the Wilson interval excludes the decision threshold", in_logs=False),
    KnownFix("data-starved-module", "data_starved", (r"has \d+ rows, needs \d+",),
             "A module was trained on too few rows: only a few dozen steps, which cannot move the model (thin-mix rule: >= 200 steps).",
             (R("hold the module until the mix has enough rows (more data jobs first)", "no GPU money wasted", "low"),
              R("merge the mix into a sibling family mix", "enough steps", "medium")),
             "planned steps >= 200 before the job is queued",
             ),
    KnownFix("pod-unreachable", "pod_unreachable", (r"\bpod-unreachable\b",),
             "The pod does not answer over the relay or direct address; syncs, runner and guardian all stall.",
             (R("probe the direct address, then the relay; retry with backoff", "reconnects if transient", "low"),
              R("restart the instance (after confirming the last sync) within the approved budget", "pod back", "medium"),
              R("create a new instance and re-stage", "only if the host is lost", "high")),
             "a trivial command returns within 10 s; sync_age falls"),
    KnownFix("runner-down", "runner_down", (r"\brunner-down\b",),
             "The module runner process is not running while work is queued.",
             (R("start the runner (low-priority launcher) unless the STOP file is present", "queue drains", "low"),),
             "runner process count > 0 and a module line 'start' appears"),
    KnownFix("sync-stale", "sync_stale", (r"\bsync-stale\(\d+min\)",),
             "Results were not synced home for a long time: a stop now would lose them.",
             (R("run the sync now and check every out_* directory", "sync_age < 5 min", "low"),
              R("restart the sync loop", "regular syncs", "low")),
             "sync_age < 30 min on the next health line"),
    KnownFix("serve-failed", "serve_failed", (r"could not be served: remote step failed \(rc \d+\)",),
             "The eval model server failed to start on the pod (exit code from the remote start step).",
             (R("read the server log tail, free VRAM or fix the argument it rejects, retry once", "tuned eval runs", "low"),
              R("score that eval at home on the PC slot", "no GPU needed", "low")),
             "the server answers a test request"),
    KnownFix("module-failed", "module_failed", (r"module \S+: FAILED rc=\d+",),
             "A module failed: at least one of its jobs returned a non-zero code.",
             (R("read the failing job's log, apply the matching known fix, re-queue once", "module reaches DONE", "low"),
              R("park the module and let the next one run", "queue keeps moving", "low")),
             "the re-queued module reaches DONE"),
    KnownFix("failure-storm-hold", "failure_storm", (r"failure storm guard",),
             "autogen stopped queuing because failed files sit in queue/hold.",
             (R("clear the cause of the held failures first, then release the hold files", "queue resumes", "low"),),
             "queue/hold is empty and autogen logs 'queued'"),
    KnownFix("autogen-crash", "autogen_crash", (r"autogen: pass failed:",),
             "An autogen pass raised: a code error in the generator, nothing was queued.",
             (R("fix the named error (missing attribute or import), rerun the pass", "queue fills", "low"),),
             "next pass logs 'queued'"),
)
BY_ID = {k.id: k for k in TABLE}
KINDS = tuple(k.kind for k in TABLE)
# the incident kinds named in MASTER_BLUEPRINT 8.7 and the P0.6 task: replay must report each one that the logs contain
BLUEPRINT_KINDS = ("vram_starvation", "smoke_release_race", "feeder_stall", "transfer_stall", "missing_lora", "unseen_kind_vram",
                   "disk_full", "pkill_self_match", "autogen_no_state", "deadman_unarmed", "base_download_wait", "eval_too_small",
                   "data_starved")


@dataclasses.dataclass(frozen=True)
class Incident:
    kind: str
    fix_id: str
    ts: str                                 # HH:MM:SS from the log line, an ISO time, or "" when the log has none
    source: str                             # log file name (no directories)
    line: int
    evidence: str                           # the matching text, trimmed
    count: int = 1                          # lines merged into this episode


def remedies_for(kind_or_id: str) -> tuple[Remedy, ...]:
    for k in TABLE:
        if kind_or_id in (k.id, k.kind):
            return k.remedies
    return ()


def parse_health(line: str) -> Optional[dict]:
    """'09:46:29 util=78 disk=8G guard=1 stop=0 runner=5 queued=0 lastfails=2 sync_age=22min problems: ...' -> dict (None if not one)."""
    if " util=" not in line or " problems:" not in line:
        return None
    m = _TS.match(line)
    kv = dict(re.findall(r"\b(util|disk|guard|stop|runner|queued|lastfails|sync_age)=(\S+)", line))

    def num(v: Optional[str]) -> Optional[int]:
        mm = re.match(r"(\d+)", v or "")
        return int(mm.group(1)) if mm else None
    return {"ts": m.group(1) if m else "", "util": num(kv.get("util")), "disk_g": num(kv.get("disk")), "guard": num(kv.get("guard")),
            "stop": num(kv.get("stop")), "runner": num(kv.get("runner")), "queued": num(kv.get("queued")),
            "sync_age_min": num(kv.get("sync_age"))}


def match_line(line: str) -> list[KnownFix]:
    """Entries whose text signature matches this one log line. Partial-transfer lines count only when got < total."""
    out = []
    for k in TABLE:
        for rx in k.compiled():
            m = rx.search(line)
            if not m:
                continue
            if k.id == "transfer-stall" and rx.pattern.startswith("sync") and (int(m.group(1)) >= int(m.group(2))):
                continue
            out.append(k)
            break
    return out


def match_metrics(rec: dict) -> list[KnownFix]:
    return [k for k in TABLE if k.metric is not None and k.metric(rec)]


def _mins(ts: str) -> Optional[int]:
    m = re.match(r"(\d\d):(\d\d)", ts or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def scan_lines(lines: Iterable[str], source: str) -> list[Incident]:
    """Incidents in one log, consecutive repeats of a kind merged into one episode (gap <= 10 min, or <= 5 lines when untimed)."""
    raw: list[Incident] = []
    ts = ""
    pending: Optional[tuple[int, int]] = None            # (line, steps) of a trainer result block awaiting its adapter line
    for no, line in enumerate(lines, 1):
        line = line.rstrip("\n")
        tm = _TS.match(line)
        if tm:
            ts = tm.group(1)
        else:
            im = _ISO.search(line)
            if im:
                ts = im.group(1)
        hits = {k.id: k for k in match_line(line)}
        rec = parse_health(line)
        if rec:
            hits.update({k.id: k for k in match_metrics(rec)})
        for k in hits.values():
            raw.append(Incident(k.kind, k.id, ts, source, no, line.strip()[:160]))
        # trainer result blocks: "steps": N ... later "adapter": ".../ft_x/adapter"; a ft with N < 200 steps is data-starved
        sm = re.match(r'\s*"steps": (\d+),?\s*$', line)
        if sm:
            pending = (no, int(sm.group(1)))
        am = re.search(r'"adapter": "[^"]*/ft_[^"]*"', line)
        if am and pending and pending[1] < 200 and no - pending[0] < 80:
            raw.append(Incident("data_starved", "data-starved-module", ts, source, pending[0], f'ft trained only {pending[1]} steps', 1))
            pending = None
        elif '"adapter"' in line:
            pending = None
    merged: list[Incident] = []
    last: dict[str, tuple[int, str, int]] = {}            # fix id -> (index in merged, ts of last member, line of last member)
    wide = {"vram-starved-trainer", "missing-lora", "serve-failed"}     # a compact and a pretty-printed copy of one result: same episode
    always_new = {"module-failed"}
    for inc in sorted(raw, key=lambda i: i.line):
        j = last.get(inc.fix_id)
        if j is not None and inc.fix_id not in always_new:
            idx, pts, pline = j
            a, b = _mins(pts), _mins(inc.ts)
            if inc.fix_id in wide:
                near = inc.line - pline <= 250
            elif a is not None and b is not None and b >= a:
                near = b - a <= 35
            else:
                near = inc.line - pline <= 5
            if near:
                merged[idx] = dataclasses.replace(merged[idx], count=merged[idx].count + 1)
                last[inc.fix_id] = (idx, inc.ts, inc.line)
                continue
        last[inc.fix_id] = (len(merged), inc.ts, inc.line)
        merged.append(inc)
    return merged


def default_log_paths(home: Optional[Path] = None) -> list[Path]:
    base = (home or Path.home()) / "creator_runtime"
    out = [base / "gpu" / n for n in ("module_runner.out", "health.log", "health_alerts.log", "sync_loop.out")]
    q = base / "gpuday" / "trainmix" / "queue" / "logs"
    out += sorted(q.glob("*.log")) if q.is_dir() else []
    return [p for p in out if p.is_file()]


def replay(paths: Optional[Sequence[Path | str]] = None) -> dict[str, Any]:
    """Scan logs and report every recognised incident. -> {incidents, counts (kind -> episodes), by_source, missed (blueprint kinds with
    no incident in these logs), files}."""
    paths = [Path(p) for p in (paths if paths is not None else default_log_paths())]
    incidents: list[Incident] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                incidents += scan_lines(fh, p.name)
        except OSError:
            continue
    counts: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for i in incidents:
        counts[i.kind] = counts.get(i.kind, 0) + 1
        by_source.setdefault(i.kind, {})
        by_source[i.kind][i.source] = by_source[i.kind].get(i.source, 0) + 1
    missed = [k for k in BLUEPRINT_KINDS if k not in counts]
    return {"incidents": incidents, "counts": counts, "by_source": by_source, "missed": missed, "files": [p.name for p in paths]}


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"files: {len(rep['files'])}; incidents (episodes): {len(rep['incidents'])}"]
    for kind, n in sorted(rep["counts"].items(), key=lambda kv: -kv[1]):
        firsts = [i for i in rep["incidents"] if i.kind == kind][:3]
        lines.append(f"{kind}: {n}  e.g. " + "; ".join(f"{i.source}:{i.line} {i.ts or '-'}" for i in firsts))
    lines.append("not seen in these logs: " + (", ".join(rep["missed"]) or "none"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(replay()))
