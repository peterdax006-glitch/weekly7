"""GPU operator (MASTER_BLUEPRINT 8.7, PHASE0 P1.7): sees pod telemetry, knows the pipeline's bottlenecks and the known-fix table,
ranks remedies by expected gain and risk, acts through a backend, verifies the effect, reverts if worse, enforces lifecycle rules
(deadman, sync before stop, stop on empty wishlist, owner budget cap) and appends every incident + remedy + measured effect as
known-fix evidence.

Two backends share one interface: ReplayBackend (records what it would do; the world is the real 3-4 Oct logs) and LiveBackend
(ssh/local commands built from a config file OUTSIDE the repo; it never executes unless explicitly armed). Nothing in this file
holds a hostname, address, port, key or home path."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from creator import knownfix as KF

LOCAL_UTC_OFFSET_H = 6          # the 3-4 Oct logs: local clock = UTC-6 (ISO "at" stamps are UTC)
EPISODE_GAP_S = 35 * 60
VERIFY_WINDOW_S = 15 * 60
FT_WAIT_PROBE_S = 120           # a trainer that holds no VRAM after this long is starved (replay: synthesised from the skip line)
THIN_STEPS = 200


# ---------------------------------------------------------------- scrubbing
_SCRUBS = ((re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
           (re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\s\"']*", re.I), "<path>"),
           (re.compile(r"/[a-z]/Users/[^\s\"']*", re.I), "<path>"),
           (re.compile(r"/(?:home|root)/[^\s\"']*"), "<path>"))


def scrub(text: str) -> str:
    for rx, rep in _SCRUBS:
        text = rx.sub(rep, text)
    return text


# ---------------------------------------------------------------- telemetry
@dataclasses.dataclass
class Telemetry:
    """One observation of the pod and its pipeline. Fields a source cannot know stay None."""
    t: Optional[float]                       # seconds since 00:00 of the first log day (local clock)
    source: str                              # health | runner | sync | autogen | job
    util: Optional[int] = None               # GPU utilisation %
    vram_mib: Optional[dict] = None          # per process: {"trainer": n, "eval_server": n, "filler": n, "free": n, "need": n}
    power_w: Optional[float] = None
    disk_g: Optional[int] = None
    guard: Optional[int] = None              # guardian/filler processes
    stop: Optional[int] = None               # STOP file present (1 = fillers held)
    runner: Optional[int] = None             # runner processes
    queued: Optional[int] = None
    sync_age_min: Optional[int] = None
    reachable: bool = True
    job: str = ""
    steps: Optional[int] = None              # trainer steps, from a finished ft result
    usd_spent: Optional[float] = None
    text: str = ""                           # the log line (scrubbed)
    synthetic: bool = False                  # reconstructed, not read verbatim

    @property
    def clock(self) -> str:
        if self.t is None:
            return ""
        s = int(self.t) % 86400
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def metrics(self) -> dict:
        return {"util": self.util, "disk_g": self.disk_g, "guard": self.guard, "stop": self.stop, "runner": self.runner,
                "queued": self.queued, "sync_age_min": self.sync_age_min}


def _tod(s: str) -> float:
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def _iso_local(s: str) -> float:
    d = dt.datetime.fromisoformat(s.replace("Z", "")[:19])
    return d.hour * 3600 + d.minute * 60 + d.second + (d.day - 4) * 86400 - LOCAL_UTC_OFFSET_H * 3600


_LINE_TS = re.compile(r"^(?:HEALTH\s+)?(\d{2}:\d{2}:\d{2})\b")


def _timed_lines(lines: Iterable[str]) -> list[tuple[Optional[float], str]]:
    """(t, line) with untimed lines taking the previous line's time (or the next one's at the file start); day rollover handled."""
    out: list[tuple[Optional[float], str]] = []
    last: Optional[float] = None
    day = 0
    for ln in lines:
        ln = ln.rstrip("\n")
        m = _LINE_TS.match(ln)
        if m:
            t = _tod(m.group(1)) + day * 86400
            if last is not None and t < last - 6 * 3600:
                day += 1
                t += 86400
            last = t
        out.append((last if not m else last, ln))
    first = next((t for t, _ in out if t is not None), None)
    return [(t if t is not None else first, ln) for t, ln in out]


def _read(path: Path) -> list[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def from_health(lines: Iterable[str]) -> list[Telemetry]:
    out = []
    for t, ln in _timed_lines(lines):
        rec = KF.parse_health(ln)
        if rec is None:
            continue
        out.append(Telemetry(t, "health", util=rec["util"], disk_g=rec["disk_g"], guard=rec["guard"], stop=rec["stop"],
                             runner=rec["runner"], queued=rec["queued"], sync_age_min=rec["sync_age_min"],
                             reachable="pod-unreachable" not in ln, text=scrub(ln)))
    return out


def from_runner(lines: Iterable[str]) -> list[Telemetry]:
    return [Telemetry(t, "runner", text=scrub(ln)) for t, ln in _timed_lines(lines)
            if ln.strip() and not ln.startswith(("nohup", "Welcome", "Have fun", "AI agents"))]


def from_sync(lines: Iterable[str]) -> list[Telemetry]:
    return [Telemetry(t, "sync", text=scrub(ln)) for t, ln in _timed_lines(lines)
            if re.search(r"closed by remote host|^\d\d:\d\d:\d\d sync ", ln)]


def from_autogen(lines: Iterable[str]) -> list[Telemetry]:
    return [Telemetry(t, "autogen", text=scrub(ln)) for t, ln in _timed_lines(lines) if _LINE_TS.match(ln)]


def module_windows(runner_lines: Iterable[str]) -> dict[str, tuple[float, float]]:
    """module -> (start, end) of its LAST run in module_runner.out."""
    win: dict[str, tuple[float, float]] = {}
    start: dict[str, float] = {}
    for t, ln in _timed_lines(runner_lines):
        m = re.match(r"\d\d:\d\d:\d\d module (\S+?): (start|DONE|FAILED)", ln)
        if not m or t is None:
            continue
        if m.group(2) == "start":
            start[m.group(1)] = t
        elif m.group(1) in start:
            win[m.group(1)] = (start[m.group(1)], t)
    return win


def from_job_log(lines: Iterable[str], module: str, window: Optional[tuple[float, float]]) -> list[Telemetry]:
    """Job result lines of one queue log. Time = module start + cumulative wall_s (module window known), else the last ISO stamp seen.
    An ft that was starved of VRAM gets a synthetic sample FT_WAIT_PROBE_S after its start (free/need read from its skip line)."""
    out: list[Telemetry] = []
    clock = window[0] if window else None
    for ln in lines:
        m = re.search(r'"at": "(\d{4}-\d\d-\d\dT[\d:]+)', ln)
        if m and not window:
            clock = _iso_local(m.group(1))
        if ln.startswith('{"job":'):
            try:
                rec = json.loads(ln)
            except ValueError:
                rec = {}
            wall = float(rec.get("wall_s") or 0)
            start = clock
            end = (clock + wall) if clock is not None else None
            if window:
                clock = end
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            mon = rec.get("monitor") or {}
            job = str(rec.get("job", "")).replace("ext:", "")
            sk = str(res.get("skipped", ""))
            sv = re.match(r"VRAM:\s*(\d+) MiB free after \d+ s,? needs (\d+)", sk)
            if sv and start is not None:
                out.append(Telemetry(start + FT_WAIT_PROBE_S, "job", util=None, job=job,
                                     vram_mib={"free": int(sv.group(1)), "need": int(sv.group(2))}, synthetic=True,
                                     text=f"{job} skipped: VRAM: {sv.group(1)} MiB free after {FT_WAIT_PROBE_S} s, needs {sv.group(2)}"))
            steps = (res.get("sft") or {}).get("steps") if isinstance(res.get("sft"), dict) else None
            out.append(Telemetry(end, "job", util=int(mon["util_mean"]) if mon.get("util_mean") is not None else None, job=job,
                                 steps=steps if job.startswith("ft_") else None, usd_spent=rec.get("usd_spent_total"),
                                 text=scrub(ln[:400])))
        elif ln.startswith(("job ext:", "server ")):
            out.append(Telemetry(clock, "job", job=module, text=scrub(ln[:400])))
    return out


def reconstruct(root: Optional[Path | str] = None) -> list[Telemetry]:
    """Telemetry stream rebuilt from the real logs (read-only). root = the runtime directory that holds gpu/ and gpuday/."""
    root = Path(root) if root else Path.home() / "creator_runtime"
    gpu, qlogs = root / "gpu", root / "gpuday" / "trainmix" / "queue" / "logs"
    recs: list[Telemetry] = []
    recs += from_health(_read(gpu / "health.log"))
    run_lines = _read(gpu / "module_runner.out")
    recs += from_runner(run_lines)
    recs += from_sync(_read(gpu / "sync_loop.out"))
    recs += from_autogen(_read(qlogs / "autogen.log"))
    wins = module_windows(run_lines)
    for p in sorted(qlogs.glob("*.log")) if qlogs.is_dir() else []:
        if p.name == "autogen.log":
            continue
        recs += from_job_log(_read(p), p.stem, wins.get(p.stem))
    return order(recs)


def order(recs: Sequence[Telemetry]) -> list[Telemetry]:
    prio = {"health": 0, "runner": 1, "sync": 2, "autogen": 3, "job": 4}
    idx = {id(r): i for i, r in enumerate(recs)}
    return sorted(recs, key=lambda r: (r.t is None, r.t or 0.0, prio.get(r.source, 9), idx[id(r)]))


# ---------------------------------------------------------------- bottleneck model
COMPONENTS = {
    "trainer": "fine-tune process; needs its VRAM block free at start, then keeps util ~99%",
    "eval_server": "llama server for held-out evals; needs its LoRA/base files on the pod",
    "fillers": "backfill models run by the guardian to keep util up; they hold VRAM the trainer needs",
    "feeder": "PC-side process that sends filler requests; if it stalls the fillers sit loaded and idle",
    "transfers": "sync loop (results home) and uploads (data/adapters to the pod); the relay stalls near 45 MB",
    "pc_eval_client": "PC-side eval client; with too few requests in flight the PC is the bottleneck",
    "guardian": "pod-side supervisor: STOP file holds the fillers, release lets them run",
    "runner": "module runner on the PC: sequences the module jobs, releases the guardian",
    "autogen": "queue generator on the PC: needs --state, queue/hold empty",
    "pod": "the rented instance: reachability, disk, deadman, spend",
}
KIND_COMPONENT = {
    "vram_starvation": "fillers", "smoke_release_race": "runner", "feeder_stall": "feeder", "transfer_stall": "transfers",
    "missing_lora": "eval_server", "serve_failed": "eval_server", "unseen_kind_vram": "trainer", "disk_full": "pod",
    "pkill_self_match": "runner", "autogen_no_state": "autogen", "autogen_crash": "autogen", "failure_storm": "autogen",
    "deadman_unarmed": "pod", "base_download_wait": "transfers", "eval_too_small": "pc_eval_client", "data_starved": "trainer",
    "pod_unreachable": "pod", "runner_down": "runner", "sync_stale": "transfers", "module_failed": "runner"}


def locate_bottleneck(r: Telemetry) -> str:
    """Which component limits the GPU right now, from one health-type sample (for symptoms no signature names)."""
    if not r.reachable or r.util is None:
        return "pod"
    if r.disk_g is not None and r.disk_g <= 3:
        return "pod"
    if (r.util or 0) <= 5 and r.stop == 1:
        return "trainer"             # fillers held on purpose; the trainer has not claimed the GPU yet
    if (r.util or 0) <= 5 and (r.guard or 0) >= 1 and (r.runner or 0) > 0:
        return "feeder"
    if (r.util or 0) < 60 and (r.runner or 0) == 0:
        return "runner"
    if (r.util or 0) < 80:
        return "pc_eval_client"
    return "trainer"


# ---------------------------------------------------------------- actions, playbook
@dataclasses.dataclass(frozen=True)
class Action:
    key: str
    params: tuple = ()
    live_only: bool = False


@dataclasses.dataclass(frozen=True)
class Option:
    remedy_idx: int                          # index into the known-fix entry's ranked remedies (text + risk label live there)
    action: str
    gain: float                              # prior expected gain 0..1 (share of the lost GPU time recovered)


PLAYBOOK: dict[str, tuple[Option, ...]] = {
    "vram_starvation": (Option(1, "hold_guardian_stop", .92), Option(0, "evict_filler", .85), Option(2, "pair_small_job", .40)),
    "smoke_release_race": (Option(0, "release_after_result", .90), Option(1, "requeue_module", .70), Option(2, "skip_smoke", .50)),
    "feeder_stall": (Option(0, "restart_feeder", .90), Option(1, "move_client_to_pod", .70), Option(2, "raise_feeder_workers", .60)),
    "transfer_stall": (Option(1, "sync_chunks_direct", .92), Option(0, "sync_resume_direct", .80), Option(2, "bucket_relay", .50)),
    "missing_lora": (Option(0, "place_lora", .95), Option(1, "rerun_ft", .50), Option(2, "skip_tuned_eval", .40)),
    "unseen_kind_vram": (Option(0, "reserve_family_peak", .90), Option(1, "run_alone_record_peak", .70)),
    "disk_full": (Option(0, "clean_disk", .90), Option(1, "move_cache", .60), Option(2, "recreate_instance", .50)),
    "pkill_self_match": (Option(0, "match_by_pidfile", .90), Option(1, "use_stop_file", .80)),
    "autogen_no_state": (Option(0, "restart_autogen_state", .95), Option(1, "require_state_flag", .60)),
    "deadman_unarmed": (Option(0, "arm_deadman", .95), Option(1, "stop_after_sync", .60)),
    "base_download_wait": (Option(0, "prestage_next_base", .90), Option(1, "reorder_queue", .60)),
    "eval_too_small": (Option(0, "rerun_full_eval", .90), Option(1, "mark_inconclusive", .60)),
    "data_starved": (Option(0, "hold_module_for_data", .90), Option(1, "merge_family_mix", .50)),
    "pod_unreachable": (Option(0, "probe_backoff", .80), Option(1, "restart_instance", .60), Option(2, "new_instance", .20)),
    "runner_down": (Option(0, "start_runner", .95),),
    "sync_stale": (Option(0, "sync_now", .90), Option(1, "restart_sync_loop", .70)),
    "serve_failed": (Option(0, "serve_retry_after_fix", .80), Option(1, "score_at_home", .60)),
    "module_failed": (Option(0, "diagnose_job_and_requeue", .80), Option(1, "park_module", .50)),
    "failure_storm": (Option(0, "clear_failure_then_release", .90),),
    "autogen_crash": (Option(0, "escalate_code_fix", .90),),
}
RISK_PENALTY = {"low": 0.02, "medium": 0.15, "high": 0.40}
LEVEL_MAX_RISK = {"A0": None, "A1": "low", "A2": "medium", "A3": "medium"}      # high risk is never automatic: the owner decides
_RISK_RANK = {"low": 0, "medium": 1, "high": 2}

# what Claude actually did on 3-4 Oct (owner-reported fix names); None = no pod remedy recorded for this kind
CLAUDE_FIX: dict[str, Optional[str]] = {
    "vram_starvation": "hold_guardian_stop",          # held the STOP file until the trainer claimed VRAM, released after the fast ft
    "smoke_release_race": "release_after_result",
    "feeder_stall": "restart_feeder",
    "transfer_stall": "sync_chunks_direct",           # byte-append chunks over the direct address
    "missing_lora": "place_lora",
    "autogen_no_state": "restart_autogen_state",
    "runner_down": "start_runner",                    # swapped the runner between modules
    "module_failed": "diagnose_job_and_requeue",
}


@dataclasses.dataclass
class Ranked:
    option: Option
    action: str
    text: str
    risk: str
    gain: float
    score: float


class Learner:
    """Known-fix evidence: JSONL of {kind, action, effect, ok}. Measured outcomes pull the prior gain of (kind, action)."""

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path else None
        self.stats: dict[tuple[str, str], list[float]] = {}
        if self.path and self.path.is_file():
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                    self.stats.setdefault((r["kind"], r["action"]), []).append(1.0 if r.get("ok") else 0.0)
                except (ValueError, KeyError):
                    continue

    def blended(self, kind: str, action: str, prior: float) -> float:
        obs = self.stats.get((kind, action), [])
        if not obs:
            return prior
        w = min(1.0, len(obs) / 5.0)
        return prior * (1 - w) + (sum(obs) / len(obs)) * w

    def record(self, entry: dict) -> None:
        self.stats.setdefault((entry["kind"], entry["action"]), []).append(1.0 if entry.get("ok") else 0.0)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")


def rank_remedies(kind: str, learner: Optional[Learner] = None, level: str = "A2") -> list[Ranked]:
    """Remedies for a kind ordered by expected gain minus risk penalty; those above the level's risk ceiling are dropped (high never)."""
    fix = next((k for k in KF.TABLE if k.kind == kind), None)
    if fix is None:
        return []
    ceiling = LEVEL_MAX_RISK.get(level)
    out: list[Ranked] = []
    for o in PLAYBOOK.get(kind, ()):
        rem = fix.remedies[o.remedy_idx]
        if ceiling is None or _RISK_RANK[rem.risk] > _RISK_RANK[ceiling]:
            continue
        g = learner.blended(kind, o.action, o.gain) if learner else o.gain
        out.append(Ranked(o, o.action, rem.action, rem.risk, g, g - RISK_PENALTY[rem.risk]))
    return sorted(out, key=lambda r: -r.score)


# ---------------------------------------------------------------- backends
@dataclasses.dataclass
class Result:
    action: str
    executed: bool
    detail: str = ""
    command: tuple = ()


class Backend:
    """Interface: do(action) applies it, revert(action) undoes it. Subclasses decide whether anything really happens."""
    name = "abstract"

    def do(self, action: Action, t: Optional[float] = None) -> Result:
        raise NotImplementedError

    def revert(self, action: Action, t: Optional[float] = None) -> Result:
        raise NotImplementedError


class ReplayBackend(Backend):
    """Records what the operator would do; touches nothing."""
    name = "replay"

    def __init__(self):
        self.log: list[tuple[str, str, Optional[float]]] = []

    def do(self, action: Action, t: Optional[float] = None) -> Result:
        self.log.append(("do", action.key, t))
        return Result(action.key, False, "recorded (replay)")

    def revert(self, action: Action, t: Optional[float] = None) -> Result:
        self.log.append(("revert", action.key, t))
        return Result(action.key, False, "revert recorded (replay)")


class LiveDisabled(RuntimeError):
    pass


# remote command templates; {guardian_dir}/{work_dir}/{lora_dir} come from the external config, never from this file
_REMOTE: dict[str, str] = {
    "hold_guardian_stop": "touch {guardian_dir}/STOP",
    "release_guardian": "rm -f {guardian_dir}/STOP",
    "release_after_result": "test -s {work_dir}/result.json && rm -f {guardian_dir}/STOP",
    "evict_filler": "{guardian_dir}/evict_largest_filler",
    "clean_disk": "{work_dir}/clean_caches --keep-prestaged",
    "arm_deadman": "{guardian_dir}/deadman arm {deadman_minutes}",
    "place_lora": "test -s {lora_dir}/{lora_file}",
    "serve_retry_after_fix": "{guardian_dir}/serve_retry",
    "probe_backoff": "true",
}
# commands run on the PC (argv templates from config["local"]), e.g. restart_feeder, sync_chunks_direct, start_runner, sync_now
_LOCAL_KEYS = ("restart_feeder", "sync_chunks_direct", "sync_resume_direct", "start_runner", "sync_now", "restart_sync_loop",
               "restart_autogen_state", "requeue_module", "stop_instance", "restart_instance", "place_lora_upload")
_REVERT = {"hold_guardian_stop": "release_guardian", "evict_filler": None}


def _push_gpu(action: "Action") -> None:
    try:
        from creator import notify as N
        why = dict(action.params).get("reason")
        if action.key == "stop_instance":
            N.gpu_event("budget_near_cap" if why == "budget" else "rental_idle", f"Rental stopped ({why}).")
        elif action.key == "release_after_result":
            N.gpu_event("job_done", "A job produced its result.")
    except Exception:
        pass


class LiveBackend(Backend):
    """Builds ssh/local commands from a config file outside the repo (GPUOP_CONFIG or the path given). Nothing runs unless armed=True
    AND the instance was created with armed=True explicitly; otherwise do() returns the command it would have run."""
    name = "live"

    def __init__(self, config_path: Optional[Path | str] = None, armed: bool = False, runner: Callable[..., Any] = subprocess.run):
        path = config_path or os.environ.get("GPUOP_CONFIG")
        self.cfg: dict = {}
        if path and Path(path).is_file():
            self.cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        self.armed = bool(armed)
        self._run = runner

    def command_for(self, action: Action) -> tuple:
        c = self.cfg
        if action.key in _REMOTE:
            tgt = c.get("ssh_target", "")
            argv = ["ssh", "-p", str(c.get("ssh_port", 22))]
            if c.get("ssh_key"):
                argv += ["-i", str(c["ssh_key"])]
            fmt = {"guardian_dir": c.get("guardian_dir", ""), "work_dir": c.get("work_dir", ""), "lora_dir": c.get("lora_dir", ""),
                   "lora_file": dict(action.params).get("lora_file", ""), "deadman_minutes": c.get("deadman_minutes", 30)}
            return tuple(argv + [tgt, _REMOTE[action.key].format(**fmt)])
        if action.key in _LOCAL_KEYS:
            tmpl = (c.get("local") or {}).get(action.key)
            return tuple(tmpl) if tmpl else ()
        return ()

    def _exec(self, action: Action, verb: str) -> Result:
        cmd = self.command_for(action)
        if not cmd:
            return Result(action.key, False, f"{verb}: no live command configured", ())
        if not self.armed:
            return Result(action.key, False, f"{verb}: dry-run (live not armed)", cmd)
        p = self._run(list(cmd), capture_output=True, text=True, timeout=120)
        return Result(action.key, True, f"{verb}: rc={getattr(p, 'returncode', '?')}", cmd)

    def do(self, action: Action, t: Optional[float] = None) -> Result:
        r = self._exec(action, "do")
        if r.executed:                                             # owner push only for a really executed live action
            _push_gpu(action)
        return r

    def revert(self, action: Action, t: Optional[float] = None) -> Result:
        back = _REVERT.get(action.key)
        return self._exec(Action(back), "revert") if back else Result(action.key, False, "no revert defined", ())


# ---------------------------------------------------------------- lifecycle
@dataclasses.dataclass
class PodState:
    usd_spent: float = 0.0
    usd_per_hour: float = 0.0
    deadman_armed: bool = False
    sync_age_min: Optional[int] = None
    wishlist_len: int = 0
    jobs_running: int = 0
    running: bool = True


def lifecycle_actions(s: PodState, budget_usd: float, sync_ok_min: int = 5, horizon_min: int = 30) -> list[Action]:
    """Rules, in priority order: deadman armed; budget cap (never above the owner budget at any level); stop when the wishlist is
    empty and nothing runs; ALWAYS sync home before a stop."""
    acts: list[Action] = []
    if s.running and not s.deadman_armed:
        acts.append(Action("arm_deadman"))
    projected = s.usd_spent + s.usd_per_hour * horizon_min / 60.0
    over_budget = projected >= budget_usd
    idle_done = s.wishlist_len == 0 and s.jobs_running == 0
    if s.running and (over_budget or idle_done):
        if s.sync_age_min is None or s.sync_age_min > sync_ok_min:
            acts.append(Action("sync_now"))
        acts.append(Action("stop_instance", (("reason", "budget" if over_budget else "wishlist_empty"),)))
    return acts


def within_budget(spent: float, cost_usd: float, budget_usd: float) -> bool:
    return spent + cost_usd <= budget_usd


# ---------------------------------------------------------------- operator
@dataclasses.dataclass
class Incident:
    kind: str
    fix_id: str
    t_onset: Optional[float]
    t_detect: Optional[float]
    t_visible: Optional[float]               # when the real log line existed (what Claude could have read)
    evidence: str
    component: str
    diagnosis: str
    ranked: list[Ranked]
    chosen: Optional[Ranked]
    refused: str = ""                        # why nothing was done (level, budget)
    claude_fix: Optional[str] = None
    verdict: str = ""
    measured: dict = dataclasses.field(default_factory=dict)
    reverted: bool = False
    t_resolved: Optional[float] = None
    job: str = ""

    @property
    def ttd_s(self) -> Optional[float]:
        return None if self.t_onset is None or self.t_detect is None else self.t_detect - self.t_onset


# per-kind "condition cleared" test over later health samples; used for the measured effect and for Claude's resolve time proxy
def _clear(kind: str, r: Telemetry) -> Optional[bool]:
    if r.source != "health":
        return None
    if kind == "feeder_stall":
        return r.util is not None and r.util >= 60
    if kind == "disk_full":
        return r.disk_g is not None and r.disk_g >= 5
    if kind == "pod_unreachable":
        return r.util is not None
    if kind == "runner_down":
        return (r.runner or 0) > 0
    if kind == "sync_stale":
        return r.sync_age_min is not None and r.sync_age_min < 30
    return None


def _util_mean(rs: Sequence[Telemetry]) -> Optional[float]:
    v = [r.util for r in rs if r.source == "health" and r.util is not None]
    return sum(v) / len(v) if v else None


class Operator:
    def __init__(self, backend: Optional[Backend] = None, learner: Optional[Learner] = None, level: str = "A2",
                 budget_usd: float = 0.0):
        self.backend = backend or ReplayBackend()
        self.learner = learner or Learner()
        self.level = level
        self.budget = budget_usd
        self.incidents: list[Incident] = []
        self.escalated: list[tuple[Optional[float], str, str]] = []       # (t, why, text) -> THINKER, then the digest
        self.spent = 0.0
        self._open: dict[str, tuple[Incident, float]] = {}                # fix id -> (episode head, t of last member)

    # --- detection
    def detect(self, r: Telemetry) -> list[KF.KnownFix]:
        hits = {k.id: k for k in KF.match_line(r.text)} if r.text else {}
        if r.source == "health":
            hits.update({k.id: k for k in KF.match_metrics(r.metrics())})
        if r.steps is not None and r.steps < THIN_STEPS:
            hits["data-starved-module"] = KF.BY_ID["data-starved-module"]
        return list(hits.values())

    def _new_episode(self, k: KF.KnownFix, r: Telemetry) -> Incident:
        ranked = rank_remedies(k.kind, self.learner, self.level)
        refused, chosen = "", None
        for cand in ranked:
            cost = 0.0                                      # pod remedies here cost no extra $; instance restarts are excluded by level
            if not within_budget(self.spent, cost, self.budget) and self.budget > 0:
                refused = "budget cap"
                if self.backend.name == "live":
                    _push_gpu(Action("stop_instance", (("reason", "budget"),)))
                break
            chosen = cand
            break
        if chosen is None and not refused:
            refused = "level %s allows no remedy for this risk" % self.level
        onset = r.t
        inc = Incident(k.kind, k.id, onset, r.t, r.t, scrub(r.text.strip()[:160]), KIND_COMPONENT.get(k.kind, "pod"), k.diagnosis,
                       ranked, chosen, refused, CLAUDE_FIX.get(k.kind))
        if chosen:
            self.backend.do(Action(chosen.action), r.t)
        inc.verdict = self._verdict(inc)
        self.incidents.append(inc)
        return inc

    @staticmethod
    def _verdict(inc: Incident) -> str:
        if inc.claude_fix is None:
            return "no Claude pod fix recorded"
        if inc.chosen is None:
            return "differs (nothing chosen)"
        if inc.chosen.action == inc.claude_fix:
            return "matches"
        ref = next((c for c in inc.ranked if c.action == inc.claude_fix), None)
        return "beats" if ref is None or inc.chosen.score > ref.score + 1e-9 else "differs"

    def observe(self, r: Telemetry) -> list[Incident]:
        """Feed one telemetry sample in time order; returns the incidents that opened on it."""
        if r.usd_spent is not None:
            self.spent = max(self.spent, r.usd_spent)
        opened: list[Incident] = []
        for k in self.detect(r):
            key = k.id + ('|' + r.job if k.id == 'vram-starved-trainer' else '')
            cur = self._open.get(key)
            if cur and r.t is not None and cur[1] is not None and r.t - cur[1] <= EPISODE_GAP_S:
                self._open[key] = (cur[0], r.t)
                continue
            inc = self._new_episode(k, r)
            if k.id == "vram-starved-trainer" and r.synthetic:
                inc.t_visible = None                           # filled at the real skip line
            inc.job = r.job
            self._open[key] = (inc, r.t if r.t is not None else 0.0)
            opened.append(inc)
        # real skip line for a starved trainer: when Claude could first have read it
        for inc in self.incidents:
            if inc.fix_id == "vram-starved-trainer" and inc.job == r.job and inc.t_visible is None and not r.synthetic and r.text and "skipped" in r.text:
                inc.t_visible = r.t
        if not self.detect(r) and r.source == "health" and (r.util is not None and r.util < 60 or r.stop == 0 and r.guard == 0):
            if "guardian-down" in r.text and r.stop == 0:
                self.escalated.append((r.t, "guardian down without a STOP hold", r.text[:120]))
        return opened

    # --- verify / learn
    def verify(self, stream: Sequence[Telemetry]) -> None:
        """For each incident: measure the next VERIFY_WINDOW_S of telemetry against the previous one; revert if the remedy made things
        worse; record the evidence. In replay the 'next minutes' are what really happened (with Claude's fix), not the operator's."""
        health = [r for r in stream if r.source == "health" and r.t is not None]
        for inc in self.incidents:
            if inc.t_detect is None or inc.chosen is None:
                continue
            pre = [r for r in health if inc.t_detect - VERIFY_WINDOW_S <= r.t < inc.t_detect]
            post = [r for r in health if inc.t_detect < r.t <= inc.t_detect + VERIFY_WINDOW_S]
            clear = [(_clear(inc.kind, r), r.t) for r in health if r.t > inc.t_detect]
            resolved = next((t for c, t in clear if c), None)
            inc.t_resolved = resolved
            up, uq = _util_mean(pre), _util_mean(post)
            ok = resolved is not None and resolved - inc.t_detect <= VERIFY_WINDOW_S if _clear(inc.kind, health[0]) is not None else None
            worse = up is not None and uq is not None and uq < up - 15 and ok is not True
            inc.measured = {"util_before": up, "util_after": uq, "cleared_within_window": ok, "samples_after": len(post),
                            "resolved_after_s": None if resolved is None else resolved - inc.t_detect}
            if worse:
                self.backend.revert(Action(inc.chosen.action), inc.t_detect)
                inc.reverted = True
            self.learner.record({"kind": inc.kind, "action": inc.chosen.action, "t": inc.t_detect, "ok": (ok is not False and not worse),
                                 "util_before": up, "util_after": uq, "reverted": inc.reverted, "backend": self.backend.name,
                                 "evidence": inc.evidence})

    def run(self, stream: Sequence[Telemetry]) -> list[Incident]:
        for r in stream:
            self.observe(r)
        self.verify(stream)
        return self.incidents


# ---------------------------------------------------------------- replay report
def claude_reference_s(inc: Incident) -> Optional[float]:
    """How long after the onset Claude's side had the facts / the problem was gone. Job-log kinds: when the log line existed (job end).
    Health kinds: when a later health sample shows the condition cleared (an upper bound on when the fix landed)."""
    if inc.kind == "vram_starvation" and inc.t_visible is not None and inc.t_onset is not None:
        return inc.t_visible - (inc.t_onset - FT_WAIT_PROBE_S)
    if inc.t_resolved is not None and inc.t_onset is not None:
        return inc.t_resolved - inc.t_onset
    return None


def replay_report(root: Optional[Path | str] = None, learner_path: Optional[Path | str] = None, level: str = "A2") -> dict:
    stream = reconstruct(root)
    op = Operator(ReplayBackend(), Learner(learner_path), level=level, budget_usd=0.0)
    incs = op.run(stream)
    rows = []
    for i in incs:
        ref = claude_reference_s(i)
        onset = i.t_onset - FT_WAIT_PROBE_S if i.kind == "vram_starvation" and i.t_onset is not None else i.t_onset
        ttd = None if onset is None or i.t_detect is None else i.t_detect - onset
        rows.append({"kind": i.kind, "clock": Telemetry(i.t_detect, "x").clock, "diagnosis": i.diagnosis.split(":")[0][:90],
                     "bottleneck": i.component, "remedy": i.chosen.action if i.chosen else "-", "risk": i.chosen.risk if i.chosen else "-",
                     "claude_fix": i.claude_fix or "-", "verdict": i.verdict, "ttd_s": ttd, "claude_ref_s": ref,
                     "reverted": i.reverted, "measured": i.measured})
    kf = KF.replay(KF.default_log_paths(Path(root).parent if root else None))
    return {"records": len(stream), "incidents": rows, "kinds_detected": sorted({r["kind"] for r in rows}),
            "knownfix_kinds": sorted(kf["counts"]), "escalated": len(op.escalated), "stream": stream, "operator": op}


def format_replay(rep: dict) -> str:
    out = [f"telemetry records: {rep['records']}; incidents: {len(rep['incidents'])}; escalated to THINKER: {rep['escalated']}"]
    for r in rep["incidents"]:
        out.append(f"{r['clock'] or '-':8} {r['kind']:20} {r['bottleneck']:11} -> {r['remedy']:24} (claude: {r['claude_fix']:24}) "
                   f"{r['verdict']:27} ttd={r['ttd_s']} claude_ref={r['claude_ref_s']}")
    return "\n".join(out)


if __name__ == "__main__":
    print(format_replay(replay_report()))
