"""THE CHECKER / SAFETY LOOP around unsupervised adoption (owner, 4 Oct 2026: "reliably trustworthy and unsupervised"; master checklist
row 1). Loaded on demand by creator.kernel.execute only (never at start). Every guard here only ADDS restrictions:

    TRUST GATE    a change made by an UNSUPERVISED worker (any worker not named in the policy's `supervised` list - today only the
                  teacher's 'claude-session') is adopted only when creator.codetrust.may_change opens its change class for that worker's
                  measured setup (attempt records named in the policy). No records = nothing open. Measuring-class changes never.
                  A refusal is a REJECT 'needs owner/teacher' with the diff saved to state/creator/pending/ for the teacher.
    RATE LIMIT    at most `max_adoptions_per_hour` unsupervised adoptions in any rolling hour, and none for `cooldown_hours` after ANY
                  rollback. A refusal is DEFERRED (not an attempt; the diff is saved; the gap is re-planned later).
    FULL SUITE    after every merge (supervised or not) the whole protected suite (codetrust.PROTECTED_SUITE: the tests of the measuring
                  code) runs on main at BelowNormal priority, with pytest-xdist from codetrust's private pylib when present, cached by
                  git tree hash. A confirmed failure (re-run serially) that the parent tree did not have reverts the merge through the
                  kernel's rollback path, which starts the cool-down.
    EVENTS        every cycle outcome with its safety facts -> state/creator/safety/events.jsonl (the source of
                  scripts/nupen_digest.py and of the rate limit).

The policy is state/creator/safety.json (owner/teacher-owned, sandbox-PROTECTED); missing keys take the defaults below."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Collection, Mapping, Optional, Sequence

POLICY_FILE = "safety.json"                         # under the kernel state dir (state/creator/)
EVENTS = Path("safety") / "events.jsonl"
SUITE_CACHE = Path("safety") / "suite_cache.json"
NEEDS = "needs owner/teacher"


@dataclasses.dataclass(frozen=True)
class Policy:
    max_adoptions_per_hour: int = 2                 # unsupervised adoptions in any rolling hour (open classes only)
    cooldown_hours: float = 6.0                     # no unsupervised adoption this long after any rollback
    supervised: tuple[str, ...] = ("claude-session",)   # workers whose changes the teacher made (trust gate / rate limit do not apply)
    trust_records: Mapping[str, str] = dataclasses.field(default_factory=dict)   # worker name -> codetrust attempt records (jsonl)
    full_suite: bool = True                         # run the protected suite after every merge
    suite_timeout: float = 3600.0


def policy(state: Path) -> Policy:
    """The policy file merged over the defaults; an unreadable file gives the defaults (which are the strict ones)."""
    try:
        raw = json.loads((Path(state) / POLICY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Policy()
    if not isinstance(raw, dict):
        return Policy()
    p = Policy()
    return Policy(max_adoptions_per_hour=max(0, int(raw.get("max_adoptions_per_hour", p.max_adoptions_per_hour))),
                  cooldown_hours=max(0.0, float(raw.get("cooldown_hours", p.cooldown_hours))),
                  supervised=tuple(str(x) for x in raw.get("supervised", p.supervised)),
                  trust_records={str(k): str(v) for k, v in dict(raw.get("trust_records") or {}).items()},
                  full_suite=bool(raw.get("full_suite", True)), suite_timeout=float(raw.get("suite_timeout", p.suite_timeout)))


# ------------------------------------------------------------------------------------------------ events
def events(state: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    f = Path(state) / EVENTS
    if f.is_file():
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if isinstance(x, dict):
                out.append(x)
    return out


def record(state: Path, row: Mapping[str, Any]) -> None:
    f = Path(state) / EVENTS
    f.parent.mkdir(parents=True, exist_ok=True)
    from creator import kernel as K                 # its line lock: swarm threads share the file
    K._append_log_line(f, json.dumps(dict(row), default=str))


def note(cfg: Any, rep: Any, plan: Any, now: Optional[float] = None) -> None:
    """One event per finished cycle (called by the kernel before it releases the adoption lock). Never raises."""
    try:
        s = dict(rep.details.get("safety") or {})
        record(cfg.state, {"at": time.time() if now is None else now, "outcome": rep.outcome, "package": rep.package,
                           "requirement": rep.requirement, "step": getattr(plan, "step", ""), "reason": str(rep.reason)[:1500],
                           "verdict": rep.verdict, "merge": rep.merge_commit, "by": s.get("by") or (rep.details.get("worker") or {}).get("by", ""),
                           "evidence": f"state/creator/cycles/{rep.package}/", "regression": rep.details.get("regression", ""), **s})
    except Exception:                                                  # noqa: BLE001 - the record is best effort, never a crash
        pass


# ------------------------------------------------------------------------------------------------ the gates
def supervised(pol: Policy, by: str) -> bool:
    return bool(by) and by in pol.supervised


def trust(pol: Policy, by: str, paths: Collection[str], message: str) -> tuple[bool, str]:
    """codetrust.may_change for this worker's measured setup (recomputed from its attempt records every call)."""
    from creator import codetrust as CT
    src = pol.trust_records.get(by, "")
    recs = CT._jsonl(Path(src)) if src else []
    ok, why = CT.may_change(paths, recs, message)
    if not src:
        why += f" (no measured setup for worker '{by}' in the safety policy)"
    return ok, why


def rate(state: Path, pol: Policy, now: Optional[float] = None) -> str:
    """Why an unsupervised adoption must wait now ('' = it may go)."""
    t = time.time() if now is None else now
    ev = events(state)
    backs = [float(e.get("at", 0)) for e in ev if e.get("outcome") == "ROLLED_BACK"]
    if backs and t - max(backs) < pol.cooldown_hours * 3600:
        return (f"cool-down after a rollback: {(pol.cooldown_hours * 3600 - (t - max(backs))) / 3600:.1f} h left of "
                f"{pol.cooldown_hours:g} h")
    recent = sum(1 for e in ev if e.get("outcome") == "ADOPTED" and not e.get("supervised") and t - float(e.get("at", 0)) < 3600)
    if recent >= pol.max_adoptions_per_hour:
        return f"{recent} unsupervised adoptions in the last hour (limit {pol.max_adoptions_per_hour})"
    return ""


def gate(cfg: Any, plan: Any, wp: Any, sb: Any, rep: Any, by: str, paths: Collection[str], at_decide: bool = False,
         now: Optional[float] = None) -> Optional[Exception]:
    """The exception the kernel raises instead of adopting (None = may go on). Called once before evaluation (trust + rate, so a
    refused change costs no test run) and again under the adoption lock just before the merge (rate, counted race-free)."""
    from creator import kernel as K
    pol = policy(cfg.state)
    info = rep.details.setdefault("safety", {})
    sup = supervised(pol, by)
    info.update(by=by, supervised=sup, paths=sorted(paths)[:60], objective=str(getattr(wp, "objective", ""))[:400])
    if "cls" not in info:
        from creator import codetrust as CT
        info["cls"] = CT.class_of_paths(list(paths), info["objective"])
    if sup:
        info["gate"] = "supervised (teacher's change): trust gate and rate limit do not apply"
        return None
    if not at_decide:
        ok, why = trust(pol, by, paths, info["objective"])
        info["trust"] = why
        if not ok:
            info["refused"] = "trust"
            K._save_pending(cfg, plan, sb, rep)
            return K._Reject(f"{NEEDS}: unsupervised worker '{by}' may not adopt this change - {why}")
    why = rate(cfg.state, pol, now)
    if why:
        info["refused"] = "rate"
        from creator import planner as P
        return K.Cancelled(f"{P.DEFERRED_PREFIX} safety rate limit: {why}")
    info["gate"] = "open: " + str(info.get("trust", ""))
    return None


# ------------------------------------------------------------------------------------------------ the full protected suite
def pytest_configure(config: Any) -> None:          # loaded as a pytest plugin (-p creator.safety): the suite runs at BelowNormal
    lower_priority()


def lower_priority() -> None:
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32                                   # type: ignore[attr-defined]
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)     # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:                                                  # noqa: BLE001 - priority is a courtesy, never a failure
        pass


def suite_files() -> tuple[str, ...]:
    from creator import codetrust as CT
    return tuple(CT.PROTECTED_SUITE)


def run_suite(root: Path, files: Sequence[str], junit: Path, timeout: float, python: str = sys.executable,
              workers: int = 0) -> dict[str, Any]:
    """{'status', 'failed': [case ids], 'files', 'seconds', 'workers'} for `files` in `root`."""
    from creator import codetrust as CT
    from creator import testrun as T
    tg = [f for f in files if (Path(root) / f).is_file()]
    if not tg:
        return {"status": "NO_TESTS", "failed": [], "files": [], "seconds": 0.0, "workers": 0}
    k = workers or CT.free_cores()
    args = ["--continue-on-collection-errors"] + (["-p", "creator.safety"] if (Path(root) / "creator" / "safety.py").is_file() else [])
    env: dict[str, str] = {}
    if k > 1 and len(tg) > 1 and (CT.PYLIB / "xdist").is_dir():
        args += ["-p", "xdist", "-n", str(k)]
        env["PYTHONPATH"] = f"{root}{os.pathsep}{CT.PYLIB}"
    else:
        k = 1
    run = T.run_pytest(root, tg, junit, label="protected_suite", tree=str(root),
                       config=T.PytestConfig(python=python, timeout=timeout, extra_args=args, env=env))
    failed = sorted(cid for cid, c in run.cases.items() if c.outcome.bad)
    return {"status": run.status.value, "failed": failed, "files": tg, "seconds": round(run.seconds, 1), "workers": k,
            "cases": len(run.cases), "problems": list(run.problems)[:5]}


def tree_key(repo: Path, rev: str, files: Sequence[str], python: str) -> str:
    from creator import sandbox as S
    tree = S.git(repo, "rev-parse", f"{rev}^{{tree}}").stdout.strip()
    return hashlib.sha256("\n".join([tree, python, *files]).encode()).hexdigest()[:24]


def _cache(state: Path) -> dict[str, Any]:
    try:
        x = json.loads((Path(state) / SUITE_CACHE).read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except (OSError, ValueError):
        return {}


def _cache_put(state: Path, key: str, val: Mapping[str, Any]) -> None:
    c = _cache(state)
    c[key] = dict(val)
    f = Path(state) / SUITE_CACHE
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(list(c.items())[-200:]), indent=1), encoding="utf-8")
    tmp.replace(f)


def _ok(r: Mapping[str, Any]) -> bool:
    return r.get("status") in ("PASSED", "FAILED", "NO_TESTS")    # a run that produced a result (TIMEOUT/CRASHED did not)


def check_tree(cfg: Any, root: Path, rev: str, files: Sequence[str], label: str, runner: Any = None) -> dict[str, Any]:
    """The protected suite on the committed tree `rev` checked out at `root`: served from the tree-hash cache, otherwise run (xdist),
    failing files re-run serially (a failure must reproduce), a run without a result retried serially. Cached when it has a result."""
    pol = policy(cfg.state)
    py = cfg.pytest.python
    key = tree_key(cfg.repo, rev, files, py)
    hit = _cache(cfg.state).get(key)
    if hit is not None:
        return dict(hit, cached=True)
    run = runner or run_suite
    jd = Path(cfg.state) / "evidence" / "safety" / label
    r = run(root, files, jd / "suite.xml", pol.suite_timeout, py)
    if not _ok(r):
        r = dict(run(root, files, jd / "suite_serial.xml", pol.suite_timeout, py, 1), first=r.get("status"))
    if r.get("failed") and _ok(r):
        again = run(root, sorted({c.split("::", 1)[0] for c in r["failed"]}), jd / "suite_rerun.xml", pol.suite_timeout, py, 1)
        r = dict(r, failed=sorted(set(r["failed"]) & set(again.get("failed", []))) if _ok(again) else r["failed"],
                 rerun=again.get("failed", []))
    r = dict(r, key=key, rev=rev, at=time.time())
    if _ok(r):
        _cache_put(cfg.state, key, r)
    return dict(r, cached=False)


def suite_failure(cfg: Any, merge_commit: str, rep: Any, runner: Any = None) -> str:
    """The post-merge verdict; a check that itself breaks is a failure too (the merge is reverted, never kept unchecked)."""
    try:
        return _suite_failure(cfg, merge_commit, rep, runner)
    except Exception as e:                                             # noqa: BLE001
        return f"protected suite check crashed after the merge: {type(e).__name__}: {e}"[:500]


def _suite_failure(cfg: Any, merge_commit: str, rep: Any, runner: Any = None) -> str:
    """Why the merge must be reverted ('' = keep it): the protected suite on main after the merge has a confirmed failure, or no
    result at all, that the merge's parent tree did not have. The parent is served from the cache (it is the previous main) or, only
    when something failed, its failing files are run in a sandbox of the parent."""
    pol = policy(cfg.state)
    info = rep.details.setdefault("safety", {})
    if not pol.full_suite:
        info["suite"] = {"skipped": "policy full_suite=false"}
        return ""
    files = suite_files()
    after = check_tree(cfg, cfg.repo, merge_commit, files, f"{rep.package}_after", runner)
    info["suite"] = {k: after.get(k) for k in ("status", "failed", "cases", "seconds", "workers", "cached", "files")}
    if not _ok(after):
        return f"protected suite gave no result after the merge ({after.get('status')}: {after.get('problems')})"
    if not after.get("failed"):
        return ""
    from creator import sandbox as S
    parent = S.git(cfg.repo, "rev-parse", f"{merge_commit}^1").stdout.strip()
    base = _cache(cfg.state).get(tree_key(cfg.repo, parent, files, cfg.pytest.python))
    if base is None:                                                   # unknown parent: run only the failing files there
        sb = S.Sandbox.open(cfg.repo, parent, cfg.scratch, label="suitebase", omit=cfg.omit)
        try:
            base = (runner or run_suite)(sb.path, sorted({c.split("::", 1)[0] for c in after["failed"]}),
                                         Path(cfg.state) / "evidence" / "safety" / f"{rep.package}_parent" / "suite.xml",
                                         pol.suite_timeout, cfg.pytest.python, 1)
        finally:
            S.discard(sb)
    new = sorted(set(after["failed"]) - set(base.get("failed") or []))
    info["suite"]["pre_existing"] = sorted(set(after["failed"]) - set(new))
    if new:
        return f"protected suite failed after the merge: {new[:8]}" + (f" (+{len(new) - 8} more)" if len(new) > 8 else "")
    return ""


# ------------------------------------------------------------------------------------------------ the rollback path
def roll_back(cfg: Any, led: Any, plan: Any, merge_commit: str, why: str) -> str:
    """Revert an adopted merge (moved here unchanged from kernel.execute 7 VERIFY): revert commit, ROLLBACK decision, Failure +
    Diagnosis + Repair (CR204: a rollback is never silent), the planner outcome. The kernel's event record starts the cool-down."""
    from creator import model as M
    from creator import planner as P
    from creator import sandbox as S
    revert = S.rollback(cfg.repo, merge_commit, why)
    led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=plan.experiment_id, verdict=M.DecisionVerdict.ROLLBACK,
                          reason=f"{why}; reverted by {revert[:12]}"))
    fid = led.append(M.Failure(created_by=M.Role.DEBUGGER, parents=(plan.work_package_id,), subject_id=plan.experiment_id,
                               symptom=f"post-merge rollback: {why}"[:500], classification="post-merge rollback",
                               reproduction=f"merge {merge_commit[:12]}, re-assess main"))
    did = led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(fid,), failure_id=fid, hypotheses=(why[:300],),
                                 root_cause=why[:500], uncertainty=M.Uncertainty.LIKELY))
    led.append(M.Repair(created_by=M.Role.KERNEL, parents=(did,), diagnosis_id=did,
                        description=f"merge {merge_commit[:12]} reverted by {revert[:12]}"))
    P.record_outcome(led, plan, False, f"rolled back: {why}")
    return revert


def status(state: Path, now: Optional[float] = None) -> dict[str, Any]:
    """The safety state right now (for the digest): policy, whether an unsupervised adoption may go, per-worker trust gate."""
    pol = policy(state)
    out: dict[str, Any] = {"policy": dataclasses.asdict(pol), "rate": rate(state, pol, now) or "open", "trust": {}}
    from creator import codetrust as CT
    for by, src in pol.trust_records.items():
        rep = CT.report(CT._jsonl(Path(src)))
        out["trust"][by] = {c: ("open" if v["open"] else "; ".join(v["why_not"])) for c, v in rep["classes"].items()}
    return out
