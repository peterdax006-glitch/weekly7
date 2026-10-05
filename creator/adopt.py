"""P1.3 build -> verify -> adopt with a REQUIRED benchmark, the autonomy ladder per change class, and the P1.4 goal driver
(MASTER_BLUEPRINT 7.5-7.8, 9.1). Loaded on demand; standard library at module level.

    BUILD    the team pipeline: SPEC -> LOCATE -> PLAN -> CODE (a diff) -> static checks + affected tests -> debug loop (<= K rounds)
             -> BENCH (the builder's benchmark spec for the claimed saving). A diff that fails creator.tools.policy.check_diff is not built on.
    VERIFY   protected, never written by the builder: the full protected tests, the codetrust gate (creator.safety.trust), the benchmark
             run by THIS module before/after (builder gives the workload spec only; timing, noise and thresholds live here), and every
             tracked metric before/after. judge() is a pure function of that evidence.
    ADOPT    only if tests pass, no tracked metric is worse beyond its noise, and the measured saving >= HIT_RATIO (50%) of the predicted
             one; a smaller saving is adopted only if it still pays back within PAYBACK_MAX_DAYS, and the miss is recorded. Then the
             autonomy level of the change class decides: A0 = proposal only. After the merge the post-merge suite runs; a failure reverts
             (auto-revert), records ROLLED_BACK (which starts creator.safety's cool-down) and drops the class one level.
    LEARN    every finished candidate appends predicted-vs-measured to state/adopt/history.jsonl (the estimator's calibration data,
             and the evidence for promoting a class up the ladder).

Autonomy levels (state/adopt/autonomy.json, owner/teacher-owned): A0 propose, A1 tools/caches/scheduler, A2 slots/data/GPU settings,
A3 the improvement engine itself; 'measuring' code is never changed. Everything starts at A0. earn() promotes A0->A1 / A1->A2 by the
measured criteria; A3 needs the owner's sign-off in the file. A post-merge regression or gate failure demotes one level with a cool-down."""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

HIT_RATIO = 0.5                       # measured saving must reach this share of the predicted one ...
PAYBACK_MAX_DAYS = 3.0                # ... else the change must still pay back within this many days (blueprint 7.3)
LEVELS = ("A0", "A1", "A2", "A3")
CLASS_NEEDS = {"tool": 1, "cache": 1, "scheduler": 1, "slot": 2, "data": 2, "gpu_setting": 2, "engine": 3}   # 'measuring': never
A1_ADOPTIONS = 30                     # blueprint 9.1: verified adoptions, 0 regressions, estimator within 50%
A1_ESTIMATOR_ERR = 0.5
A2_ECE = 0.10
A3_DAYS = 90.0
DEMOTE_COOLDOWN_H = 24.0
DIR = "adopt"


# ------------------------------------------------------------------------------------------------ change classes and the ladder levels
def change_class(paths: Sequence[str]) -> str:
    """The most demanding class among the changed paths: measuring (protected: never) > engine > slot/data/gpu_setting > scheduler > cache > tool."""
    from creator import sandbox as S
    order = ["tool", "cache", "scheduler", "gpu_setting", "data", "slot", "engine", "measuring"]
    got = "tool"
    for raw in paths:
        p = raw.replace("\\", "/")
        if S.is_protected(p):
            c = "measuring"
        elif p.startswith("creator/lm") or "train" in p.rsplit("/", 1)[-1]:
            c = "data"
        elif "gpu" in p.rsplit("/", 1)[-1]:
            c = "gpu_setting"
        elif p.startswith("creator/") and p.rsplit("/", 1)[-1] in ("kernel.py", "decide.py", "gaps.py", "safety.py", "adopt.py", "team.py",
                                                                    "planner.py", "codetrust.py", "constraints.py"):
            c = "engine"
        elif "sched" in p.rsplit("/", 1)[-1]:
            c = "scheduler"
        elif "cache" in p.rsplit("/", 1)[-1]:
            c = "cache"
        else:
            c = "tool"
        if order.index(c) > order.index(got):
            got = c
    return got


def _adir(state: Path) -> Path:
    return Path(state) / DIR


def _jsonl(f: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if f.is_file():
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if isinstance(x, dict):
                out.append(x)
    return out


def _append(f: Path, row: Mapping[str, Any]) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def _levels(state: Path) -> dict[str, Any]:
    try:
        x = json.loads((_adir(state) / "autonomy.json").read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except (OSError, ValueError):
        return {}


def _put_levels(state: Path, d: Mapping[str, Any]) -> None:
    f = _adir(state) / "autonomy.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(f)


def level(state: Path, cls: str) -> int:
    """The class's autonomy level 0..3 (A0 when unknown or unreadable: the strict default)."""
    if cls == "measuring":
        return -1
    return max(0, min(3, int((_levels(state).get(cls) or {}).get("level", 0))))


def cooldown_left(state: Path, cls: str, now: float) -> float:
    return max(0.0, float((_levels(state).get(cls) or {}).get("cooldown_until", 0.0)) - now)


def grant(state: Path, cls: str, lvl: int, why: str, now: Optional[float] = None) -> None:
    """Owner/teacher (or earn()) sets a class level. 'measuring' can never be granted."""
    if cls == "measuring":
        raise ValueError("measuring code is never self-changed")
    d = _levels(state)
    row = dict(d.get(cls) or {})
    row.update(level=int(lvl), why=why, since=time.time() if now is None else now)
    d[cls] = row
    _put_levels(state, d)


def demote(state: Path, cls: str, why: str, now: float, cooldown_h: float = DEMOTE_COOLDOWN_H) -> int:
    """A regression or gate failure: one level down and a cool-down. Returns the new level."""
    d = _levels(state)
    row = dict(d.get(cls) or {})
    new = max(0, int(row.get("level", 0)) - 1)
    row.update(level=new, why=f"demoted: {why}"[:300], since=now, cooldown_until=now + cooldown_h * 3600)
    d[cls] = row
    _put_levels(state, d)
    return new


def history(state: Path) -> list[dict[str, Any]]:
    return _jsonl(_adir(state) / "history.jsonl")


def estimator_error(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Median |measured - predicted| / predicted per-event saving over finished candidates (None without data)."""
    errs = [abs(float(r["measured"]) - float(r["predicted"])) / float(r["predicted"]) for r in rows
            if r.get("predicted") and r.get("measured") is not None and float(r["predicted"]) > 0]
    return statistics.median(errs) if errs else None


def earn(state: Path, cls: str, *, gate_open: bool = False, ece: Optional[float] = None, now: Optional[float] = None) -> int:
    """Promote `cls` by the blueprint's measured criteria; returns the (possibly new) level. Never above A2 here (A3 is the owner's
    sign-off: a 'signoff' key in the class row), never during a cool-down, never for 'measuring'."""
    t = time.time() if now is None else now
    if cls == "measuring" or cooldown_left(state, cls, t) > 0:
        return level(state, cls)
    cur = level(state, cls)
    rows = [r for r in history(state) if r.get("cls") == cls]
    ok = [r for r in rows if r.get("outcome") == "ADOPTED"]
    bad = [r for r in rows if r.get("outcome") == "ROLLED_BACK" or r.get("regression")]
    if cur == 0 and len(ok) >= A1_ADOPTIONS and not bad and (estimator_error(ok) or 1.0) <= A1_ESTIMATOR_ERR:
        grant(state, cls, 1, f"earned A1: {len(ok)} verified adoptions, 0 regressions, estimator error {estimator_error(ok):.2f}", t)
        return 1
    if cur == 1 and gate_open and ece is not None and ece <= A2_ECE and not bad:
        grant(state, cls, 2, f"earned A2: trust-gate class open, ECE {ece:.3f}", t)
        return 2
    row = _levels(state).get(cls) or {}
    if cur == 2 and row.get("signoff") and not bad and t - float(row.get("since", t)) >= A3_DAYS * 86400:
        grant(state, cls, 3, "A3: 90 days at A2, no unresolved regression, owner sign-off", t)
        _d = _levels(state)
        _d[cls]["signoff"] = row["signoff"]
        _put_levels(state, _d)
        return 3
    return cur


# ------------------------------------------------------------------------------------------------ VERIFY (pure judgement)
def _med(xs: Sequence[float]) -> float:
    return float(statistics.median(xs))


def _noise(xs: Sequence[float]) -> float:
    m = _med(xs)
    return 1.4826 * float(statistics.median([abs(x - m) for x in xs]))


def measure(before: Sequence[float], after: Sequence[float]) -> dict[str, Any]:
    """Per-event saving (cost units; lower cost = better) with its noise. A saving inside the noise counts as none."""
    if not before or not after:
        return {"saving": 0.0, "noise": 0.0, "significant": False, "before": _med(before) if before else None,
                "after": _med(after) if after else None}
    b, a = _med(before), _med(after)
    noise = max(_noise(before), _noise(after))
    s = b - a
    return {"before": b, "after": a, "saving": s if s > noise else 0.0, "raw_saving": s, "noise": noise, "significant": s > noise,
            "n": [len(before), len(after)]}


def regressions(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Tracked metrics ({'value', 'noise', 'better': 'higher'|'lower'}) that got worse by more than their noise (a metric that vanished too)."""
    out: list[str] = []
    for name, b in before.items():
        a = after.get(name)
        if a is None:
            out.append(f"{name}: tracked metric missing after the change")
            continue
        worse = float(b["value"]) - float(a["value"]) if b.get("better", "higher") == "higher" else float(a["value"]) - float(b["value"])
        noise = max(float(b.get("noise", 0.0)), float(a.get("noise", 0.0)))
        if worse > noise:
            out.append(f"{name}: {b['value']} -> {a['value']} (worse by {worse:.4g}, noise {noise:.4g})")
    return out


@dataclass
class Verdict:
    adopt: bool
    reasons: list[str] = field(default_factory=list)
    saving: float = 0.0               # measured per-event saving
    ratio: Optional[float] = None     # measured / predicted
    miss: bool = False                # measured < HIT_RATIO of predicted (recorded; adopted only if still payback-positive)
    payback_days: Optional[float] = None
    regressed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def judge(cand: Mapping[str, Any], *, tests_ok: bool, tests_why: str = "", trust_ok: bool, trust_why: str = "", bench: Optional[Mapping[str, Any]],
          metrics_before: Mapping[str, Mapping[str, Any]], metrics_after: Mapping[str, Mapping[str, Any]]) -> Verdict:
    """The protected verdict. cand: predicted (per-event saving), f (events/day), B (build cost, same units as the saving x events)."""
    v = Verdict(False)
    if bench is None:
        v.reasons.append("no benchmark artifact: nothing measures the claimed saving")
    if not tests_ok:
        v.reasons.append(f"protected tests failed: {tests_why}"[:300])
    if not trust_ok:
        v.reasons.append(f"codetrust gate closed: {trust_why}"[:300])
    v.regressed = regressions(metrics_before, metrics_after)
    v.reasons += [f"tracked metric regressed: {r}" for r in v.regressed]
    if bench is None or v.reasons:
        return v
    m = measure(bench["before"], bench["after"])
    v.saving = float(m["saving"])
    pred = float(cand.get("predicted", 0.0))
    v.ratio = v.saving / pred if pred > 0 else None
    v.miss = v.ratio is None or v.ratio < HIT_RATIO
    if v.saving <= 0:
        v.reasons.append(f"measured saving is within the noise ({m['raw_saving']:.4g} vs noise {m['noise']:.4g}): nothing was saved")
        return v
    per_day = float(cand.get("f", 0.0)) * v.saving
    v.payback_days = float(cand.get("B", 0.0)) / per_day if per_day > 0 else None
    if v.miss:
        if v.payback_days is not None and v.payback_days <= PAYBACK_MAX_DAYS:
            v.reasons.append(f"MISS recorded: measured {v.saving:.4g} is {0 if v.ratio is None else v.ratio:.0%} of predicted {pred:.4g}, "
                             f"adopted because it still pays back in {v.payback_days:.2f} days")
        else:
            v.reasons.append(f"measured saving {v.saving:.4g} is {0 if v.ratio is None else v.ratio:.0%} of predicted {pred:.4g} (< "
                             f"{HIT_RATIO:.0%}) and the payback {v.payback_days if v.payback_days is None else round(v.payback_days, 2)} days "
                             f"exceeds {PAYBACK_MAX_DAYS:g}")
            return v
    v.adopt = True
    return v


def adopt_gate(state: Path, paths: Sequence[str], by: str, now: Optional[float] = None) -> tuple[bool, str]:
    """DECIDE's question before any build time is spent (creator/decide.py is protected, so the gate lives here): the class of the changed
    paths (measuring = never), its autonomy level (below the level the change needs it may only PROPOSE), the class cool-down, and for an
    unsupervised worker the h70 rate limit. Only restricts; any failure is a refusal."""
    try:
        from creator import safety as SF
        t = time.time() if now is None else now
        cls = change_class(paths)
        if cls == "measuring":
            return False, "measuring code is never self-changed"
        lvl, need = level(state, cls), CLASS_NEEDS.get(cls, 3)
        if cooldown_left(state, cls, t) > 0:
            return False, f"class {cls} is cooling down after a demotion"
        pol = SF.policy(state)
        why = "" if SF.supervised(pol, by) else SF.rate(state, pol, t)
        if why:
            return False, f"safety rate limit: {why}"
        return (lvl >= need), f"class {cls} at A{lvl}, change needs A{need}" + ("" if lvl >= need else ": proposal only")
    except Exception as e:                                             # noqa: BLE001
        return False, f"adopt gate could not decide ({type(e).__name__}): refused"


# ------------------------------------------------------------------------------------------------ the cycle
@dataclass
class Env:
    """Everything the cycle touches, injected: fakes in tests, the kernel's sandbox/merge/suite in production."""
    state: Path                                           # state/creator
    team: Any                                             # creator.team.Team with CHECKER / CODER / THINKER / locate actors
    by: str                                               # the worker's name (trust gate, rate limit)
    check: Callable[[str], Mapping[str, Any]]             # patch -> {'ok', 'why'}: apply in the candidate sandbox + static checks + affected tests
    bench: Callable[[Mapping[str, Any], str], list[float]]   # (spec, 'before'|'after') -> cost samples per event; run by VERIFY, not the builder
    protected: Callable[[], Mapping[str, Any]]            # full protected tests on the candidate: {'status', 'failed'}
    tracked: Callable[[str], Mapping[str, Mapping[str, Any]]]   # 'before'|'after' -> tracked metrics
    merge: Callable[[str, Mapping[str, Any]], str]        # (patch, candidate) -> merge commit
    revert: Callable[[str, str], str]                     # (commit, why) -> revert commit
    post_merge: Callable[[str], str] = lambda commit: ""  # '' or the reason the merged tree is bad (the protected suite after the merge)
    clock: Callable[[], float] = time.time
    k_debug: int = 3


def _event(env: Env, goal: str, step: str, t0: float, outcome: str, **extra: Any) -> None:
    try:
        from creator import slowpath as SP
        SP.event("adopt", goal_id=goal, step=step, wall_s=time.perf_counter() - t0, outcome=outcome, state=env.state, **extra)
    except Exception:                                       # noqa: BLE001 - measurement never breaks the work
        pass


def _paths(diff: str) -> list[str]:
    from creator.tools import policy as PO
    return [f[0] for f in PO._diff_files(diff)]


def build(env: Env, cand: Mapping[str, Any], rung: str = "start") -> dict[str, Any]:
    """The team pipeline. Returns {'ok', 'patch', 'bench_spec', 'why', 'rounds', 'ids'}; no patch is returned unless the checks passed."""
    from creator import team as TM
    from creator.tools import policy as PO
    gid = str(cand["id"])
    env.team.set_goal(gid, str(cand.get("title", "")))
    task = {k: cand[k] for k in ("id", "title", "claim") if k in cand}
    ids = {"task": env.team.board.put(gid, "task", json.dumps(task, sort_keys=True))}
    cons = ([f"rung:{rung}"] if rung != "start" else []) + ([f"try:{cand['attempt']}"] if cand.get("attempt") else [])

    def send(step: str, inputs: list[str], prior: str = "", extra: Sequence[str] = ()) -> str:
        out = env.team.dispatch(TM.Envelope(gid, step, inputs=inputs, constraints=cons + list(extra), prior=prior,
                                            success_test=str(cand.get("claim", ""))[:60] if step in ("CODE", "DEBUG_FIX", "BENCH") else ""))
        ids[step.lower()] = env.team.board.put(gid, step.lower(), out)
        return out

    t0 = time.perf_counter()
    send("SPEC", [ids["task"]])
    where = send("LOCATE", [ids["task"], ids["spec"]])
    ranges = [r for r in where.splitlines() if TM._RANGE.match(r)][:2]
    send("PLAN", [ids["task"], ids["spec"]] + ranges)
    patch = send("CODE", [ids["task"], ids["plan"]] + ranges)
    rounds = 0
    why = ""
    ok = False
    while True:
        dv = PO.check_diff(patch)
        if not dv.ok:
            why = "diff refused by policy: " + "; ".join(dv.reasons)[:200]
        else:
            res = env.check(patch)
            ok, why = bool(res.get("ok")), str(res.get("why", ""))
        if ok or rounds >= env.k_debug:
            break
        rounds += 1                                                        # the debug loop: pinpoint (code tool) -> fix (CODER), K rounds
        err = env.team.board.put(gid, "failure", why[:400])
        send("DEBUG_PINPOINT", [ids["task"], err])
        patch = send("DEBUG_FIX", [ids["task"], ids["debug_pinpoint"], err] + ranges, prior=f"round {rounds}: {why[:60]}")
    _event(env, gid, "build", t0, "ok" if ok else "failed", rounds=rounds)
    if not ok:
        return {"ok": False, "why": why, "rounds": rounds, "ids": ids}
    spec_txt = send("BENCH", [ids["task"], ids["plan"], env.team.board.put(gid, "patch", patch)])
    try:
        spec = json.loads(spec_txt)
        spec = spec if isinstance(spec, dict) and spec else None
    except ValueError:
        spec = None
    return {"ok": True, "patch": patch, "bench_spec": spec, "rounds": rounds, "ids": ids, "why": ""}


def _history(env: Env, cand: Mapping[str, Any], cls: str, outcome: str, v: Optional[Verdict], extra: Optional[Mapping[str, Any]] = None) -> None:
    _append(_adir(env.state) / "history.jsonl",
            {"at": env.clock(), "id": cand["id"], "cls": cls, "outcome": outcome, "predicted": cand.get("predicted"),
             "measured": None if v is None else v.saving, "ratio": None if v is None else v.ratio, "miss": bool(v and v.miss),
             "regression": bool(v and v.regressed), **dict(extra or {})})


def run_candidate(env: Env, cand: Mapping[str, Any], rung: str = "start") -> dict[str, Any]:
    """One candidate improvement through BUILD -> VERIFY -> ADOPT. Outcome: ADOPTED | PROPOSED (class at A0) | REFUSED | BUILD_FAILED |
    ROLLED_BACK | DEFERRED (rate limit / cool-down: not an attempt). 'progress' = measured saving per day (None when nothing was measured)."""
    from creator import safety as SF
    t_all = time.perf_counter()
    cid = str(cand["id"])
    now = env.clock()
    pol = SF.policy(env.state)
    sup = SF.supervised(pol, env.by)
    declared = str(cand.get("cls", "tool"))
    out: dict[str, Any] = {"id": cid, "cls": declared, "rung": rung}

    def finish(outcome: str, reason: str, v: Optional[Verdict] = None, **kw: Any) -> dict[str, Any]:
        out.update(outcome=outcome, reason=reason, verdict=None if v is None else v.to_dict(), **kw)
        out.setdefault("progress", (v.saving * float(cand.get("f", 0.0))) if v is not None and v.saving > 0 and outcome in ("ADOPTED", "PROPOSED") else None)
        refused = {"REFUSED": "verify", "BUILD_FAILED": "build"}.get(outcome)
        SF.record(env.state, {"at": now, "outcome": outcome, "package": cid, "reason": reason[:500], "by": env.by, "supervised": sup,
                              **({"refused": refused} if refused else {}), "cls": out["cls"], "level": f"A{max(level(env.state, out['cls']), 0)}"})
        if outcome != "DEFERRED":
            _history(env, cand, out["cls"], outcome, v)
        _event(env, cid, "cycle", t_all, outcome)
        return out

    if declared == "measuring" or level(env.state, declared) < 0:
        return finish("REFUSED", "measuring code is never self-changed (autonomy ladder: never)")
    wait = cooldown_left(env.state, declared, now)
    if wait > 0:
        return finish("DEFERRED", f"class {declared} is in its cool-down after a demotion ({wait / 3600:.1f} h left)")
    if not sup:
        why = SF.rate(env.state, pol, now)
        if why:
            return finish("DEFERRED", f"safety rate limit: {why}")
    b = build(env, cand, rung)
    out["rounds"] = b["rounds"]
    if not b["ok"]:
        return finish("BUILD_FAILED", b["why"])
    patch = b["patch"]
    paths = _paths(patch)
    cls = change_class(paths)                                                # the diff decides the class, not the builder's claim
    out["cls"] = cls
    if cls == "measuring":
        return finish("REFUSED", "the diff touches protected measuring code (autonomy ladder: never)")
    if b["bench_spec"] is None:
        return finish("REFUSED", "builder produced no benchmark spec: the claimed saving cannot be measured")
    lvl = level(env.state, cls)
    wait = cooldown_left(env.state, cls, now)
    if wait > 0:
        return finish("DEFERRED", f"class {cls} is in its cool-down after a demotion ({wait / 3600:.1f} h left)")
    if sup:
        trust_ok, trust_why = True, "supervised change"
    else:
        trust_ok, trust_why = SF.trust(pol, env.by, paths, str(cand.get("title", "")))
    t0 = time.perf_counter()
    bench = {"spec": b["bench_spec"], "before": list(env.bench(b["bench_spec"], "before")), "after": list(env.bench(b["bench_spec"], "after"))}
    _event(env, cid, "bench", t0, "ok", samples=len(bench["before"]) + len(bench["after"]))
    tests = env.protected()
    mb, ma = env.tracked("before"), env.tracked("after")
    v = judge(cand, tests_ok=tests.get("status") in ("PASSED", "NO_TESTS") and not tests.get("failed"), tests_why=str(tests.get("failed") or tests.get("status")),
              trust_ok=trust_ok, trust_why=trust_why, bench=bench, metrics_before=mb, metrics_after=ma)
    art = _adir(env.state) / cid
    art.mkdir(parents=True, exist_ok=True)
    (art / "candidate.diff").write_text(patch, encoding="utf-8")
    (art / "benchmark.json").write_text(json.dumps({**bench, "measure": measure(bench["before"], bench["after"]), "predicted": cand.get("predicted")},
                                                   indent=1, sort_keys=True), encoding="utf-8")
    (art / "verdict.json").write_text(json.dumps({"cls": cls, "level": lvl, "verdict": v.to_dict(), "tests": dict(tests)}, indent=1, sort_keys=True,
                                                 default=str), encoding="utf-8")
    out["artifact"] = str(art)
    if not v.adopt:
        return finish("REFUSED", "; ".join(v.reasons), v)
    if lvl < CLASS_NEEDS.get(cls, 3):
        return finish("PROPOSED", f"class {cls} is at A{lvl}, needs A{CLASS_NEEDS.get(cls, 3)} to adopt: proposal + benchmark saved "
                      + "; ".join(v.reasons), v)
    commit = env.merge(patch, cand)
    bad = env.post_merge(commit)
    if bad:
        env.revert(commit, bad)
        newl = demote(env.state, cls, bad, now)
        return finish("ROLLED_BACK", f"post-merge check failed, reverted; class {cls} demoted to A{newl}: {bad}"[:500], v, merge=commit)
    return finish("ADOPTED", "; ".join(v.reasons) or "tests pass, no metric regressed, saving at or above 50% of predicted", v, merge=commit,
                  after_metrics=dict(ma))


def monitor(env: Env, adopted: Mapping[str, Any], baseline: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """Later regression watch (auto-revert): compare the tracked metrics now with the baseline taken at adoption; on a regression revert
    the merge, record ROLLED_BACK (cool-down) and demote the class. Returns the revert commit or None."""
    from creator import safety as SF
    bad = regressions(baseline, env.tracked("after"))
    if not bad:
        return None
    why = "later regression: " + "; ".join(bad)
    rev = env.revert(str(adopted["merge"]), why)
    now = env.clock()
    SF.record(env.state, {"at": now, "outcome": "ROLLED_BACK", "package": adopted["id"], "reason": why[:500], "by": env.by, "regression": why[:300]})
    _history(env, adopted, str(adopted["cls"]), "ROLLED_BACK", None, {"regression": True})
    demote(env.state, str(adopted["cls"]), why, now)
    return rev


# ------------------------------------------------------------------------------------------------ the goal driver (P1.4)
def run_goal(env: Env, goal: str, make_candidate: Callable[[str], Optional[Mapping[str, Any]]], fingerprints: Callable[[], Mapping[str, str]],
             kind: str = "", deps: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """One engine cycle for `goal`: re-try it if a parked dependency changed, skip it while parked, else build a candidate at the current
    ladder rung and feed the measured progress to the stuck detector (creator.gaps ladder). Returns the candidate result + the ladder action."""
    from creator import gaps as G
    now = env.clock()
    G.ladder_due(env.state, fingerprints(), now)
    rung = G.ladder_rung(env.state, goal)
    if (G._lad_load(env.state).get(goal) or {}).get("status") == "parked":
        return {"id": goal, "outcome": "PARKED", "ladder": {"action": "parked"}}
    cand = make_candidate(rung)
    if cand is None:
        return {"id": goal, "outcome": "NO_CANDIDATE", "ladder": G.ladder_note(env.state, goal, None, now, kind=kind, deps=deps or fingerprints())}
    res = run_candidate(env, cand, rung)
    if res["outcome"] == "DEFERRED":                       # not an attempt: it must not count as a stuck cycle
        return dict(res, ladder={"action": "continue", "reason": "deferred"})
    return dict(res, ladder=G.ladder_note(env.state, goal, res.get("progress"), now, kind=kind, deps=deps or fingerprints()))


def run_bench_command(root: Path, argv: Sequence[str], repeats: int = 5, timeout: float = 120.0) -> list[float]:
    """Run a benchmark command in `root` `repeats` times; it prints one JSON number (cost per event) on its last line. Used by the real
    wiring; the process runs at the caller's priority (the kernel starts it under scripts/lowprio.py)."""
    import subprocess
    out: list[float] = []
    for _ in range(repeats):
        r = subprocess.run(list(argv), cwd=str(root), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"benchmark failed: {r.stderr[-300:]}")
        out.append(float(r.stdout.strip().splitlines()[-1]))
    return out
