"""P1.3 / P1.4: build -> verify -> adopt with a required benchmark, the autonomy ladder, the fallback ladder + parking. FAKE builder actors
(no model), fake sandbox/merge; the safety loop (creator.safety) and the ladder (creator.gaps) are the real ones on a temporary state dir."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from creator import adopt as A
from creator import gaps as G
from creator import safety as SF
from creator import team as TM
from creator.tools import policy as PO

REAL_SAFETY = True                                  # tests/conftest.py: run the real safety loop (trust gate, rate limit, cool-down)
DIFF = ("diff --git a/creator/tools/fastfoo.py b/creator/tools/fastfoo.py\n--- a/creator/tools/fastfoo.py\n+++ b/creator/tools/fastfoo.py\n"
        "@@ -1,2 +1,2 @@\n def foo(xs):\n-    return sorted(xs)[0]\n+    return min(xs)\n")
BENCH_SPEC = {"argv": ["python", "bench_foo.py"], "unit": "cpu_ms/event", "claim": "foo is faster"}
BEFORE = [100.0, 101.0, 99.0, 100.0, 100.5]


def _records(path: Path) -> Path:
    rows = [{"cls": c, "passed": True, "confidence": 0.97, "task": f"{c}{i}"}
            for c in ("tests_only", "docs", "refactor", "bugfix", "feature") for i in range(30)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class World:
    """The fake builder + fake repo: scripted behaviour per test."""

    def __init__(self) -> None:
        self.patch = DIFF
        self.bench_spec: Any = dict(BENCH_SPEC)
        self.bench = {"before": list(BEFORE), "after": [40.0, 41.0, 39.0, 40.0, 40.5]}
        self.check_results: list[dict[str, Any]] = []
        self.protected = {"status": "PASSED", "failed": []}
        self.metrics_before = {"pass_rate": {"value": 0.99, "noise": 0.005, "better": "higher"}}
        self.metrics_after = {"pass_rate": {"value": 0.99, "noise": 0.005, "better": "higher"}}
        self.post_merge = ""
        self.merged: list[str] = []
        self.reverted: list[str] = []
        self.constraints: list[list[str]] = []
        self.coder: Any = None                      # optional fn(constraints) -> patch text
        self.t = 1_000_000.0


def make_env(tmp_path: Path, w: World, by: str = "fake-coder", trusted: bool = True) -> A.Env:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    pol: dict[str, Any] = {"full_suite": False, "supervised": [], "max_adoptions_per_hour": 50}
    if trusted:
        pol["trust_records"] = {by: str(_records(tmp_path / "recs.jsonl"))}
    (state / SF.POLICY_FILE).write_text(json.dumps(pol), encoding="utf-8")

    def checker(env: TM.Envelope, team: TM.Team) -> str:
        return "spec: min instead of sorted"

    def thinker(env: TM.Envelope, team: TM.Team) -> str:
        return "plan: replace sorted()[0] by min()"

    def locate(env: TM.Envelope, team: TM.Team) -> str:
        return "creator/tools/fastfoo.py:1-2"

    def coder(env: TM.Envelope, team: TM.Team) -> str:
        w.constraints.append(list(env.constraints))
        if env.step == "BENCH":
            return json.dumps(w.bench_spec) if w.bench_spec is not None else "no benchmark"
        return w.coder(env.constraints) if w.coder else w.patch

    team = TM.Team(tmp_path / "team", [TM.actor("CHECKER", checker), TM.actor("THINKER", thinker), TM.actor("locate", locate),
                                      TM.actor("CODER", coder)])

    def check(patch: str) -> dict[str, Any]:
        return w.check_results.pop(0) if w.check_results else {"ok": True}

    def merge(patch: str, cand: Any) -> str:
        w.merged.append(cand["id"])
        return f"commit-{cand['id']}"

    def revert(commit: str, why: str) -> str:
        w.reverted.append(commit)
        return "rev-" + commit

    return A.Env(state=state, team=team, by=by, check=check, bench=lambda spec, which: list(w.bench[which]), protected=lambda: w.protected,
                 tracked=lambda which: w.metrics_before if which == "before" else w.metrics_after, merge=merge, revert=revert,
                 post_merge=lambda c: w.post_merge, clock=lambda: w.t)


def cand(**kw: Any) -> dict[str, Any]:
    c = {"id": "c1", "title": "faster foo", "claim": "foo costs 50% less", "cls": "tool", "predicted": 50.0, "f": 100.0, "B": 1000.0}
    c.update(kw)
    return c


# ------------------------------------------------------------------------------------------------ P1.3
def test_good_change_is_adopted_with_its_benchmark_recorded(tmp_path: Path) -> None:
    w = World()
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "test: earned")
    r = A.run_candidate(env, cand())
    assert r["outcome"] == "ADOPTED", r
    assert w.merged == ["c1"] and not w.reverted
    bench = json.loads((env.state / "adopt" / "c1" / "benchmark.json").read_text(encoding="utf-8"))
    assert bench["before"] == BEFORE and bench["measure"]["saving"] == pytest.approx(60.0) and bench["spec"] == BENCH_SPEC
    assert (env.state / "adopt" / "c1" / "candidate.diff").read_text(encoding="utf-8") == DIFF
    assert r["verdict"]["ratio"] == pytest.approx(1.2) and not r["verdict"]["miss"]
    h = A.history(env.state)[-1]
    assert (h["outcome"], h["predicted"], h["measured"]) == ("ADOPTED", 50.0, pytest.approx(60.0))
    ev = SF.events(env.state)[-1]
    assert ev["outcome"] == "ADOPTED" and ev["supervised"] is False and ev["cls"] == "tool"
    assert r["progress"] == pytest.approx(6000.0)                      # saving/day = f x measured saving
    # the team pipeline ran every step through envelopes, within the token target
    assert env.team.stats["dispatched"] == 5 and max(env.team.stats["env_tokens"]) <= TM.ENVELOPE_TARGET_TOKENS * 2


def test_class_at_a0_only_proposes_but_still_measures(tmp_path: Path) -> None:
    w = World()
    env = make_env(tmp_path, w)
    r = A.run_candidate(env, cand())
    assert r["outcome"] == "PROPOSED" and not w.merged and "A0" in r["reason"]
    assert json.loads((env.state / "adopt" / "c1" / "benchmark.json").read_text(encoding="utf-8"))["measure"]["saving"] == pytest.approx(60.0)


def test_a_change_that_regresses_a_tracked_metric_is_refused(tmp_path: Path) -> None:
    w = World()
    w.metrics_after = {"pass_rate": {"value": 0.90, "noise": 0.005, "better": "higher"}}
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    r = A.run_candidate(env, cand())
    assert r["outcome"] == "REFUSED" and "pass_rate" in r["reason"] and not w.merged
    # inside the noise is not a regression
    w2 = World()
    w2.metrics_after = {"pass_rate": {"value": 0.987, "noise": 0.005, "better": "higher"}}
    assert A.run_candidate(make_env(tmp_path / "b", w2), cand())["outcome"] == "PROPOSED"


def test_saving_below_half_of_predicted_is_adopted_only_if_payback_positive(tmp_path: Path) -> None:
    w = World()
    w.bench["after"] = [70.0, 71.0, 69.0, 70.0, 70.5]                   # measured 30 vs predicted 50 (60%): a hit
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    assert A.run_candidate(env, cand())["outcome"] == "ADOPTED"
    w.bench["after"] = [80.0, 81.0, 79.0, 80.0, 80.5]                   # measured 20 vs 50 (40%): a miss ...
    r = A.run_candidate(env, cand(id="c2", f=1000.0, B=1000.0))         # ... payback 1000/(1000*20) = 0.05 days: adopt, record the miss
    assert r["outcome"] == "ADOPTED" and r["verdict"]["miss"] and "MISS recorded" in r["reason"]
    assert A.history(env.state)[-1]["miss"] is True
    r = A.run_candidate(env, cand(id="c3", f=1.0, B=1000.0))            # same miss, payback 50 days: refused
    assert r["outcome"] == "REFUSED" and "payback" in r["reason"] and "c3" not in w.merged


def test_no_saving_inside_the_noise_and_no_benchmark_are_refused(tmp_path: Path) -> None:
    w = World()
    w.bench["after"] = [99.0, 101.0, 100.0, 100.0, 99.5]
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    assert "within the noise" in A.run_candidate(env, cand())["reason"]
    w.bench_spec = None
    r = A.run_candidate(env, cand(id="c2"))
    assert r["outcome"] == "REFUSED" and "no benchmark" in r["reason"] and not w.merged


def test_failing_tests_and_a_closed_trust_gate_refuse(tmp_path: Path) -> None:
    w = World()
    w.protected = {"status": "FAILED", "failed": ["tests/test_x.py::t"]}
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    assert "protected tests failed" in A.run_candidate(env, cand())["reason"]
    w2 = World()
    env2 = make_env(tmp_path / "u", w2, trusted=False)
    A.grant(env2.state, "tool", 1, "t")
    r = A.run_candidate(env2, cand())
    assert r["outcome"] == "REFUSED" and "codetrust gate closed" in r["reason"] and not w2.merged


def test_debug_loop_runs_k_rounds_and_a_policy_violating_diff_is_never_built_on(tmp_path: Path) -> None:
    w = World()
    w.check_results = [{"ok": False, "why": "NameError x"}, {"ok": False, "why": "NameError y"}, {"ok": True}]
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    r = A.run_candidate(env, cand())
    assert r["outcome"] == "ADOPTED" and r["rounds"] == 2
    w2 = World()
    w2.check_results = [{"ok": False, "why": "always"}] * 10
    env2 = make_env(tmp_path / "k", w2)
    r = A.run_candidate(env2, cand())
    assert r["outcome"] == "BUILD_FAILED" and r["rounds"] == env2.k_debug and not w2.merged
    w3 = World()
    w3.patch = DIFF.replace("creator/tools/fastfoo.py", "creator/sandbox.py")           # protected path
    r = A.run_candidate(make_env(tmp_path / "p", w3), cand(id="c9"), "start")
    assert r["outcome"] == "BUILD_FAILED" and "policy" in r["reason"]


def test_measuring_code_is_never_changed_and_classes_come_from_the_diff(tmp_path: Path) -> None:
    assert A.change_class(["creator/tools/x.py"]) == "tool" and A.change_class(["creator/kernel.py", "creator/tools/x.py"]) == "engine"
    assert A.change_class(["creator/sandbox.py"]) == "measuring" and A.change_class(["creator/resultcache.py"]) == "cache"
    with pytest.raises(ValueError):
        A.grant(tmp_path, "measuring", 3, "no")
    w = World()
    w.patch = DIFF.replace("creator/tools/fastfoo.py", "creator/planner.py")          # builder claims 'tool', the diff says engine (A3)
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    r = A.run_candidate(env, cand())
    assert r["cls"] == "engine" and r["outcome"] == "PROPOSED" and not w.merged


def test_post_merge_failure_reverts_demotes_and_starts_the_cooldown(tmp_path: Path) -> None:
    w = World()
    w.post_merge = "protected suite failed after the merge: ['tests/test_gate.py::t']"
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    r = A.run_candidate(env, cand())
    assert r["outcome"] == "ROLLED_BACK" and w.reverted == ["commit-c1"] and A.level(env.state, "tool") == 0
    assert SF.rate(env.state, SF.policy(env.state), w.t + 60).startswith("cool-down")      # h70's cool-down, started by the event
    w.post_merge = ""
    A.grant(env.state, "tool", 1, "re-granted")
    w.t += 25 * 3600                                                                         # h70 cool-down over; the class cool-down too
    assert A.run_candidate(env, cand(id="c2"))["outcome"] == "ADOPTED"


def test_class_cooldown_and_later_regression_auto_revert(tmp_path: Path) -> None:
    w = World()
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    r = A.run_candidate(env, cand())
    assert r["outcome"] == "ADOPTED"
    base = r["after_metrics"]
    assert A.monitor(env, {"id": "c1", "merge": r["merge"], "cls": "tool"}, base) is None
    w.metrics_after = {"pass_rate": {"value": 0.8, "noise": 0.005, "better": "higher"}}
    assert A.monitor(env, {"id": "c1", "merge": r["merge"], "cls": "tool"}, base) == "rev-commit-c1"
    assert A.level(env.state, "tool") == 0 and A.cooldown_left(env.state, "tool", w.t) > 0
    A.grant(env.state, "tool", 1, "again")
    assert A.run_candidate(env, cand(id="c2"))["outcome"] == "DEFERRED"                    # class cool-down / h70 cool-down: not an attempt


def test_rate_limit_defers_unsupervised_adoptions(tmp_path: Path) -> None:
    w = World()
    env = make_env(tmp_path, w)
    (env.state / SF.POLICY_FILE).write_text(json.dumps({"full_suite": False, "supervised": [], "max_adoptions_per_hour": 1,
                                                       "trust_records": {"fake-coder": str(tmp_path / "recs.jsonl")}}), encoding="utf-8")
    A.grant(env.state, "tool", 1, "t")
    assert A.run_candidate(env, cand())["outcome"] == "ADOPTED"
    r = A.run_candidate(env, cand(id="c2"))
    assert r["outcome"] == "DEFERRED" and "limit" in r["reason"]


def test_autonomy_is_earned_by_measurement_and_a3_needs_the_owner(tmp_path: Path) -> None:
    st = tmp_path
    for i in range(29):
        A._append(st / "adopt" / "history.jsonl", {"cls": "cache", "outcome": "ADOPTED", "predicted": 10, "measured": 9})
    assert A.earn(st, "cache", now=1.0) == 0                                                # 29 < 30
    A._append(st / "adopt" / "history.jsonl", {"cls": "cache", "outcome": "ADOPTED", "predicted": 10, "measured": 12})
    assert A.earn(st, "cache", now=1.0) == 1
    assert A.earn(st, "cache", gate_open=True, ece=0.2, now=2.0) == 1                       # ECE too high
    assert A.earn(st, "cache", gate_open=True, ece=0.05, now=2.0) == 2
    assert A.earn(st, "cache", gate_open=True, ece=0.05, now=2.0 + 100 * 86400) == 2        # no owner sign-off: stays A2
    d = A._levels(st)
    d["cache"]["signoff"] = "owner"
    A._put_levels(st, d)
    assert A.earn(st, "cache", now=2.0 + 100 * 86400) == 3
    A._append(st / "adopt" / "history.jsonl", {"cls": "slot", "outcome": "ADOPTED", "predicted": 10, "measured": 1})
    assert A.earn(st, "slot", now=1.0) == 0 and A.level(st, "measuring") == -1
    bad = tmp_path / "bad"
    for i in range(30):
        A._append(bad / "adopt" / "history.jsonl", {"cls": "tool", "outcome": "ADOPTED", "predicted": 10, "measured": 9})
    A._append(bad / "adopt" / "history.jsonl", {"cls": "tool", "outcome": "ROLLED_BACK", "regression": True})
    assert A.earn(bad, "tool", now=1.0) == 0                                                # a regression blocks promotion


def test_the_benchmark_command_runner_and_the_diff_of_this_work_pass_policy() -> None:
    assert A.run_bench_command(Path.cwd(), [sys.executable, "-c", "print('x');print(2.5)"], repeats=2) == [2.5, 2.5]
    root = Path(__file__).resolve().parents[1]
    assert (root / "creator" / "adopt.py").is_file()


# ------------------------------------------------------------------------------------------------ P1.4
def test_a_stuck_goal_walks_the_ladder_is_parked_and_retried_when_a_dependency_changes(tmp_path: Path) -> None:
    w = World()
    world = {"dep": "v1"}

    def coder(cons: list[str]) -> str:
        return DIFF if world["dep"] == "v2" else "this is not a diff at all"            # cannot be built until the dependency changes

    w.coder = coder
    env = make_env(tmp_path, w)
    A.grant(env.state, "tool", 1, "t")
    seen: list[str] = []
    attempt = {"n": 0}

    def make(rung: str) -> dict[str, Any]:
        seen.append(rung)
        attempt["n"] += 1
        return cand(id="g1", attempt=attempt["n"])

    fp = lambda: {"model_slot": world["dep"]}                                          # noqa: E731
    actions = []
    t_start = w.t
    for _ in range(30):
        r = A.run_goal(env, "g1", make, fp, kind="skill")
        actions.append(r["ladder"]["action"])
        if r["outcome"] == "PARKED":
            break
        w.t += 120
    assert w.t - t_start < 3600                                                         # no stall longer than an hour
    assert [a for a in actions if a not in ("continue",)][:3] == ["smaller_step", "different_approach", "bigger_rung"]
    assert "park" in actions and actions[-1] == "parked"
    assert [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]] == ["start", "smaller_step", "different_approach", "bigger_rung"]
    jobs = [json.loads(x) for x in (env.state / G.JOBS_FILE).read_text(encoding="utf-8").splitlines()]
    dig = [json.loads(x) for x in (env.state / G.DIGEST_FILE).read_text(encoding="utf-8").splitlines()]
    assert jobs[0]["job"] == "gpu_module" and dig[0]["kind"] == "parked" and "dependency" in dig[0]["retry"]
    n = len(seen)
    assert A.run_goal(env, "g1", make, fp)["outcome"] == "PARKED" and len(seen) == n     # parked: nothing is tried while waiting
    world["dep"] = "v2"                                                                  # the dependency changed
    r = A.run_goal(env, "g1", make, fp, kind="skill")
    assert r["outcome"] == "ADOPTED" and r["ladder"]["action"] == "continue" and len(seen) == n + 1 and seen[-1] == "start"
    assert [json.loads(x)["kind"] for x in (env.state / G.DIGEST_FILE).read_text(encoding="utf-8").splitlines()] == ["parked", "retried"]


def test_progress_resets_the_ladder_and_the_stall_limit_parks_at_once(tmp_path: Path) -> None:
    st = tmp_path
    assert G.ladder_note(st, "a", 1.0, 0)["action"] == "continue"
    assert G.ladder_note(st, "a", None, 10)["action"] == "continue"
    assert G.ladder_note(st, "a", None, 20)["action"] == "continue"
    assert G.ladder_note(st, "a", None, 30)["action"] == "smaller_step"
    assert G.ladder_note(st, "a", 2.0, 40)["reason"] == "progress" and G.ladder_rung(st, "a") == "start"
    r = G.ladder_note(st, "a", None, 40 + G.STALL_MAX_S + 1, deps={"x": "1"})            # an hour without progress: parked from any rung
    assert r["action"] == "park" and "stall limit" in r["reason"]
    assert G.ladder_due(st, {"x": "1"}, 99999) == [] and G.ladder_due(st, {"x": "2"}, 99999) == ["a"]
    r = G.ladder_note(st, "b", None, 0, kind="data")                                      # a data gap queues a data job when parked
    for t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        r = G.ladder_note(st, "b", None, t, kind="data")
    assert r["action"] in ("park", "parked") and json.loads((st / G.JOBS_FILE).read_text(encoding="utf-8").splitlines()[-1])["job"] == "data_job"
    assert G.ladder_due(st, {}, 90000 + 86400) == ["b"]                                   # no dependency recorded: timer re-try


def test_this_modules_own_diff_style_passes_the_policy_check() -> None:
    assert PO.check_diff(DIFF).ok


def test_decide_adopt_gate_restricts(tmp_path: Path) -> None:
    D = A
    assert D.adopt_gate(tmp_path, ["creator/sandbox.py"], "w")[0] is False
    assert D.adopt_gate(tmp_path, ["creator/tools/x.py"], "w") == (False, "class tool at A0, change needs A1: proposal only")
    A.grant(tmp_path, "tool", 1, "t")
    assert D.adopt_gate(tmp_path, ["creator/tools/x.py"], "w")[0] is True
    A.demote(tmp_path, "tool", "r", 100.0)
    assert "cooling" in D.adopt_gate(tmp_path, ["creator/tools/x.py"], "w", now=200.0)[1]
