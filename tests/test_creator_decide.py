"""Trusted predictions in Nupen's decisions (creator.decide, owner 3 Oct 2026). Untrusted => exactly the old behaviour; trusted => only a
REORDER (schedule) or STRICTER checking (full suite in the kernel); a holdout share stays on the old rule; every decision is logged with its
counterfactual; nothing can reduce testing."""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

import pytest

from creator import decide as D
from creator import registry as REG
from creator import sandbox as SB
from creator import schedule as S
from creator import testrun as TR
from creator import thinking as T

GOOD = {"n": 120, "n_live": 12, "gain_ci95": [0.01, 0.03], "ece": 0.02, "best_baseline": "base"}
BAD = {"n": 120, "n_live": 4, "gain_ci95": [0.01, 0.03], "ece": 0.02, "best_baseline": "base"}


def now_iso(age_s: float = 0.0) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_s)).isoformat(timespec="seconds")


def write_trust(state: Path, topics: dict[str, tuple[bool, dict[str, Any]]] = {}, drills: dict[str, tuple[bool, dict[str, Any]]] = {},
                age_s: float = 0.0) -> None:
    state.mkdir(parents=True, exist_ok=True)
    rep = {"at": now_iso(age_s), "topics": {t: {"trusted": ok, "score": sc} for t, (ok, sc) in topics.items()},
           "drills": {s: {"trusted": ok, "heldout": sc} for s, (ok, sc) in drills.items()}}
    (state / "trust.json").write_text(json.dumps(rep), encoding="utf-8")


@pytest.fixture(autouse=True)
def decide_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUPEN_DECIDE_OFF", raising=False)


# ------------------------------------------------------------------------------------------------ the one "use a prediction?" helper
def test_trusted_only_when_fresh_flagged_and_the_gate_passes_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert D.trusted_now(tmp_path, "verdict") is False                                      # no report at all
    write_trust(tmp_path, {"verdict": (True, GOOD), "duration": (True, BAD), "cost": (False, GOOD)}, {"git_fixed": (True, GOOD)})
    assert D.trusted_now(tmp_path, "verdict") and D.trusted_now(tmp_path, "git_fixed")
    assert not D.trusted_now(tmp_path, "duration")                 # flagged trusted but its score fails the gate: a hand edit is not trust
    assert not D.trusted_now(tmp_path, "cost") and not D.trusted_now(tmp_path, "nonsense")
    write_trust(tmp_path, {"verdict": (True, GOOD)}, age_s=D.TRUST_MAX_AGE_S + 3600)
    assert not D.trusted_now(tmp_path, "verdict")                                            # stale
    write_trust(tmp_path, {"verdict": (True, GOOD)})
    monkeypatch.setenv("NUPEN_DECIDE_OFF", "1")
    assert not D.trusted_now(tmp_path, "verdict")                                            # kill switch
    monkeypatch.delenv("NUPEN_DECIDE_OFF")
    calls: list[int] = []
    assert D.use(tmp_path, "verdict", lambda: calls.append(1) or 0.7) == 0.7
    assert D.use(tmp_path, "duration", lambda: calls.append(2) or 0.7) is None and calls == [1]   # untrusted: never even computed


def test_holdout_is_deterministic_and_about_thirty_percent() -> None:
    units = [f"CP{i:05d}" for i in range(20000)]
    share = sum(D.holdout(u) for u in units) / len(units)
    assert abs(share - D.HOLDOUT_SHARE) < 0.015
    assert [D.holdout(u) for u in units[:50]] == [D.holdout(u) for u in units[:50]]


# ------------------------------------------------------------------------------------------------ REORDER
def node(i: str, comp: str, cost: float = 100.0, deps: tuple[str, ...] = (), files: tuple[str, ...] = (), status: str = "ready",
         step: str = "exists", value: float = 0.5, capacity: bool = False) -> S.Node:
    return S.Node(i, comp, step, value, cost, frozenset(files or (f"{comp}.py",)), deps, status, capacity)


def old_order_ids(nodes: list[S.Node]) -> list[str]:
    a = S.analyse(nodes)
    return [n.id for n in sorted((n for n in nodes if n.status == "ready"),
                                 key=lambda n: (not n.capacity, a.slack[n.id], -a.tail[n.id], -n.value / max(n.cost, 1.0), n.id))]


def random_nodes(rnd: random.Random) -> list[S.Node]:
    ids = [f"g{i}" for i in range(rnd.randint(2, 9))]
    out = []
    for k, i in enumerate(ids):
        deps = tuple(d for d in ids[:k] if rnd.random() < 0.2)
        files = tuple(rnd.sample(["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"], rnd.randint(1, 2)))
        out.append(node(i, f"C{rnd.randint(0, 5)}", cost=rnd.choice([100.0, 300.0, 900.0]), deps=deps, files=files,
                        status=rnd.choice(["ready"] * 5 + ["running", "blocked"]), step=rnd.choice(["exists", "tested", "validated"]),
                        value=rnd.random(), capacity=rnd.random() < 0.2))
    return out


def feasible(b: S.Batch, nodes: list[S.Node], slots: int, steps: tuple[str, ...]) -> None:
    by = {n.id: n for n in nodes}
    busy_c = {n.component for n in nodes if n.status == "running"}
    busy_f = {f for n in nodes if n.status == "running" for f in n.files}
    assert len(b.picks) <= slots
    for p in b.picks:
        assert p.status == "ready" and p.step in steps and not any(d in by for d in p.deps)
        assert p.component not in busy_c and not (p.files & busy_f)
        busy_c.add(p.component)
        busy_f |= p.files


def test_without_predictions_the_schedule_is_the_old_one_and_with_them_only_reorders_within_class() -> None:
    rnd = random.Random(7)
    steps = ("exists", "tested")
    for _ in range(300):
        nodes = random_nodes(rnd)
        old = S.schedule(nodes, 2, steps=steps)
        # byte-identical: the picks follow the OLD key exactly
        expect: list[str] = []
        for i in old_order_ids(nodes):
            if i in {p.id for p in old.picks}:
                expect.append(i)
        assert [p.id for p in old.picks] == expect
        pred = {n.id: rnd.random() for n in nodes}
        new = S.schedule(nodes, 2, steps=steps, pred=pred)
        feasible(new, nodes, 2, steps)                                          # dependencies, exclusions, slots: still enforced
        a = new.analysis
        caps = [(not p.capacity, a.slack[p.id] != 0) for p in new.picks]
        assert caps == sorted(caps)                                             # never a lower class before a higher one
    # a capacity gap with the WORST prediction still goes before a non-capacity one with the best; the critical path before slack
    nodes = [node("cap", "A", capacity=True, cost=100.0), node("crit", "B", cost=900.0), node("slack", "C", cost=100.0)]
    assert [p.id for p in S.schedule(nodes, 3, pred={"cap": 0.0, "crit": 0.1, "slack": 9.0}).picks] == ["cap", "crit", "slack"]


def test_trusted_verdict_reorders_equal_class_gaps_and_logs_the_counterfactual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # two equal-cost independent gaps: the old rule prefers the higher stored value; the trusted prediction says it rarely succeeds
    nodes = [node("hi", "A", value=0.9, step="tested"), node("lo", "B", value=0.5, step="exists")]
    hist = S.History()
    old = S.schedule(nodes, 1)
    assert [p.id for p in old.picks] == ["hi"]
    assert S._predicted_batch(tmp_path, nodes, hist, old, 1, (), [], None) is old            # untrusted: the very same batch
    assert not D.log_path(tmp_path).exists()                                                   # ... and nothing written
    monkeypatch.setattr(D, "rank_scores", lambda state, steps, now=None: {"tested": {"p_adopted": 0.05}, "exists": {"p_adopted": 0.9}})
    monkeypatch.setattr(D, "holdout", lambda unit: False)
    b = S._predicted_batch(tmp_path, nodes, hist, old, 1, (), [], None)
    assert [p.id for p in b.picks] == ["lo"]
    row = json.loads(D.log_path(tmp_path).read_text().splitlines()[-1])
    assert row["kind"] == "rank" and row["arm"] == "treated" and row["old_rule"] == ["hi"] and row["did"] == ["lo"] and row["changed"]
    monkeypatch.setattr(D, "holdout", lambda unit: True)
    b2 = S._predicted_batch(tmp_path, nodes, hist, old, 1, (), [], None)
    assert [p.id for p in b2.picks] == ["hi"]                                                 # holdout round: the old rule acts
    row = json.loads(D.log_path(tmp_path).read_text().splitlines()[-1])
    assert row["arm"] == "holdout" and row["new_rule"] == ["lo"] and row["did"] == ["hi"] and not row["changed"]
    monkeypatch.setattr(D, "rank_scores", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")))
    assert S._predicted_batch(tmp_path, nodes, hist, old, 1, (), [], None) is old            # a broken predictor changes nothing


def history_world(st: Path, n: int = 60) -> None:
    """'exists' packages are adopted, 'tested' ones are not (planted)."""
    led, klog = [], []
    t0 = 1_790_000_000.0
    for i in range(n):
        kind = "exists" if i % 2 == 0 else "tested"
        pkg = f"CP{i:04d}"
        iso = dt.datetime.fromtimestamp(t0 + i * 3600, dt.timezone.utc).isoformat(timespec="seconds")
        end = dt.datetime.fromtimestamp(t0 + i * 3600 + 900, dt.timezone.utc).isoformat(timespec="seconds")
        led.append({"rtype": "WorkPackage", "id": f"WP-{i}", "provenance": {"timestamp": iso},
                    "data": {"package_id": pkg, "why_it_exists": f"gap (K07.{kind})", "outputs": ["a.py"]}})
        led.append({"rtype": "StrategyOutcome", "id": f"SO-{i}", "provenance": {"timestamp": end}, "data": {"subject_id": f"WP-{i}"}})
        klog.append({"package": pkg, "outcome": "ADOPTED" if kind == "exists" else "REJECTED", "requirement": f"K07.{kind}",
                     "seconds": 900.0 if kind == "exists" else 5000.0})
    st.mkdir(parents=True, exist_ok=True)
    (st / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in led) + "\n", encoding="utf-8")
    (st / "kernel_log.jsonl").write_text("\n".join(json.dumps(r) for r in klog) + "\n", encoding="utf-8")


def test_rank_scores_come_only_from_trusted_topics_and_learn_from_history(tmp_path: Path) -> None:
    history_world(tmp_path)
    assert D.rank_scores(tmp_path, ["exists", "tested"]) is None
    write_trust(tmp_path, {"verdict": (True, GOOD), "duration": (False, GOOD)})
    sc = D.rank_scores(tmp_path, ["exists", "tested"])
    assert sc is not None and set(sc["exists"]) == {"p_adopted"}                             # duration untrusted: no estimate
    assert sc["exists"]["p_adopted"] > 0.8 > 0.2 > sc["tested"]["p_adopted"]
    write_trust(tmp_path, {"verdict": (True, GOOD), "duration": (True, GOOD)})
    sc2 = D.rank_scores(tmp_path, ["exists", "tested"])
    assert sc2 is not None and sc2["tested"]["est_s"] > sc2["exists"]["est_s"]
    assert D.ev_per_cost(1.0, sc2["exists"], 0.5, 600) > D.ev_per_cost(1.0, sc2["tested"], 0.5, 600)
    assert D.ev_per_cost(2.0, {}, 0.5, 100.0) == pytest.approx(0.01)                          # nothing trusted: the old estimate


# ------------------------------------------------------------------------------------------------ STRICTER
def graph_of(tmp: Path) -> TR.ImportGraph:
    for rel, text in {"pkg/__init__.py": "", "pkg/a.py": "X = 1\n", "pkg/b.py": "Y = 2\n", "tests/__init__.py": "",
                      "tests/test_a.py": "from pkg.a import X\n\ndef test_a():\n    assert X == 1\n",
                      "tests/test_b.py": "from pkg.b import Y\n\ndef test_b():\n    assert Y == 2\n"}.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(text, encoding="utf-8")
    return TR.ImportGraph.build(tmp)


def test_widening_only_ever_adds_tests_and_never_widens_an_empty_selection(tmp_path: Path) -> None:
    g = graph_of(tmp_path)
    sel = TR.select_tests(g, ["pkg/a.py"])
    assert sel.tests == ("tests/test_a.py",)
    w = SB.widen_selection(sel, g, "git_fixed P=0.6")
    assert set(sel.tests) < set(w.tests) and w.select_all and w.reasons["tests/test_a.py"] == sel.reasons["tests/test_a.py"]
    assert w.reasons["tests/test_b.py"].startswith("predicted risk")
    empty = TR.select_tests(g, ["README.md"])
    assert empty.empty and SB.widen_selection(empty, g, "x") is empty                       # empty = rejected today: stays so
    allsel = TR.select_tests(g, ["pkg/data.csv"])
    assert allsel.select_all and SB.widen_selection(allsel, g, "x") is allsel
    rnd = random.Random(3)
    for _ in range(50):                                                                     # property: never fewer tests
        ch = rnd.sample(["pkg/a.py", "pkg/b.py", "tests/test_a.py", "README.md", "pkg/__init__.py"], rnd.randint(1, 3))
        s0 = TR.select_tests(g, ch)
        assert set(s0.tests) <= set(SB.widen_selection(s0, g, "r").tests)


def test_risk_band_flags_twenty_to_thirty_percent() -> None:
    rnd = random.Random(1)
    ps = [rnd.random() for _ in range(300)]
    thr = D.risk_band(ps)
    assert 0.2 <= sum(p >= thr for p in ps) / len(ps) <= 0.3
    tied = [0.1] * 60 + [0.5] * 25 + [0.9] * 15                                            # ties: never more than 30%
    assert D.risk_band(tied) == 0.9 and D.risk_band([]) == float("inf")
    assert sum(p >= D.risk_band([0.1] * 50 + [0.9] * 50) for p in [0.1] * 50 + [0.9] * 50) == 0   # 50% cannot be the band: flag nothing


def git_state(st: Path, n: int = 400) -> None:
    """A cached history where commits in risky/ are fixed soon after (the model sees the top directory, file count, size and message
    class, never the file name), and a git_fixed drill result for the variant."""
    rows = []
    for i in range(n):
        risky = i % 4 == 0
        fix = i % 4 == 1
        files = ["risky/r.py"] if (risky or fix) else [f"pkg/m{i % 7}.py"]
        rows.append({"h": f"{i:040x}", "t": 1_700_000_000.0 + i * 600, "s": "fix: risky" if fix else "add: thing", "files": files,
                     "lines": 10, "add": 5, "del": 5, "au": "", "body": ""})
    p = st / "thinking" / "git_history.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    run = {"source": "git_fixed", "variant": {"decay": 1.0, "k": 2.0}, "digest": "d", "items": n, "resolved": n,
           "select": {"n": 100, "brier": 0.1}, "heldout": GOOD}
    (st / "thinking" / "drill_runs.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")


def test_risk_gate_untrusted_is_a_no_op_and_trusted_flags_only_the_top_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git_state(tmp_path)
    j = tmp_path / "no_journal.md"
    assert D.risk_gate(tmp_path, "CP1", ["risky/r.py"], "+x\n", "CP1 K07.exists", journal=j) == ""
    assert not D.log_path(tmp_path).exists()
    write_trust(tmp_path, drills={"git_fixed": (True, GOOD)})
    r = D.git_fixed_risk(tmp_path, ["risky/r.py"], "+x\n-y\n", "CP1 K07.exists", journal=j)
    safe = D.git_fixed_risk(tmp_path, ["pkg/m3.py"], "+x\n", "CP2 K07.exists", journal=j)
    assert r is not None and safe is not None and r["high"] and not safe["high"] and r["p"] > safe["p"]
    assert 0.0 < r["ref_flag_share"] <= 0.3
    monkeypatch.setattr(D, "holdout", lambda unit: unit == "CP_H")
    why = D.risk_gate(tmp_path, "CP_T", ["risky/r.py"], "+x\n", "CP_T K07.exists", journal=j)
    assert why.startswith("git_fixed P=")
    assert D.risk_gate(tmp_path, "CP_H", ["risky/r.py"], "+x\n", "CP_H K07.exists", journal=j) == ""      # holdout: today's testing
    assert D.risk_gate(tmp_path, "CP_S", ["pkg/m3.py"], "+x\n", "CP_S K07.exists", journal=j) == ""         # low risk: today's testing
    rows = [json.loads(x) for x in D.log_path(tmp_path).read_text().splitlines()]
    assert [(r["unit"], r["arm"], r["flagged"], r["did"], r["old_rule"]) for r in rows] == [
        ("CP_T", "treated", True, "full_suite", "selection"), ("CP_H", "holdout", True, "selection", "selection"),
        ("CP_S", "treated", False, "selection", "selection")]
    monkeypatch.setattr(D, "git_fixed_risk", lambda *a, **k: 1 / 0)
    assert D.risk_gate(tmp_path, "CP_E", ["risky/r.py"], "", "m", journal=j) == ""                       # broken: today's testing


def test_no_decide_output_can_reduce_testing() -> None:
    """The only thing the kernel takes from creator.decide is a reason to WIDEN; evaluate never narrows on it, and the decide module has no
    name that could be read as a skip / fewer-tests / verdict switch."""
    import inspect
    src = inspect.getsource(D)
    for bad in ("select_tests(", "Verdict.", "timeout=", "flaky_reruns", "run_tests="):
        assert bad not in src, bad
    ev = inspect.getsource(SB.Sandbox.evaluate)
    assert ev.count("widen_reason") == 4 and "widen_selection(selection, graph, widen_reason)" in ev


def test_summary_compares_the_arms_with_intervals(tmp_path: Path) -> None:
    rows, klog = [], []
    rnd = random.Random(5)
    for i in range(80):
        arm = "holdout" if i % 3 == 0 else "treated"
        rows.append({"kind": "risk", "unit": f"CP{i}", "arm": arm, "flagged": True, "changed": arm == "treated"})
        caught = rnd.random() < (0.4 if arm == "treated" else 0.1)
        klog.append({"package": f"CP{i}", "outcome": "REJECTED" if caught else "ADOPTED", "seconds": 1000.0,
                     "details": {"regression": "REGRESSION" if caught else "CLEAN"}})
    (tmp_path / "thinking").mkdir()
    D.log_path(tmp_path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "kernel_log.jsonl").write_text("\n".join(json.dumps(r) for r in klog) + "\n", encoding="utf-8")
    s = D.summary(tmp_path)
    arms = s["risk"]["arms"]
    assert s["risk"]["flagged"] == 80 and arms["treated"]["n"] + arms["holdout"]["n"] == 80
    d = arms["treated_minus_holdout"]["regression_caught"]
    assert d["diff"] > 0 and d["ci95"][0] < d["diff"] < d["ci95"][1]
    assert arms["treated"]["regression_caught"]["ci95"][0] <= arms["treated"]["regression_caught"]["rate"]
    assert D.summary(tmp_path / "empty")["decisions"] == 0


def test_registered_and_in_the_trust_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import anticipation as A
    monkeypatch.setattr(A, "DEFAULT_OWNER_DIR", tmp_path / "owner")
    assert REG.get("decide") is D
    rep = T.trust(tmp_path / "st", write=False)
    assert rep["decisions"]["decisions"] == 0 and "risk" in rep["decisions"]


# ------------------------------------------------------------------------------------------------ the kernel hook, end to end
from tests.test_creator_kernel import USER, USER_TEST, Scripted, cached_provenance, cfg  # noqa: E402,F401 - fixtures + helpers
from creator import kernel as K  # noqa: E402


def _evaluation(c: K.KernelConfig, pkg: str) -> dict[str, Any]:
    return json.loads((c.state / "cycles" / pkg / "evaluation.json").read_text(encoding="utf-8"))


def test_kernel_runs_the_full_suite_only_for_a_trusted_high_risk_candidate(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Untrusted: the affected-test selection exactly as before. Trusted + high predicted risk (treated arm): every test, with the old
    selection kept inside it, and the decision recorded on the cycle and in decisions.jsonl. Fails on the old kernel (no hook)."""
    risk = {"p": 0.8, "threshold": 0.5, "high": True, "variant": {}, "n_ref": 300, "ref_flag_share": 0.25}
    monkeypatch.setattr(D, "git_fixed_risk", lambda *a, **k: dict(risk))
    monkeypatch.setattr(D, "holdout", lambda unit: False)
    w = Scripted("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST})
    write_trust(cfg.state, drills={"git_fixed": (True, GOOD)})
    rep = K.cycle(cfg, w)
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details)
    sel = _evaluation(cfg, rep.package)["selection"]
    assert sel["select_all"] and sel["why_all"].startswith("predicted risk: git_fixed P=0.80")
    assert set(sel["tests"]) == {"tests/test_base.py", "tests/test_user.py"} and sel["reasons"]["tests/test_user.py"] == "test file changed"
    assert rep.details["decide"]["full_suite"].startswith("git_fixed")
    row = json.loads(D.log_path(cfg.state).read_text().splitlines()[-1])
    assert row["unit"] == rep.package and row["did"] == "full_suite" and row["old_rule"] == "selection"


def test_kernel_untrusted_keeps_the_selection_and_writes_nothing(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "git_fixed_risk", lambda *a, **k: {"p": 0.9, "threshold": 0.1, "high": True})
    write_trust(cfg.state, drills={"git_fixed": (False, BAD)})
    rep = K.cycle(cfg, Scripted("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details)
    sel = _evaluation(cfg, rep.package)["selection"]
    assert not sel["select_all"] and sel["tests"] == ["tests/test_user.py"] and "decide" not in rep.details
    assert not D.log_path(cfg.state).exists()
