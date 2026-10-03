"""The learning loop (owner, 3 Oct 2026: "if we arent testing on the data to see what it remembers and then improving the system afterward what good
was collecting the data in the first place"): progress curves, retention probes that never leak, a diagnosis that finds planted weaknesses, improvements
that are re-measured with a recorded verdict, the need-data signal, and no write to any measuring code or the frozen benchmark."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from creator import drillsources as D
from creator import focus as F
from creator import goals as GO
from creator import learnloop as L
from creator import thinking as T

ROOT = Path(__file__).resolve().parents[1]
V0 = {"decay": 1.0, "k": 2.0}


# ------------------------------------------------------------------------------------------------ 1 the progress curve
def series(vals: list[float], half: float) -> list[tuple[float, float, Optional[list[float]]]]:
    return [(1000.0 + 3600 * i, v, [v - half, v + half]) for i, v in enumerate(vals)]


def test_progress_curve_classifies_synthetic_series() -> None:
    assert L.classify(series([0.00, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], 0.01))["trend"] == "improving"
    assert L.classify(series([0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, -0.01], 0.01))["trend"] == "regressing"
    rnd = random.Random(3)
    flat = [0.02 + rnd.uniform(-0.004, 0.004) for _ in range(12)]
    assert L.classify(series(flat, 0.01))["trend"] == "plateau"
    noisy_rise = [0.0, 0.01, 0.02, 0.03]                              # a rise inside wide intervals is not called improvement
    assert L.classify(series(noisy_rise, 0.08))["trend"] == "plateau"
    assert L.classify(series([0.1, 0.2], 0.01))["trend"] == "insufficient"
    no_ci = [(float(i), 0.01 * i, None) for i in range(9)]            # no CIs: the series' own scatter is the uncertainty
    assert L.classify(no_ci)["trend"] == "improving"


def bench_file(d: Path, name: str, at: str, per_item: dict[str, float], role: str = "thinker") -> None:
    n = len(per_item)
    part = {"n": n, "metric": "Brier", "higher_is_better": False, "score": sum(per_item.values()) / n, "baseline": 0.2, "gain": 0.2 - sum(per_item.values()) / n,
            "gain_ci95": [0.0, 0.1], "per_item": per_item}
    (d / name).write_text(json.dumps({"at": at, "items_hash": "h", "versions": {"local_model": {"role": role}}, "parts": {"b_judgment": part}}), encoding="utf-8")


def test_benchmark_curve_is_paired_on_the_frozen_items(tmp_path: Path) -> None:
    d = tmp_path / "thinkbench"
    d.mkdir()
    rnd = random.Random(1)
    base = {f"b{i}": 0.2 + rnd.uniform(-0.02, 0.02) for i in range(40)}
    bench_file(d, "baseline_20261003T010000Z.json", "2026-10-03T01:00:00+00:00", base)
    bench_file(d, "now_20261003T050000Z.json", "2026-10-03T05:00:00+00:00", {k: v + 0.05 for k, v in base.items()})   # every item worse
    c = L.bench_curve(tmp_path)["bench:b_judgment:thinker"]
    assert c["paired"] and c["trend"] == "regressing" and c["first_vs_last"]["n_paired"] == 40
    rows = L.measure(tmp_path, 2e9, tj={})
    assert {r["bench_file"] for r in rows} == {"baseline_20261003T010000Z.json", "now_20261003T050000Z.json"}
    assert L.measure(tmp_path, 2e9 + 10, tj={}) == []                  # a run is recorded once


def test_measure_records_every_trust_skill_and_plugins(tmp_path: Path) -> None:
    sc = {"n": 300, "gain_vs_best": 0.02, "gain_ci95": [0.01, 0.03], "brier": 0.17}
    tj = {"at": "2026-10-03T08:00:00+00:00", "topics": {"verdict": {"score": sc}}, "drills": {"git_fixed": {"heldout": sc, "live": {}}},
          "judgment": {"verdict": {"judge_calibrated": sc, "gain_vs_statistical": [0.004, -0.01, 0.02], "n": 43}},
          "fast_topics": {"test_file_slow": {"score": sc}}, "anticipation": {"anticipated": 1, "eligible_directives": 20},
          "reasoning_mc": {"order": {"score": sc}}}                                     # an unknown section (a plug-in such as h27's)
    (tmp_path / "thinking" / "skill_scores").mkdir(parents=True)
    (tmp_path / "thinking" / "skill_scores" / "h27.json").write_text(json.dumps({"at": "x", "skills": {"mc": {"n": 50, "value": 0.3, "ci": [0.2, 0.4]}}}))
    skills = {r["skill"] for r in L.measure(tmp_path, 2e9, tj=tj)}
    assert {"thinking:verdict", "drill:git_fixed:heldout", "judgment:verdict", "judgment:verdict:vs_statistical", "fast:test_file_slow",
            "anticipation:rate", "reasoning_mc:order", "h27:mc"} <= skills
    assert L.measure(tmp_path, 2e9 + 5, tj=tj) == []                   # unchanged measurements are not repeated


# ------------------------------------------------------------------------------------------------ 2 retention never leaks
def items(n: int, seed: int = 5) -> list[D.BItem]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        a = i % 2 == 0
        out.append(D.BItem((f"k:{'a' if a else 'b'}", f"z:{i % 3}"), 1000.0 + 10 * i, 1000.0 + 10 * i + 5, int(rnd.random() < (0.85 if a else 0.15)), f"c{i:04d}"))
    return out


class FakeData:
    def __init__(self, its: list[D.BItem], variant: dict[str, Any], source: str = "git_fixed") -> None:
        self.d = {source: {"items": its, "variant": variant, "preds": L.predict(its, source, variant, Path("-"))}}

    def get(self, s: str) -> Optional[dict[str, Any]]:
        return self.d.get(s)


def test_fresh_probe_scores_only_never_selected_items_and_never_sees_their_own_outcome(tmp_path: Path) -> None:
    its = items(100)
    fd = FakeData(its, V0)
    rep = L.retention(tmp_path, fd, 1e9, ["git_fixed"])                # type: ignore[arg-type]
    pr = [e for e in L._read(tmp_path, "retention") if e["event"] == "probe"]
    assert len(pr) == 1 and rep["git_fixed"]["probes"] == 1
    cut = L.select_cut(L.predict(its, "git_fixed", V0, Path("-")))
    keys = pr[0]["scores"]
    assert len(keys) == 40 and pr[0]["never_selected_on"] is True        # the held-out 40%: no variant selection ever used them
    assert all(float(k.split("@")[1]) > cut for k in keys)
    # the probe's probability of an item never depends on that item's own outcome (scored strictly before it is learned from)
    last = max(keys, key=lambda k: float(k.split("@")[1]))
    flipped = [D.BItem(i.keys, i.created, i.resolved, 1 - i.y if f"{i.subject}@{i.created}" == last else i.y, i.subject) for i in its]
    again = {f"{it.subject}@{it.created}": p.p for it, p in L.predict(flipped, "git_fixed", V0, Path("-"))}
    assert again[last] == pytest.approx(keys[last][0])
    L.retention(tmp_path, fd, 1e9 + 60, ["git_fixed"])                 # type: ignore[arg-type]
    assert sum(1 for e in L._read(tmp_path, "retention") if e["event"] == "probe") == 1   # the same items are never probed twice


def test_probe_is_rescored_paired_after_the_predictor_changes(tmp_path: Path) -> None:
    its = items(100)
    L.retention(tmp_path, FakeData(its, V0), 1e9, ["git_fixed"])      # type: ignore[arg-type]
    v1 = {"decay": 0.9, "k": 10.0}
    L.retention(tmp_path, FakeData(its, v1), 1e9 + 100, ["git_fixed"])  # type: ignore[arg-type]
    assert not [e for e in L._read(tmp_path, "retention") if e["event"] == "rescore"]      # too early
    rep = L.retention(tmp_path, FakeData(its, v1), 1e9 + 4000, ["git_fixed"])  # type: ignore[arg-type]
    rs = [e for e in L._read(tmp_path, "retention") if e["event"] == "rescore"]
    assert len(rs) == 1 and rs[0]["n"] == 40 and rs[0]["selected_since"] is True
    pr = [e for e in L._read(tmp_path, "retention") if e["event"] == "probe"][0]
    now = {f"{it.subject}@{it.created}": p for it, p in L.predict(its, "git_fixed", v1, Path("-"))}
    want = sum(T.brier(v[0], v[1]) - T.brier(now[k].p, v[1]) for k, v in pr["scores"].items()) / 40
    assert rs[0]["gain_since_probe"] == pytest.approx(want, abs=1e-4)
    assert rep["git_fixed"]["rescored"] == 1
    L.retention(tmp_path, FakeData(its, v1), 1e9 + 9000, ["git_fixed"])  # type: ignore[arg-type]
    assert len([e for e in L._read(tmp_path, "retention") if e["event"] == "rescore"]) == 1   # nothing changed: no new re-score


# ------------------------------------------------------------------------------------------------ 3 diagnosis finds planted weaknesses
def test_diagnosis_ranks_planted_weaknesses_by_score_lost() -> None:
    rnd = random.Random(7)
    pairs = []
    for i in range(600):
        fam = "bad" if i % 6 == 0 else "big" if i % 6 == 1 else "ok"
        y = int(rnd.random() < 0.5)
        if fam == "bad":
            p = 0.9 if y == 0 else 0.1                                  # badly wrong where key f:bad: a large loss on few items
        elif fam == "big":
            p = 0.62 if y == 0 else 0.38                                # mildly wrong on as many items: smaller loss
        else:
            p = 0.8 if y else 0.2                                       # good
        it = D.BItem((f"f:{fam}", f"g:{i % 2}"), float(i), float(i) + 0.5, y, f"s{i}")
        pairs.append((it, T.Pred("git_fixed", it.subject, it.created, p, 0.5, 0.5, y)))
    pairs.append((D.BItem(("f:tiny",), 9e3, 9e3, 1, "t"), T.Pred("git_fixed", "t", 9e3, 0.0, 0.5, 0.5, 1)))   # one item: below MIN_GROUP_N
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    L.drill_groups("git_fixed", pairs, acc)
    ranked = L.rank(acc)
    feats = [w["where"] for w in ranked if w["kind"] == "feature"]
    assert feats[:2] == ["git_fixed|f:bad", "git_fixed|f:big"]
    assert "git_fixed|f:ok" not in [w["where"] for w in ranked]          # gains are not weaknesses
    assert "git_fixed|f:tiny" not in feats
    top = ranked[0]
    assert top["significant"] and top["pos"] == 0 and top["loss_total"] > 0
    assert all(w["significant"] for w in ranked[:3])


# ------------------------------------------------------------------------------------------------ 4 improvement proposals and the re-measure bookkeeping
def run_row(source: str, variant: dict[str, Any], sel: float, held_gain: float, digest: str = "d1", at: str = "2026-10-03T01:00:00+00:00") -> dict[str, Any]:
    return {"source": source, "variant": variant, "search": bool(variant.get("search")), "digest": digest, "items": 500, "resolved": 480,
            "select": {"n": 288, "brier": sel}, "heldout": {"n": 192, "brier": 0.17, "gain_vs_best": held_gain, "gain_ci95": [held_gain - 0.01, held_gain + 0.01]},
            "at": at}


def write_runs(state: Path, rows: list[dict[str, Any]]) -> None:
    p = D.runs_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def tj_for(gain: float) -> dict[str, Any]:
    return {"drills": {"git_fixed": {"heldout": {"n": 192, "gain_vs_best": gain, "gain_ci95": [gain - 0.01, gain + 0.01]}}}}


FEATURE_W = {"kind": "feature", "where": "git_fixed|m:checklist", "source": "git_fixed", "pos": 0, "family": "m", "n": 21, "loss_total": 1.4,
             "loss_mean": 0.07, "loss_ci95": [0.015, 0.12], "significant": True, "id": "W-feat"}


def test_a_feature_weakness_queues_measured_search_variants_and_is_judged_helped(tmp_path: Path) -> None:
    write_runs(tmp_path, [run_row("git_fixed", V0, 0.20, 0.010)])
    out = L.improve(tmp_path, {"weaknesses": [FEATURE_W]}, tj_for(0.010), None, 1e9)   # type: ignore[arg-type]
    rec = out["applied"][0]
    assert rec["action"] == "queue_variants" and rec["skill"] == "drill:git_fixed:heldout" and rec["before"]["value"] == 0.010
    assert rec["weakness"]["id"] == "W-feat" and rec["variants"]
    assert any(v["mask"] & 1 for v in rec["variants"])                  # the diagnosed feature family (key position 0) is dropped
    q = L.queued_variants(tmp_path)
    assert [v for _s, v in q] == rec["variants"] and all(s == "git_fixed" and v["learn"] == rec["id"] for s, v in q)
    assert L.improve(tmp_path, {"weaknesses": [FEATURE_W]}, tj_for(0.010), None, 1e9 + 1)["applied"] == []   # one open improvement per skill
    write_runs(tmp_path, [run_row("git_fixed", v, 0.19 if i == 0 else 0.25, 0.018 if i == 0 else 0.0) for i, v in enumerate(rec["variants"])])
    assert L.queued_variants(tmp_path) == []
    assert L.verify(tmp_path, rec, tj_for(0.018), 1e9 + 60) is None     # not due yet
    out = L.improve(tmp_path, {"weaknesses": []}, tj_for(0.018), None, 1e9 + L.VERIFY_AFTER_S + 1)   # type: ignore[arg-type]
    v = out["verified"][0]
    assert v["verdict"] == "helped" and v["adopted"] is True and v["after"]["value"] == 0.018
    assert [e["event"] for e in L._read(tmp_path, "improvements")] == ["applied", "verdict"]


def test_not_adopted_is_no_effect_and_adopted_but_worse_is_flagged_never_silent(tmp_path: Path) -> None:
    write_runs(tmp_path, [run_row("git_fixed", V0, 0.20, 0.010)])
    rec = L.improve(tmp_path, {"weaknesses": [FEATURE_W]}, tj_for(0.010), None, 1e9)["applied"][0]   # type: ignore[arg-type]
    write_runs(tmp_path, [run_row("git_fixed", v, 0.25, 0.0) for v in rec["variants"]])            # none beats the best on select
    v = L.verify(tmp_path, rec, tj_for(0.010), 1e9 + L.VERIFY_AFTER_S + 1)
    assert v is not None and v["verdict"] == "no_effect" and v["adopted"] is False
    other = tmp_path / "b"
    write_runs(other, [run_row("git_fixed", V0, 0.20, 0.010)])
    rec2 = L.improve(other, {"weaknesses": [FEATURE_W]}, tj_for(0.010), None, 1e9)["applied"][0]   # type: ignore[arg-type]
    write_runs(other, [run_row("git_fixed", v, 0.10 if i == 0 else 0.3, -0.02) for i, v in enumerate(rec2["variants"])])
    v2 = L.verify(other, rec2, tj_for(-0.02), 1e9 + L.VERIFY_AFTER_S + 1)
    assert v2 is not None and v2["verdict"] == "hurt_flag_revert" and "revert" in v2["flag"]
    # a small fix that had no effect is escalated: the next time the same weakness becomes a goal proposal
    L._append(tmp_path, "improvements", [v])
    out = L.improve(tmp_path, {"weaknesses": [FEATURE_W]}, tj_for(0.010), None, 1e9 + 2 * L.VERIFY_AFTER_S)   # type: ignore[arg-type]
    assert out["applied"][0]["action"] == "goal_proposal"


JUDGE_W = {"kind": "judgment", "where": "git_fixed|Qwen3-1.7B", "topic": "git_fixed", "model": "Qwen3-1.7B", "n": 299, "loss_total": 4.0,
           "loss_mean": 0.0135, "loss_ci95": [0.0027, 0.0242], "significant": True, "id": "W-judge"}


def test_a_bigger_weakness_becomes_a_thinking_goal_proposal_with_evidence_and_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tj = {"judgment": {"git_fixed": {"gain_vs_statistical": [-0.0135, -0.0242, -0.0027], "n": 299}}}
    out = L.improve(tmp_path, {"weaknesses": [JUDGE_W]}, tj, None, 1e9)   # type: ignore[arg-type]
    rec = out["applied"][0]
    assert rec["action"] == "goal_proposal" and rec["skill"] == "judgment:git_fixed:vs_statistical"
    p = [x for x in GO.listing(tmp_path) if x["id"] == rec["proposal"]][0]
    assert p["source"] == "weakness" and p["status"] == "PENDING" and "W-judge" in p["evidence"]
    assert p["metric"] == {"skill": "judgment:git_fixed:vs_statistical", "before": {"value": -0.0135, "ci": [-0.0242, -0.0027], "n": 299}}
    assert p["spec"]["modules"][0].startswith(F.THINKING_MODULE_PREFIX)
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    assert F.allowed({"component": p["spec"]["modules"][0], "step": "exists"}, tmp_path)      # the thinking focus lets it be built
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps({"capabilities": [{"id": "K31", "modules": p["spec"]["modules"]}]}), encoding="utf-8")
    monkeypatch.setattr(F, "APPROVED_FILE", caps)
    monkeypatch.setattr(F.approved_thinking_components, "__defaults__", (caps,))
    assert F.allowed({"requirement_key": "K31.exists", "component": "K31"}, tmp_path)          # ... also once approved as a K-id
    assert not F.allowed({"requirement_key": "K28.exists", "component": "K28"}, tmp_path)
    assert L.improve(tmp_path, {"weaknesses": [JUDGE_W]}, tj, None, 1e9 + 5)["applied"] == []  # never a duplicate
    GO.reject(tmp_path, rec["proposal"], "teacher: not now")
    v = L.improve(tmp_path, {"weaknesses": []}, tj, None, 1e9 + L.VERIFY_AFTER_S + 1)["verified"]   # type: ignore[arg-type]
    assert v and v[0]["verdict"] == "rejected"


def test_every_weakness_maps_to_the_progress_skill_that_must_move() -> None:
    assert L._skill_of(FEATURE_W) == "drill:git_fixed:heldout"
    assert L._skill_of(JUDGE_W) == "judgment:git_fixed:vs_statistical"
    assert L._skill_of({"kind": "live_key", "where": "fast|drill_beats_best|research_bool|search"}) == "fast:drill_beats_best"
    assert L._skill_of({"kind": "live", "where": "drill_live|git_churn"}) == "drill:git_churn:live"
    assert L._skill_of({"kind": "live", "where": "predictions|verdict"}) == "thinking:verdict"
    assert L._skill_of({"kind": "regression", "where": "bench:b_judgment:thinker"}) == "bench:b_judgment:thinker"


# ------------------------------------------------------------------------------------------------ 5 need data
def test_need_data_event_when_the_search_converged_and_no_fresh_items(tmp_path: Path) -> None:
    lim = {"git_fixed": {"data_limited": True, "converged": True, "fresh_items": 0}, "git_churn": {"data_limited": False}}
    diag = {"weaknesses": [{"id": "W1", "kind": "feature", "where": "git_fixed|m:x", "source": "git_fixed", "loss_total": 1.0}], "limits": lim}
    ev = L.need_data(tmp_path, diag, 1e9)
    assert ev is not None and ev["data_limited_sources"] == ["git_fixed"] and ev["top_weakness"]["id"] == "W1"
    assert json.loads(L._out(tmp_path, "need_data_now").read_text())["need_data"] is True
    assert L.need_data(tmp_path, diag, 1e9 + 60) is None                 # not repeated within NEED_DATA_EVERY_S
    L._append(tmp_path, "queue", [{"source": "git_fixed", "variant": {"decay": 0.5, "k": 1.0}, "improvement": "IM-1"}])
    assert L.need_data(tmp_path, diag, 1e9 + 2 * L.NEED_DATA_EVERY_S) is None    # a remedy is queued: still something to learn
    assert json.loads(L._out(tmp_path, "need_data_now").read_text())["need_data"] is False


# ------------------------------------------------------------------------------------------------ the drill filler runs queued remedies (old code: never)
def plan_rows(n: int) -> str:
    rnd = random.Random(4)
    return "".join(json.dumps({"at": f"2026-10-0{1 + i // 40}T{(i % 24):02d}:{i % 60:02d}:00Z", "chosen": rnd.random() < 0.3, "step": "exists",
                               "component": f"K{i % 5}", "value": 0.5, "node": f"n{i}"}) + "\n" for i in range(n))


def test_drill_filler_hands_out_queued_variants_even_after_the_search_stopped(tmp_path: Path) -> None:
    (tmp_path / "plan_explanations.jsonl").write_text(plan_rows(80), encoding="utf-8")
    dg = D.source_digest("plan_choice", tmp_path, tmp_path, tmp_path / "J.md", tmp_path)
    rows = [run_row("plan_choice", v, 0.2, 0.0, dg) for v in D.VARIANTS]
    rows += [run_row("plan_choice", {"decay": 0.8, "k": float(i) + 0.5, "search": 1}, 0.3, 0.0, dg) for i in range(D.SEARCH_STOP_K)]
    write_runs(tmp_path, rows)
    assert D.search_state(rows, dg)["stopped"]
    nxt = D.drill_filler(tmp_path, tmp_path, tmp_path / "J.md", tmp_path, sources=["plan_choice"], seed=1)
    assert nxt() is None                                                # converged, nothing queued: no work
    qv = {"decay": 0.97, "k": 7.0, "search": 1, "learn": "IM-test"}
    L._append(tmp_path, "queue", [{"source": "plan_choice", "variant": qv, "improvement": "IM-test"}])
    nxt = D.drill_filler(tmp_path, tmp_path, tmp_path / "J.md", tmp_path, sources=["plan_choice"], seed=1)
    job = nxt()
    assert job is not None
    job()
    last = T._jsonl(D.runs_path(tmp_path))[-1]
    assert last["variant"] == qv and last["select"]["n"] > 0            # measured exactly like any other variant
    assert L.queued_variants(tmp_path) == []
    reopened = last["select"]["brier"] < 0.2                            # a queued remedy that beats the best re-opens the stopped search
    assert (nxt() is not None) == reopened


# ------------------------------------------------------------------------------------------------ wiring and the read-only guarantee
def _swarm_script():                                                 # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("creator_swarm_ll", ROOT / "scripts" / "creator_swarm.py")
    mod = importlib.util.module_from_spec(spec)                      # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                     # type: ignore[union-attr]
    return mod


def test_the_thinking_filler_runs_the_learning_loop_on_its_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cs = _swarm_script()
    monkeypatch.setattr(cs, "STATE", tmp_path)
    calls: list[str] = []
    mods = {"thinking": SimpleNamespace(run=lambda state: calls.append("run")),
            "drillsources": SimpleNamespace(drill_filler=lambda *a, **k: (lambda: (lambda: calls.append("drill"))), live_pass=lambda *a: None)}
    learn = SimpleNamespace(run=lambda state, root: calls.append("learn"))
    monkeypatch.setattr(cs.REG, "get", lambda name: mods[name])
    monkeypatch.setattr(cs.REG, "optional", lambda name: F if name == "focus" else learn if name == "learnloop" else None)
    monkeypatch.setattr(cs, "_THINK_LAST", {"run": 1e18, "blueprint": 1e18})
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    nxt = cs.make_filler()
    nxt()()
    nxt()()
    assert calls == ["learn", "drill"]                                  # due first, then not again within LEARN_EVERY_S


def _digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_a_round_never_writes_measuring_code_the_benchmark_or_trust(tmp_path: Path) -> None:
    st = tmp_path / "state"
    (st / "thinkbench").mkdir(parents=True)
    (st / "thinkbench" / "items.json").write_text(json.dumps({"hash": "1f520b202ac4", "a": []}), encoding="utf-8")
    bench_file(st / "thinkbench", "baseline_20261003T010000Z.json", "2026-10-03T01:00:00+00:00", {"b1": 0.2, "b2": 0.1})
    (st / "trust.json").write_text(json.dumps({"at": "2026-10-03T08:00:00+00:00", "drills": {"git_fixed": {"heldout": {"n": 50, "gain_vs_best": 0.01,
                                   "gain_ci95": [0.0, 0.02]}}}}), encoding="utf-8")
    (st / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    write_runs(st, [run_row("git_fixed", V0, 0.2, 0.01)])
    code = {f: hashlib.sha256((ROOT / "creator" / f).read_bytes()).hexdigest() for f in ("thinking.py", "thinkbench.py", "judgment.py", "drillsources.py")}
    before = _digest(st)
    launched: list[int] = []

    def launcher(s: Path, r: Path) -> int:
        launched.append(1)
        return 4242
    out = L.run(st, repo=tmp_path, journal=tmp_path / "J.md", now=L._ts("2026-10-03T09:00:00+00:00"), sources=(), launcher=launcher)
    after = _digest(st)
    changed = {k for k in after if before.get(k) != after[k]}
    allowed = {str(Path("thinking") / f) for f in L.FILES.values()} | {"goal_proposals.jsonl"}
    assert changed <= allowed, changed - allowed
    for k in ("trust.json", str(Path("thinkbench") / "items.json"), str(Path("thinkbench") / "baseline_20261003T010000Z.json"), "ledger.jsonl",
              str(Path("thinking") / "drill_runs.jsonl")):
        assert after[k] == before[k]
    assert code == {f: hashlib.sha256((ROOT / "creator" / f).read_bytes()).hexdigest() for f in code}
    assert not out["errors"], out["errors"]
    src = (ROOT / "creator" / "learnloop.py").read_text(encoding="utf-8")
    assert "trust.json\").write" not in src and "write=True" not in src  # the loop only reads the measurements
