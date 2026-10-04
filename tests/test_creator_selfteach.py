"""h58: the self-teaching trust gate (creator.selfteach) and the three measurement fixes it found - the walk-forward tie leak, frozen benchmark
items in the select part / the diagnosis, and the missing auto-revert of a regression - plus the protected measuring code and cost per drill row."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import fastwalk as FW
from creator import learnloop as L
from creator import sandbox as S
from creator import selfteach as ST
from creator import thinking as T


# ------------------------------------------------------------------------------------------------ fix 2: the tie leak
def test_canary_an_item_is_never_predicted_with_its_own_outcome() -> None:
    can = ST.leak_canaries()
    assert can["leak"] is False, can
    assert can["in_use"]["p_tied"] == can["in_use"]["p_honest"] == can["reference"]["p_tied"]


def test_plan_like_items_resolved_at_creation_do_not_see_their_own_label() -> None:
    # every item resolves at its creation instant (plan_choice / research_bool): a unique key per item, outcome 1 -> before h58 p > global
    items = [D.BItem((f"u{i}",), float(i), float(i), 1, f"s{i}") for i in range(30)]
    for p in D.walk_forward_reference(items, "x", 1.0) + D.walk_forward(items, "x", 1.0):
        glob = (sum(1 for it in items if it.created < p.made_at) + 1.0) / (sum(1 for it in items if it.created < p.made_at) + 2.0)
        assert p.p == pytest.approx(min(1 - 0.02, max(0.02, glob)))       # only EARLIER items (the global rate), never its own key
        assert p.last == (0.5 if p.made_at == 0.0 else 0.75)             # the last-value baseline is the previous item, not itself


@pytest.mark.parametrize("seed", range(4))
def test_compiled_and_reference_agree_with_ties(seed: int) -> None:
    rnd = random.Random(seed)
    items = [D.BItem(tuple(f"k{rnd.randrange(8)}" for _ in range(2)), float(i // 4), float(i // 4) + rnd.choice([0.0, 0.0, 1.0, 3.0]),
                     int(rnd.random() < 0.4), f"s{i}") for i in range(300)]
    a, b = D.walk_forward_reference(items, "x"), FW.walk_forward(items, "x")
    assert [(p.subject, p.made_at) for p in a] == [(p.subject, p.made_at) for p in b]
    assert all(x.p == pytest.approx(y.p, abs=1e-9) and x.base == pytest.approx(y.base) and x.last == y.last for x, y in zip(a, b))


def test_old_rows_are_marked_superseded_and_never_trusted(tmp_path: Path) -> None:
    assert D.superseded({"digest": "abc"}) and not D.superseded({"digest": f"abc.{D.WALK_VERSION}"})
    assert set(D.LEAKY_BEFORE_W2) == {"plan_choice", "research_bool"}
    good = {"n": 400, "n_live": 0, "gain_vs_best": 0.05, "gain_ci95": [0.03, 0.07], "ece": 0.01, "brier": 0.1, "best_baseline": "base"}
    row = {"source": "plan_choice", "variant": {"decay": 1.0, "k": 2.0}, "digest": "123", "items": 900, "resolved": 890,
           "select": {"n": 500, "brier": 0.1}, "heldout": good, "at": "2026-10-03T01:00:00+00:00"}
    p = D.runs_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(row) + "\n")
    sec = D.trust_section(tmp_path)["plan_choice"]
    assert sec["superseded"] is True and sec["trusted"] is False and any("LEAKY" in w for w in sec["why_not"])


# ------------------------------------------------------------------------------------------------ fix 3: the frozen benchmark
def _freeze(state: Path, a: list[dict[str, Any]], pin: bool = True) -> str:
    body: dict[str, Any] = {"version": 1, "a": a, "b": [], "c": [], "d": []}
    h = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    body["hash"] = h if pin else "0" * 64
    (state / "thinkbench").mkdir(parents=True, exist_ok=True)
    (state / "thinkbench" / "items.json").write_text(json.dumps(body), encoding="utf-8")
    return h


def test_frozen_subjects_are_read_and_the_hash_is_checked_from_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _freeze(tmp_path, [{"id": "a:git_fixed:s3", "source": "git_fixed", "subject": "s3", "created": 3.0, "y": 1}])
    assert D.frozen_subjects(tmp_path, "git_fixed") == {"s3"} and D.frozen_subjects(tmp_path, "git_churn") == frozenset()
    assert D.frozen_hash_ok(tmp_path) is False                           # a test set is not the pinned real one
    monkeypatch.setattr(D, "PINNED_BENCH_HASH", h[:12])
    D._FROZEN_MEMO.clear()
    assert D.frozen_hash_ok(tmp_path) is True
    _freeze(tmp_path / "edited", [{"source": "git_fixed", "subject": "s3"}], pin=False)   # the file's own hash field was not updated
    assert D.frozen_hash_ok(tmp_path / "edited") is False and D.frozen_subjects(tmp_path / "edited", "git_fixed") == {"s3"}
    assert D.frozen_hash_ok(tmp_path / "none") is None


def test_select_part_never_contains_frozen_items_or_the_heldout_tail() -> None:
    preds = [T.Pred("x", f"s{i}", float(i), 0.5, 0.5, 0.5, i % 2) for i in range(100)]
    sp = D.split_score(preds, {"s3", "s70"})
    assert sp["select"]["n"] == 59 and sp["heldout"]["n"] == 40          # s3 (in select) dropped; s70 (held-out) stays measured
    sel = D.select_part(list(reversed(preds)), {"s3"})
    assert len(sel) == 59 and max(p.made_at for p in sel) < 60 and all(p.subject != "s3" for p in sel)


def test_learnloop_diagnosis_and_react_see_only_the_select_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze(tmp_path, [{"source": "git_fixed", "subject": "s5"}, {"source": "git_fixed", "subject": "s90"}])
    items = [D.BItem(("k",), float(i), float(i) + 0.5, i % 2, f"s{i}") for i in range(100)]
    pairs = [(it, T.Pred("git_fixed", it.subject, it.created, 0.5, 0.5, 0.5, it.y)) for it in items]
    dp = L.diag_pairs(tmp_path, "git_fixed", pairs)
    assert {it.subject for it, _p in dp} == {f"s{i}" for i in range(60)} - {"s5"}

    seen: list[set[str]] = []
    real = L.drill_groups

    def spy(source: str, ps: Any, acc: Any) -> None:
        seen.append({it.subject for it, _p in ps})
        real(source, ps, acc)
    monkeypatch.setattr(L, "drill_groups", spy)
    L.diagnose(tmp_path, tmp_path, type("Dt", (), {"get": lambda self, s: {"preds": pairs} if s == "git_fixed" else None})(), {}, {},
               ["git_fixed"], 1e9)
    from creator import trialerror as TE
    row = {"source": "git_fixed", "variant": {"decay": 1.0, "k": 2.0}, "digest": "d", "items": 100, "resolved": 100,
           "select": {"n": 59, "brier": 0.2}, "heldout": {"n": 40, "brier": 0.2}}
    p = D.runs_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(D, "load", lambda *a, **k: items)
    TE.react(tmp_path, row, tmp_path, tmp_path / "J.md", tmp_path, now=1e9)
    assert len(seen) == 2
    for s in seen:
        assert "s5" not in s and "s90" not in s and not any(int(x[1:]) >= 60 for x in s)


# ------------------------------------------------------------------------------------------------ fix 1: auto-revert
def _row(v: dict[str, Any], sel: float, held: float) -> dict[str, Any]:
    return {"source": "x_git_fixed", "variant": v, "search": True, "digest": "d.w2", "items": 500, "resolved": 480,
            "select": {"n": 288, "brier": sel}, "heldout": {"n": 192, "brier": 0.17, "gain_vs_best": held, "gain_ci95": [held - 0.001, held + 0.001]},
            "at": "2026-10-03T01:00:00+00:00"}


def test_a_regression_never_stays_the_best_and_the_revert_is_recorded(tmp_path: Path) -> None:
    prev = {"agg": "mean", "cap": 0.02, "decay": 0.97, "extra": "none", "k": 2.0, "mask": 0, "search": 1, "learn": "IM-12cf051cc5"}
    bad = {"agg": "logit", "cap": 0.02, "decay": 1.0, "extra": "none", "k": 2.0, "learn": "IM-d2e9a3990c", "mask": 13, "search": 1}
    p = D.runs_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("\n".join(json.dumps(r) for r in (_row(prev, 0.20, 0.0259), _row(bad, 0.19, 0.0107))) + "\n")
    assert D.best_variant(tmp_path, "x_git_fixed")["variant"] == bad      # type: ignore[index]
    applied = {"event": "applied", "id": "IM-d2e9a3990c", "action": "queue_variants", "t": 0.0, "skill": "drill:x_git_fixed:heldout",
               "weakness": {"id": "W-1", "source": "x_git_fixed"}, "before": {"value": 0.0259, "ci": [0.025, 0.0268]}, "variants": [bad]}
    verdict = {"event": "verdict", "id": "IM-d2e9a3990c", "verdict": "hurt_flag_revert", "skill": "drill:x_git_fixed:heldout"}
    L._append(tmp_path, "improvements", [applied, verdict])
    b = D.best_variant(tmp_path, "x_git_fixed")
    assert b is not None and b["variant"] == prev and b["reverted_skipped"] == 1
    imps = L.improvements(tmp_path)
    recs = L.record_reverts(tmp_path, imps, 1e9)
    assert len(recs) == 1 and recs[0]["best_now"] == prev and recs[0]["heldout_now"] == 0.0259
    assert L.record_reverts(tmp_path, L.improvements(tmp_path), 1e9 + 1) == []      # recorded once
    rep = ST.check_heldout(tmp_path, {})
    assert rep["evidence"]["open_regressions"] == []


# ------------------------------------------------------------------------------------------------ safe fixes + the gate
def test_measuring_code_is_protected_from_nupens_packages() -> None:
    for p in ST.MEASURING_PATHS:
        assert S.is_protected(p), p
    assert ST.check_own_ruler()["pass"]


def test_a_drill_row_records_its_cost_and_versioned_digest(tmp_path: Path) -> None:
    (tmp_path / "plan_explanations.jsonl").write_text("\n".join(json.dumps({"at": f"2026-10-03T00:00:{i:02d}Z", "chosen": i % 3 == 0, "step": i % 4,
                                                                             "component": "c", "value": 1.0, "node": f"n{i}"}) for i in range(40)))
    row = D.compute_row("plan_choice", {"decay": 1.0, "k": 2.0}, tmp_path, tmp_path, tmp_path / "J.md", tmp_path)
    assert row["cpu_s"] >= 0 and row["wall_s"] >= 0 and row["digest"].endswith("." + D.WALK_VERSION) and not D.superseded(row)


def test_the_gate_runs_on_an_empty_state_and_writes_its_reports(tmp_path: Path) -> None:
    rep = ST.gate(tmp_path, journal=tmp_path / "none.md", now=1e9, out_dir=tmp_path / "out")
    assert rep["self_teach_trusted"] is False and set(rep["checks"]) >= {k for k, _d in ST.ORDER}
    assert rep["checks"]["b_leakproof"]["evidence"]["canaries"]["leak"] is False
    assert all(rep["checks"][k]["fix"] for k in rep["failing"])            # every failing check names its fix
    assert (tmp_path / "out" / "selfteach_gate.json").exists() and "SELF_TEACH_TRUSTED" in (tmp_path / "out" / "selfteach_gate.md").read_text()
    assert not (tmp_path / "trust.json").exists() and not D.runs_path(tmp_path).exists()      # read-only on the records


# ------------------------------------------------------------------------------------------------ the GPU job (dry run with a stub)
def test_the_heldout_gate_job_spec_is_valid_for_the_runner() -> None:
    from creator import gpupulse as GP
    from creator import gpuselfteach as G
    j = GP.parse_job(G.HELDOUT_GATE_JOB)
    assert j["kind"] == "ext" and j["via"] == "call" and j["model"] == "Qwen3-1.7B-gpuday-ft1.gguf"
    mod, _, fn = j["call"].partition(":")
    assert callable(getattr(__import__(mod, fromlist=[fn]), fn))


def test_heldout_gate_dry_run_with_a_stub_model() -> None:
    from creator import gpuselfteach as G
    from creator import reasondrills as R
    qs = [R.Question(f"r:which_first:p:{i}", "which_first", "p", f"s{i}", float(i), f"Which came first? #{i}", ["a", "b", "c", "d"], i % 4, "easy")
          for i in range(200)]
    bank = {"r:which_first:p:5": {"t": 99.0, "trace": "x", "model": "m", "gpu_pulse": "p"}}
    rows = [{"qid": q.qid, "strategy": "plain", "model": "home.gguf", "correct": int(i % 2 == 0)} for i, q in enumerate(qs)]
    pairs, cut = G.heldout_set(qs, rows, bank, "home.gguf", 500)
    assert cut == 99.0 and all(q.t > 99.0 for q, _h in pairs) and len(pairs) == 100       # only newer than every banked (training) question
    answers = {q.qid: q.answer for q in qs}

    class Stub:                                                       # the "tuned" model: always right
        model = "tuned.gguf"

        def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
            q = next(x for x in qs if x.stem in messages[-1]["content"] and x.prompt() in messages[-1]["content"])
            return f"ANSWER: {'ABCD'[answers[q.qid]]}"
    good = G.run_gate(Stub(), pairs, workers=2, ask=lambda m, msgs: m.chat(msgs))
    assert good["verdict"] == "ADOPT" and good["n"] == 100 and good["gain_ci95"][0] > 0
    bad = G.run_gate(Stub(), pairs[:20], workers=2, ask=lambda m, msgs: m.chat(msgs))
    assert bad["verdict"] == "KEEP_HOME"                              # too few held-out pairs: never adopted on a small sample
