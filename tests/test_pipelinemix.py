"""h61 creator.pipelinemix: the multi-role pipeline adapter's data (role rows, Aider distill, balance, per-role held-out scoring)."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import pytest

from creator import pipelinemix as PM
from creator import trainmix as TM

HELD = {"cut_ts": 1000.0, "held_out_commits": [], "tasks": []}
STUB = "def f(*args, **kwargs):\n    raise NotImplementedError\n"
INPUT = f"ROLE: CODE\nRequest: write f\n\nCurrent solution.py:\n```python\n{STUB}```\nEdit format: ..."
GOOD = "<<<<<<< SEARCH\n    raise NotImplementedError\n=======\n    return args[0] * 2\n>>>>>>> REPLACE"
TEST = {"name": "f", "cases": [{"args": [2], "expect": 4}, {"args": [5], "expect": 10}]}


def row(role: str, tid: str, model: str, verified: Any, out: str = "x", **kw: Any) -> dict[str, Any]:
    return {"role": role, "task_id": tid, "model": model, "verified": verified, "input": INPUT, "output": out, **kw}


def test_role_rows_are_verified_only_and_prefer_the_strongest_model() -> None:
    raw = [row("CODE", "hf:a", "Qwen3-1.7B-Q4_K_M.gguf", True, "small"), row("CODE", "hf:a", "Qwen3.6-27B-Q4_K_M.gguf", True, "big"),
           row("CODE", "hf:b", "Qwen3.6-27B-Q4_K_M.gguf", False), row("REVIEW", "hf:b", "Qwen3-4B-Q4_K_M.gguf", True, "VERDICT: correct"),
           row("PLAN", "nupen:c", "Qwen3.6-27B-Q4_K_M.gguf", True), row("trajectory", "hf:a", "Qwen3.6-27B-Q4_K_M.gguf", True)]
    d: collections.Counter[str] = collections.Counter()
    out = PM.src_roles(None, d, [r for r in raw if PM.role_of(r)])
    by = {(r.group, r.meta["role"]): r for r in out}
    assert set(by) == {("hf:a", "CODE"), ("hf:b", "REVIEW")}
    assert by[("hf:a", "CODE")].body["messages"][-1]["content"] == "big"                     # 27B row preferred
    assert by[("hf:a", "CODE")].body["messages"][0]["content"].startswith("[ROLE: CODE]")
    assert d["roles: not verified"] == 1 and d["roles: not a public task"] == 1


def test_code_and_debug_are_scored_by_running_the_tests_after_applying_the_edits() -> None:
    meta = {"check": "py", "test": TEST, "current": STUB}
    assert PM.apply_edits(STUB, GOOD) == "def f(*args, **kwargs):\n    return args[0] * 2\n"
    assert PM.score("CODE", meta, "", GOOD)["correct"] is True
    assert PM.score("CODE", meta, "", GOOD.replace("* 2", "* 3"))["correct"] is False
    bad = PM.score("DEBUG", meta, "", GOOD.replace("raise NotImplementedError", "missing line"))
    assert bad == {"correct": False, "applied": False}                                     # a SEARCH that does not match fails
    assert PM.score("CODE", meta, "", "```python\ndef f(x):\n    return x * 2\n```")["correct"] is True


def test_review_and_validate_are_scored_against_the_recorded_truth() -> None:
    assert PM.score("REVIEW", {"label": True}, "", "VERDICT: correct\nISSUES: - none")["correct"] is True
    assert PM.score("REVIEW", {"label": False}, "", "VERDICT: correct")["correct"] is False
    assert PM.score("REVIEW", {"label": False}, "", "no verdict line") == {"correct": False, "parsed": False}
    assert PM.score("VALIDATE", {"label": False}, "", "<think>MEETS SPEC: yes</think>MEETS SPEC: no\nGAPS: - x")["correct"] is True
    assert PM.score("PLAN", {}, "", "STEPS: 1.") is None                                    # no automatic metric


def test_heldout_eval_rows_carry_tests_or_labels_and_never_train(tmp_path: Path) -> None:
    raw = [row("CODE", "hf:h", "Qwen3-4B-Q4_K_M.gguf", False), row("REVIEW", "hf:h", "Qwen3-4B-Q4_K_M.gguf", False, real_pass=True),
           row("VALIDATE", "hf:h", "Qwen3-4B-Q4_K_M.gguf", True, truth=False), row("PLAN", "hf:h", "Qwen3-4B-Q4_K_M.gguf", True),
           row("CODE", "hf:t", "Qwen3-4B-Q4_K_M.gguf", True)]
    ev = PM.heldout_eval_rows(raw, {"hf:h"}, {"hf:h": TEST})
    assert set(ev) == {"CODE", "REVIEW", "VALIDATE"}
    assert ev["CODE"][0]["meta"]["current"] == STUB and ev["REVIEW"][0]["meta"]["label"] is True and ev["VALIDATE"][0]["meta"]["label"] is False


def test_balance_caps_each_role() -> None:
    rows = [TM.Row(f"r{i}", "s", f"g{i}", {}, meta={"role": "PLAN" if i < 50 else "DEBUG"}) for i in range(60)]
    bal, counts = PM.balance(rows, cap=20)
    assert counts == {"DEBUG": 10, "PLAN": 20} and len(bal) == 30


def test_aider_rows_are_read_filtered_and_tagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PM, "aider_dir", lambda: tmp_path)
    msgs = [{"role": "system", "content": "aider system"}, {"role": "user", "content": "edit it"}, {"role": "assistant", "content": GOOD}]
    rows = [{"messages": msgs, "task_id": "hf:x", "passed": True}, {"messages": msgs, "task_id": "hf:y", "passed": False},
            {"messages": msgs, "meta": {"task_id": "pub:z", "role": "DEBUG"}}, {"messages": msgs, "task_id": "local:q"}]
    (tmp_path / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    d: collections.Counter[str] = collections.Counter()
    out = PM.src_aider(None, d, "sft")
    assert [(r.group, r.meta["role"]) for r in out] == [("hf:x", "CODE"), ("pub:z", "DEBUG")]
    assert out[0].body["messages"][0]["content"].startswith("[ROLE: CODE]") and out[0].meta["aider"]
    assert d["aider: not passed"] == 1 and d["aider: not a public task id"] == 1
    assert PM.src_aider(None, d, "pref") == []                                              # no pref file yet: nothing, no error


def test_pipeline_jobs_have_per_role_eval_for_base_and_tuned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import gpuday as GD
    from creator import gpupulse as GP
    monkeypatch.setattr(TM, "load_heldout", lambda path=None: HELD)
    monkeypatch.setattr(TM, "frozen", lambda: GD.Frozen.empty())
    for n in ("pipeline_17b", "pipeline_4b"):
        d = tmp_path / n
        d.mkdir()
        (d / "train.jsonl").write_text("{}\n" * 400, encoding="utf-8")
        (d / "dev.jsonl").write_text("{}\n", encoding="utf-8")
        (d / "pref.jsonl").write_text("", encoding="utf-8")
        (d / "MANIFEST.json").write_text(json.dumps({"rows": {"train": 400, "dev": 1, "pref": 0}, "tokens_train": 400_000}), encoding="utf-8")
    js = TM.jobs({}, root=tmp_path, targets=["pipeline_17b", "pipeline_4b"])
    for j in js:
        GP.ext_job(j)
    names = [j["name"] for j in js]
    i = names.index("ft_pipeline_17b")
    assert names[i + 1:i + 6] == ["register_pipeline_17b", "roleeval_pipeline_17b_base", "roleeval_pipeline_17b_tuned", "stopeval_pipeline_17b",
                                  "delete_pipeline_17b"]
    assert js[names.index("roleeval_pipeline_4b_base")]["model"] == "Qwen3-4B-Q4_K_M.gguf"
    ft4 = js[names.index("ft_pipeline_4b")]["remote"]
    assert "HF_HOME=\"/dev/shm/nupen_train/hf_cache\"" in ft4 and "needs 27" in ft4              # 4B base on the RAM disk; skip below 27 GB


def test_pipeline_eval_compare_is_paired_per_role() -> None:
    base = [{"id": f"c{i}", "role": "CODE", "scored": True, "correct": i % 2 == 0} for i in range(60)]
    tuned = [{"id": f"c{i}", "role": "CODE", "scored": True, "correct": True} for i in range(60)] + \
        [{"id": "p1", "role": "PLAN", "scored": False}]
    res = PM.compare(base, tuned)
    assert res["roles"]["CODE"]["verdict"] == "ADOPT" and res["roles"]["CODE"]["n"] == 60
    assert res["roles"]["PLAN"]["verdict"] == "NO_METRIC" and res["adopt_roles"] == ["CODE"]
