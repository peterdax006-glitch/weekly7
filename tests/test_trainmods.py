"""h61 creator.trainmods: the training-module bank (calibration, brevity, promptbake, draft, locate, aider, judge) - data rules and evals."""
from __future__ import annotations

import collections
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from creator import pipelinemix as PM
from creator import trainmix as TM
from creator import trainmods as M

HELD = {"cut_ts": 1000.0, "held_out_commits": [], "tasks": []}


def rrow(role: str, tid: str, verified: bool, out: str, model: str = "Qwen3-4B-Q4_K_M.gguf", inp: str = "ROLE: X\nrequest") -> dict[str, Any]:
    return {"role": role, "task_id": tid, "model": model, "verified": verified, "input": inp, "output": out}


def tasks(split: str, n: int) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        t = f"hf:t{i}"
        if PM.split(t) == split:
            out.append(t)
        i += 1
    return out


def test_calibration_targets_are_empirical_rates_from_training_tasks_only() -> None:
    tr, ho = tasks("train", 20), tasks("heldout", 5)
    raw = [rrow("REVIEW", t, i % 4 != 0, "VERDICT: correct") for i, t in enumerate(tr)] + \
          [rrow("REVIEW", t, False, "VERDICT: correct") for t in ho]                  # held-out outcomes must not move the rate
    rates = M.calib_rates(raw)
    k = ("REVIEW", "Qwen3-4B-Q4_K_M.gguf", "correct", 0)
    assert abs(rates[k] - 0.75) < 1e-9                                                # 15/20 (shrunk toward the same role rate)
    rows = M.src_calib(None, collections.Counter(), raw)
    assert {r.group for r in rows} == set(tr) and rows[0].body["messages"][-1]["content"] == "CONFIDENCE: 0.75"
    held = M.calib_heldout(raw)
    assert {h["meta"]["task"] for h in held} == set(ho) and all(h["meta"]["y"] == 0 for h in held)


def test_calibration_compare_uses_brier_and_reports_ece() -> None:
    base = [{"id": f"c{i}", "role": "CALIB", "scored": True, "correct": True, "p": 0.95, "y": i % 2, "brier": (0.95 - i % 2) ** 2} for i in range(80)]
    tuned = [{"id": f"c{i}", "role": "CALIB", "scored": True, "correct": True, "p": 0.5, "y": i % 2, "brier": 0.25} for i in range(80)]
    res = M._pm().compare(base, tuned, "calibration")["roles"]["CALIB"]
    assert res["verdict"] == "ADOPT" and res["brier_gain"] > 0 and res["ece_tuned"] == 0.0 and res["ece_base"] > 0.4


def test_brevity_sft_takes_the_shortest_verified_and_pairs_need_a_real_gap() -> None:
    raw = [rrow("PLAN", "hf:a", True, "x" * 100), rrow("PLAN", "hf:a", True, "y" * 400, "Qwen3.6-27B-Q4_K_M.gguf"),
           rrow("PLAN", "hf:a", False, "z" * 10), rrow("CODE", "hf:b", True, "c" * 100), rrow("CODE", "hf:b", True, "d" * 120)]
    sft = M.src_brevity(None, collections.Counter(), raw)
    assert {r.group: r.body["messages"][-1]["content"][0] for r in sft} == {"hf:a": "x", "hf:b": "c"}      # unverified short one ignored
    pref = M.src_brevity(None, collections.Counter(), raw, pref=True)
    assert [(r.group, r.body["chosen"][0]["content"][0], r.body["rejected"][0]["content"][0]) for r in pref] == [("hf:a", "x", "y")]


def test_brevity_adopts_fewer_tokens_only_at_an_equal_pass_rate() -> None:
    base = [{"id": f"r{i}", "role": "CODE", "scored": True, "correct": i % 2 == 0, "tok_out": 300} for i in range(80)]
    short = [dict(r, tok_out=120) for r in base]
    assert PM.compare(base, short, "brevity")["roles"]["CODE"]["verdict"] == "ADOPT"
    worse = [dict(r, tok_out=120, correct=False) for r in base]
    assert PM.compare(base, worse, "brevity")["roles"]["CODE"]["verdict"] == "KEEP_BASE"   # shorter but failing more: never adopted


def test_brief_sections_give_the_long_role_text(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Brief\nshared rules\n\n## Roles\n\n### SPEC (understand)\nlong spec text\n\n### CODE\ncode text\n## End\n",
                                   encoding="utf-8")
    b = PM.brief_sections(tmp_path / "b.md")
    assert set(b) == {"SPEC", "CODE"} and "shared rules" in b["SPEC"] and "long spec text" in b["SPEC"] and "code text" not in b["SPEC"]


def test_spec_compare_pairs_acceptance_per_prompt() -> None:
    base = [{"id": f"p{i}", "draft_n": 100, "accepted": 40, "tps": 200} for i in range(60)]
    tuned = [{"id": f"p{i}", "draft_n": 100, "accepted": 60 + i % 3, "tps": 260} for i in range(60)]
    res = M.spec_compare(base, tuned)
    assert res["verdict"] == "ADOPT" and res["n"] == 60 and res["acceptance_tuned"] > res["acceptance_base"]
    assert M.spec_compare(base[:30], tuned[:30])["verdict"] == "KEEP_BASE"


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_spec_remote_is_valid_bash() -> None:
    s = M.spec_remote("Qwen3-1.7B-Q4_K_M.gguf", "draft.gguf", "base", "gpuday/trainmix/draft_06b/spec_prompts.jsonl")
    r = subprocess.run([str(shutil.which("bash")), "-n"], input=s, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "kill $PID" in s and "--port 18990" in s                               # its own server, its own port, killed at exit


def test_module_jobs_are_valid_and_self_contained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import gpuday as GD
    from creator import gpupulse as GP
    monkeypatch.setattr(TM, "load_heldout", lambda path=None: HELD)
    monkeypatch.setattr(TM, "frozen", lambda: GD.Frozen.empty())
    names = [t.name for t in TM.TARGETS if t.module and not t.blocked]
    for t in TM.TARGETS:                                    # a blocked target (parked on purpose) is built and audited but never gets a GPU job
        if t.module and t.blocked:
            (tmp_path / t.name).mkdir()
            (tmp_path / t.name / "MANIFEST.json").write_text(json.dumps({"rows": {"train": 400, "dev": 400, "pref": 0}, "tokens_train": 400_000}), encoding="utf-8")
            assert not [j for j in TM.jobs({}, root=tmp_path, targets=[t.name], cleanup=False) if j["name"].startswith("ft_")]
    for n in names:
        d = tmp_path / n
        d.mkdir()
        for f in ("train.jsonl", "dev.jsonl"):
            (d / f).write_text("{}\n" * 400, encoding="utf-8")
        (d / "pref.jsonl").write_text("", encoding="utf-8")
        (d / "MANIFEST.json").write_text(json.dumps({"rows": {"train": 400, "dev": 400, "pref": 0}, "tokens_train": 400_000}), encoding="utf-8")
    for n in names:
        js = TM.jobs({}, root=tmp_path, targets=[n], cleanup=False)
        for j in js:
            GP.ext_job(j)
            if j.get("call"):
                mod, _, fn = j["call"].partition(":")
                assert callable(getattr(__import__(mod, fromlist=[fn]), fn))
        jn = [j["name"] for j in js]
        assert jn[0] == "trainmix_upload" and jn[1] == f"ft_{n}" and jn[-1] == f"delete_{n}" and "trainmix_cleanup" not in jn
        assert all("hf_cache" not in str(j.get("remote", "")).split("HF_HOME")[0] for j in js[1:])   # nothing deletes the base cache
    pb = TM.jobs({}, root=tmp_path, targets=["promptbake_17b"], cleanup=False)
    assert next(j for j in pb if j["name"] == "roleeval_promptbake_17b_base")["args"]["long_system"] is True
    dr = [j["name"] for j in TM.jobs({}, root=tmp_path, targets=["draft_06b"], cleanup=False)]
    assert dr.index("spec_draft_06b_tuned") < dr.index("delete_draft_06b") and "specgate_draft_06b" in dr


def test_adapter_entries_are_served_as_base_plus_lora(tmp_path: Path) -> None:
    from creator import gpupulse as GP
    sha = "a" * 64
    cfg = {"extra_models": {"Qwen3-1.7B-nupen-calib-Q4_K_M.gguf": {"bytes": 50, "sha256": sha, "base": "Qwen3-1.7B-Q4_K_M.gguf",
                                                                   "lora": "Qwen3-1.7B-nupen-calib-lora.gguf"}},
           "extra_models_file": str(tmp_path / "none.json")}
    e = GP.extra_models(cfg)["Qwen3-1.7B-nupen-calib-Q4_K_M.gguf"]
    assert e["lora"] == "Qwen3-1.7B-nupen-calib-lora.gguf" and e["base"] == "Qwen3-1.7B-Q4_K_M.gguf"
    assert GP.vram_need_gb("Qwen3-1.7B-nupen-calib-Q4_K_M.gguf", 4, 4096) > 1.0            # the base weights count, not the 50-byte adapter
    s = GP.server_script("/w", "/x/llama-server", "Qwen3-1.7B-nupen-calib-Q4_K_M.gguf", 18130, 4, 4096, base=e["base"], lora=e["lora"])
    assert "-m /w/models/Qwen3-1.7B-Q4_K_M.gguf" in s and "--lora /w/models/Qwen3-1.7B-nupen-calib-lora.gguf" in s
    assert "Qwen3-1.7B-nupen-calib-Q4_K_M.gguf|" in s                                     # the signature keeps the served name
    with pytest.raises(GP.PulseError):
        GP.extra_models({"extra_models": {"X.gguf": {"bytes": 5, "sha256": sha, "base": "../evil.gguf", "lora": "l.gguf"}},
                         "extra_models_file": str(tmp_path / "none.json")})


def test_register_lora_job_writes_an_adapter_entry(tmp_path: Path) -> None:
    from creator import gpupulse as GP
    st = tmp_path / "state"
    (st / "thinking").mkdir(parents=True)
    row = {"job": "ext:ft_calib_17b", "rc": 0, "gpu_pulse": "p", "result": {"lora_gguf": {"files": {"adapter.gguf": {"bytes": 77, "sha256": "b" * 64}}}}}
    (st / "thinking" / "gpu_pulse_runs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    reg = tmp_path / "extra.json"
    res = TM.register_lora_job({"state": str(st), "pulse": "p", "args": {"source": "ft_calib_17b", "serve_as": "S.gguf", "base": "Qwen3-1.7B-Q4_K_M.gguf",
                                                                          "lora": "L.gguf", "extra_models_file": str(reg)}})
    assert res["registered"] and GP.extra_models({"extra_models_file": str(reg)})["S.gguf"]["lora"] == "L.gguf"
    assert not TM.register_lora_job({"state": str(st), "pulse": "q", "args": {"source": "ft_calib_17b", "serve_as": "S.gguf", "base": "B.gguf",
                                                                               "lora": "L.gguf", "extra_models_file": str(reg)}})["registered"]


def test_specs_are_config_and_unknown_keys_are_refused() -> None:
    t = TM.target_from_spec({"name": "x_17b", "size": "1.7b", "sources": ["roles"], "filter": {"role": ["DEBUG"]}, "module": "roles",
                             "packing": True, "lora_serve": True, "target_minutes": [45, 60]})
    assert t.sources == ("roles",) and dict(t.filter or ()) == {"role": ("DEBUG",)} and t.target_minutes == (45, 60)
    with pytest.raises(ValueError, match="unknown keys"):
        TM.target_from_spec({"name": "y", "size": "1.7b", "srcs": []})
    names = [t.name for t in TM.TARGETS if t.module]
    idx = [names.index(n) for n in ("calib_17b", "brevity_17b", "promptbake_17b", "locate_17b")]
    assert idx == sorted(idx)                                                              # value order from the spec priorities
    lo = TM.BY_NAME["calib_17b"]
    ft = TM.ft_remote(lo, {"rows": {"pref": 0}, "tokens_train": 2_000_000})
    assert "--no-merge" in ft and "--packing" not in ft and "-lora.gguf" in ft and ".prestage_" in ft   # packing off: it broke under Unsloth (4 Oct)
    js = TM.jobs({}, root=TM.mix_dir(), targets=["judge_06b"], cleanup=False) if (TM.mix_dir() / "judge_06b" / "MANIFEST.json").is_file() else []
    if js:
        n = [j["name"] for j in js]
        assert n.index("fetch_judge_06b_basegguf") < n.index("register_judge_06b") and n[-1] == "delete_judge_06b"


def test_epochs_fill_the_window_and_autogen_respects_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = TM.BY_NAME["calib_17b"]
    assert TM.epochs_for(t, {"tokens_train": 2_000_000, "rows": {"train": 1}}) == 2.0                # two passes fit 60 min
    assert TM.epochs_for(t, {"tokens_train": 30_000_000, "rows": {"train": 1}}) == 1.0
    st = tmp_path / "s"
    (st / "thinking").mkdir(parents=True)
    ok, why = M.gates_ok(TM.BY_NAME["draft_06b"], None, st)
    assert not ok and "pipeline_17b" in why
    (st / "thinking" / "train_gate.jsonl").write_text(json.dumps({"target": "pipeline_17b", "compare": {"adopt_roles": ["CODE"]}}) + "\n",
                                                       encoding="utf-8")
    assert M.gates_ok(TM.BY_NAME["draft_06b"], None, st)[0]


def test_speed_paths_need_a_gpu_measurement_and_stages_split_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "peaks_path", lambda: tmp_path / "peaks.json")
    t17, t06 = TM.BY_NAME["calib_17b"], TM.BY_NAME["judge_06b"]
    assert M.ckpt_plan(t06)[0] is False                                               # never on an estimate
    (tmp_path / "peaks.json").write_text(json.dumps({"0.6b": {"ckpt": 8.0, "nockpt": 12.0}, "1.7b": {"ckpt": 25.0, "nockpt": 38.0}}), encoding="utf-8")
    assert M.ckpt_plan(t06) == (True, 13.0) and M.ckpt_plan(t17) == (False, 26.0)    # 1.7B no-ckpt + an eval server would not fit
    assert not M.can_pair(t17, t06) and M.can_pair(t06, t06)
    jobs = [{"name": "trainmix_upload", "remote": "x", "minutes": 1}, {"name": "ft_a", "remote": "x", "minutes": 1, "free_gpu": True},
            {"name": "register_a", "call": "m:f", "minutes": 1}, {"name": "roleeval_a_base", "call": "m:f", "minutes": 1},
            {"name": "stopeval_a", "remote": "x", "minutes": 1}, {"name": "delete_a", "remote": "x", "minutes": 1},
            {"name": "prestage_b", "remote": "x", "minutes": 1}]
    tr, ev = M.split_stages(jobs)
    assert [j["name"] for j in tr] == ["trainmix_upload", "ft_a", "register_a", "prestage_b"] and "free_gpu" not in tr[1]
    assert [j["name"] for j in ev] == ["roleeval_a_base", "stopeval_a", "delete_a"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_stopeval_stops_only_servers_of_its_models(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    procs = {}
    for port, model in ((18130, "Qwen3-1.7B-nupen-calib-Q4_K_M.gguf"), (18140, "Other.gguf")):
        p = subprocess.Popen([str(shutil.which("bash")), "-c", "sleep 60"])
        procs[port] = p
        (run / f"{port}.pid").write_text(str(p.pid), encoding="utf-8")
        (run / f"{port}.sig").write_text(f"{model}|8|4096", encoding="utf-8")
    s = TM.stop_model_servers_remote(["Qwen3-1.7B-nupen-calib-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf"])
    r = subprocess.run([str(shutil.which("bash")), "-s"], input=s, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert "18130" in r.stdout and not (run / "18130.sig").exists() and (run / "18140.sig").exists()
    for p in procs.values():
        p.kill()


def test_pod_exec_runs_the_programs_in_one_batch(tmp_path: Path) -> None:
    from creator import gpupulse as GP
    if shutil.which("bash") is None:
        pytest.skip("no bash")
    sh = GP.Shell([str(shutil.which("bash")), "-s"])
    progs = {"ok": "def f(x):\n    return x * 2\nassert f(2) == 4\n", "bad": "assert 1 == 2\n"}
    try:
        got = PM.pod_exec(progs, shell=sh)
    except RuntimeError as e:
        pytest.skip(f"no python3 / nice in this bash: {e}")
    assert got == {"ok": True, "bad": False}


def test_safe_mode_retry_and_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = TM.target_from_spec({"name": "x_17b", "size": "1.7b", "sources": ["roles"], "module": "roles", "packing": True, "batch": 8, "accum": 2})
    ft = TM.ft_remote(t, {"rows": {"pref": 0}, "tokens_train": 1_000_000})
    first, retry = ft.split("safe mode retry", 1)
    assert "--packing" in first and "--batch 8 --accum 2" in first
    assert "--packing" not in retry and "--batch 4 --accum 4" in retry and "safe_mode" in retry
    st = tmp_path / "s"
    (st / "thinking").mkdir(parents=True)
    rows = [{"job": "ext:ft_x_17b", "rc": 5, "tail": "ValueError: boom"}, {"job": "ext:ft_x_17b", "rc": 0}, {"job": "ext:smoke_x_17b", "rc": 6}]
    (st / "thinking" / "gpu_pulse_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert len(M.failures(st, "x_17b")) == 2
    monkeypatch.setattr(TM, "mix_dir", lambda: tmp_path / "mix")
    f = M.suspend(t, "ValueError: boom")
    assert json.loads(f.read_text())["suspended"] == "ValueError: boom" and TM.target_from_spec(json.loads(f.read_text())).suspended
    h1 = M.code_hash(t)
    assert h1 == M.code_hash(t) and h1 != M.code_hash(TM.target_from_spec({"name": "x_17b", "size": "1.7b", "sources": ["roles"], "batch": 2}))
