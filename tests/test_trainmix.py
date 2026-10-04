"""h61 creator.trainmix: training mixes for the small home models (exclusion rule, dedupe vs eval sets, paired adoption rule, GPU job specs)."""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from creator import gpuday as GD
from creator import trainmix as TM

ROOT = Path(__file__).resolve().parents[1]
HELD = {"cut_ts": 1000.0, "cut_utc": "1970-01-01T00:16:40+00:00", "held_out_commits": ["abcdef1234567890"], "tasks": [{"id": "ct-1", "task": "x"}]}


def chat(user: str, answer: str, system: str = "sys") -> dict[str, Any]:
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def test_exclusion_rule_matches_the_trust_gate_semantics() -> None:
    assert TM.excluded(1000.0, "", HELD) and TM.excluded(5000.0, "", HELD)            # at or after the cut
    assert not TM.excluded(999.0, "", HELD) and not TM.excluded(0.0, "", HELD)
    assert TM.excluded(0.0, "abcdef1", HELD) and TM.excluded(0.0, "abcdef1234567890ff", HELD)   # held-out commit, prefix either way
    assert not TM.excluded(0.0, "abcde", HELD)                                          # under 7 chars: never a match


def test_codetrust_audit_agrees_when_available() -> None:
    try:
        from creator import codetrust as CT  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("creator.codetrust is not merged into this checkout")
    for ts, sha in ((1000.0, ""), (999.0, ""), (0.0, "abcdef1"), (0.0, "abcde")):
        assert CT.excluded(ts, sha, HELD) == TM.excluded(ts, sha, HELD)


def test_near_duplicates_are_found_and_distinct_texts_are_not() -> None:
    base = " ".join(f"word{i}" for i in range(300))
    near = base.replace("word150", "changed")
    idx = TM.NearIndex()
    idx.add("a", base)
    assert idx.match(base) == "a"                         # exact
    assert idx.match(near) == "a"                         # one word changed in 300: near-duplicate
    assert idx.match(" ".join(f"other{i}" for i in range(300))) is None
    assert idx.match("WORD0 word1, word2!! " + base[len("word0 word1 word2 "):]) == "a"     # case/punctuation-normalised exact key


def test_screen_drops_eval_overlap_privacy_frozen_long_and_internal_duplicates() -> None:
    evals = {"ladder_eval": {"texts": [("e1", "Which commit came first in repository alpha? A. one B. two C. three D. four")],
                             "ids": {"r:q:eval"}, "hash": "h"},
             "bakeoff_tasks": {"texts": [], "ids": {"5c2f69d928f692cf"}, "hash": None, "fn_names": ["wilson"]}}
    rows = [TM.Row("ok", "s", "g1", chat("Write a function that adds two numbers together please", "```python\ndef add(a, b):\n    return a + b\n```")),
            TM.Row("dup", "s", "g2", chat("Write a function that adds two numbers together please", "```python\ndef add(a, b):\n    return a + b\n```")),
            TM.Row("evaltext", "s", "g3", chat("Which commit came first in repository alpha? A. one B. two C. three D. four", "ANSWER: A")),
            TM.Row("evalid", "s", "r:q:eval", chat("some other question text that is long enough", "ANSWER: B")),
            TM.Row("bakeoffsha", "s", "g5", chat("implement the thing in the codebase now", "done"), sha="5c2f69d928f6"),
            TM.Row("bakeofffn", "s", "g6", chat("write the interval function here now", "```python\ndef wilson(k, n):\n    return 0\n```")),
            TM.Row("private", "s", "g7", chat("read Masterstock notes and summarise them", "no")),
            TM.Row("email", "s", "g8", chat("contact someone@example.com about this", "ok")),
            TM.Row("frozen", "s", "g9", chat("FROZEN QUESTION TEXT " * 5, "ANSWER: C")),
            TM.Row("long", "s", "g10", chat("long " * 2000, "x")),
            TM.Row("empty", "s", "g11", chat("a question with an empty answer here", ""))]
    fz = GD.Frozen(set(), set(), set(), ["FROZEN QUESTION TEXT " * 2])
    drops: collections.Counter[str] = collections.Counter()
    kept = TM.screen(rows, evals, fz, 1000, drops)
    assert [r.id for r in kept] == ["ok"]
    assert drops["near-duplicate inside the mix"] == 1 and drops["near-duplicate of eval set ladder_eval"] == 1
    assert drops["eval/held-out item id or commit"] == 2 and drops["bake-off function"] == 1
    assert drops["private marker"] == 1 and drops["e-mail address"] == 1 and drops["frozen thinkbench"] == 1
    assert drops["longer than 1000 tokens"] == 1 and drops["empty answer"] == 1


def test_reasoning_rows_check_the_answer_and_the_cut() -> None:
    from creator import reasondrills as R
    qs = {"r:which_first:pub:1": R.Question("r:which_first:pub:1", "which_first", "pub", "s", 5000.0, "Which first?", ["a", "b", "c", "d"], 2, "easy"),
          "r:which_first:nupen:2": R.Question("r:which_first:nupen:2", "which_first", "nupen", "s", 5000.0, "Which?", ["a", "b", "c", "d"], 1, "easy"),
          "r:which_first:nupen:3": R.Question("r:which_first:nupen:3", "which_first", "nupen", "s", 10.0, "Old?", ["a", "b", "c", "d"], 0, "easy")}
    ctx = TM.Ctx(Path("."), Path("."), HELD, qs)
    d: collections.Counter[str] = collections.Counter()
    pub = TM._reason_rows(ctx, "r:which_first:pub:1", "Because c is older.\nANSWER: C", "trace_bank", d, ("cot", "plain"))
    assert [r.body["messages"][-1]["content"] for r in pub] == ["Because c is older.\nANSWER: C", "ANSWER: C"]
    assert not pub[0].own and pub[0].group == "r:which_first:pub:1"           # public history: not judged by the Nupen cut
    assert TM._reason_rows(ctx, "r:which_first:pub:1", "ANSWER: A", "trace_bank", d, ("cot",)) == []
    assert TM._reason_rows(ctx, "r:which_first:nupen:2", "ANSWER: B", "trace_bank", d, ("cot",)) == []   # Nupen history after the cut
    assert len(TM._reason_rows(ctx, "r:which_first:nupen:3", "ANSWER: A", "trace_bank", d, ("cot",))) == 1
    assert TM._reason_rows(ctx, "r:unknown", "ANSWER: A", "trace_bank", d, ("cot",)) == []
    assert d["reasoning: answer line != known answer"] == 1 and d["exclusion rule: Nupen's own history at/after the cut"] == 1


def _rows(model: str, suite: str, config: str, correct: list[int], tok: float) -> list[dict[str, Any]]:
    return [{"items_hash": "h", "model": model, "suite": suite, "config": config, "split": "eval", "item": f"i{k}", "correct": bool(c),
             "tok_out": tok} for k, c in enumerate(correct)]


def test_compare_adopts_only_a_significant_paired_gain_with_enough_items() -> None:
    n = 120
    base = _rows("base.gguf", "reasoning", "plain@32", [i % 2 for i in range(n)], 5) + \
        _rows("base.gguf", "reasoning", "think@2048", [int(i % 4 != 0) for i in range(n)], 1100)
    good = _rows("tuned.gguf", "reasoning", "plain@32", [int(i % 10 != 0) for i in range(n)], 5)
    res = TM.compare(base + good, "tuned.gguf", "base.gguf", ["reasoning"])
    same = next(p for p in res["pairs"] if p["base_config"] == "plain@32")
    assert res["verdict"] == "ADOPT" and same["n"] == n and same["gain_ci95"][0] > 0
    assert any("think@2048" in s for s in res["cheaper_not_worse"])            # 5 tokens not worse than 1100 tokens: the saving
    small = TM.compare(base + good[:30], "tuned.gguf", "base.gguf", ["reasoning"])
    assert small["verdict"] == "KEEP_BASE"                                        # n < 50: never adopted
    worse = TM.compare(base + _rows("tuned.gguf", "reasoning", "plain@32", [0] * n, 5), "tuned.gguf", "base.gguf", ["reasoning"])
    assert worse["verdict"] == "KEEP_BASE" and worse["cheaper_not_worse"] == []
    other = TM.compare(base + [dict(r, items_hash="other") for r in good], "tuned.gguf", "base.gguf", ["reasoning"])
    assert all(p["n"] == 0 for p in other["pairs"])                               # different item sets are never paired


def test_heldout_gate_excludes_the_training_mix_questions_too() -> None:
    from creator import gpuselfteach as G
    from creator import reasondrills as R
    qs = [R.Question(f"r:which_first:p:{i}", "which_first", "p", f"s{i}", float(i), f"Q{i}", ["a", "b", "c", "d"], i % 4, "easy") for i in range(200)]
    rows = [{"qid": q.qid, "strategy": "plain", "model": "home.gguf", "correct": 1} for q in qs]
    bank = {"r:which_first:p:5": {"t": 99.0}}
    pairs, cut = G.heldout_set(qs, rows, bank, "home.gguf", 500)
    assert cut == 99.0 and len(pairs) == 100
    pairs2, cut2 = G.heldout_set(qs, rows, bank, "home.gguf", 500, trained=["r:which_first:p:150", "r:which_first:p:160"])
    assert cut2 == 160.0 and len(pairs2) == 39 and all(q.t > 160.0 for q, _h in pairs2)   # only ever stricter


def _mix(root: Path, name: str, n_train: int, ts: float = 0.0) -> None:
    d = root / name
    d.mkdir(parents=True)
    with (d / "train.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n_train):
            f.write(json.dumps(chat(f"question {i} about history", "ANSWER: A")) + "\n")
    (d / "dev.jsonl").write_text(json.dumps(chat("dev question", "ANSWER: B")) + "\n", encoding="utf-8")
    (d / "pref.jsonl").write_text("", encoding="utf-8")
    src = "talk_speak" if name.startswith("voice") else "trace_bank_cot"
    (d / "sft_train.ids.jsonl").write_text(json.dumps({"id": "x", "source": src, "ts": ts, "sha": ""}) + "\n", encoding="utf-8")
    (d / "train_qids.json").write_text("[]", encoding="utf-8")
    man = {"rows": {"train": n_train, "dev": 1, "pref": 0}, "tokens_train": 1000 * n_train}
    (d / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")


def test_jobs_are_valid_runner_specs_in_value_order_and_blocked_targets_stay_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import gpupulse as GP
    monkeypatch.setattr(TM, "load_heldout", lambda path=None: HELD)
    monkeypatch.setattr(TM, "frozen", lambda: GD.Frozen.empty())
    for t in TM.TARGETS:
        _mix(tmp_path, t.name, 300)
    js = TM.jobs({}, root=tmp_path)
    for j in js:
        GP.ext_job(j)
        if j.get("call"):
            mod, _, fn = j["call"].partition(":")
            assert callable(getattr(__import__(mod, fromlist=[fn]), fn))
    names = [j["name"] for j in js]
    assert names[0] == "trainmix_upload" and names[-1] == "trainmix_cleanup"
    assert names.index("ft_thinker_17b") < names.index("ft_thinker_06b") < names.index("ft_coder_17b") < names.index("ft_reviewer_17b")
    v = names.index("ft_voice_17b")                                               # voice-only adapter: talk eval, base first, no ladder/gate
    assert names[v + 1:v + 5] == ["register_voice_17b", "talkeval_voice_17b_base", "talkeval_voice_17b_tuned", "delete_voice_17b"]
    i = names.index("ft_thinker_17b")
    assert names[i + 1:i + 5] == ["register_thinker_17b", "eval_thinker_17b", "gate_thinker_17b", "delete_thinker_17b"]
    ft = js[i]
    assert ft["free_gpu"] and "--eval-data gpuday/trainmix/thinker_17b/dev.jsonl" in ft["remote"] and "/dev/shm/nupen_train" in ft["remote"]
    gate = js[names.index("gate_thinker_17b")]
    assert gate["args"]["train_qids"].endswith("train_qids.json") and gate["model"] == TM.serve_name(TM.BY_NAME["thinker_17b"])
    ev = js[names.index("eval_thinker_06b")]
    assert ev["args"]["base"] == "Qwen3-0.6B-Q4_K_M.gguf,Qwen3-1.7B-Q4_K_M.gguf"


def test_the_upload_is_refused_when_a_mix_breaks_the_exclusion_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TM, "load_heldout", lambda path=None: HELD)
    monkeypatch.setattr(TM, "frozen", lambda: GD.Frozen.empty())
    _mix(tmp_path, "thinker_17b", 300, ts=2000.0)
    with pytest.raises(ValueError, match="exclusion rule"):
        TM.upload_bundle(tmp_path, ["thinker_17b"])


BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_the_fine_tune_script_runs_end_to_end_against_stubs(tmp_path: Path) -> None:
    """The pod script with a stub finetune.py and a stub nvidia-smi: checks pass, the GGUF lands in models/ with its .ok hash, the adapter
    and result.json are kept for the trip home, the scratch is removed, and the printed result registers (gpuday.tuned_gguf)."""
    import sys
    t = TM.BY_NAME["thinker_06b"]
    man = {"rows": {"train": 300, "dev": 10, "pref": 0}, "tokens_train": 1000}
    shm = (tmp_path / "shm" / "nupen_train").as_posix()
    (tmp_path / "shm").mkdir()
    pod = tmp_path / "pod"
    (pod / "gpuday" / "trainmix" / t.name).mkdir(parents=True)
    (pod / "gpuday" / "trainmix" / t.name / "train.jsonl").write_text("{}\n" * 300, encoding="utf-8")
    real = Path(sys.executable).as_posix()
    (pod / "py").write_text(f"#!/bin/sh\n[ \"$1\" = -c ] && exit 0\nexec \"{real}\" \"$@\"\n", encoding="utf-8")   # the stack check passes
    (pod / "q").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (pod / "gpuday" / "python").write_text((pod / "py").as_posix(), encoding="utf-8")
    (pod / "gpuday" / "quantize_path").write_text((pod / "q").as_posix(), encoding="utf-8")
    (pod / "gpuday" / "llama.cpp").mkdir()
    (pod / "gpuday" / "llama.cpp" / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    (pod / "gpuday" / "finetune.py").write_text(
        "import json, sys, pathlib, hashlib\n"
        "a = sys.argv; out = pathlib.Path(a[a.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'adapter').mkdir(exist_ok=True); (out / 'adapter' / 'adapter_model.safetensors').write_bytes(b'w')\n"
        "g = out / 'model-Q4_K_M.gguf'; g.write_bytes(b'GGUF' * 10); (out / 'adapter.gguf').write_bytes(b'A')\n"
        "r = {'sft': {'rows': 300, 'dev_rows': 10}, 'gguf': {'files': {g.name: {'bytes': 40, 'sha256': hashlib.sha256(g.read_bytes()).hexdigest()}}}}\n"
        "(out / 'result.json').write_bytes(json.dumps(r, indent=1).replace(chr(10), chr(13) + chr(10)).encode())\n", encoding="utf-8")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 30000\n", encoding="utf-8")
    script = TM.ft_remote(t, man, shm=shm)
    bp = bin_.as_posix()
    if len(bp) > 1 and bp[1] == ":":                                              # Git bash on Windows: C:/x -> /c/x (a ':' would split PATH)
        bp = "/" + bp[0].lower() + bp[2:]
    env = {"PATH": bp + ":/usr/bin:/bin"}
    r = subprocess.run([BASH, "-c", f"chmod +x {bp}/nvidia-smi py q; export PATH={env['PATH']}:$PATH; {script}"], cwd=pod,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    sv = TM.serve_name(t)
    assert f"@@served_as={sv}" in r.stdout and (pod / "models" / sv).is_file()
    assert (pod / "models" / f"{sv}.ok").read_text().strip() == hashlib.sha256((pod / "models" / sv).read_bytes()).hexdigest()
    keep = pod / "gpuday" / "runs" / f"ft_{t.name}"
    assert (keep / "adapter" / "adapter_model.safetensors").is_file() and (keep / "result.json").is_file() and (keep / "adapter.gguf").is_file()
    assert not Path(shm, f"ft_{t.name}").exists()
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("@@result="))
    e = GD.tuned_gguf({"rc": 0, "result": json.loads(line[len("@@result="):])})
    assert e is not None and e["bytes"] == 40
    # too little VRAM (the 27B coder stage still loaded): skipped with the reason, nothing trained
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 2000\n", encoding="utf-8")
    r2 = subprocess.run([BASH, "-c", f"export PATH={env['PATH']}:$PATH; {script}"], cwd=pod, capture_output=True, text=True, timeout=60)
    assert r2.returncode == 0 and '"skipped": "VRAM: 2000 MiB free' in r2.stdout


def test_trainmix_stays_out_of_the_eager_start_load() -> None:
    swarm = (ROOT / "scripts" / "creator_swarm.py").read_text(encoding="utf-8")
    assert "trainmix" not in swarm
    import ast
    tree = ast.parse((ROOT / "creator" / "trainmix.py").read_text(encoding="utf-8"))
    top = {a.name.split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or "")])}
    assert not top & {"numpy", "torch", "transformers", "creator"}                 # heavy and package imports stay inside functions


def test_eval_job_dry_run_records_one_verdict_per_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval job with a stub ladder (no server): the tuned model's rows land in effladder.jsonl and a verdict per base model is recorded."""
    from creator import effladder as EL
    state = tmp_path / "state"
    n = 80
    out = EL.results_path(state)
    out.parent.mkdir(parents=True)
    with out.open("w", encoding="utf-8") as f:
        for b in ("base.gguf", "big.gguf"):
            for r in _rows(b, "reasoning", "plain@32", [i % 2 for i in range(n)], 5):
                f.write(json.dumps(r) + "\n")
    seen: dict[str, Any] = {}

    def stub(ctx: Any) -> dict[str, Any]:
        seen.update(ctx["args"])
        with out.open("a", encoding="utf-8") as f:
            for r in _rows(ctx["args"]["model"], "reasoning", "plain@32", [1] * n, 5):
                f.write(json.dumps(r) + "\n")
        return {"calls": n, "items_hash": "h", "model": ctx["args"]["model"]}
    monkeypatch.setattr(EL, "ladder_job", stub)
    res = TM.eval_job({"state": str(state), "model": "tuned.gguf", "pulse": "p", "args": {"target": "thinker_17b", "suites": ["reasoning"],
                                                                                           "configs": {"reasoning": "plain@32"}, "base": "base.gguf,big.gguf"}})
    assert seen["model"] == "tuned.gguf" and seen["configs"] == {"reasoning": "plain@32"}
    assert res["verdicts"] == {"base.gguf": "ADOPT", "big.gguf": "ADOPT"} and res["n"]["base.gguf"] == n
    recs = [json.loads(ln) for ln in TM.gate_path(state).read_text(encoding="utf-8").splitlines()]
    assert [r["base"] for r in recs] == ["base.gguf", "big.gguf"] and all(r["target"] == "thinker_17b" for r in recs)


def test_talk_rows_stay_in_the_voice_only_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TM, "frozen", lambda: GD.Frozen.empty())
    d = tmp_path / "voice_17b"
    d.mkdir()
    (d / "sft_train.ids.jsonl").write_text(json.dumps({"id": "talk:speak:p:0", "source": "talk_speak", "ts": 0.0}) + "\n", encoding="utf-8")
    assert TM.audit_mix(d, HELD) == []
    with (d / "sft_train.ids.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "trace_bank:cot:x", "source": "trace_bank_cot", "ts": 0.0}) + "\n")
    assert any("not a talk row" in b for b in TM.audit_mix(d, HELD))
    bad = TM.Target("thinker_x", "1.7b", ("trace_bank", "talk_speak"), "", "")
    with pytest.raises(ValueError, match="voice-only"):
        TM.build_target(bad, TM.Ctx(tmp_path, tmp_path, HELD, {}), {}, GD.Frozen.empty(), tmp_path)


def test_talk_compare_is_paired_and_never_adopts_below_min_n() -> None:
    base = [{"q": f"q{i}", "intent_ok": i % 2 == 0, "grounded": True, "answered": True, "tokens_out": 40} for i in range(60)]
    tuned = [{"q": f"q{i}", "intent_ok": True, "grounded": True, "answered": True, "tokens_out": 30} for i in range(60)]
    res = TM.talk_compare(base, tuned)
    assert res["n"] == 60 and res["verdict"] == "ADOPT" and res["intent_ok"]["gain"] == 0.5
    assert TM.talk_compare(base[:45], tuned[:45])["verdict"] == "INSUFFICIENT_N"
    worse = [dict(r, grounded=False) for r in tuned]
    assert TM.talk_compare(base, worse)["verdict"] == "KEEP_BASE"


def test_epochs_and_caps_scale_with_the_data() -> None:
    t = TM.BY_NAME["coder_17b"]
    small = {"rows": {"train": 4318, "dev": 493, "pref": 99}, "tokens_train": 2_884_551}
    big = {"rows": {"train": 87_323, "dev": 9508, "pref": 0}, "tokens_train": 19_517_346}
    assert TM.epochs_for(t, small) == 2.0 and TM.epochs_for(TM.BY_NAME["thinker_17b"], big) == 1.0     # a big mix gets one pass
    m = TM.minutes(t, small)
    assert m["steps"] == 540 and 20 < m["train"] < 45 and TM.cap_minutes(m["train"]) > 2 * m["train"]
