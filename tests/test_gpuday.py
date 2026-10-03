"""GPU-day prep (creator/gpuday.py): privacy + frozen-benchmark filters on every export, diffs -> edits, time-ordered splits,
the evaluation-only coder harness (fake model, temporary git repo, real sandbox) and the embedding index (fake embedder)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest

from creator import generator as G
from creator import gpuday as GD

FROZEN_REAL = Path.home() / "weekly7" / "state" / "creator" / "thinkbench" / "items.json"


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout


def frozen_file(p: Path, h: str = "1f520b202ac4" + "0" * 52) -> Path:
    d = {"hash": h, "a": [{"id": "a:verdict:CP0009", "source": "verdict", "subject": "CP0009", "y": 0},
                          {"id": "a:git_fixed:abcdef1234", "source": "git_fixed", "subject": "abcdef1234", "y": 1},
                          {"id": "a:journal_persist:j1", "source": "journal_persist", "subject": "j1", "y": 1}],
         "b": [], "c": [{"id": "c:test_file:creator/x.py", "kind": "test_file", "answer": 2,
                         "prompt": "Which test file covers the module creator/secret_question.py?\nA. a\nB. b\nC. c\nD. d"}],
         "d": [{"id": "d:1", "candidates": [{"pkg": "CP0002", "text": "x"}], "context": "planning context " * 10}]}
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ privacy
def test_private_reason_catches_markers_and_secrets_but_not_the_protected_path_list() -> None:
    assert GD.private_reason("see ~/Masterstock/JOURNAL.md")
    assert GD.private_reason("rows from state/livesim/trades.json are")  # data path mention outside the protected list form
    assert GD.private_reason("key = 'sk-ant-abcdefghijklmnopqrstuvwxyz0123'")
    assert GD.private_reason("hf_" + "a" * 30)
    assert GD.private_reason("-----BEGIN OPENSSH PRIVATE KEY-----")
    ok = "- Never edit these protected paths: canon/*, state/creator/ledger.jsonl, state/livesim/*"
    assert GD.private_reason(ok) == ""
    assert GD.private_reason("def add(a, b):\n    return a + b") == ""


def test_scrub_removes_names_emails_and_home_paths() -> None:
    s = GD.scrub("Peter (peterdax006@gmail.com) ran C:\\Users\\Peter\\weekly7\\x.py and /c/Users/peter/y; Dax agreed; peterdax006-glitch")
    assert "Peter" not in s and "@" not in s and "Users" not in s and "Dax" not in s and "peterdax006" not in s
    obj = GD.scrub_obj({"m": [{"content": "mail a@b.co"}]})
    assert obj == {"m": [{"content": "mail <email>"}]}


# ------------------------------------------------------------------------------------------------ frozen benchmark
def test_frozen_load_refuses_an_unknown_benchmark(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        GD.Frozen.load(frozen_file(tmp_path / "items.json", h="deadbeef" * 8))
    f = GD.Frozen.load(frozen_file(tmp_path / "i2.json"))
    assert "CP0009" in f.packages and "CP0002" not in f.packages and "abcdef1234" in f.commits
    assert f.leak(text="xx Which test file covers the module creator/secret_question.py?\nA. a\nB. b\nC. c\nD. d yy")
    assert GD.Frozen.load(frozen_file(tmp_path / "i3.json"), strict=True).leak(package="CP0002")


@pytest.mark.skipif(not FROZEN_REAL.is_file(), reason="the live frozen thinkbench is not on this machine")
def test_the_real_frozen_items_load_with_the_pinned_hash() -> None:
    f = GD.Frozen.load(FROZEN_REAL)
    assert f.hash.startswith(GD.FROZEN_HASH) and len(f.packages) >= 20 and f.texts


# ------------------------------------------------------------------------------------------------ diffs -> edits
def test_diff_to_edits_reproduces_the_change(tmp_path: Path) -> None:
    r = tmp_path / "r"
    (r / "pkg").mkdir(parents=True)
    before = "".join(f"line{i} = {i}\n" for i in range(40))
    (r / "pkg" / "m.py").write_text(before, encoding="utf-8")
    (r / "notes.md").write_text("a\n", encoding="utf-8")
    sh(r.parent, "init", "-q", str(r))
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    after = before.replace("line3 = 3\n", "line3 = 33\n").replace("line30 = 30\n", "line30 = 30\nextra = 1\n")
    (r / "pkg" / "m.py").write_text(after, encoding="utf-8")
    (r / "pkg" / "new.py").write_text("X = 1\n", encoding="utf-8")
    (r / "notes.md").write_text("b\n", encoding="utf-8")
    sh(r, "add", "-A")
    patch = subprocess.run(["git", "diff", "--cached"], cwd=r, capture_output=True, text=True).stdout
    edits, paths = GD.diff_to_edits(patch)
    assert sorted(paths) == ["pkg/m.py", "pkg/new.py"]                  # the .md change is not an EDIT_RE edit
    w = tmp_path / "w"
    (w / "pkg").mkdir(parents=True)
    (w / "pkg" / "m.py").write_text(before, encoding="utf-8")
    applied, refused = G.apply_edits(w, G.parse_edits("REASONING: x\n" + edits))
    assert not refused and (w / "pkg" / "m.py").read_text(encoding="utf-8") == after
    assert (w / "pkg" / "new.py").read_text(encoding="utf-8") == "X = 1\n"


# ------------------------------------------------------------------------------------------------ splits
def test_time_split_is_strictly_earlier_and_groups_never_straddle() -> None:
    xs = [GD.Example(f"e{i}", "k", float(i + 1), f"g{i // 2}", {}) for i in range(10)]
    xs.append(GD.Example("late-member-of-g0", "k", 100.0, "g0", {}))
    tr, ev, cut = GD.time_split(xs, 0.3)
    assert tr and ev and max(x.ts for x in tr) < min(x.ts for x in ev) and cut == min(x.ts for x in ev)
    assert not ({x.group for x in tr} & {x.group for x in ev})
    assert "late-member-of-g0" not in {x.id for x in tr + ev}           # a started group never leaks into eval
    assert GD.time_split([], 0.2) == ([], [], 0.0)


# ------------------------------------------------------------------------------------------------ the export
def _state(tmp: Path, repo: Path) -> Path:
    st = tmp / "state"
    (st / "thinkbench").mkdir(parents=True)
    (st / "thinking").mkdir()
    frozen_file(st / "thinkbench" / "items.json")
    base = sh(repo, "rev-parse", "HEAD").strip()
    ctx = ("You are working in a git worktree of a Python project (the Creator). Package {p}.\nObjective: the module(s) of K{k} exist\n"
           "Why: gap GAP-1 (K{k}.exists): missing: pkg/k{k}.py\nRules:\n- Never edit these protected paths: canon/*, state/livesim/*")
    led, les = [], []
    for i, (pkg, k, outcome) in enumerate([("CP0001", 7, "REJECTED"), ("CP0002", 7, "ADOPTED"), ("CP0009", 9, "ADOPTED"),
                                           ("CP0010", 10, "ADOPTED"), ("CP0011", 11, "ADOPTED")]):
        led.append({"data": {"package_id": pkg, "objective": f"K{k}", "why_it_exists": f"gap (K{k}.exists)"},
                    "provenance": {"timestamp": f"2026-10-01T0{i}:00:00+00:00", "git_commit": base}})
        d = st / "cycles" / pkg
        d.mkdir(parents=True)
        body = "def f():\n    return 1\n" if outcome == "ADOPTED" else "def f(:\n"
        if pkg == "CP0011":
            body = "# from Masterstock: owner said\nX = 1\n"                 # private: must be dropped
        patch = (f"diff --git a/pkg/k{k}.py b/pkg/k{k}.py\nnew file mode 100644\n--- /dev/null\n+++ b/pkg/k{k}.py\n@@ -0,0 +1,2 @@\n"
                 + "".join("+" + ln + "\n" for ln in body.rstrip("\n").split("\n")))
        (d / "diff.patch").write_text(patch, encoding="utf-8")
        (d / "cycle.json").write_text(json.dumps({"outcome": outcome, "verdict": "X", "requirement": f"K{k}.exists"}), encoding="utf-8")
        (d / "evaluation.json").write_text(json.dumps({"base": base}), encoding="utf-8")
        les.append({"lesson_id": f"L{i}", "package_id": pkg, "objective": f"K{k}", "context": ctx.format(p=pkg, k=k), "at": f"2026-10-01T0{i}:00:00",
                    "files_before": {}, "files_after": {}, "solver": "claude", "task_kind": "gap"})
    (st / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in led), encoding="utf-8")
    (st / "lessons.jsonl").write_text("".join(json.dumps(r) + "\n" for r in les), encoding="utf-8")
    bank = [{"qid": "q1", "kind": "co_change", "source": "core", "prompt": "Which file changes with a.py?\nA. x\nB. y", "reasoning": "a imports y",
             "answer": "B", "correct": 1, "model": "m", "created": 1.0},
            {"qid": "q2", "kind": "co_change", "source": "core", "prompt": "Q2?", "reasoning": "r", "answer": "A", "correct": 0, "model": "m",
             "created": 2.0},
            {"qid": "c:test_file:creator/x.py", "kind": "test_file", "source": "nupen", "prompt": "frozen", "reasoning": "r", "answer": "C",
             "correct": 1, "model": "m", "created": 3.0},
            {"qid": "q4", "kind": "x", "source": "core", "prompt": "Q4 from Masterstock?", "reasoning": "r", "answer": "A", "correct": 1,
             "model": "m", "created": 4.0}]
    (st / "thinking" / "worked_bank.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bank), encoding="utf-8")
    (st / "thinking" / "judgment.jsonl").write_text("".join(json.dumps({"subject": s, "created": t}) + "\n" for s, t in
                                                            [("CP0009", 1.0), ("CP0020", 2.0), ("CP0021", 3.0), ("CP0022", 4.0)]), encoding="utf-8")
    return st


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "pkg" / "mathx.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (r / "tests" / "test_mathx.py").write_text("from pkg.mathx import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    (r / "tests" / "test_other.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    sh(tmp_path, "init", "-q", str(r))
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    (r / "pkg" / "extra.py").write_text("def two():\n    return 2\n", encoding="utf-8")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "add extra helper (by Peter, peterdax006@gmail.com)")
    (r / "pkg" / "secret.py").write_text("X = 'see Masterstock'\n", encoding="utf-8")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "private one")
    return r


def test_export_excludes_frozen_and_private_rows_and_writes_a_manifest(tmp_path: Path, repo: Path) -> None:
    st = _state(tmp_path, repo)
    out = tmp_path / "export"
    man = GD.export(st, repo, out, eval_frac=0.25, commit_limit=0)
    every = "".join(p.read_text(encoding="utf-8") for p in out.glob("*.jsonl"))
    assert "CP0009" not in every                                          # frozen verdict subject: its outcome is the label
    assert "Masterstock" not in every and "peterdax006" not in every and "@gmail" not in every
    assert "Which test file covers the module creator/secret_question.py" not in every
    assert "c:test_file:creator/x.py" not in every and "Q4 from" not in every and "Q2?" not in every
    assert GD.audit_export(out, GD.Frozen.load(st / "thinkbench" / "items.json")) == []
    f = man["files"]
    assert f["handoff_train"]["rows"] + f["handoff_eval"]["rows"] == 2   # CP0002 + CP0010 (CP0009 frozen, CP0011 private)
    assert f["pref_train"]["rows"] + f["pref_eval"]["rows"] == 1         # K07: CP0002 adopted > CP0001 rejected
    assert f["worked_train"]["rows"] + f["worked_eval"]["rows"] == 1
    assert f["commit_train"]["rows"] + f["commit_eval"]["rows"] == 1     # 'add extra helper'; the private commit is dropped
    assert man["frozen_hash"] == "1f520b202ac4" and man["dropped"]
    pref = json.loads((out / "pref_train.jsonl").read_text(encoding="utf-8").splitlines()[0]) if f["pref_train"]["rows"] else \
        json.loads((out / "pref_eval.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(pref) == {"prompt", "chosen", "rejected"} and "return 1" in pref["chosen"][0]["content"]
    sft = json.loads((out / "coder_sft_mix.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in sft["messages"]] == ["system", "user", "assistant"]
    sp = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    assert "CP0009" not in sp["judgment"]["train"] + sp["judgment"]["eval"] and sp["judgment"]["eval"] == ["CP0022"]
    with pytest.raises(ValueError):
        GD.export(st, repo, GD.ROOT / "state" / "gpuday_export")         # never inside the public repo


# ------------------------------------------------------------------------------------------------ coder harness
class FakeModel:
    def __init__(self, replies: Sequence[str]) -> None:
        self.replies, self.calls = list(replies), 0

    def __call__(self, messages: Sequence[dict[str, str]], temperature: float, max_tokens: int, seed: int) -> str:
        assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
        r = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        return r


def test_harness_best_of_n_tests_pick_and_nothing_is_merged(tmp_path: Path, repo: Path) -> None:
    head = sh(repo, "rev-parse", "HEAD").strip()
    branches = sh(repo, "branch", "--list")
    case = GD.Case("c1", "CP0100", head, "fix add", GD.coder_messages("fix add", {"pkg/mathx.py": "def add(a, b):\n    return a - b\n"}),
                   ["pkg/mathx.py"])
    wrong = "REASONING: no\nFILE: pkg/mathx.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a +\n>>>>>>> REPLACE\n"
    junk = "I think it is fine."
    right = "REASONING: plus\nFILE: pkg/mathx.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE\n"
    m = FakeModel([wrong, junk, right])
    res = GD.replay_case(repo, case, m, n=3, scratch=tmp_path / "scratch")
    assert m.calls == 3 and res["picked"] == 2 and res["pass_best_of_n"] and not res["pass_at_1"]
    assert res["candidates"][0]["report"] != "CLEAN" and res["candidates"][1]["would_pass"] is False
    assert sh(repo, "rev-parse", "HEAD").strip() == head and sh(repo, "branch", "--list") == branches   # evaluation only
    assert "return a - b" in (repo / "pkg" / "mathx.py").read_text(encoding="utf-8")
    rep = GD.harness_report([res])
    assert rep["rate_best_of_n"] == 1.0 and rep["rate_at_1"] == 0.0


def test_harness_cases_round_trip_from_an_export(tmp_path: Path, repo: Path) -> None:
    st = _state(tmp_path, repo)
    out = tmp_path / "export"
    GD.export(st, repo, out, eval_frac=0.5, commit_limit=0)
    cases = GD.harness_cases(out / "handoff_eval.jsonl")
    assert cases and all(c.messages[-1]["role"] == "user" and c.base for c in cases)
    assert cases[0].task.startswith("You are working in a git worktree")


# ------------------------------------------------------------------------------------------------ embedding index
def fake_embed(texts: Sequence[str]) -> list[list[float]]:
    vocab = ["sandbox", "ledger", "gguf", "test", "budget"]
    return [[float(t.lower().count(w)) + 0.01 for w in vocab] for t in texts]


def test_embedding_index_builds_small_and_queries_on_cpu(tmp_path: Path) -> None:
    docs = [{"id": "a", "kind": "commit", "ts": 1.0, "text": "sandbox sandbox worktree"},
            {"id": "b", "kind": "lesson", "ts": 2.0, "text": "ledger hash chain ledger"},
            {"id": "c", "kind": "plan", "ts": 3.0, "text": "gguf quantize gguf"}]
    meta = GD.build_index(docs, fake_embed, tmp_path / "idx", "fake", batch=2)
    assert meta["rows"] == 3 and meta["dim"] == 5
    hits = GD.query_index(tmp_path / "idx", fake_embed, "where is the ledger", k=2)
    assert hits[0]["id"] == "b" and hits[0]["score"] > hits[1]["score"]
    assert GD.query_index(tmp_path / "idx", fake_embed, "ledger", k=3, kinds=["plan"])[0]["id"] == "c"


def test_embed_corpus_filters_private_and_frozen(tmp_path: Path, repo: Path) -> None:
    st = _state(tmp_path, repo)
    d = GD.Drops()
    docs = GD.embed_corpus(st, repo, GD.Frozen.load(st / "thinkbench" / "items.json"), d)
    text = json.dumps(docs)
    assert "Masterstock" not in text and "peterdax006" not in text and "CP0009" not in text
    assert any(x["kind"] == "commit" for x in docs)


# ------------------------------------------------------------------------------------------------ the plan
def test_day_plan_covers_24_hours_without_gaps() -> None:
    p = GD.day_plan()
    assert p["total_hours"] == 24.0 and abs(p["total_usd"] - 24 * GD.USD_PER_HR) < 0.05
    ends = [(b.start_h, b.end_h) for b in GD.DAY]
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:])) and ends[0][0] == 0.0
    assert all(b.ready for b in GD.DAY)


def _unused(_: Any) -> None:
    """(keeps Any imported for the helper signatures above)"""


# ------------------------------------------------------------------------------------------------ the runner hook
def test_day_jobs_are_valid_runner_jobs_and_the_upload_unpacks(tmp_path: Path, repo: Path) -> None:
    from creator import gpupulse as GP
    st = _state(tmp_path, repo)
    out = tmp_path / "export"
    GD.export(st, repo, out, eval_frac=0.5, commit_limit=0)
    jobs = GD.day_jobs({"gpuday_export": str(out), "gpuday_frozen": str(st / "thinkbench" / "items.json")})
    parsed = [GP.parse_job(j) for j in jobs]
    names = [p.get("name") for p in parsed if p["kind"] == "ext"]
    assert names[0] == "gpuday_upload" and "ft2_coder" in names and len(set(names)) == len(names)
    assert all(p["free_gpu"] for p in parsed if p.get("name", "").startswith(("ft", "rl_", "embed")))
    up = next(p for p in parsed if p.get("name") == "gpuday_upload")
    GP.outbound_ok([{"content": up["remote"]}])                           # the runner's private-marker guard passes
    pod = tmp_path / "pod"
    pod.mkdir()
    r = subprocess.run(["bash", "-s"], input=up["remote"], cwd=pod, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (pod / "gpuday" / "gpuday_lib.py").is_file() and (pod / "gpuday" / "finetune.py").is_file()
    assert (pod / "gpuday" / "data" / "coder_sft_mix.jsonl").is_file() and "@@result=" in r.stdout
    assert "CP0009" not in "".join(p.read_text(encoding="utf-8") for p in (pod / "gpuday" / "data").glob("*.jsonl"))


def test_upload_is_refused_when_the_export_audit_fails(tmp_path: Path) -> None:
    out = tmp_path / "export"
    out.mkdir()
    (out / "coder_sft_mix.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "from ~/Masterstock"}]}) + "\n",
                                             encoding="utf-8")
    with pytest.raises(ValueError):
        GD.upload_bundle(out, GD.Frozen.empty())


def test_ft_script_skips_thin_data_and_places_the_gguf_for_serving() -> None:
    s = GD.ft_script("ft1", "Qwen/Qwen3-1.7B", "worked_train.jsonl", min_rows=200, serve_as="X.gguf", merge_base="Qwen/Qwen3-1.7B")
    assert "-lt 200" in s and "models/X.gguf.ok" in s and "--merge-base Qwen/Qwen3-1.7B" in s and "finetune.py pipeline" in s


# ------------------------------------------------------------------------------------------------ provider- and GPU-agnostic choices
def test_gpu_profiles_follow_vram_and_compute_capability() -> None:
    g3090 = GD.parse_gpu("NVIDIA GeForce RTX 3090, 24576 MiB, 8.6")
    assert g3090 == {"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "cc": 8.6}
    p = GD.gpu_profile(g3090)
    assert p["coder"] == "14b" and p["coder_qlora"] and p["coder_batch"] == 1 and p["dtype"] == "bf16" and not p["fp8"]
    assert p["train_speed"] < 0.7 and p["gen_speed"] > 0.85                   # trains ~1.5-2x slower, generates about as fast
    p5 = GD.gpu_profile(GD.parse_gpu("NVIDIA GeForce RTX 5090, 32607 MiB, 12.0"))
    assert p5["min_cuda"] == "12.8" and p5["coder_batch"] == 2 and p5["vram_gb"] > 31
    p48 = GD.gpu_profile(GD.parse_gpu("NVIDIA GeForce RTX 4090, 49140 MiB, 8.9"))
    assert p48["coder"] == "coder30b" and p48["ft1_batch_4b"] > GD.gpu_profile(None)["ft1_batch_4b"]
    assert GD.gpu_profile({"name": "Tesla V100", "vram_gb": 16, "cc": 7.0})["dtype"] == "fp16"
    assert GD.parse_gpu("RTX 3090,24,8.6")["cc"] == 8.6


def test_day_plan_scales_training_blocks_with_the_card() -> None:
    p4, p3 = GD.day_plan(), GD.day_plan("NVIDIA GeForce RTX 3090, 24576 MiB, 8.6", usd_per_hr=0.16)
    ft4 = next(b for b in p4["blocks"] if b["name"].startswith("FINE-TUNE 1"))
    ft3 = next(b for b in p3["blocks"] if b["name"].startswith("FINE-TUNE 1"))
    assert ft4["overflow_hours"] == 0 and ft3["work_hours"] > 1.6 * ft3["hours"] and ft3["overflow_hours"] > 0
    assert p3["total_usd"] == round(24 * 0.16, 2) and p3["total_hours"] == 24


def test_day_jobs_size_the_coder_from_the_card(tmp_path: Path, repo: Path) -> None:
    st = _state(tmp_path, repo)
    out = tmp_path / "export"
    GD.export(st, repo, out, eval_frac=0.5, commit_limit=0)
    base = {"gpuday_export": str(out), "gpuday_frozen": str(st / "thinkbench" / "items.json")}
    j48 = {j["name"]: j for j in GD.day_jobs(dict(base, gpuday_gpu="NVIDIA GeForce RTX 4090, 49140 MiB, 8.9")) if isinstance(j, dict)}
    j24 = {j["name"]: j for j in GD.day_jobs(dict(base, gpuday_gpu="NVIDIA GeForce RTX 3090, 24576 MiB, 8.6")) if isinstance(j, dict)}
    assert "Qwen3-Coder-30B-A3B" in j48["ft2_coder"]["remote"] and "Qwen3-14B" in j24["ft2_coder"]["remote"]
    assert j24["ft2_coder"]["minutes"] > j48["ft2_coder"]["minutes"]          # the 3090 is given more time for the same work


def test_pod_scripts_parse_and_finetune_trains_on_the_answer_only() -> None:
    import importlib.util
    root = Path(__file__).resolve().parents[1] / "scripts" / "gpuday"
    r = subprocess.run(["bash", "-n", str(root / "pod_setup.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert b"\r\n" not in (root / "pod_setup.sh").read_bytes()               # sent to bash on Linux: LF only
    spec = importlib.util.spec_from_file_location("gpuday_finetune", root / "finetune.py")
    assert spec and spec.loader
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)                                               # no torch needed to import it
    pc = ft.to_prompt_completion([{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                                {"role": "assistant", "content": "a"}]}, {"messages": [{"role": "user", "content": "x"}]}])
    assert pc == [{"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                   "completion": [{"role": "assistant", "content": "a"}]}]


# ------------------------------------------------------------------------------------------------ hook H3: a tuned model becomes servable
def _ft_row(pulse: str = "P1", job: str = "ext:ft1_17b", rc: int = 0, **result: Any) -> dict[str, Any]:
    return {"job": job, "rc": rc, "gpu_pulse": pulse, "result": result}


def _runs(state: Path, *rows: dict[str, Any]) -> None:
    (state / "thinking").mkdir(parents=True, exist_ok=True)
    (state / "thinking" / "gpu_pulse_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_register_tuned_job_records_the_gguf_of_this_pulse_in_an_injected_file(tmp_path: Path) -> None:
    from creator import gpupulse as GP
    sha_old, sha = "11" * 32, "AB" * 32
    gg = lambda s, n: {"quant": "Q4_K_M", "files": {"model-Q4_K_M.gguf": {"bytes": n, "sha256": s}}}  # noqa: E731
    _runs(tmp_path, _ft_row("P0", gguf=gg(sha_old, 7)), _ft_row("P1", gguf=gg(sha, 1234)), _ft_row("P1", job="ext:other", gguf=gg(sha_old, 9)))
    reg = tmp_path / "rt" / "extra_models.json"
    reg.parent.mkdir()
    reg.write_text(json.dumps({"Keep.gguf": {"bytes": 3, "sha256": "cd" * 32}}), encoding="utf-8")
    ctx = {"state": tmp_path, "pulse": "P1", "args": {"source": "ft1_17b", "serve_as": "Qwen3-1.7B-gpuday-ft1.gguf", "extra_models_file": str(reg)}}
    r = GD.register_tuned_job(ctx)
    assert r["registered"] and r["bytes"] == 1234 and r["sha256"] == sha.lower()
    got = GP.extra_models({"extra_models_file": str(reg)})                     # exactly what the runner will read
    assert got["Qwen3-1.7B-gpuday-ft1.gguf"]["sha256"] == sha.lower() and "Keep.gguf" in got
    assert not (tmp_path / "rt" / "pulse.json").exists()


def test_register_tuned_job_refuses_skipped_failed_or_absent_fine_tunes(tmp_path: Path) -> None:
    reg = tmp_path / "extra_models.json"
    base = {"state": tmp_path, "pulse": "P1", "args": {"source": "ft1_17b", "serve_as": "T.gguf", "extra_models_file": str(reg)}}
    _runs(tmp_path, _ft_row(skipped="fewer than 200 rows in worked_train.jsonl"))
    r = GD.register_tuned_job(base)
    assert not r["registered"] and "fewer than 200" in r["reason"]
    _runs(tmp_path, _ft_row(rc=5, gguf={"files": {"model-Q4_K_M.gguf": {"bytes": 1, "sha256": "ab" * 32}}}))
    assert "rc 5" in GD.register_tuned_job(base)["reason"]
    _runs(tmp_path, _ft_row("P0", gguf={"files": {"model-Q4_K_M.gguf": {"bytes": 1, "sha256": "ab" * 32}}}))
    assert "no run of ft1_17b in this pulse" in GD.register_tuned_job(base)["reason"]       # an earlier pulse's model is never registered
    _runs(tmp_path, _ft_row(gguf={"files": {"model-Q4_K_M.gguf": {"bytes": 1, "sha256": "not-hex"}}}))
    with pytest.raises(Exception):
        GD.register_tuned_job(base)                                               # validated the runner's way before anything is written
    assert not reg.exists()
    assert not GD.register_tuned_job({"state": tmp_path, "args": {}})["registered"]


def test_day_jobs_register_each_tuned_model_before_serving_it(tmp_path: Path, repo: Path) -> None:
    from creator import gpupulse as GP
    st = _state(tmp_path, repo)
    out = tmp_path / "export"
    GD.export(st, repo, out, eval_frac=0.5, commit_limit=0)
    base = {"gpuday_export": str(out), "gpuday_frozen": str(st / "thinkbench" / "items.json"), "gpuday_gpu": "NVIDIA GeForce RTX 5090, 32607 MiB, 12.0"}
    jobs = GD.day_jobs(base)
    specs = [GP.parse_job(j) for j in jobs]
    label = [s.get("name") or s["spec"] for s in specs]
    for src, model in (("ft1_17b", "Qwen3-1.7B-gpuday-ft1.gguf"), ("ft1_4b", "Qwen3-4B-gpuday-ft1.gguf"), ("ft2_coder", "Qwen3-14b-gpuday-coder.gguf")):
        reg = label.index(f"register_{src}")
        assert label.index(src) < reg < min(i for i, s in enumerate(specs) if s["model"] == model)
        assert specs[reg]["args"] == {"source": src, "serve_as": model}
    ft2 = next(s for s in specs if s.get("name") == "ft2_coder")["remote"]
    assert "-lt 110 ]" in ft2 and "gpuday/python" in ft2 and "ln -f" in ft2
    assert all("gpuday/python" in s["remote"] for s in specs if s.get("via") == "remote" and s["name"] != "gpuday_upload")
    small = {s.get("name") or s["spec"]: s for s in (GP.parse_job(j) for j in GD.day_jobs(dict(base, gpuday_ft_gguf=False)))}
    assert not any(k.startswith(("register_", "coder_trial_tuned", "thinkbench:Qwen3-1.7B-gpuday")) for k in small)
    assert "--llama-cpp" not in small["ft1_4b"]["remote"] and "-lt 10 ]" in small["ft1_4b"]["remote"]


def test_ft_script_skips_itself_when_the_pod_disk_is_short(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    (pod / "gpuday" / "data").mkdir(parents=True)
    (pod / "gpuday" / "data" / "d.jsonl").write_text("{}\n" * 5, encoding="utf-8")
    s = GD.ft_script("x", "base", "d.jsonl", min_rows=2, disk_gb=10 ** 7)        # more than any disk has
    r = subprocess.run(["bash", "-s"], input=s, cwd=pod, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.split("@@result=", 1)[1].splitlines()[0])
    assert res["skipped"].startswith("disk: ") and "needs ~10000000 GB" in res["skipped"]
    assert not (pod / "gpuday" / "runs").exists()


def test_a_second_run_of_the_day_starts_with_a_fresh_upload(tmp_path: Path, repo: Path) -> None:
    st = _state(tmp_path, repo)
    out = tmp_path / "export"
    GD.export(st, repo, out, eval_frac=0.5, commit_limit=0)
    base = {"gpuday_export": str(out), "gpuday_frozen": str(st / "thinkbench" / "items.json"), "gpuday_start_at": "ft1_17b"}
    names = [j["name"] if isinstance(j, dict) else j for j in GD.day_jobs(base)]
    assert names[:3] == ["gpuday_upload", "ft1_17b", "register_ft1_17b"] and "gpuday_reexport" not in names and names[-1] == "rl_poc"
    with pytest.raises(ValueError):
        GD.day_jobs(dict(base, gpuday_start_at="nope"))
