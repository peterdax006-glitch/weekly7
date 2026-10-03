"""EFFICIENCY LADDER (creator.effladder) with local fakes only: an in-process OpenAI-compatible fake server, fake datasets, a fake bench.
Nothing is rented, downloaded or served."""
from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from creator import effladder as E
from creator import gpupulse as GP

ROOT = Path(__file__).resolve().parents[1]

CODE_OK = "```python\ndef add(a, b):\n    return a + b\n```"


class _H(http.server.BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []
    lock = threading.Lock()

    def log_message(self, *a: Any) -> None:
        return None

    def do_POST(self) -> None:
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        with self.lock:
            self.seen.append(d)
        last = d["messages"][-1]["content"]
        if "ANSWER" in json.dumps(d["messages"]):
            out = "<think>hmm</think>\nB fits.\nANSWER: B"
        elif "PROBABILITY" in json.dumps(d["messages"]):
            out = "Big commit.\nPROBABILITY: 0.8"
        elif "add" in last:
            out = CODE_OK
        else:
            out = "```python\ndef nope(:\n```"
        trunc = d["max_tokens"] < 20
        body = {"choices": [{"message": {"content": "<think>unfinished" if trunc else out}, "finish_reason": "length" if trunc else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": min(d["max_tokens"], 30)}, "timings": {"predicted_per_second": 321.0}}
        b = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture()
def server() -> Iterator[int]:
    _H.seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield int(srv.server_address[1])
    srv.shutdown()


def _mc(i: int, split: str = "eval", answer: int = 1) -> dict[str, Any]:
    m = [{"role": "system", "content": "Finish with 'ANSWER: <letter>'."}, {"role": "user", "content": f"Question {i}?\nA. x\nB. y\nC. z\nD. w"}]
    return {"id": f"reasoning:q{i}", "suite": "reasoning", "kind": "files_changed", "split": split, "answer": answer, "check": "mc",
            "messages": {"plain": m, "cot": m, "think": m, "retrieval": m}}


def _code(i: int, split: str, prompt: str = "Write add(a, b).") -> dict[str, Any]:
    m = [{"role": "system", "content": E.CODE_SYSTEM}, {"role": "user", "content": prompt}]
    return {"id": f"coding:t{i}", "suite": "coding", "kind": "mbpp", "split": split, "check": "py",
            "test": {"asserts": ["assert add(2, 3) == 5"], "imports": []}, "messages": {"plain": m, "retrieval": m, "think": m}}


def _body(items: list[dict[str, Any]]) -> dict[str, Any]:
    b: dict[str, Any] = {"version": E.VERSION, "items": items}
    b["hash"] = E._sha({"version": E.VERSION, "items": items})
    return b


def test_scoring_multiple_choice_probability_and_executed_code() -> None:
    it = _mc(0)
    assert E.score(it, "<think>A? no</think>ANSWER: B")["correct"] is True
    assert E.score(it, "<think>still thinking, ANSWER: B")["correct"] is False          # truncated thinking leaves no answer
    j = {"check": "prob", "answer": 1}
    assert E.score(j, "PROBABILITY: 0.7") == {"correct": True, "p": 0.7, "brier": 0.09}
    assert E.score(j, "no number")["correct"] is False
    assert E.score(_code(0, "eval"), CODE_OK)["correct"] is True
    assert E.score(_code(0, "eval"), "```python\ndef add(a, b):\n    return a - b\n```")["correct"] is False
    assert E.score(_code(0, "eval"), "```python\nwhile True: pass\n```")["correct"] is False             # no def add: nothing runs
    rl = {"check": "py", "test": {"name": "sq", "cases": [{"args": [3], "expect": 9}]}}
    assert E.score(rl, "```python\ndef sq(x):\n    return x * x\n```")["correct"] is True
    he = {"check": "py", "test": {"prefix": "import math\n\ndef root(x):\n    \"\"\"sqrt\"\"\"\n", "entry": "root",
                                  "test": "def check(f):\n    assert f(9) == 3\n"}}
    assert E.score(he, "```python\n    return int(math.sqrt(x))\n```")["correct"] is True                # a body-only completion
    assert E.score(he, "```python\nimport math\ndef root(x):\n    return 4\n```")["correct"] is False


def test_run_records_every_config_resumes_and_distills_only_verified_distill_items(server: int, tmp_path: Path) -> None:
    secret = _code(9, "distill", "Write add(a, b). Masterstock")                   # a private marker: never sent, never a row
    items = [_mc(1), _mc(2, answer=0), _code(3, "eval"), _code(4, "distill", "Write add(a, b) please."), _code(5, "distill", "Write add(a, b) now.")]
    body = _body(items)
    out, dist = tmp_path / "r.jsonl", tmp_path / "d.jsonl"
    ep = E.Endpoint(server, "Qwen3-0.6B-Q4_K_M.gguf")
    res = E.run(body, ep, tmp_path, suites=["reasoning", "coding"], configs={"reasoning": "plain@16,cot@256"}, workers=3, distill=True,
                say=lambda s: None, out=out, distill_out=dist)
    rows = E.read_rows(out)
    assert res["errors"] == 0 and len(rows) == 2 * 2 + 4 * 1 + 2       # 2 mc x 2 configs + 1 coding eval x 4 configs + 2 distill
    plain = [r for r in rows if r["config"] == "plain@16"]
    assert all(r["truncated"] and not r["correct"] for r in plain)    # the budget was too small: recorded as such
    cot = {r["item"]: r["correct"] for r in rows if r["config"] == "cot@256"}
    assert cot == {"reasoning:q1": True, "reasoning:q2": False}
    r0 = rows[0]
    assert r0["quant"] == "Q4_K_M" and r0["params_b"] == 0.6 and r0["tok_in"] == 100 and r0["gen_tps"] == 321.0
    assert all(m["messages"][-1]["content"].endswith(("/no_think", "/think")) for m in _H.seen)
    assert any(m["chat_template_kwargs"]["enable_thinking"] for m in _H.seen)
    d = E.read_rows(dist)
    assert [x["meta"]["id"] for x in sorted(d, key=lambda x: x["meta"]["id"])] == ["coding:t4", "coding:t5"]
    assert d[0]["messages"][-1]["role"] == "assistant" and "def add" in d[0]["messages"][-1]["content"]
    n = len(_H.seen)
    E.run(body, ep, tmp_path, suites=["reasoning", "coding"], configs={"reasoning": "plain@16,cot@256"}, distill=True, say=lambda s: None,
          out=out, distill_out=dist)
    assert len(_H.seen) == n                                          # everything already recorded: nothing asked again
    with pytest.raises(GP.PulseError):
        E.Endpoint(server, "x").call(secret["messages"]["plain"], 50, False)


def test_distill_never_writes_eval_frozen_or_private_items() -> None:
    c = E.Config("plain", 1024)
    ev = _code(1, "eval")
    assert E.distill_row(ev, c, CODE_OK, "m", set()) is None
    fz = dict(_code(2, "distill"), frozen=True)
    assert E.distill_row(fz, c, CODE_OK, "m", set()) is None
    same = _code(3, "distill")
    assert E.distill_row(same, c, CODE_OK, "m", {"Write add(a, b)."}) is None              # its prompt is an eval prompt
    priv = _code(4, "distill", "Write add(a, b) for the livesim trader.")
    assert E.distill_row(priv, c, CODE_OK, "m", set()) is None
    ok = E.distill_row(_code(5, "distill"), c, CODE_OK, "m", set())
    assert ok is not None and ok["meta"]["verified"] is True


def test_items_file_hash_refuses_an_edit_and_coding_items_split_cleanly(tmp_path: Path) -> None:
    pages = {
        ("google-research-datasets/mbpp", "train"): [{"task_id": 1, "prompt": "Write add of two numbers.", "code": "def add(a,b): return a+b",
                                                       "test_list": ["assert add(1,2)==3"], "test_imports": []}],
        ("google-research-datasets/mbpp", "validation"): [], ("google-research-datasets/mbpp", "prompt"): [],
        ("google-research-datasets/mbpp", "test"): [{"task_id": 7, "prompt": "Write add numbers twice.", "code": "x",
                                                      "test_list": ["assert add2(1,2)==6"], "test_imports": []}],
        ("openai/openai_humaneval", "test"): [{"task_id": "HumanEval/0", "prompt": "def f(x):\n    \"\"\"d\"\"\"\n", "test": "def check(c): pass",
                                                "entry_point": "f"}]}

    def fetch(url: str) -> bytes:
        ds = url.split("dataset=")[1].split("&")[0]
        split = url.split("split=")[1].split("&")[0]
        rows = pages[(ds, split)] if "offset=0" in url else []
        return json.dumps({"rows": [{"row": r} for r in rows], "num_rows_total": len(pages[(ds, split)])}).encode()

    exp = tmp_path / "export"
    exp.mkdir()
    (exp / "rl_tasks.jsonl").write_text(json.dumps({"id": "m:f", "name": "f", "prompt": "Write f.", "tests": [], "split": "eval"}) + "\n" +
                                        json.dumps({"id": "m:g", "name": "g", "prompt": "Write g.", "tests": [], "split": "train"}) + "\n")
    items = E.coding_items(exp, fetch)
    split = {it["id"]: it["split"] for it in items}
    assert split == {"coding:mbpp:1": "distill", "coding:rl:m:f": "eval", "coding:rl:m:g": "distill", "coding:mbpp:7": "eval",
                     "coding:HumanEval/0": "eval"}
    ret = next(it for it in items if it["id"] == "coding:mbpp:7")["messages"]["retrieval"]
    assert ret[2]["content"].startswith("```python\ndef add")             # the similar distill-pool task, with its reference code
    p = tmp_path / "items.json"
    b = _body(items)
    p.write_text(json.dumps(b))
    assert E.load_items(p)["hash"] == b["hash"]
    b["items"][0]["answer"] = 3
    p.write_text(json.dumps(b))
    with pytest.raises(ValueError):
        E.load_items(p)


def test_thinkbench_items_are_read_only_and_marked_frozen() -> None:
    src = (ROOT / "creator" / "effladder.py").read_text(encoding="utf-8")
    assert "TB.load_frozen(state)" in src and "GD.FROZEN_HASH" in src
    assert "freeze(" not in src.split("def thinkbench_items")[1].split("\ndef ")[0]             # never re-freezes the benchmark


def test_job_list_is_valid_disk_safe_and_never_deletes_runner_models() -> None:
    js = E.jobs()
    for j in js + [E.coder_job()]:
        GP.parse_job(j)
    names = [j["name"] for j in js]
    assert "ladder_delete_Qwen3-14B-Q4_K_M" not in names and "ladder_fetch_Qwen3-4B-Q4_K_M" not in names
    i = names.index("ladder_Qwen3-4B-Q8_0")
    assert names[i - 1] == "ladder_fetch_Qwen3-4B-Q8_0" and names[i + 1] == "ladder_delete_Qwen3-4B-Q8_0"
    fetched = 0                                                           # at most one ladder-only file on the pod at any time
    for j in js:
        fetched += 1 if j["name"].startswith("ladder_fetch_") else -1 if j["name"].startswith("ladder_delete_") else 0
        assert fetched <= 1
    assert "rm -f" not in E.delete_script("/w", "Qwen3-1.7B-Q4_K_M.gguf")
    sc = E.fetch_script("/workspace/nupen", "Qwen3-0.6B-Q8_0.gguf")
    assert E.LADDER["Qwen3-0.6B-Q8_0.gguf"]["sha256"] in sc and "@@result=" in sc
    GP.outbound_ok([{"content": sc}])
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n"], input=sc, text=True).returncode == 0


def test_summary_picks_the_cheapest_setup_within_the_points() -> None:
    rows = []
    for i in range(20):
        for m, c, ok, out in (("Qwen3-14B-Q4_K_M.gguf", "cot@256", True, 40), ("Qwen3-0.6B-Q4_K_M.gguf", "cot@256", i != 0, 40),
                              ("Qwen3-0.6B-Q4_K_M.gguf", "think@1024", True, 600), ("Qwen3-0.6B-Q4_K_M.gguf", "plain@32", i % 2 == 0, 5)):
            rows.append({"split": "eval", "suite": "reasoning", "model": m, "config": c, "item": f"q{i}", "correct": ok, "tok_in": 300,
                         "tok_out": out, "latency_s": 0.1})
    rows.append({"split": "eval", "suite": "reasoning", "model": "Qwen3-14B-Q4_K_M.gguf", "config": "cot@256", "item": "only14", "correct": True})
    hc = {"rows": {"Qwen3-0.6B-Q4_K_M.gguf": {"tg_tok_s": 40.0, "pp_tok_s": 300.0}, "Qwen3-14B-Q4_K_M.gguf": {"tg_tok_s": 2.0, "pp_tok_s": 20.0}}}
    sm = E.summary(rows, hc, points=5.0)["reasoning"]
    assert sm["common_items"] == 20 and sm["best_acc"] == 1.0
    assert (sm["cheapest_within"]["model"], sm["cheapest_within"]["config"]) == ("Qwen3-0.6B-Q4_K_M.gguf", "cot@256")
    assert "CHEAPEST" in E.format_summary(E.summary(rows, hc))


def test_homecost_measures_local_files_and_labels_estimates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    f.write_bytes(b"0")
    monkeypatch.setattr(E, "local_ggufs", lambda: {f.name: f})

    def bench(p: Path, threads: int) -> dict[str, Any]:
        return {"model": p.name, "disk_bytes": E.LADDER[p.name]["bytes"], "tg_tok_s": 40.0, "pp_tok_s": 300.0, "peak_rss_mb": 600}

    body = E.homecost(4, ["Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "Qwen3-4B-Q8_0.gguf"], say=lambda s: None, bench=bench,
                      out=tmp_path / "hc.json")
    r = body["rows"]
    assert r["Qwen3-0.6B-Q4_K_M.gguf"]["measured"] is True
    q8 = r["Qwen3-0.6B-Q8_0.gguf"]
    assert q8["measured"] is False and q8["estimated_from"] == "Qwen3-0.6B-Q4_K_M.gguf" and q8["tg_tok_s"] < 40.0 and q8["peak_rss_mb"] > 600
    assert "tg_tok_s" not in r["Qwen3-4B-Q8_0.gguf"] and "contended" in body
    assert json.loads((tmp_path / "hc.json").read_text())["threads"] == 4


def test_effladder_stays_out_of_the_eager_start_load() -> None:
    swarm = (ROOT / "scripts" / "creator_swarm.py").read_text(encoding="utf-8")
    assert "effladder" not in swarm
