"""P0.9: real slot actors (creator.slotteam): grammars, prompts, budgets, tool confinement, one event per step. No model server needed."""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from creator import modelpool as MP
from creator import slotteam as S
from creator import team as TM


def test_every_form_has_a_grammar_with_a_root():
    for form, path in S.grammars().items():
        text = path.read_text(encoding="utf-8")
        assert text.strip() and text.startswith("root ::=") or "\nroot ::=" in text, form
    assert set(S.FORMS) >= {"diff", "confidence", "verdict", "fileline"}


def test_diff_grammar_bodies_cannot_swallow_the_markers():
    g = (S.GRAMMAR_DIR / "diff.gbnf").read_text(encoding="utf-8")
    assert "[^\\n=<>]" in g                      # a body line never starts with two marker characters


def test_chatml_ends_with_the_empty_think_block():
    p = S.chatml("sys", "user")
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and "<|im_start|>user\nuser<|im_end|>" in p


def test_pack_headers_become_fenced_file_blocks():
    out = S._fenced("### a/b.py:3-5 (Foo.bar)\nx = 1\n### c.py:1-2 (y)\nz\n")
    assert out.startswith("FILE: a/b.py (lines 3-5)") and "FILE: c.py (lines 1-2)" in out and out.count("```") == 4


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def complete(self, slot, prompt, form, max_tokens, timeout):
        self.calls.append((slot, form, max_tokens, timeout, prompt))
        return {"content": "STEPS: 1. x", "in_tok": 11, "out_tok": 5, "prompt_s": 0.1, "gen_s": 0.2, "slot": slot, "form": form, "stop": "eos"}


def _team(tmp_path: Path, runner: FakeRunner, sink=None) -> S.MeasuredTeam:
    t = S.MeasuredTeam(tmp_path / "t", S.actors(runner), goal_id="g", sink=sink)    # type: ignore[arg-type]
    return t


def test_budget_caps_the_model_call_and_the_event_carries_real_tokens(tmp_path):
    ev: list[tuple] = []
    runner = FakeRunner()
    t = _team(tmp_path, runner, lambda a, **kw: ev.append((a, kw)))
    task = t.board.put("g", "task", json.dumps({"id": "x", "family": "fn", "request": "r", "stub": "def f(): ...", "query": "f"}))
    env = TM.Envelope("g", "PLAN", inputs=[task], budget={"max_tok": 50, "max_s": 7})
    assert t.dispatch(env).startswith("STEPS")
    slot, form, max_tokens, timeout, prompt = runner.calls[0]
    assert (slot, form, max_tokens, timeout) == ("THINKER", "plan", 50, 7.0)       # envelope budget beat the profile (192 / 120)
    assert prompt.startswith("<|im_start|>system\n[ROLE: PLAN]")
    a, kw = ev[0]
    assert a == "THINKER" and kw["in_tok"] == 11 and kw["out_tok"] == 5 and kw["model"] is True and kw["cpu_s"] >= 0 and kw["prompt_s"] == 0.1


def test_repeat_is_served_by_the_cache_without_a_model_call(tmp_path):
    runner = FakeRunner()
    t = _team(tmp_path, runner)
    task = t.board.put("g", "task", json.dumps({"id": "x", "family": "fn", "request": "r", "stub": "s", "query": "f"}))
    for _ in range(2):
        t.dispatch(TM.Envelope("g", "PLAN", inputs=[task]))
    assert len(runner.calls) == 1


def test_code_prompt_is_the_trained_role_format():
    t = {"family": "fn", "request": "Write f", "examples": "f(1) == 2", "stub": "def f(): ..."}
    u = S.code_user(t, "STEPS: 1. do it", "")
    assert u.startswith("ROLE: CODE\nPlan:\nSTEPS: 1. do it\n\nRequest:\nWrite f") and "Current solution.py:" in u and "<<<<<<< SEARCH" in u


def test_workspace_patching_is_confined_and_tests_are_read_only(tmp_path):
    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "solution.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (ws / "tests" / "test_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    f = S.WsFiles(ws)
    ok = f.apply_patch("<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n", "solution.py")
    assert ok["ok"] and "return 2" in (ws / "solution.py").read_text(encoding="utf-8")
    den = f.apply_patch("tests/test_a.py\n<<<<<<< SEARCH\n    pass\n=======\n    assert False\n>>>>>>> REPLACE\n")
    assert not den["ok"] and den["rule"] == "tests"
    out = f.apply_patch("../evil.py\n<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n")
    assert not out["ok"] and not (tmp_path / "evil.py").exists()


def test_runner_reports_a_server_error_instead_of_raising(tmp_path, monkeypatch):
    spec = MP.SlotSpec("X", tmp_path / "x.gguf")
    r = S.SlotRunner({"X": spec}, tmp_path / "w")
    monkeypatch.setattr(r, "use", lambda slot, load_timeout=0: 1234)

    def boom(*a, **k):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(MP, "slot_call", boom)
    out = r.complete("X", "p", None, 8, 1.0)
    assert out["content"] == "" and out["error"] == "URLError" and out["wall_s"] >= 0


def test_unknown_model_step_is_refused(tmp_path):
    t = _team(tmp_path, FakeRunner())
    task = t.board.put("g", "task", "{}")
    with pytest.raises(TM.Refused):
        t.dispatch(TM.Envelope("g", "SUMMARIZE", inputs=[task]))


def test_canonical_patch_reads_the_models_native_format():
    nl = chr(10)
    fence = chr(96) * 3
    text = nl.join([fence, "CAUSE: x", "FIX:", "FILE: minishop/cart.py", "<<<<<<< SEARCH", "a", "=======", "b", ">>>>>>> REPLACE",
                    "FILE: tests/test_x.py", "<<<<<<< SEARCH", "=======", "t", ">>>>>>> REPLACE", "EXPECT: y", fence, ""])
    out, dropped = S.canonical_patch(text, "solution.py")
    assert out == "minishop/cart.py" + nl + nl.join(["<<<<<<< SEARCH", "a", "=======", "b", ">>>>>>> REPLACE"]) + nl and dropped == 1
    bare, n = S.canonical_patch(nl.join(["<<<<<<< SEARCH", "q", "=======", "r", ">>>>>>> REPLACE", ""]), "solution.py")
    assert bare.startswith("solution.py" + nl) and n == 0


def test_fast_prefill_holds_search_and_replace_head():
    stub = "from __future__ import annotations\n\ndef f(x):\n    \"\"\"Doc.\"\"\"\n    raise NotImplementedError\n"
    pre = S.fast_prefill({"family": "fn", "stub": stub})
    assert pre.startswith("<<<<<<< SEARCH\ndef f(x):") and pre.count("raise NotImplementedError") == 1
    assert pre.endswith("=======\ndef f(x):\n    \"\"\"Doc.\"\"\"\n")
    assert S.fast_prefill({"family": "app"}) == ""


# ---- h94 speed: tests decide, router slot, fixed-prefix-first prompt
def test_tests_decided_only_when_runnable_tests_gave_a_verdict(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    ctx = type("C", (), {"ws": ws})()
    assert not S.tests_decided(ctx, "PASS")                       # no test file: the checker is still needed
    (ws / "tests" / "test_x.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    assert S.tests_decided(ctx, "PASS") and S.tests_decided(ctx, "FAIL FAILED: x") and S.tests_decided(ctx, "FAIL apply: no match")
    assert not S.tests_decided(ctx, "")


def test_prefix_first_puts_the_task_last_and_keeps_the_fixed_text_identical():
    t1 = {"family": "fn", "name": "foo", "examples": "foo(1) == 2", "stub": "def foo(x):\n    raise NotImplementedError\n"}
    t2 = dict(t1, name="bar", examples="bar(3) == 9")
    old = S.PREFIX_FIRST
    try:
        S.PREFIX_FIRST = True
        a, b = S.fast_code_user(t1, ""), S.fast_code_user(t2, "")
        n = len(__import__("os").path.commonprefix([a, b]))
        assert "Edit format" in a[:n] and a.rstrip().endswith("foo(1) == 2")
        S.PREFIX_FIRST = False
        c = S.fast_code_user(t1, "")
        assert len(__import__("os").path.commonprefix([c, S.fast_code_user(t2, "")])) < n
    finally:
        S.PREFIX_FIRST = old


def test_coder06_slot_exists_and_slot_constraint_is_in_the_cache_key():
    assert "CODER06" in S.default_specs() and S.default_specs()["CODER06"].model.name.startswith("Qwen3-0.6B")
    a = TM.Envelope("g", "CODE", inputs=["b:abcdef12"])
    b = TM.Envelope("g", "CODE", inputs=["b:abcdef12"], constraints=[S.COUPLED_SLOT + "CODER06"])
    assert TM._sig(a) != TM._sig(b)
