"""creator.talk (layer 2): the voice is a stub here (no real model). Understanding falls back to the rules on a bad model answer; a clear rule
hit costs no model call; replies with an invented number fall back to the rule report; evidence is always appended by us; open questions are
answered from the BM25 index of docs and records; a missing voice never breaks the conversation."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import pytest

from creator import conversation as CV
from creator import talk as T
from creator import talkeval as TE

NOW = dt.datetime(2026, 10, 2, 12, 0, 0)


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    st = tmp_path / "state" / "creator"
    _w(st / "nupen_service.heartbeat", "x")
    t = NOW.timestamp() - 20
    os.utime(st / "nupen_service.heartbeat", (t, t))
    _w(st / "constraints.jsonl", json.dumps({"event": "snapshot", "ranked": [
        {"name": "wasted_cycles", "loss": 0.4, "score": 0.28, "value": 0.4, "unit": "share", "remedy": "goal"}]}) + "\n")
    _w(tmp_path / "creator" / "pulsething.py", '"""pulsething: routes model calls to the rented GPU pod while its tunnel is healthy."""\nx = 1\n')
    _w(tmp_path / "creator" / "other.py", '"""other: counts apples in the orchard for the harvest report."""\n')
    _w(tmp_path / "docs" / "GUIDE.md", "# Planner\nThe planner orders work packages by the critical path.\n# Secret\nMasterstock notes live here.\n")
    _w(st / "lessons.jsonl", json.dumps({"lesson_id": "abcdef123456", "package_id": "CP0007", "task_kind": "gap", "component": "K05", "solver": "claude",
                                          "objective": "make the planner skip duplicate packages", "verdict": "adopted"}) + "\n")
    T._INDEX.clear()
    return tmp_path


class Stub:
    """A fake model client: answers from a queue (or a function of the messages) and records every call."""

    def __init__(self, answers: list[str] | None = None, fn: Any = None) -> None:
        self.answers = list(answers or [])
        self.fn = fn
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 0, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        if self.fn is not None:
            return str(self.fn(messages))
        return self.answers.pop(0) if self.answers else ""


def _voice(stub: Stub) -> T.Voice:
    return T.Voice(Path("Qwen3-1.7B-Q4_K_M.gguf"), factory=lambda: stub)


def _conv(root: Path, stub: Stub | None, mode: str = "hybrid") -> CV.Conversation:
    c = T.conversation(root, _voice(stub) if stub is not None else None, mode=mode, log=False)
    c.clock = lambda: NOW
    return c


def test_ungrounded_flags_invented_numbers_and_ids_only() -> None:
    src = "Top constraint: wasted_cycles (loss 0.4, 3 packages). CP0001 adopted. lesson abcdef123456"
    assert T.ungrounded("My worst limit is wasted_cycles with loss 0.4 and 3 packages; CP0001 was adopted.", src) == []
    assert T.ungrounded("Loss is 0.5 and CP0009 failed.", src) == ["CP0009", "0.5"]
    assert T.ungrounded("1. wasted cycles\n2. nothing else", src) == []          # list markers are not claims
    assert T.ungrounded("lesson abcdef123456 says so", src) == []


def test_clear_rule_hit_costs_no_model_call_and_reply_is_grounded(root: Path) -> None:
    stub = Stub(["My biggest limit is wasted_cycles: loss 0.4."])
    c = _conv(root, stub)
    out = c.reply("what is limiting you")
    assert len(stub.calls) == 1                                        # speak only: the rules understood
    assert out.startswith("My biggest limit is wasted_cycles: loss 0.4.")
    assert out.endswith("[evidence: constraints.jsonl:wasted_cycles]")   # appended by us
    assert c.last["parsed"].intent == "constraints"
    assert c.talk_session.turn["spoken_by"] == "model"  # type: ignore[attr-defined]


def test_invented_number_falls_back_to_the_rule_report(root: Path) -> None:
    stub = Stub(["My biggest limit is wasted_cycles with loss 0.9."])
    c = _conv(root, stub)
    out = c.reply("what is limiting you")
    assert "What limits me, worst first:" in out and "0.9" not in out
    assert c.talk_session.turn["ungrounded"] == ["0.9"]  # type: ignore[attr-defined]


def test_model_routes_free_text_and_bad_json_falls_back_to_rules(root: Path) -> None:
    stub = Stub(['{"intent": "constraints"}', "wasted_cycles holds me back (loss 0.4)."])
    c = _conv(root, stub)
    c.reply("anything getting in your way lately?")
    assert c.last["parsed"].intent == "constraints"
    assert "JSON only" in stub.calls[0][0]["content"]                   # the router prompt
    stub2 = Stub(["no idea", ""])
    c2 = _conv(root, stub2)
    out = c2.reply("anything getting in your way lately?")
    assert c2.talk_session.turn["understand_by"].startswith("rules")  # type: ignore[attr-defined]
    assert out                                                           # the rules still answered


def test_open_question_retrieves_docs_and_records_never_private_text(root: Path) -> None:
    ix = T.index_for(root)
    hits = ix.search("how are model calls routed to the GPU pod")
    assert hits and hits[0]["id"] == "doc:creator/pulsething.py"
    assert not any("Masterstock" in d["text"] for d in ix.docs)          # private text is never indexed
    assert ix.search("planner duplicate packages")[0]["id"] == "lesson:abcdef123456"
    stub = Stub(fn=lambda m: '{"intent": "ask"}' if "JSON only" in m[0]["content"]
                else "It routes model calls to the rented GPU pod while its tunnel is healthy.")
    c = _conv(root, stub)
    out = c.reply("what does pulsething do?")
    assert c.last["parsed"].intent == "ask"
    assert out.endswith("]") and "doc:creator/pulsething.py" in out


def test_without_a_voice_layer2_still_answers(root: Path) -> None:
    v = T.Voice(Path("missing.gguf"), factory=lambda: (_ for _ in ()).throw(FileNotFoundError("no weights")))
    c = T.conversation(root, v, log=False)
    assert "What limits me" in c.reply("what is limiting you")
    assert "Nupen" in c.reply("hello")                                   # small talk by rule, rule speaker
    assert "pulsething" in c.reply("tell me about the pulsething routes")   # unknown -> search


def test_actions_stay_literal_and_need_confirmation(root: Path) -> None:
    stub = Stub(["Sure, paused!"])
    c = _conv(root, stub)
    out = c.reply("pause")
    assert "Yes or no?" in out and not stub.calls                        # never paraphrased, never a model call
    assert not (root / "state" / "creator" / "NUPEN_STOP").exists()


def test_history_is_kept_and_follow_ups_reach_the_router(root: Path) -> None:
    seen: list[str] = []

    def fn(m: list[dict[str, str]]) -> str:
        if "JSON" in m[0]["content"]:
            seen.append(m[-1]["content"])
            return '{"intent": "constraints"}'
        return "wasted_cycles, loss 0.4."
    c = _conv(root, Stub(fn=fn))
    c.reply("what is limiting you")
    c.reply("and why is that?")
    assert "Previous turn" in seen[-1] and "what is limiting you" in seen[-1]
    assert len(c.talk_session.history) == 2  # type: ignore[attr-defined]


def test_logged_exchange_carries_layer_and_voice(root: Path) -> None:
    c = T.conversation(root, _voice(Stub(["wasted_cycles, loss 0.4."])), log=True)
    c.reply("what is limiting you")
    rec = json.loads((root / "state" / "creator" / CV.CONVERSATIONS_FILE).read_text(encoding="utf-8").splitlines()[-1])
    assert rec["layer"] == 2 and rec["voice"] == "Qwen3-1.7B-Q4_K_M.gguf" and rec["spoken_by"] == "model"


def test_eval_questions_and_scoring(root: Path) -> None:
    assert len(TE.QUESTIONS) >= 50 and len({q.text for q in TE.QUESTIONS}) == len(TE.QUESTIONS)
    rep = TE.run_layer(root, "1", TE.QUESTIONS[:6], voice=None)
    assert rep["n"] == 6 and 0.0 <= rep["intent_acc"] <= 1.0 and rep["grounded"] == 1.0
    assert not (root / "state" / "creator" / "goal_proposals.jsonl").exists()      # the eval never drafts a real proposal


def test_state_verbs_must_match_the_record() -> None:
    nothing = "My last plan was made at T1. Chosen now: nothing.\n- not chosen: K28 exists: deferred"
    assert T.state_claims("I am working on K28.", nothing)                       # the 3 Oct overstatement
    assert T.state_claims("I considered K28; it is deferred and nothing is chosen right now.", nothing) == []
    chosen = "Chosen now: GAP-1 (K05 tested)."
    assert T.state_claims("I am working on K05.", chosen) == []
    assert T.state_claims("I am working on K28.", chosen)
    assert T.state_claims("I adopted CP0002.", "CP0002 not adopted: rejected") != []
    assert T.state_claims("I adopted CP0001.", "CP0001 adopted: merged") == []
    assert T.state_claims("I have learned a lot.", "Free RAM: 5 GB.") != []


def test_an_overstated_status_is_repaired_once_then_the_record_speaks(root: Path) -> None:
    st = root / "state" / "creator"
    _w(st / "plan_explanations.jsonl", json.dumps({"at": "T1", "node": "GAP-9", "component": "K28", "step": "exists", "chosen": False,
                                                    "slack_s": 0.0, "critical_path_s": 60.0, "why": "deferred"}) + "\n")
    stub = Stub(["I am working on K28.", "Nothing is chosen right now; I considered K28 and deferred it."])
    c = _conv(root, stub)
    out = c.reply("what are you working on")
    assert out.startswith("Nothing is chosen right now; I considered K28") and len(stub.calls) == 2
    assert "overstates" in stub.calls[1][-1]["content"]
    assert c.talk_session.turn["spoken_by"] == "model (corrected)"  # type: ignore[attr-defined]
    stub2 = Stub(["I am working on K28.", "I am busy with K28."])
    c2 = _conv(root, stub2)
    out2 = c2.reply("what are you working on")
    assert "Chosen now: nothing" in out2 and c2.talk_session.turn["spoken_by"] == "rules (voice overstated)"  # type: ignore[attr-defined]


def test_gpu_pairs_pipeline_keeps_only_verified_grounded_private_free_rows(root: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    from creator import talkdata as TD
    out = tmp_path_factory.mktemp("talkout")

    def chat(msgs: list[dict[str, str]], max_tokens: int, temp: float) -> str:
        sys_ = msgs[0]["content"]
        last = msgs[-1]["content"]
        if msgs[0]["role"] == "user" and "might type" in sys_:
            if "limiting" in sys_:
                return "what's holding you back?\nwhat slows you down these days?\n1. which bottleneck hurts most?\nMasterstock is limiting you?"
            return ""
        if msgs[0]["role"] == "user":
            return "what routes model calls to the pod?"
        if "JSON only" in sys_:
            if "bottleneck" in last or "slows" in last:
                return '{"intent": "constraints"}'
            if "routes" in last:
                return '{"intent": "ask"}'
            return '{"intent": "status"}'
        if "bottleneck" in last:
            return "wasted_cycles limits me, loss 0.9."                        # invented number: dropped
        if "routes" in last:
            return "pulsething routes model calls to the rented GPU pod."
        return "wasted_cycles limits me most (loss 0.4)."
    rep = TD.build(root, chat, out, per_intent=4, docs=1, per_doc=1, workers=2, model="stub")
    st = rep["stats"]
    assert st["held_out"] >= 1                                                  # 'what is limiting you' is an eval question
    assert st["private"] >= 1                                                   # the Masterstock line never left
    assert st["ungrounded"] >= 1
    rows = [json.loads(x) for x in (out / "talk_speak_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["intent"] for r in rows} <= {"constraints", "ask"} and rows
    assert all(r["messages"][-1]["role"] == "assistant" and "0.9" not in r["messages"][-1]["content"] for r in rows)
    dia = [json.loads(x) for x in (out / "talk_dialogues.jsonl").read_text(encoding="utf-8").splitlines()]
    assert dia and all(d["source"] == "talk-gpu" and d["reviewed"] is False and d["dialogue"][0]["role"] == "user" for d in dia)
    und = (out / "talk_understand_sft.jsonl").read_text(encoding="utf-8")
    assert "Masterstock" not in und and "Masterstock" not in (out / "talk_speak_sft.jsonl").read_text(encoding="utf-8")


def test_talk_jobs_is_a_valid_call_job() -> None:
    from creator import gpupulse as GP
    from creator import talkdata as TD
    jobs = TD.talk_jobs({})
    j = GP.parse_job(jobs[0])
    assert j["via"] == "call" and j["call"] == "creator.talkdata:pairs_job" and j["model"] == "Qwen3-14B-Q4_K_M.gguf"
    assert 5 <= j["minutes"] <= j["max_minutes"] <= 120


def test_chooser_stats_are_measured_only_when_asked_then_cached(root: Path, tmp_path_factory: pytest.TempPathFactory,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import shadow as SH
    monkeypatch.setenv("NUPEN_RUNTIME", str(tmp_path_factory.mktemp("rt")))
    calls: list[int] = []

    def fake(state: Path, les: list[Any]) -> dict[str, Any]:
        calls.append(1)
        return {"n": 3, "chooser_acc": 0.5, "default_acc": 0.25}
    monkeypatch.setattr(SH, "stats", fake)
    c = CV.Ctx(root, NOW)
    assert "not measured yet" in "\n".join(CV.h_lessons(c, {"question": "what have you learned"}).lines) and not calls
    assert "chooser accuracy 0.5 vs default 0.25." in "\n".join(CV.h_lessons(c, {"question": "how good is the chooser?"}).lines)
    assert "chooser accuracy 0.5" in "\n".join(CV.h_lessons(c, {"question": "what have you learned"}).lines) and len(calls) == 1
