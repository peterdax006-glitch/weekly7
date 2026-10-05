"""P0.8 team plumbing: routing, jurisdiction, envelopes, board, goal gate, cache, end-to-end on 20 synthetic tasks."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from creator import team as T
from creator.tools.index import Index

NAMES = ["parse_config", "load_ledger", "merge_rows", "score_patch", "rank_hits", "pack_lines", "split_words", "hash_blob",
         "fetch_state", "write_report", "trim_cache", "plan_steps", "check_budget", "sort_events", "join_paths", "scan_files",
         "count_tokens", "build_index", "drop_stale", "sum_costs"]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    for i, n in enumerate(NAMES):
        (r / "pkg" / f"m{i}.py").write_text(f"def {n}(x):\n    \"\"\"{n.replace('_', ' ')} helper\"\"\"\n    return x + {i}\n")
    return r


def fake_actors(ix, calls):
    def slot(name):
        def fn(env, team):
            calls.append((name, env.step))
            assert env.step in team.actors[name].accepts
            if name == "CODER" and env.step == "DEBUG_FIX" and "defer" in env.success_test:
                return "HANDOFF: THINKER"
            body = "".join((team.board.get(r) or {"body": ""})["body"] for r in env.inputs if r.startswith("b:"))
            return f"{name}:{env.step}:{len(body)}\n" + "ok"
        return fn
    def tests(env, team):
        calls.append(("tests", env.step))
        return "pass"
    acts = [T.actor("CHECKER", slot("CHECKER")), T.actor("CODER", slot("CODER")),
            T.actor("THINKER", slot("THINKER"), accepts=["PLAN", "DESIGN", "DEBUG_FIX"]),
            T.locate_actor(ix), T.actor("tests", tests)]
    return acts


def test_routing_is_deterministic_and_total():
    assert T.route("CODE") == "CODER" and T.route("LOCATE") == "locate" and T.route("PLAN") == "THINKER"
    assert all(T.route(s) == T.route(s) for s in T.ROUTES)
    with pytest.raises(T.Refused):
        T.route("NOPE")


def test_jurisdiction_refused_before_send(tmp_path, repo):
    calls: list = []
    ix = Index(repo, db=tmp_path / "ix.sqlite"); ix.update()
    team = T.Team(tmp_path / "st", fake_actors(ix, calls), goal_id="g1")
    env = T.Envelope("g1", "CODE")
    with pytest.raises(T.Refused):
        team.dispatch(env, actor_name="CHECKER")
    assert calls == [] and team.stats["out_of_jurisdiction"] == 1


def test_off_goal_and_bad_envelopes_refused(tmp_path, repo):
    ix = Index(repo, db=tmp_path / "ix.sqlite")
    team = T.Team(tmp_path / "st", fake_actors(ix, []), goal_id="g1")
    with pytest.raises(T.Refused):
        team.dispatch(T.Envelope("other", "SPEC"))
    assert team.stats["off_goal"] == 1
    with pytest.raises(T.Refused):
        team.dispatch(T.Envelope("g1", "SPEC", prior="a\nb\nc"))
    with pytest.raises(T.Refused):
        team.dispatch(T.Envelope("g1", "SPEC", inputs=["not a ref"]))
    with pytest.raises(T.Refused):
        team.dispatch(T.Envelope("g1", "SPEC", inputs=["b:abcdef012345"] * 200))   # oversize


def test_handoff_rerouted_and_refused_when_out_of_jurisdiction(tmp_path, repo):
    calls: list = []
    ix = Index(repo, db=tmp_path / "ix.sqlite")
    team = T.Team(tmp_path / "st", fake_actors(ix, calls), goal_id="g1")
    r = team.dispatch(T.Envelope("g1", "DEBUG_FIX", success_test="defer"))
    assert r.startswith("THINKER:DEBUG_FIX") and team.stats["handoffs"] == 1
    team.actors["CODER"].fn = lambda e, t: "HANDOFF: CHECKER"          # CHECKER does not accept CODE
    with pytest.raises(T.Refused):
        team.dispatch(T.Envelope("g1", "CODE"))


def test_board_and_cache(tmp_path):
    b = T.Board(tmp_path / "b.sqlite")
    i = b.put("g", "fact", "hello")
    assert i == b.put("g", "fact", "hello") and b.get(i)["body"] == "hello" and len(b.by_goal("g")) == 1
    c = T.ResultCache(b)
    e = T.Envelope("g", "SPEC", inputs=[i], success_test="make the parser handle empty config files without raising an error when the file is missing or unreadable or locked by another process on windows")
    assert c.get(e) is None
    c.put(e, "R")
    assert c.get(e) == ("R", "exact")
    e2 = T.Envelope("g", "SPEC", inputs=[i], success_test="make the parser handle empty config files without raising any error when the file is missing or unreadable or locked by another process on windows")
    assert c.get(e2) == ("R", "near")
    e3 = T.Envelope("g", "SPEC", inputs=[i], success_test="completely unrelated words about training a tokenizer on spanish text")
    assert c.get(e3) is None


def _tasks():
    return [{"query": NAMES[i].replace("_", " "), "success_test": f"test_{NAMES[i]} passes", "constraints": ["no new deps"],
             "want": NAMES[i]} for i in range(20)]


def test_end_to_end_20_tasks(tmp_path, repo):
    calls: list = []
    ix = Index(repo, db=tmp_path / "ix.sqlite"); ix.update()
    team = T.Team(tmp_path / "st", fake_actors(ix, calls), goal_id="g1")
    tasks = _tasks()
    found = 0
    for t in tasks:
        out = team.run_task(t, ix)
        found += any(t["want"] in ln or ln.startswith("pkg/") for ln in out["LOCATE"].splitlines())
        assert out["VALIDATE"] == "pass"
    s = team.stats
    med = statistics.median(s["env_tokens"])
    assert s["misroutes"] == 0 and s["out_of_jurisdiction"] == 0 and s["refused"] == 0
    assert med <= 300 and found == 20
    first_calls = s["slot_calls"]
    for t in tasks:                                                     # repeated batch: slot calls come from the cache
        team.run_task(t, ix)
    assert s["slot_calls"] == first_calls
    st = team.cache.stats()
    assert st["hit_rate"] >= 0.4
    print("E2E", {"median_env_tokens": med, "max_env_tokens": max(s["env_tokens"]), "dispatched": s["dispatched"], **st})
