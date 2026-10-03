"""Public-commit judgment topics (3 Oct 2026): public-only text, no outcome or future in a case or its prompt, retrieval and own-answer memory
strictly earlier, rounds of fresh subjects with the winners re-tested out of sample, the filler hands the topics out, scoring with n and CIs."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import judgment as J
from creator import publiccases as P
from creator import reasonmethods as RM
from creator import thinking as T


def synth(n: int = 160, repos: tuple[str, ...] = ("alpha", "beta")) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r, name in enumerate(repos):
        cs = []
        for i in range(n):
            f = {f"src/m{i % 7}.py"} if i % 3 else {f"docs/d{i % 5}.md"}
            s = f"fix crash in m{i % 7} ({name} c{i:04d})" if i % 4 == 0 else f"add feature {name} c{i:04d}"
            cs.append({"h": f"{name[0]}{i:09d}abcdef", "t": 1_000_000.0 + i * 600 + r * 7, "s": s, "files": f, "lines": 5 + i % 40})
        out[name] = cs
    return out


@pytest.fixture()
def pub(monkeypatch: Any) -> dict[str, list[dict[str, Any]]]:
    data = synth()
    monkeypatch.setattr(P, "public_repos", lambda: [Path(k) for k in data])
    monkeypatch.setattr(P, "commits", lambda repo: data[Path(repo).name])
    monkeypatch.setattr(J, "_STAT_CACHE", {})
    return data


# ------------------------------------------------------------------------------------------------ public only, no author, no body
def _git(repo: Path, *args: str, env: Any = None) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _repo(path: Path, n: int) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    for i in range(n):
        (path / f"f{i % 3}.py").write_text(f"x = {i}\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="Jane Secretname", GIT_AUTHOR_EMAIL="jane@private.example", GIT_COMMITTER_NAME="Jane Secretname",
                   GIT_COMMITTER_EMAIL="jane@private.example", GIT_COMMITTER_DATE=f"{1_600_000_000 + i * 60} +0000",
                   GIT_AUTHOR_DATE=f"{1_600_000_000 + i * 60} +0000")
        _git(path, "add", "-A", env=env)
        _git(path, "commit", "-q", "-m", f"fix thing {i} thanks @octocat mail bob@corp.example", "-m", "BODYSECRET line", env=env)


def test_text_is_read_only_from_public_repos_and_keeps_no_author_or_body(tmp_path: Path, monkeypatch: Any) -> None:
    rt = tmp_path / "rt"
    monkeypatch.setenv("NUPEN_RUNTIME", str(rt))
    _repo(rt / "public_repos" / "openproj", 6)
    _repo(tmp_path / "oldpc" / "privproj", 4)
    _repo(rt / "public_repos" / "cloning.tmp", 2)                              # an unfinished clone is not read
    assert [r.name for r in P.public_repos()] == ["openproj"]
    with pytest.raises(PermissionError):
        P.refresh(tmp_path / "oldpc" / "privproj")
    assert not P.is_public(rt / "public_repos" / "cloning.tmp") and not P.is_public(tmp_path / "oldpc" / "privproj")
    assert P.refresh_all() == {"openproj": 6}
    blob = P.text_cache_path(rt / "public_repos" / "openproj").read_text(encoding="utf-8")
    assert "Jane" not in blob and "Secretname" not in blob and "private.example" not in blob and "BODYSECRET" not in blob
    assert "corp.example" not in blob and "@octocat" not in blob and "fix thing 5" in blob
    assert P.text_cache_path(rt / "public_repos" / "openproj").is_relative_to(rt)            # outside the Nupen repository
    # a cache file planted under a private repository's name is never read: commits() of a non-public repo is empty
    P.text_cache_path(tmp_path / "oldpc" / "privproj").write_text(json.dumps({"h": "x", "t": 1.0, "s": "PRIVATE SUBJECT", "files": ["a"]}) + "\n")
    assert P.commits(tmp_path / "oldpc" / "privproj") == []
    texts = [x[3] for x in P.raw_cases("pub_git_churn")]
    assert texts and not any("PRIVATE SUBJECT" in t or "privproj" in t for t in texts)
    # incremental: new commits are appended, nothing re-read twice
    p = rt / "public_repos" / "openproj"
    env = dict(os.environ, GIT_AUTHOR_NAME="A", GIT_AUTHOR_EMAIL="a@b.c", GIT_COMMITTER_NAME="A", GIT_COMMITTER_EMAIL="a@b.c")
    (p / "new.py").write_text("y = 1\n")
    _git(p, "add", "-A", env=env)
    _git(p, "commit", "-q", "-m", "late commit", env=env)
    assert P.refresh(p) == 1 and P.refresh(p) == 0 and len(P.commits(p)) == 7


def test_case_text_never_depends_on_the_future(pub: dict[str, list[dict[str, Any]]], monkeypatch: Any) -> None:
    """Changing everything after a commit changes its outcome but never its setup text; outcomes match drillsources' git events."""
    for topic in P.TOPICS:
        before = {x[1].subject: x for x in P.raw_cases(topic)}
        mode, w = P.MODE[topic]
        ev = D._git_events(pub["alpha"], w, mode)
        assert [before[f"alpha:{e.subject}"][1].y for e in ev if e.resolved is not None] == [e.y for e in ev if e.resolved is not None]
        alt = {k: [dict(c) for c in v] for k, v in pub.items()}
        for c in alt["alpha"][80:]:
            c["s"], c["files"] = "revert everything", {"src/m1.py", "src/m2.py", "src/m3.py", "docs/d0.md", "docs/d1.md"}
        monkeypatch.setattr(P, "commits", lambda repo: alt[Path(repo).name])
        after = {x[1].subject: x for x in P.raw_cases(topic)}
        monkeypatch.setattr(P, "commits", lambda repo: pub[Path(repo).name])
        for k, (_r, it, _m, text) in before.items():
            if k.startswith("alpha:") and int(k.split(":")[1][1:10]) < 80:
                assert after[k][3] == text                             # setup unchanged
                assert "fixed" not in text.split("?")[-1] and str(it.resolved) not in text
        assert any(after[k][1].y != before[k][1].y for k in before if k.startswith("alpha:"))     # the outcomes did depend on the future


def test_public_prompt_has_nothing_created_or_resolved_at_or_after_the_case(pub: dict[str, list[dict[str, Any]]]) -> None:
    cs = J.load_cases("pub_git_fixed", Path("nostate"), Path("norepo"))
    assert len(cs) > 250 and all(c.group.split("/")[0] in ("alpha", "beta") for c in cs)
    idx = RM.Index(RM.build_corpus(Path("nostate"), cs))
    recs = [{"topic": c.topic, "subject": c.subject, "strategy": J.REF_STRATEGY, "created": c.created, "resolved": c.resolved, "y": c.y,
             "p": 0.9 if c.y else 0.1} for c in cs]                     # its own answers, including the case's own and later ones
    own = J.own_index(recs, cs)
    c = cs[200]
    tag = c.subject.split(":")[1]

    def marker(x: J.Case) -> str:                                 # the unique words of a case's commit subject
        name, h = x.subject.split(":")
        return f"{name} c{int(h[1:10]):04d}"
    for s in J.PUB_STRATEGIES:
        if "knn" in s:
            continue
        blob = json.dumps(J.build_prompt(s, cs, c, idx, own))
        shown = [x for x in cs if x is not c and marker(x) in blob]
        assert all(x.resolved < c.created for x in shown), s             # nothing created or resolved at/after the case
        if s.get("retrieve") or s.get("shots") or s.get("own"):
            assert shown, s
        assert blob.count(marker(c)) == 1 and tag not in blob and str(c.resolved) not in blob
    assert "Your own earlier forecasts" in json.dumps(J.build_prompt({"shots": 0, "hint": 1, "own": 6}, cs, c, idx, own))
    got = own.search(c.text, c.created, 50, kinds=("answer",), topic=c.topic)
    assert got and all(r.known < c.created for r in got)


def test_calibration_matches_the_exact_refit_on_short_records() -> None:
    recs: list[dict[str, Any]] = [{"subject": f"s{i}", "created": 100.0 + i, "resolved": 100.0 + i + (i % 7), "y": int(i % 3 == 0), "p": 0.2 + 0.6 * ((i * 37) % 11) / 10}
            for i in range(120)]
    fast = J.calibrated(recs)
    for r, p, ok in fast:
        prior = [x for x in recs if x["resolved"] < r["created"]]
        if len(prior) >= J.MIN_CAL:
            a, b = J.platt_fit([J._logit(x["p"]) for x in prior], [x["y"] for x in prior])
            assert ok and math.isclose(p, min(0.97, max(0.03, 1 / (1 + math.exp(-(a * J._logit(r["p"]) + b))))), rel_tol=1e-9)
        else:
            assert not ok and p == r["p"]


# ------------------------------------------------------------------------------------------------ rounds and the filler
class StratLLM:
    """A fake model that knows the truth (an ORACLE, test only) but uses it only when the prompt is a plain retrieval prompt: a deterministic
    winner for the round mechanics."""
    model = "fake-thinker.gguf"
    oracle: dict[str, int] = {}

    def __enter__(self) -> "StratLLM":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def chat(self, messages: Any, **kw: Any) -> str:
        blob = json.dumps(messages)
        q = messages[-1]["content"].replace("\n/no_think", "")
        y = next((v for k, v in self.oracle.items() if q.endswith(k)), None)
        if y is not None and "Similar past cases" in blob and "Your own" not in blob and "QUESTION" not in blob:
            return f"PROBABILITY: {0.8 if y else 0.2}"
        return "PROBABILITY: 0.5"


def test_filler_hands_out_public_rounds_and_retests_the_winner(tmp_path: Path, pub: dict[str, list[dict[str, Any]]], monkeypatch: Any) -> None:
    st = tmp_path / "st"
    monkeypatch.setattr(J, "active_tag", lambda: "fake-thinker.gguf")
    monkeypatch.setattr(J, "ROUND_SIZE", 24)
    monkeypatch.setattr(J, "PUB_HALVING", ((6, 6), (12, 3), (24, 1)))
    monkeypatch.setattr(J, "PUB_ALLOCATION", "halving")                  # the plain halving mechanics (racing / online: the tests below)
    monkeypatch.setattr(J, "PUB_SPAN", 60)
    monkeypatch.setattr(J, "PUB_REFRESH_S", 1e12)
    monkeypatch.setattr(StratLLM, "oracle", {c.text: c.y for c in J.load_cases("pub_git_fixed", st, tmp_path)})
    nj = J.judgment_filler(st, tmp_path, max_servers=2, llm_factory=StratLLM, topics=("pub_git_fixed",))
    for _ in range(400):
        j = nj()
        if j is None:
            break
        j()
        rounds = J.load_rounds(st, "pub_git_fixed", "fake-thinker.gguf")
        if len(rounds) >= 2 and J.round_state(rounds[1], T._jsonl(J.path(st)))["winner"] is not None:
            break
    rounds = J.load_rounds(st, "pub_git_fixed", "fake-thinker.gguf")
    assert len(rounds) >= 2
    r0, r1 = rounds[0], rounds[1]
    assert not set(r0["subjects"]) & set(r1["subjects"])                     # every round is fresh
    assert {k.split(":")[0] for k in r0["subjects"]} == {"alpha", "beta"}       # repositories interleaved
    cs = {c.subject: c for c in J.load_cases("pub_git_fixed", st, tmp_path)}
    for name in ("alpha", "beta"):                                             # forward in time: the next round comes after this one
        assert max(cs[k].created for k in r0["subjects"] if k.startswith(name)) < min(cs[k].created for k in r1["subjects"] if k.startswith(name))
    recs = [r for r in T._jsonl(J.path(st)) if r.get("topic") == "pub_git_fixed"]
    w0 = J.round_state(r0, recs)["winner"]
    assert w0 is not None and w0.get("retrieve") and not w0.get("structured")
    assert w0 in r1["pinned"] and J.REF_STRATEGY in r1["pinned"] and {"knn": 8} in r1["pinned"]
    asked_w0 = {r["subject"] for r in recs if r.get("round") == 1 and r["strategy"] == w0}
    assert asked_w0 == set(r1["subjects"])                                     # the pinned winner answered every fresh subject
    rep = J.rounds_report(st, "pub_git_fixed", "fake-thinker.gguf")
    rt = rep[1]["retest"][0]
    assert rt["strategy"] == w0 and rt["out_of_sample_gain_vs_reference"][-1] == len(r1["subjects"]) and rt["out_of_sample_gain_vs_reference"][0] > 0
    assert rt["in_sample_gain_when_chosen"] is not None
    per: dict[str, set[str]] = {}
    for r in recs:
        per.setdefault(json.dumps(r["strategy"], sort_keys=True), set()).add(r["subject"])
    assert min(len(v) for v in per.values()) >= 6                               # no strategy is judged on fewer than the first cutoff


def test_filler_alternates_own_and_public_topics(tmp_path: Path, pub: dict[str, list[dict[str, Any]]], monkeypatch: Any) -> None:
    st = tmp_path / "st"
    real = J.load_cases
    own = [J.Case("verdict", f"p{i}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, i % 2, "K", f"case {i}") for i in range(30)]
    monkeypatch.setattr(J, "load_cases", lambda t, s, r: own if t == "verdict" else (real(t, s, r) if t in J.PUB_TOPICS else []))
    monkeypatch.setattr(J, "active_tag", lambda: "fake-thinker.gguf")
    monkeypatch.setattr(J, "PUB_REFRESH_S", 1e12)
    nj = J.judgment_filler(st, tmp_path, max_servers=1, llm_factory=StratLLM)
    topics = []
    for _ in range(6):
        j = nj()
        assert j is not None
        j()
        topics.append(T._jsonl(J.path(st))[-1]["topic"])
    assert topics[:3] == ["verdict", "pub_git_fixed", "pub_git_churn"] and topics[3] == "verdict"
    rows = T._jsonl(J.path(st))
    assert sum(1 for r in rows if r["topic"] == "pub_git_fixed") == 2 * J.PUB_BATCH and all("round" in r for r in rows if r["topic"] in J.PUB_TOPICS)
    assert all(r["model"] == "fake-thinker.gguf" for r in rows)
    assert J.PUB_TOPICS == P.TOPICS


def test_trust_section_reports_public_topics_replay_only_with_n_and_cis(tmp_path: Path, pub: dict[str, list[dict[str, Any]]],
                                                                       monkeypatch: Any) -> None:
    st = tmp_path / "st"
    monkeypatch.setattr(J, "active_tag", lambda: "fake-thinker.gguf")
    monkeypatch.setattr(J, "ROUND_SIZE", 30)
    monkeypatch.setattr(J, "PUB_REFRESH_S", 1e12)
    monkeypatch.setattr(J, "load_cases", (lambda real: lambda t, s, r: real(t, s, r) if t == "pub_git_fixed" else [])(J.load_cases))
    nj = J.judgment_filler(st, tmp_path, max_servers=1, llm_factory=StratLLM, topics=("pub_git_fixed",))
    for _ in range(30):
        j = nj()
        if j is None:
            break
        j()
    sec = J.trust_section(st, tmp_path)
    s = sec["pub_git_fixed"]
    assert s["trusted"] is False and s["replay_only"] and s["rounds"] and s["n"] > 0
    row = next(iter(s["strategies"].values()))
    assert row["n"] > 0 and len(row["brier_ci"]) == 2 and row["brier_ci"][0] <= row["brier_calibrated"] <= row["brier_ci"][1] + 1e-9
    assert len(row["gain_vs_statistical"]) == 3 and "verdict" not in sec


class WrongPlainLLM(StratLLM):
    """Like StratLLM, but every prompt without retrieval answers CONFIDENTLY WRONG: clearly worse than the statistical predictor."""
    def chat(self, messages: Any, **kw: Any) -> str:
        blob = json.dumps(messages)
        q = messages[-1]["content"].replace("\n/no_think", "")
        y = next((v for k, v in self.oracle.items() if q.endswith(k)), None)
        if y is not None and "Similar past cases" not in blob:
            return f"PROBABILITY: {0.05 if y else 0.95}"
        return super().chat(messages, **kw)


def _run_rounds(tmp_path: Path, monkeypatch: Any, allocation: str, rounds_needed: int = 2, llm: Any = StratLLM) -> tuple[Path, list[dict[str, Any]]]:
    st = tmp_path / "st"
    monkeypatch.setattr(J, "active_tag", lambda: "fake-thinker.gguf")
    monkeypatch.setattr(J, "ROUND_SIZE", 24)
    monkeypatch.setattr(J, "PUB_HALVING", ((6, 6), (12, 3), (24, 1)))
    monkeypatch.setattr(J, "PUB_ALLOCATION", allocation)
    monkeypatch.setattr(J, "ONLINE_MIN_N", 4)
    monkeypatch.setattr(J, "PUB_SPAN", 60)
    monkeypatch.setattr(J, "PUB_REFRESH_S", 1e12)
    monkeypatch.setattr(StratLLM, "oracle", {c.text: c.y for c in J.load_cases("pub_git_fixed", st, tmp_path)})
    nj = J.judgment_filler(st, tmp_path, max_servers=2, llm_factory=llm, topics=("pub_git_fixed",))
    for _ in range(600):
        j = nj()
        if j is None:
            break
        j()
        rounds = J.load_rounds(st, "pub_git_fixed", "fake-thinker.gguf")
        if len(rounds) >= rounds_needed and J.round_state(rounds[rounds_needed - 1], T._jsonl(J.path(st)))["winner"] is not None:
            break
    return st, J.load_rounds(st, "pub_git_fixed", "fake-thinker.gguf")


def test_racing_rounds_pair_with_the_stored_statistical_predictor_and_drop_clear_losers(tmp_path: Path, pub: dict[str, list[dict[str, Any]]],
                                                                                       monkeypatch: Any) -> None:
    from creator import trialerror as TE
    monkeypatch.setattr(TE, "RACE_MIN", 5)
    st, rounds = _run_rounds(tmp_path, monkeypatch, "racing", llm=WrongPlainLLM)
    assert len(rounds) >= 2 and rounds[0]["allocation"] == "racing"
    stat = {p.subject: p.p for p in D.walk_forward(P.items("pub_git_fixed"), "pub_git_fixed")}
    assert set(rounds[0]["stat"]) == set(rounds[0]["subjects"]) and all(abs(rounds[0]["stat"][k] - stat[k]) < 1e-3 for k in rounds[0]["subjects"])
    recs = [r for r in T._jsonl(J.path(st)) if r.get("topic") == "pub_git_fixed"]
    s0 = J.round_state(rounds[0], recs)
    assert s0["dropped"] and all(d not in rounds[0]["pinned"] for d in s0["dropped"])        # the 0.5-answering prompts lose to the predictor
    dkeys = {json.dumps(d, sort_keys=True) for d in s0["dropped"]}
    g = J.online_gains(rounds[0], s0["recs"], s0["dropped"])
    assert all(TE.raced_out(g[k]) for k in dkeys)
    w0 = s0["winner"]
    assert w0 is not None and w0 in rounds[1]["pinned"]                                       # promoted only through the next round's re-test
    rep = J.rounds_report(st, "pub_git_fixed", "fake-thinker.gguf")
    rt = rep[1]["retest"][0]
    assert rt["strategy"] == w0 and rt["out_of_sample_gain_vs_statistical"][-1] == len(rounds[1]["subjects"])
    assert not set(rounds[0]["subjects"]) & set(rounds[1]["subjects"])


def test_online_rounds_ask_pinned_plus_thompson_picks_and_retest_the_winner(tmp_path: Path, pub: dict[str, list[dict[str, Any]]],
                                                                           monkeypatch: Any) -> None:
    st, rounds = _run_rounds(tmp_path, monkeypatch, "online")
    assert len(rounds) >= 2 and rounds[0]["allocation"] == "online"
    recs = [r for r in T._jsonl(J.path(st)) if r.get("topic") == "pub_git_fixed"]
    for rnd in rounds[:2]:
        per: dict[str, set[str]] = {}
        for r in recs:
            if r.get("round") == rnd["round"]:
                per.setdefault(r["subject"], set()).add(json.dumps(r["strategy"], sort_keys=True))
        pins = {json.dumps(s, sort_keys=True) for s in rnd["pinned"]}
        assert per and all(pins <= ss and len(ss - pins) <= J.ONLINE_PICKS for ss in per.values())
    w0 = J.round_state(rounds[0], recs)["winner"]
    assert w0 is not None and w0 in rounds[1]["pinned"]
    asked = {r["subject"] for r in recs if r.get("round") == 1 and r["strategy"] == w0}
    assert asked == set(rounds[1]["subjects"])                                               # the out-of-sample re-test on every fresh subject
    rep = J.online_report(st, "pub_git_fixed", "fake-thinker.gguf")
    assert rep["allocation"] == "online" and rep["gains_vs_control"]


def test_anchor_prompt_carries_the_walk_forward_probability(tmp_path: Path, pub: dict[str, list[dict[str, Any]]]) -> None:
    cases = J.load_cases("pub_git_fixed", tmp_path, tmp_path)
    c = cases[120]
    ap = J._anchor_p(tmp_path, c)
    stat = {p.subject: p.p for p in D.walk_forward(P.items("pub_git_fixed"), "pub_git_fixed")}
    assert ap is not None and abs(ap - stat[c.subject]) < 1e-9
    msgs = J.build_prompt(J.ANCHOR_STRATEGY, cases, c, None, None, ap)
    assert f"P = {ap:.2f}" in msgs[-1]["content"]
    assert "P = " not in J.build_prompt(J.REF_STRATEGY, cases, c, None, None, ap)[-1]["content"]
